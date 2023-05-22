import os
import random
import numpy as np
import pickle
import torch
import logging
from tqdm import tqdm, trange
from models.model import ASAP
from utils.data_utils import *
from utils.evaluation_utils import model_evaluation, round_pre
from utils.setup_utils import set_seed
from transformers import AdamW, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler


def train(args):
    print('[TRAIN]')

    data_name = args.data
    name = f'{data_name}/DL{args.max_dial_len}_TL{args.max_seq_len}_BS{args.batch_size}_EP{args.epoch_num}_LR{args.lr}_gm{args.gamma}_TF{args.tf_heads}_{args.tf_layers}_HP{args.hp_heads}_{args.hp_layers}'
    print('TRAIN ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO, 
                        filename=save_path + '/log.txt', filemode='a')
    logging.info(args)
    
    # random seed
    set_seed(args.seed)
    
    # data preparation
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    data = load_data(args, tokenizer)
    
    train_data = DataFrame(data['train'], args)
    batch_size = args.batch_size # * max(1, len(args.device_id))
    num_train_steps = int(len(data['train']['input_ids']) / batch_size * args.epoch_num)
    
    logging.info("***** Run training *****")
    logging.info(" Num of examples = %d", len(data['train']['input_ids']))
    logging.info(" Overall batch size = %d", batch_size)
    logging.info(" Num of steps = %d", num_train_steps)
    
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data,
                              sampler=train_sampler, 
                              batch_size=batch_size, 
                              collate_fn=collate_fn)
    
    dev_data = DataFrame(data['valid'], args)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_data = DataFrame(data['test'], args)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # build model
    act_num = len(data['act_list'])
    args.act_num = act_num
    logging.info(" Num of acts = %d", act_num)
    sat_num = args.sat_num
    model = ASAP(args=args, act_num=act_num, sat_num=sat_num)
    model.to(args.device)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    
    ## prepare optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                int(num_train_steps * args.warmup), 
                                                num_train_steps)

    logging.info(optimizer)
    
    act_best_result = [0. for _ in range(4)]
    sat_best_result = [0. for _ in range(8)]
    trained_steps = 0
    epoch_loss = [[] for _ in range(2)]
    for i in trange(args.epoch_num, desc='Epoch'):
        logging.info('train epoch, {}, {}'.format(i, name))
        for j, batch in enumerate(tqdm(train_loader)):
            model.train()
            input_ids = batch['input_ids'].to(args.device)
            act_seq = batch['act_seq'].to(args.device)
            sat = batch['sat'].to(args.device)
            mask = batch['mask'].to(args.device)
            
            if act_num <= 1: # no act prediction task
                act_seq = None
            
            _, _, sat_loss, act_loss = model(input_ids=input_ids, 
                                             act_seq=act_seq, 
                                             sat=sat,
                                             mask=mask,
                                             use_hp=args.use_hp
                                            )

            if len(args.device_id) > 1:
                act_loss = act_loss.mean()  # mean() to average on multi-gpu parallel training
                sat_loss = sat_loss.mean()

            epoch_loss[0].append(sat_loss.item())
            epoch_loss[1].append(act_loss.item())
            
            loss = sat_loss + args.gamma * act_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            trained_steps += 1
            
            if trained_steps % args.eval_steps == 0:
                epoch_loss = [np.mean(x) for x in epoch_loss]
                logging.info(f'sat loss: {epoch_loss[0]}, act loss: {epoch_loss[1]}, total loss: {epoch_loss[0]+args.gamma*epoch_loss[1]}')
                epoch_loss = [[] for _ in range(2)]
                # evaluation during training
                sat_dev_result, act_dev_result = model_evaluation(model, dev_loader, args)
                if sat_dev_result[-1] > sat_best_result[-1]:
                    sat_best_result = sat_dev_result
                    act_best_result = act_dev_result
                    model_to_save = model.module if hasattr(model,
                                              'module') else model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), save_path+'/best_model.bin')
                logging.info('UAR\tkappa\trho\tbi_f1\tacc\tprecision\trecall\tf1')
                logging.info(f'sat: dev_result={round_pre(sat_dev_result, 4)}, act: dev_result={act_dev_result}')
                logging.info(f'sat: best_dev_result={round_pre(sat_best_result, 4)}, act: best_dev_result={act_best_result}')
                sat_test_result, act_test_result = model_evaluation(model, test_loader, args)
                logging.info(f'sat: test_result={round_pre(sat_test_result, 4)}, act: test_result={act_test_result}')
                logging.info('\n')
                
    # best model evaluation
    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_model.bin')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    sat_test_result, act_test_result =  model_evaluation(model, test_loader, args)
    logging.info(f'sat: final test_result={round_pre(sat_test_result, 4)}, act: final test_result={act_test_result}')


def evaluate(args):
    print('[EVALUATE]')

    data_name = args.data
    name = f'{data_name}/DL{args.max_dial_len}_TL{args.max_seq_len}_BS{args.batch_size}_EP{args.epoch_num}_LR{args.lr}_gm{args.gamma}_TF{args.tf_heads}_{args.tf_layers}_HP{args.hp_heads}_{args.hp_layers}'
    print('EVAL ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO, 
                        filename=save_path + '/log.txt', filemode='a')
    logging.info(args)
    
    # random seed
    set_seed(args.seed)
    
    # data preparation
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    data = load_data(args, tokenizer)
    
    train_data = DataFrame(data['train'], args)
    batch_size = args.batch_size # * max(1, len(args.device_id))
    num_train_steps = int(len(data['train']['input_ids']) / batch_size * args.epoch_num)
    
    logging.info("***** Run training *****")
    logging.info(" Num of examples = %d", len(data['train']['input_ids']))
    logging.info(" Overall batch size = %d", batch_size)
    logging.info(" Num of steps = %d", num_train_steps)
    
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data,
                              sampler=train_sampler, 
                              batch_size=batch_size, 
                              collate_fn=collate_fn)
    
    dev_data = DataFrame(data['valid'], args)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_data = DataFrame(data['test'], args)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # build model
    act_num = len(data['act_list'])
    args.act_num = act_num
    logging.info(" Num of acts = %d", act_num)
    sat_num = args.sat_num
    model = ASAP(args=args, act_num=act_num, sat_num=sat_num)
    model.to(args.device)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
                
    # best model evaluation
    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_model.bin')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        
    sat_test_results, _ =  model_evaluation(model, test_loader, args)
    print(sat_test_results)
#     sat_test_results =  model_evaluation_turns(model, test_loader, args)
#     for ll in range(3, 16):
#         logging.info(f'turn={ll}')
#         logging.info(f'sat: final test_result={round_pre(sat_test_results[str(ll)], 4)}')
