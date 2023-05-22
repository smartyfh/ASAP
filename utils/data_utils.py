import random
import torch
import numpy as np
import os
import pickle
import json
import re
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_data(args, tokenizer):
    dirname = f'dataset/{args.data}'
    print(dirname)

    if not os.path.exists(f'{dirname}/tokenized'):
        os.makedirs(f'{dirname}/tokenized')

    if os.path.exists(f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl') and not args.rewrite_data:
        return read_pkl(f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl')

    print('tokenize data')

    act_list = {}
    with open(os.path.join(dirname, f'act_{args.data}.txt'), 'r', encoding='utf-8') as infile:
        for line in infile:
            items = line.strip('\n').split('\t')
            act_list[items[0]] = items[1]

    data = {'act_list': act_list}
    for set_name in ['train', 'valid', 'test']:
        max_utt_len = 0
        max_dia_len = 0
        avg_utt_len = []
        avg_dia_len = []
        data[set_name] = {'input_ids':[], 'input_text':[], 'act_seq':[], 'sat':[], 'mask':[]}
        with open(os.path.join(dirname, f'{set_name}_{args.data}.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                items = line.strip('\n').split('\t')
                input_text = eval(items[0])
                act_seq = eval(items[1])
                sat = int(items[2]) 
                input_ids = []
                for text in input_text:
                    ids = []
                    for sent in text.split('|||'):
                        ids += tokenizer.encode(sent)[1:]
                        if len(ids) >= args.max_seq_len-1:
                            ids = ids[:args.max_seq_len-2] + [102]
                            break
                    
                    avg_utt_len.append(len(ids)+1)
                    max_utt_len = max(max_utt_len, len(ids)+1)
                    input_ids.append([101] + ids) # [CLS] + (max_len-1) tokens
                
                avg_dia_len.append(len(input_ids))
                max_dia_len = max(max_dia_len, len(input_ids))
                data[set_name]['input_ids'].append(input_ids)
                data[set_name]['input_text'].append(input_text)
                data[set_name]['act_seq'].append(act_seq)
                data[set_name]['sat'].append(sat)
                data[set_name]['mask'].append(list(np.arange(len(input_text))))
        print('{} set, max_utt_len: {}, max_dia_len: {}, avg_utt_len: {}, avg_dia_len: {}'.format(set_name, max_utt_len, max_dia_len, float(sum(avg_utt_len))/len(avg_utt_len), float(sum(avg_dia_len))/len(avg_dia_len)))

    write_pkl(data, f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl')

    return data
    

class DataFrame(Dataset):
    def __init__(self, data, args):
        self.input_ids = data['input_ids']
        self.act_seq = data['act_seq']
        self.sat = data['sat']
        self.mask = data['mask']
        self.max_len = args.max_dial_len
    
    def __getitem__(self, index):
        return self.input_ids[index][-self.max_len:], self.act_seq[index][-self.max_len:], self.sat[index], self.mask[index][-self.max_len:]
    
    def __len__(self):
        return len(self.input_ids)
    

def collate_fn(data):
    input_ids, act_seq, sat, mask = zip(*data)
    batch_size = len(input_ids)
    act_seq = [torch.tensor(item).long() for item in act_seq]
    act_seq = pad_sequence(act_seq, batch_first=True, padding_value=-1)
    mask = [torch.tensor(item).long() for item in mask]
    mask = pad_sequence(mask, batch_first=True, padding_value=-1)
    dialog_len = len(mask[0])
    assert dialog_len == mask.size(1)

    pad_input_ids = []
    for dialog in input_ids:
        x = dialog + [[101, 102]] * (dialog_len - len(dialog))
        pad_input_ids.append(x)
    input_ids = [torch.tensor(item).long() for dialog in pad_input_ids for item in dialog]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.view(batch_size, dialog_len, -1)

    return {'input_ids': input_ids,
            'act_seq': act_seq,
            'sat': torch.tensor(sat).long(),
            'mask': mask.ne(-1)
           }
