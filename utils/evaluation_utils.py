from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score, precision_score, recall_score
from .spearman import spearman
from tqdm import tqdm
import torch


def round_pre(tp, de):
    tp = list(tp)
    new_tp = []
    for p in tp:
        new_tp.append(round(p, de))
    return tuple(new_tp)


def sat_evaluation(pred, label, sat_num):
    acc = sum([int(p == l) for p, l in zip(pred, label)]) / len(label)
    precision = precision_score(label, pred, average='macro', zero_division=0)
    sk_recall = recall_score(label, pred, average='macro', zero_division=0)
    f1 = f1_score(label, pred, average='macro', zero_division=0)
#     sat_result = (acc, precision, sk_recall, f1)
    
    recall = [[0, 0] for _ in range(sat_num)]
    for p, l in zip(pred, label):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]
    
    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(pred, label)
    rho = spearman(pred, label)

    bi_pred = [int(item < sat_num//2) for item in pred]
    bi_label = [int(item < sat_num//2) for item in label]
    bi_recall = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if l == 1]) / max(bi_label.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if p == 1]) / max(bi_pred.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)
        
    sat_result = (UAR, kappa, rho, bi_f1, acc, precision, sk_recall, f1)
    return sat_result
  

def act_evaluation(act_pred, act_label):
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for ap, al in zip(act_pred, act_label):
        al = [x for x in al if x != -1]
        ap = ap[:len(al)]
        acc_list.append(sum([int(p == l) for p, l in zip(ap, al)]) / len(al))
        precision_list.append(precision_score(al, ap, average='macro', zero_division=0))
        recall_list.append(recall_score(al, ap, average='macro', zero_division=0))
        f1_list.append(f1_score(al, ap, average='macro', zero_division=0))
    acc = sum(acc_list)/len(acc_list)
    precision = sum(precision_list)/len(precision_list)
    recall = sum(recall_list)/len(recall_list)
    f1 = sum(f1_list)/len(f1_list)
    act_result = (acc, precision, recall, f1)
    return round_pre(act_result, 4)


def act_evaluation_v2(pred, label):
    acc = sum([int(p == l) for p, l in zip(pred, label)]) / len(label)
    precision = precision_score(label, pred, average='macro', zero_division=0)
    recall = recall_score(label, pred, average='macro', zero_division=0)
    f1 = f1_score(label, pred, average='macro', zero_division=0)
    
    act_result = (acc, precision, recall, f1)
    return round_pre(act_result, 4)

    
def model_evaluation(model, test_loader, args):
    act_pred = []
    sat_pred = []
    act_label = []
    sat_label = []

    model.eval()
    for i, batch in enumerate(tqdm(test_loader)):
        input_ids = batch['input_ids'].to(args.device)
        act_seq = batch['act_seq'].to(args.device)
        sat = batch['sat'].to(args.device)
        mask = batch['mask'].to(args.device)
        if args.act_num <= 1:
            act_seq = None
        with torch.no_grad():
            sat_probs, act_probs, _, _ = model(input_ids=input_ids, 
                                               act_seq=act_seq,
                                               sat=sat,
                                               mask=mask,
                                               use_hp=args.use_hp
                                               )
            last_index = torch.sum(mask, dim=-1, keepdim=True) - 1
            sat_prob = torch.gather(sat_probs, 1, last_index[:, :, None].expand(-1, -1, sat_probs.size(-1)))
            sat_prob = sat_prob.squeeze(1)
            
#             act_prob = torch.gather(act_probs, 1, last_index[:, :, None].expand(-1, -1, act_probs.size(-1)))
#             act_prob = act_prob.squeeze(1)
#             act = torch.gather(act_seq, 1, last_index)
                
        sat_pred.extend(sat_prob.argmax(dim=-1).view(-1).cpu().tolist())
        sat_label.extend(sat.view(-1).cpu().tolist())
        
        if args.act_num > 1:
            act_pred.extend(act_probs.argmax(dim=-1).cpu().tolist())
            act_label.extend(act_seq.cpu().tolist())
#             act_pred.extend(act_prob.argmax(dim=-1).view(-1).cpu().tolist())
#             act_label.extend(act.view(-1).cpu().tolist())
            
    # satisfaction evaluation
    print(f'Number of turns: {len(sat_pred)}')
    sat_result = sat_evaluation(sat_pred, sat_label, args.sat_num)

    # act evaluation
    act_result = (0, 0, 0, 0)
    if args.act_num > 1:
        act_result = act_evaluation(act_pred, act_label)

    return sat_result, act_result
