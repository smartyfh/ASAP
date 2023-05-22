import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from transformers import AutoModel
from .transformer import TransformerEncoder
import torchcrf


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    

class BaseModelBackbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        model_name = config.get('model_name', 'bert-base-uncased')
        self.base_model = AutoModel.from_pretrained(model_name)
        self.d_model = self.base_model.config.hidden_size

    def forward(self, input_ids):
        attention_mask = input_ids.ne(0).detach()
        outputs = self.base_model(input_ids, attention_mask)
        h = universal_sentence_embedding(outputs.last_hidden_state, attention_mask)
        cls = outputs.pooler_output

        out = torch.cat([cls, h], dim=-1)
        return out # [batch_size, d_model * 2]


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, din):
        dout = self.dropout(F.relu(self.fc1(din)))
        dout = self.fc2(dout)
        return dout
    
    
class HawkesTransformer(nn.Module):
    def __init__(self, d_model, num_types, num_layers, num_heads, dropout):
        super().__init__()
        self.num_types = num_types
        self.encoder = TransformerEncoder(d_model, d_model*4, num_heads, num_layers, dropout)
        self.event_emb = nn.Embedding(num_types, d_model)
        
    def forward(self, event_type, attention_mask):
        # we consider soft event types
        batch_size = event_type.size(0)
        event_idx = torch.LongTensor([[i for i in range(self.num_types)]]).expand(batch_size, -1)
        event_embs = self.event_emb(event_idx.to(event_type.device)) # [bs, num_types, d]
        event_input = torch.bmm(event_type, event_embs) # [bs, l, d]
        event_output = self.encoder(event_input, attention_mask)
        return event_output


class Intensity(nn.Module):
    def __init__(self, d_model, sat_num):
        super().__init__()
        self.hawkes_mlp = MLP(d_model, sat_num, d_model//4)
        self.base_mlp = MLP(d_model, sat_num, d_model//4)
        self.actv = nn.Softplus(beta=1)
        
    def forward(self, seq, base):
        out = self.hawkes_mlp(seq) + self.base_mlp(base)
        out = self.actv(out)
        return out
    
    
class ASAP(nn.Module):
    def __init__(self, args, act_num=1, sat_num=5):
        super(ASAP, self).__init__()
        self.turn_encoder = BaseModelBackbone(model_name=args.base_model_name)
        d_model = self.turn_encoder.d_model
        self.dial_encoder = TransformerEncoder(d_model*2, d_model*4, args.tf_heads, args.tf_layers, args.dropout)
        self.layer_norm = nn.LayerNorm(d_model*2, eps=1e-6)
        self.dim_reduction = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(args.dropout)
        
        self.hawkes = HawkesTransformer(d_model, sat_num, args.hp_layers, args.hp_heads, args.dropout)
        self.intensity = Intensity(d_model, sat_num)
        
        self.act_num = act_num
        self.sat_num = sat_num
        self.act_pred = MLP(d_model, act_num, d_model//4)
        self.sat_pred = MLP(d_model, sat_num, d_model//4)
        
        self.crf = torchcrf.CRF(act_num, batch_first=True)
        self.train_base = args.train_base

    def forward(self, input_ids, act_seq=None, sat=None, mask=None, use_hp=True):
        batch_size, dialog_len, utt_len = input_ids.size()
        
        # exchange-level
        input_ids = input_ids.view(-1, utt_len)
        turn_out = self.turn_encoder(input_ids=input_ids)
        turn_out = turn_out.view(batch_size, dialog_len, -1) # [bsz, nturn, hsz*2]
        
        # dialogue-level
        attention_mask = mask.unsqueeze(-2).repeat(1, dialog_len, 1)
        attention_mask &= subsequent_mask(dialog_len).to(input_ids.device)
        dial_out = self.dial_encoder(turn_out, attention_mask) # [bsz, nturn, hsz*2]
        
        dial_out = self.layer_norm(dial_out)
        dial_out = self.dim_reduction(dial_out) # reduce to dimension size = 768
        dial_out = self.dropout(dial_out) # [bsz, nturn, hsz]
        
        sat_logits_ctx = self.sat_pred(dial_out) # logits based only on dialogue context
        sat_probs_ctx = F.softmax(sat_logits_ctx, -1) # [bsz, nturn, sat_num]
        
        if use_hp:
            # Hawkes process
            hawkes_out = self.hawkes(sat_probs_ctx, attention_mask) # [bsz, nturn, hsz]
            sat_logits_hp = self.intensity(hawkes_out, dial_out) # [bsz, nturn, sat_num]
            sat_probs_hp = F.softmax(sat_logits_hp, -1)
            
            sat_logits = sat_logits_hp
            sat_probs = sat_probs_hp
        else:
            sat_logits = sat_logits_ctx
            sat_probs = sat_probs_ctx
        
        # dialog act prediction
        act_probs = None
        if self.act_num > 1:
            act_logits = self.act_pred(dial_out)
            act_probs = F.softmax(act_logits, -1)
        
        # satisfaction loss
        sat_loss = torch.tensor(0)
        if sat is not None:            
            last_index = torch.sum(mask, dim=-1, keepdim=True) - 1
            last_sat_logits = torch.gather(sat_logits, 1, last_index[:, :, None].expand(-1, -1, sat_logits.size(-1)))
            sat_loss = F.cross_entropy(last_sat_logits.squeeze(1), sat)
            if self.train_base and use_hp:
                # in this case, we use the ground-truth labels to optimize the base model as well
                last_sat_logits_base = torch.gather(sat_logits_ctx, 1, last_index[:, :, None].expand(-1, -1, sat_logits_ctx.size(-1)))
                sat_loss_base = F.cross_entropy(last_sat_logits_base.squeeze(1), sat)
                sat_loss += sat_loss_base
                
        # action loss    
        act_loss = torch.tensor(0)
        if act_seq is not None:
#             last_act_logits = torch.gather(act_logits, 1, last_index[:, :, None].expand(-1, -1, act_logits.size(-1)))
#             last_act_seq = torch.gather(act_seq, 1, last_index)
#             act_loss = F.cross_entropy(last_act_logits.squeeze(1), last_act_seq.view(-1))
            act_loss = -1*self.crf(act_logits, act_seq, mask=mask, reduction='token_mean')
             
        return sat_probs, act_probs, sat_loss, act_loss
