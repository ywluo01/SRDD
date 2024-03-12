
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

from model.models.base_clip_fsl import Fewclip_model as FewShotModel
def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
class Snatcher(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        elif args.backbone_class == 'clip':
            hdim = 128
        else:
            raise ValueError('')
        # hdim = 512
        
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)  
            
        
    def _forward(self, support, kquery, uquery):
        emb_dim = support.size(-1)

        # organize support/query data
        num_query = self.args.query*self.args.closed_way
        support = support.view(1,self.args.shot,self.args.closed_way, -1)
        kquery = kquery.view(num_query,-1)
        uquery = uquery.view(num_query,-1)

        # get mean of the support
        bproto = support.mean(dim=1) # Ntask x NK x d
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)        
        if self.args.use_euclidean:
            kquery = kquery.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            uquery = uquery.view(-1, emb_dim).unsqueeze(1)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            klogits = - torch.sum((proto - kquery) ** 2, 2) / self.args.temperature
            ulogits = - torch.sum((proto - uquery) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            kquery = kquery.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
            uquery = uquery.view(-1, emb_dim).unsqueeze(1)

            klogits = torch.bmm(kquery, proto.permute([0,2,1])) / self.args.temperature
            klogits = klogits.view(-1, num_proto)

        
        """ Snatcher """
        with torch.no_grad():
            snatch_known = []
            for j in range(klogits.shape[0]):
                pproto = bproto.clone().detach()
                """ Algorithm 1 Line 1 """
                c = klogits.argmax(1)[j]
                """ Algorithm 1 Line 2 """
                pproto[0][c] = kquery.reshape(-1, emb_dim)[j]
                """ Algorithm 1 Line 3 """
                pproto = self.slf_attn(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                snatch_known.append(pdiff)
                
            snatch_unknown = []
            for j in range(ulogits.shape[0]):
                pproto = bproto.clone().detach()
                """ Algorithm 1 Line 1 """
                c = ulogits.argmax(1)[j]
                """ Algorithm 1 Line 2 """
                pproto[0][c] = uquery.reshape(-1, emb_dim)[j]
                """ Algorithm 1 Line 3 """
                pproto = self.slf_attn(pproto, pproto, pproto)[0]
                pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                """ pdiff: d_SnaTCHer in Algorithm 1 """
                snatch_unknown.append(pdiff)
                
                pkdiff = torch.stack(snatch_known)
                pudiff = torch.stack(snatch_unknown)
                pkdiff = pkdiff.cpu().detach().numpy()
                pudiff = pudiff.cpu().detach().numpy()
                
                snatch_auroc = calc_auroc(pkdiff, pudiff)

        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.closed_way, emb_dim), 
                                  kquery.view(1, self.args.query, self.args.closed_way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.closed_way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
                # print(logits_reg.shape)
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return [klogits, logits_reg],   [ulogits,snatch_auroc]          
        else:
            return klogits,[ulogits ,snatch_auroc]  
