import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.base_clip_fsl import Fewclip_model as FewShotModel
from model.utils import one_hot

# Note: This is the MatchingNet without FCE
#       it predicts an instance based on nearest neighbor rule (not Nearest center mean)

class MatchNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, support, kquery, uquery):
        emb_dim = support.size(-1)
        # print(support.shape)

        # get mean of the support
        num_query = self.args.query*self.args.closed_way
        num_support = self.args.shot*self.args.closed_way
        support =  support.view(1,self.args.shot,self.args.closed_way, -1)
        kquery  =   kquery.view(num_query,-1)
        uquery  =   uquery.view(num_query,-1)

        # # organize support/query data
        # support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        # query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # support_fea, support_emb = support_instance_embs
        # kq_vec,  kquery_emb  = kquery_instance_embs

        if self.training:
            label_support = torch.arange(self.args.closed_way).repeat(self.args.shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.closed_way)
        else:
            label_support = torch.arange(self.args.closed_way).repeat(self.args.eval_shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.closed_way)
        if torch.cuda.is_available():
            label_support_onehot = label_support_onehot.cuda() # KN x N

        # get mean of the support
        num_batch =  support.shape[0]
        num_way = self.args.closed_way
        num_support = np.prod(support.shape[1:3])
        # num_query = np.prod(query_idx.shape[-2:])
        support = support.view(num_batch, num_support, emb_dim) # Ntask x NK x d
        label_support_onehot = label_support_onehot.unsqueeze(0).repeat(num_batch, 1, 1)
        
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)

        support = F.normalize(support, dim=-1) # normalize for cosine distance

        kquery = kquery.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        klogits = torch.bmm(kquery, support.permute([0,2,1])) 

        # print(klogits.shape, label_support_onehot.shape)
        klogits = torch.bmm(klogits, label_support_onehot) / self.args.temperature # KqN x N
        klogits = klogits.view(-1, num_way)

        uquery = uquery.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
        ulogits = torch.bmm(uquery, support.permute([0,2,1])) 
        ulogits = torch.bmm(ulogits, label_support_onehot) / self.args.temperature # KqN x N
        ulogits = ulogits.view(-1, num_way)

        if self.training:
            return [klogits, None], [ulogits,None]  
        else:
            return klogits, [ulogits,None]
