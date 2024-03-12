import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.models.base_clip_fsl import Fewclip_model as FewShotModel



# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

        

    def _forward(self, support_embs, kquery_embs, uquery_embs):
        emb_dim = support_embs.size(-1)


        # get mean of the support
        num_query = self.args.query*self.args.closed_way
        support_embs =  support_embs.view(1,self.args.shot,self.args.closed_way, -1)
        kquery_embs =   kquery_embs.view(num_query,-1)
        uquery_embs =   uquery_embs.view(num_query,-1)

            
        proto = support_embs.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        # num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if  self.args.use_euclidean:
            kquery_embs = kquery_embs.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            uquery_embs = uquery_embs.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            klogits = - torch.sum((proto - kquery_embs) ** 2, 2) / self.args.temperature
            ulogits = - torch.sum((proto - uquery_embs) ** 2, 2) / self.args.temperature
                        
            # logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            return [klogits, None], [ulogits,None]
        else:
            return klogits, [ulogits,None]
