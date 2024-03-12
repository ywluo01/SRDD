from ast import arg
from math import nan
from pickle import FALSE
from random import shuffle
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.models.base_clip import Fewclip_model as FewShotModel
from model.algorithm.similar_mask import SMGBlock
from model.algorithm.Dynamic_conv import Dynamic_conv2d, ConvBasis2d,ConvBasis2d_split,CondConv2D
from model.algorithm.gmm import GaussianMixture
from model.algorithm.kmeans import KMeans
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.autograd import Variable
from model.algorithm.Con_conv import CondConv2D
from model.networks.nets import BasicBlock
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def test_pre(logits):
    pred = torch.argmax(logits, dim=1)
    return pred

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
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SetFunction(nn.Module):
    def __init__(self, args, input_dimension, output_dimension):
        super(SetFunction, self).__init__()
        self.args = args
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # self.post  = Bottleneck(self.input_dimension, self.input_dimension//2)

        self.psi =   nn.Sequential(
            nn.Linear(input_dimension, input_dimension  ),
            nn.ReLU(),
            nn.Linear(input_dimension , input_dimension ),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(input_dimension,input_dimension),
            nn.ReLU(),
            nn.Linear(input_dimension , output_dimension),
        )
        
    def forward(self, x, level,shot=None,way=None,neg_pre=None,ratio=1):
        # x = F.adaptive_avg_pool2d(self.post(x),(1,1)).squeeze() # B, width
        if level == 'class':
            psi_output = x + self.psi(x)
            rho_input = psi_output.view(ratio, shot* way, 128).mean(0)
            rho_input = torch.mean(rho_input.view(shot, way, 128), dim=0)
            rho_output =self.rho(rho_input)
            # rho_output = F.relu6(self.rho(rho_input)) / 6 * self.args.delta
            return rho_output
        elif level == 'sample':
            return x
        elif level == 'balance':
            psi_output = self.psi(x)
            rho_input = torch.cat([psi_output, x], dim=1)
            rho_input = torch.sum(rho_input, dim=0, keepdim=True)
            rho_output = F.relu(self.rho(rho_input))
            return rho_output
        elif level == 'neg':
            psi_output = x + self.psi(x)
            rho_input = [psi_output[neg_pre == i] for i in range(self.args.ood_num)] #rho_input.view(shot, way, -1)
            rho_output =  torch.stack([self.rho(item).mean(0) for item in rho_input])
            # rho_output =  torch.stack([(F.relu6(self.rho(item)) / 6 * self.args.delta).mean(0) for item in rho_input])
            # print('rho_output',rho_output.shape) # 3, 128
            return  rho_output

        else: raise ValueError('Do not havee the target level!')


class indicator_layer(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.fc1 = nn.Linear(self.input_dimension, self.input_dimension)
        self.fc2 = nn.Linear(self.input_dimension ,output_dimension)
        # nn.init.xavier_normal_(self.fc1.weight)
        
        # nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x,x_gen,level):
        
        if level == 'sample':
            x = torch.cat([x,x_gen],dim = -1)
            x = F.relu(self.fc1(x)) + x
            x = F.relu(self.fc2(x))
            return x
        elif level == 'support':
            out = []
            for item in x_gen:
                x_t = torch.cat([x,item],dim = -1)
                x_t = F.relu(self.fc1(x_t)) + x_t
                x_t = F.relu(self.fc2(x_t))
                out.append(x_t)
            return torch.stack(out)

class post_layer(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.fc1 = nn.Linear(self.input_dimension, self.input_dimension,bias=False)
        self.fc2 = nn.Linear(self.input_dimension ,output_dimension, bias=False)
        self.drop = nn.Dropout(p=0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        #= nn.Dropout(p=0.2) nn.init.xavier_normal_(self.fc1.weight)
        
        # nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x):

        x = F.relu(self.fc1(x)) + x
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Relaclip_base(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':        hdim = 64
        elif args.backbone_class == 'Res12':        hdim = 640
        elif args.backbone_class == 'WRN':          hdim = 640
        elif args.backbone_class == 'dino':          hdim = 384
        elif args.backbone_class == 'clip':          hdim = 512
        elif 'Res12' in args.backbone_class:hdim = 640
        self.train_step = 0
        # else:
        #     raise ValueError('')
        # num_patches = (args.imgsize // args.patch_size) * (args.imgsize // args.patch_size)
        self.args = args
        self.emb_dim = 128
        self.hdim = 128
        self.width = 197
        self.alpha = 1.0
        # self.open_layer = indicator_layer(self.emb_dim*2, 2)


        self.post_sample =  post_layer(self.emb_dim*2, self.emb_dim)
        # nn.init.xavier_normal_(self.post_sample.weight)
        # self.support_attn = MultiHeadAttention(1, self.hdim, self.hdim, self.hdim, dropout=0.2) 
    
    def gen_semantic_logit(self,support, kquery,uquery):
        proto = support.view(self.args.shot,self.args.closed_way, self.emb_dim).mean(dim=0)
        num_query = kquery.squeeze().shape[0]

        kquery = kquery.view(num_query, -1).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        uquery = uquery.view(num_query, -1).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(0).expand(num_query, self.args.closed_way , self.emb_dim).contiguous()
        klogits = - torch.sum((proto - kquery) ** 2, 2) / self.args.temperature
        ulogits = - torch.sum((proto - uquery) ** 2, 2) / self.args.temperature 
        return klogits, ulogits


    def _forward(self, support_embs, kquery_embs, uquery_embs, gt_label=None):
        
        self.train_step +=1
        num_support = self.args.shot*self.args.closed_way
        num_query = self.args.query*self.args.closed_way
        # image_feature, text_feature, attention
        support_semantic, support_template = support_embs[0]
        B,width = support_semantic.shape    #;print('support_embs', support_embs[0].shape )
        nq, width = kquery_embs[0][0].shape    #;print('kquery_embs',  kquery_embs[0].shape )
        nuq,width = uquery_embs[0][0].shape   #;print('uquery_embs', uquery_embs[0].shape )
        kquery_se, kquery_temp  =   kquery_embs[0];
        uquery_se, uquery_temp  =   uquery_embs[0];
        
        text_proto = support_embs[1].view(self.args.shot,self.args.closed_way, width).mean(dim=0).unsqueeze(0)
        se_proto = support_semantic.view(self.args.shot,self.args.closed_way, self.emb_dim).mean(dim=0)
        torch.cuda.empty_cache()

        se_klogits, se_ulogits = self.gen_semantic_logit(support_semantic, kquery_se,uquery_se)

        support_fea = torch.cat([self.alpha *support_template,support_semantic] ,dim=-1)
        support_fea = self.post_sample(support_fea)
        kquery_fea = torch.cat([self.alpha *kquery_temp,kquery_se],dim=-1)
        uquery_fea = torch.cat([self.alpha *uquery_temp,uquery_se],dim=-1)
        kquery_fea = self.post_sample(kquery_fea )
        uquery_fea = self.post_sample(uquery_fea )


        if self.args.use_ood:
            support_pos ,support_neg = self.generate_sample(support_semantic, support_template,calib=False) #self.args.neg_ratio, B,L,-1
            kquery_pos ,  kquery_neg = self.generate_sample(kquery_se, kquery_temp,calib=False) #self.args.neg_ratio, B,L,-1
            support_neg = self.post_sample(support_neg.view(self.args.neg_ratio* B,width*2)).view(self.args.neg_ratio, B,width)
            support_pos = self.post_sample(support_pos.view(self.args.pos_ratio* B,width*2)).view(self.args.pos_ratio, B,width)
            kquery_neg =  self.post_sample(kquery_neg.view(self.args.neg_ratio* nq,width*2)).view(self.args.neg_ratio, nq,width)
            kquery_pos =  self.post_sample(kquery_pos.view(self.args.pos_ratio* nq,width*2)).view(self.args.pos_ratio, nq,width)
            # print(kquery_neg.shape)
            uquery_gen = self.generate_pred_sample(se_ulogits,uquery_temp, se_proto); uquery_gen  =  self.post_sample(uquery_gen)
            kquery_gen = self.generate_pred_sample(se_klogits,kquery_temp, se_proto); kquery_gen = self.post_sample(kquery_gen)

            
            
            

            # F.cross_entropy(torch.stack([pos_logit,neg_logit]).t(), torch.zeros(len(pos_logit)).type(torch.LongTensor).cuda())
            if self.args.simple_cluster:
                if self.args.closed_way==2 and self.args.shot==1:
                    cluster_resulte = torch.randint(0,self.args.ood_num,(self.args.neg_ratio* B,))
                    print(cluster_resulte)
                else:
                    model_temp = KMeans(n_clusters=self.args.ood_num, mode='euclidean', verbose=0)
                    cluster_resulte = model_temp.fit_predict(support_neg.view(self.args.neg_ratio* B,width)) #sampled_data[sample_group == i]
                self.args.ood_num = len(torch.unique(cluster_resulte))
            elif self.args.cluster:
                global_mean = torch.mean(support_fea,0)
                global_var  = torch.var(support_fea,0)
                sampler = MultivariateNormal(global_mean,torch.diag(global_var))
                temp_sample = sampler.rsample(sample_shape=torch.zeros((self.args.neg_ratio* B)).size()).cuda()
                # print(temp_sample.shape)
                # print(support_neg.view(self.args.neg_ratio* B,width,L).mean(-1).shape)
    
                model_temp = KMeans(n_clusters=self.args.ood_num, mode='euclidean', verbose=0)
                cluster_resulte = model_temp.fit_predict(support_neg.view(self.args.neg_ratio* B,width)+ 0.1*temp_sample) #sampled_data[sample_group == i]
                self.args.ood_num = len(torch.unique(cluster_resulte))
            else:
                cluster_resulte = torch.randint(0,self.args.ood_num,(self.args.neg_ratio* B,))

            pos_logit_support = self.indicator_cosine(support_fea,support_pos); neg_logit_support  = self.indicator_cosine(support_fea,support_neg)
            pos_logit_kquery = self.indicator_cosine(kquery_fea,kquery_pos);    neg_logit_skquery  = self.indicator_cosine(kquery_fea,kquery_neg)
            
            kquery_indicate = self.indicator_cosine(kquery_fea,kquery_gen.unsqueeze(0))
            neg_logit_uquery = self.indicator_cosine(uquery_fea,uquery_gen.unsqueeze(0))
            # print(pos_logit_kquery.shape)

            loss_indic = neg_logit_uquery.mean()+neg_logit_support.mean()+neg_logit_skquery.mean() - pos_logit_support.mean()-pos_logit_kquery.mean()
            # loss_indic = loss_indic + F.cross_entropy(torch.cat([pos_logit_kquery.unsqueeze(0),neg_logit_skquery.unsqueeze(0)],dim=0).t(), torch.zeros(len(pos_logit_kquery)).type(torch.LongTensor).cuda()) 
            # loss_indic = -pos_logit_support.mean()+ neg_logit_support.mean() - pos_logit_kquery.mean() +neg_logit_skquery.mean() +neg_logit_uquery.mean()
            torch.cuda.empty_cache()

            if self.args.expand:
                support_expand = torch.cat([support_fea.unsqueeze(0),support_pos ], dim = 0).mean(0).view(B,128)
                proto = support_expand.view( self.args.shot,self.args.closed_way,width ).mean(0)
            else:
                proto = support_fea.view( self.args.shot,self.args.closed_way, width).mean(dim=0)
            torch.cuda.empty_cache()

            if cluster_resulte == None:
                proto_neg   = support_neg.mean(0).view(self.args.shot,self.args.closed_way,width).mean(0)
                support_neg = support_neg.view(self.args.neg_ratio, B,width)
            else:
                support_neg = support_neg.view(self.args.neg_ratio* B,width)
                proto_neg = [support_neg[cluster_resulte == i] for i in range(self.args.ood_num)] #rho_input.view(shot, way, -1) 
                proto_neg = torch.stack([item.mean(0) for item in proto_neg]) 
                support_neg = support_neg.view(self.args.neg_ratio, B,width)
            torch.cuda.empty_cache()

            show = False
            if show:
                from sklearn.decomposition import PCA
                import matplotlib.pyplot as plt

                color = ['brown','gold' ,'forestgreen','teal','orchid', 'gray', 'darkorange']

                emb_dim  = 128
                fig_path = '/home/ywluo7/files/OSFS/os/fig/5w/'
                pca_tool = PCA(n_components=2)
                num_neg =  len(support_neg.view(-1,emb_dim)) #len(kquery_neg.view(-1,emb_dim)) +
            
                
                label_ = torch.arange(self.args.closed_way, dtype=torch.int8).repeat(self.args.shot + self.args.query)
                label_ = torch.cat([label_,torch.ones(num_neg)*self.args.closed_way]).cpu().detach().numpy()
                # label = []
                # for item in label_:
                #     label.append(color[int(item)])

                out_1 = torch.cat([support_fea.view(1, self.args.shot, self.args.closed_way,  emb_dim), 
                                                                    kquery_fea.view(1, self.args.query, self.args.closed_way, emb_dim)], 1).view(-1, 128) 
                s_vec_p_d = support_fea.view(1, self.args.shot, self.args.closed_way,  emb_dim).mean(dim=1).squeeze()
                # u_vec_p_d = s_u_vec.mean(dim=1).squeeze()
                # uq_vec = torch.cat([kquery_neg.view(-1,emb_dim), support_neg.view(-1,emb_dim)], 0)
                uq_vec = support_neg.view(-1,emb_dim)
                print('uq_vec',uq_vec.shape)

                out_1 =  torch.cat([out_1,uq_vec.view(-1, emb_dim),s_vec_p_d,proto_neg],0)


                reduced_data_list = pca_tool.fit_transform(out_1.cpu().detach().numpy())
                data_num = num_support+num_query+num_neg
                label_proto_ = [0,1,2]
                # label_proto = []
                # for item in label_proto_:
                #     label_proto.append(color[int(item)])

                # ax.scatter(reduced_data_list[:, 0], reduced_data_list[:, 1],reduced_data_list[:, 2],s=1,c = all_select_label)
                plt.scatter(reduced_data_list[:data_num, 0], reduced_data_list[:data_num, 1],s=40,c = label_,marker='o',alpha=0.45)
                plt.scatter(reduced_data_list[-(self.args.closed_way+self.args.ood_num):-self.args.ood_num, 0], reduced_data_list[-(self.args.closed_way+self.args.ood_num):-self.args.ood_num, 1],s=48,c = 'r',marker='^')
                plt.scatter(reduced_data_list[-self.args.ood_num:, 0], reduced_data_list[-self.args.ood_num:, 1],s=48,c = 'dimgrey',marker='^')


                plt.savefig(fig_path+str(self.train_step)+'_2.png',dpi=200)
                plt.clf()

            # print('support_pos',support_pos.shape)
            # print('support_neg', support_neg.shape)
            if self.args.closed_way == 2:
                neg_num = self.args.neg_ratio * B; len_ =  neg_num//self.args.ood_num
                inx =  torch.randperm(neg_num)
                support_neg = support_neg.view(self.args.neg_ratio*B,width)
                proto_neg = torch.stack([ support_neg[inx[i*len_: (i+1)*len_],:].mean(0) for i in range(self.args.ood_num)])
                support_neg = support_neg.view(self.args.neg_ratio, B,width)

            
            con_loss = self.contrastive_loss(support_fea, support_pos, support_neg)
            torch.cuda.empty_cache()
            # proto_neg = None
            # proto_neg = proto.mean(0).unsqueeze(0)
            


        else: 
            # support_fea = support_semantic
            proto = support_fea.view( self.args.shot,self.args.closed_way, width).mean(dim=0) # Ntask x NK x d            
            con_loss,loss_indic = None,None; 
            proto_neg = None#support_fea.mean(0).unsqueeze(0)

        
        
        
        

        torch.cuda.empty_cache()
        if proto_neg is not None:
            allproto = torch.cat([proto,proto_neg],dim=0).unsqueeze(0)
        else: allproto = proto
        

        # allproto = self.support_attn(allproto,allproto,allproto).squeeze()
        # text_proto = self.text_attn(text_proto,text_proto,text_proto).squeeze()
        allproto = allproto.squeeze()
        text_proto = text_proto.squeeze()
        # print(allproto.shape)
        proto_num = len(allproto)
        topy = self.topology_loss(allproto[:self.args.closed_way],text_proto)

        
        
        # query: (B, num_query, num_proto, num_emb)
        # proto: (B, num_proto, num_emb)
        if  self.args.use_euclidean:
            kquery_fea = kquery_fea.view(num_query, width).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            uquery_fea = uquery_fea.view(num_query, width).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = allproto.unsqueeze(0).expand(num_query, proto_num , width).contiguous()
            klogits = - torch.sum((proto - kquery_fea) ** 2, 2) / self.args.temperature
            ulogits = - torch.sum((proto - uquery_fea) ** 2, 2) / self.args.temperature 
            # print(klogits.shape)

            
            proto = allproto.unsqueeze(0).expand(B, proto_num , width).contiguous()
            support_fea = support_fea.view(B, width).unsqueeze(1)
            slogits = - torch.sum((proto - support_fea) ** 2, 2) / self.args.temperature
            if self.args.use_ood:
                support_pos = support_pos.mean(0).view(B, width).unsqueeze(1)
                support_neg = support_neg.mean(0).view(B, width).unsqueeze(1)
                poslogit =  - torch.sum((proto - support_pos) ** 2, 2) / self.args.temperature 
                neglogit =  - torch.sum((proto - support_neg) ** 2, 2) / self.args.temperature 
            else: poslogit,neglogit=None,None ;kquery_indicate,neg_logit_uquery = None,None 



            if self.args.use_ood and proto_num > (self.args.closed_way +1):
                klogits = torch.cat([klogits[:,:self.args.closed_way], torch.max(klogits[:,self.args.closed_way:], dim=1, keepdim=True)[0]],dim=1)
                ulogits = torch.cat([ulogits[:,:self.args.closed_way], torch.max(ulogits[:,self.args.closed_way:], dim=1, keepdim=True)[0]],dim=1)
                slogits = torch.cat([slogits[:,:self.args.closed_way], torch.max(slogits[:,self.args.closed_way:], dim=1, keepdim=True)[0]],dim=1)
                poslogit = torch.cat([poslogit[:,:self.args.closed_way], torch.max(poslogit[:,self.args.closed_way:], dim=1, keepdim=True)[0]],dim=1)
                neglogit = torch.cat([neglogit[:,:self.args.closed_way], torch.max(neglogit[:,self.args.closed_way:], dim=1, keepdim=True)[0]],dim=1)



        if self.training:
            # print('finish--------------------')
            
            return [[klogits, ulogits],[poslogit,neglogit],[se_klogits, se_ulogits],[kquery_indicate,neg_logit_uquery]],  slogits, [con_loss,loss_indic,topy] #[fea_sim_loss,con_loss,topy,loss_sim]
        else:
            return [klogits, ulogits],[se_klogits, se_ulogits],[kquery_indicate,neg_logit_uquery]
    
    def cal_distance(self,data,proto):
        pass

    
    def indicator_loss(self,support,fea):
        support = F.normalize(support, dim=-1)
        fea = F.normalize(fea, dim=-1)
        ratio = fea.shape[0]
        out = []
        for i in range(ratio):
            x = self.indicator(torch.cat([support, fea[i]], dim=1))
            out.append(x)
        return torch.cat(out)
    def indicator_cosine(self,support,fea):
        support = F.normalize(support, dim=-1)
        fea = F.normalize(fea, dim=-1)

        ratio = fea.shape[0]
        out = []
        for i in range(ratio):
            x = F.cosine_similarity(support, fea[i], dim=-1)
            out.append(x)
        return torch.stack(out).mean(0)



    def contrastive_loss(self, out_1, out_2, out_3):
        # support pos, neg
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        out_3 = F.normalize(out_3, dim=-1)
        bs = out_1.size(0)
        loss_pos = 0
        temp = 0.25
        for i in range(self.args.pos_ratio):

            out = torch.cat([out_1, out_2[i]], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
            # [2B, 2B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)
            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2[i], dim=-1) / temp)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss_pos += (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # for negative
        # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.mm(out_1, out_3.t().contiguous()) / temp)
        # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
        # [2B, 2B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

        neg_sim = []
        for i in range(self.args.pos_ratio):
            out_temp = out_3[i]
            # compute loss
            neg_sim.append(torch.exp(torch.sum(out_1 * out_temp, dim=-1) / temp)) # [2*B]
            # neg_sim = torch.cat([neg_sim, neg_sim], dim=0)
        neg_sim = torch.cat(neg_sim)
        # print('neg_sim', neg_sim.shape)
        loss_neg =  torch.log(neg_sim).mean()

        # print('loss_neg',loss_neg)
        # print('loss_pos',loss_pos)
        return [loss_pos , loss_neg]
    

    def calculate_indicator_logit(self, support, pos,neg):
        pos_logits = []
        neg_logits = []        
        support = F.normalize(support,dim=-1)
        for i in range(self.args.neg_ratio):
            pos_temp = F.normalize(pos[i], dim=-1)
            
            neg_temp = F.normalize(neg[i], dim=-1)

            # transform cosine similarity calculation into normalized matrix production
            pos_logits.append(F.cosine_similarity(support, pos_temp))
            neg_logits.append(F.cosine_similarity(support, neg_temp))
        return torch.cat(pos_logits), torch.cat(neg_logits)

            
   

    def topology_loss(self, text, image):
        text = text / text.norm(dim=1, keepdim=True)
        image = image / image.norm(dim=1, keepdim=True)
        center_ = text.mean(0)

        text_matrix = torch.exp(torch.mm(text, text.t().contiguous()))
        image_matrix = torch.exp(torch.mm(image, image.t().contiguous()))
        # print('text_matrix', text_matrix)
        # print('image_matrix', image_matrix)
        # print(text_matrix - image_matrix)

        loss = torch.logsumexp(text_matrix-image_matrix, dim=-1).sum()
        # print('topology', loss)
        return loss

    
    def pos_neg_loss(self, support , pos, neg):
        support = support.view(self.args.shot*self.args.closed_way,-1)
        pos = pos.view(self.args.shot*self.args.closed_way,-1)
        neg = neg.view(self.args.shot*self.args.closed_way,-1)

        support = support / support.norm(dim=1, keepdim=True)
        pos = pos / pos.norm(dim=1, keepdim=True)
        neg = neg / neg.norm(dim=1, keepdim=True)
        pos_loss = F.cosine_similarity(support,pos)
        neg_loss = F.cosine_similarity(support,neg)

        # out_pos = self.open_layer(torch.stack([support,pos]).permute(1,0,2))
        # out_neg = self.open_layer(torch.stack([support,neg]).permute(1,0,2))


    
    
    def generate_pred_sample(self,logits,template, proto):
        pred = test_pre(logits)
        gen_sample = []; 
        for ii,item in enumerate(template):
            gen_sample.append(torch.cat([self.alpha*item,proto[pred[ii]]],dim=-1))

        return  torch.stack(gen_sample)
    

    def generate_sample(self, semantic_support, masked_support, calib=False):

        B,width = semantic_support.shape # c  #(B,self.args.mask_num,3,3)
        nn = int(B//self.args.closed_way)
        semantic_support= semantic_support.squeeze().view(nn,self.args.closed_way,128)
        masked_support=   masked_support.squeeze().view(nn,self.args.closed_way,128)
   
        pos = []; neg = [] ; 
        for ii in range(self.args.closed_way):
            temp_mask = masked_support[:,ii].view(nn,128)  #mask,hw
            class_semantic = semantic_support[:,ii].view(nn,128) #shot,nc,hw

            pos.append( self.shuffle_intra(temp_mask,class_semantic))

            other_class_index = [i for i in range(self.args.closed_way) if i!=ii]
            # print(other_class_index)
            others_class_semantic = semantic_support[:,other_class_index]
            # print(others_class_semantic.shape)
            neg.append(self.shuffle_inter( temp_mask, others_class_semantic))

        # pos = torch.stack(pos).permute(1,2,0,3); 
        # neg = torch.stack(neg).permute(1,2,0,3); 
        pos = torch.stack(pos).permute(1,0,2); 
        neg = torch.stack(neg).permute(1,0,2); 

        # print('out neg', neg.shape)
        # print('out pos', pos.shape)

        return pos.contiguous().view(self.args.pos_ratio, B,width*2),neg.contiguous().view(self.args.neg_ratio, B,width*2)#, semantic_support.mean(-2).view(self.args.shot*self.args.closed_way,width)

    
    def shuffle_intra(self, masked_fea,similarity_part ):
        ns,  _ = masked_fea.shape
        out= []
        for i in range(self.args.pos_ratio):
            index =  torch.randperm(ns)
            # shuffle_similarity_part = similarity_part.mean(0).expand(ns,_)
            shuffle_similarity_part = similarity_part[index,:]
            out.append(torch.cat([self.alpha *masked_fea,shuffle_similarity_part] , dim=-1))
        return torch.stack(out).squeeze(0)


    
    def shuffle_inter(self, masked_fea, unsimilarity_part):
        out = []
        ns, _ = masked_fea.shape
        nss, nw, _ = unsimilarity_part.shape
        # if self.args.neg_ratio > self.args.closed_way-1:
        negative_num = self.args.neg_ratio * ns
        unsimilarity_part = unsimilarity_part.view(nss* nw,  128)
        index =  torch.randperm(nss* nw)[:negative_num]
        if self.args.closed_way <= self.args.neg_ratio:
            # out.append((masked_fea+unsimilarity_part[index]).mean(-1))
            for i in range(self.args.neg_ratio):
                index =  torch.randperm(nss)[:ns]
                out.append(torch.cat([self.alpha *masked_fea,unsimilarity_part[index,:]],dim=-1))
        else:
            for i in range(self.args.neg_ratio):
                out.append(torch.cat([self.alpha *masked_fea,unsimilarity_part[index[i*ns:(i+1)*ns],:]],dim=-1))#.mean(-1))
        return torch.stack(out).squeeze(0)

        

        