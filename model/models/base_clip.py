import torch
import torch.nn as nn
import numpy as np
import clip
import torch.nn.functional as F
import math, cv2
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class se_mask(nn.Module):
    def __init__(self, input_dimension,mid_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        # self.output_dimension = output_dimension
        self.mid_dimension = mid_dimension

        self.fc1 = nn.Linear(input_dimension, mid_dimension)
        self.fc2 = nn.Linear(mid_dimension*2, mid_dimension)
        # self.fc3 = nn.Linear(196, 196)
        # self.fc4 = nn.Linear(196, 196)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.xavier_normal_(self.fc4.weight)
        
    def forward(self, x):
        x = torch.cat([x,F.relu(self.fc1(x))],dim=-1) #.view(nb,self.mid_dimension,14,14)
        x = self.fc2(x)
        x = F.relu6(x) / 6 #* self.args.delta
        return x


class Fewclip_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.encoder , preprocess  =  clip.load('/home/ywluo7/files/OSFS/clip/pretrain_model/ViT-B-16.pt', "cpu")      
        # pre_weight = "/home/ywluo7/files/OSFS/clip/saves/Skin_SD260-Clip-pre_train/0.0001_0.3_[20, 40, 70]_1025_17_35/max_acc_dist.pth"
        # state = torch.load(pre_weight)['state_dict']
        # self.encoder.load_state_dict(state)  
        
        self.proj_mask = se_mask(197, 197)
        self.proj_sample = nn.Linear(768, 128)
        self.proj_text   = nn.Linear(512, 128)


    def extract_feature(self,x,text):
        x_fea, text_fea, x_atten = [],[],[]
        for i, item in enumerate(x):
            out = self.return_imgtext_atten([x[i].unsqueeze(0) , [text[i]] ])

            x_fea.append(out[0])
            text_fea.append(out[1])
            x_atten.append(out[2])
        return  torch.stack(x_fea), torch.stack(text_fea), torch.stack(x_atten)


    def forward(self, x, gt_label, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            support_idx, query_idx = self.split_instances()
            # print('support_idx', support_idx.view(-1).numpy())
            # print('query_idx',query_idx.view(-1).numpy())
            if self.args.ori_shot ==1:
                x0,x1,x2 = x[0]
                s0 = x0[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + x0[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.ori_shot ,self.args.closed_way,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1])#(40)
                s1 = x1[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + x0[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.ori_shot ,self.args.closed_way,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1])#(40)
                s2 = x2[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + x0[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.ori_shot ,self.args.closed_way,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1])#(40)
                self.args.shot = self.args.ori_shot+2
                support_embs = torch.cat([s0,s1,s2],dim = 0).view(self.args.shot *self.args.closed_way,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1]).cuda()
                kquery_embs  =   x0[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape + x0[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.closed_way*self.args.query,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1]).cuda()
                uquery_embs  =   x0[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape + x0[0].shape[-3:]))[:, :, self.args.closed_way:].contiguous().view(self.args.closed_way*self.args.query,x0[0].shape[-3],x0[0].shape[-2],x0[0].shape[-1]).cuda()
                del x0,x1,x2, s0,s1,s2
                # print('support shape', support_embs.shape)
                # print('query shape', uquery_embs.shape)
                support_txt =   x[1][support_idx.view(support_idx.shape).cpu().numpy()][:, :, :self.args.closed_way].reshape(-1)#.view(-1)#.view( *(support_idx.shape))[:, :, :self.args.closed_way].contiguous()#(1,5,5,640)
                support_txt = np.array([support_txt for i in range(self.args.shot)]).reshape(-1)
                # print('shot', self.args.shot)
            else:
                support_embs = x[0][support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + x[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.closed_way*self.args.shot ,x[0].shape[-3],x[0].shape[-2],x[0].shape[-1])#(40)
                kquery_embs  =   x[0][query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape + x[0].shape[-3:]))[:, :, :self.args.closed_way].contiguous().view(self.args.closed_way*self.args.query,x[0].shape[-3],x[0].shape[-2],x[0].shape[-1])
                uquery_embs  =   x[0][query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape + x[0].shape[-3:]))[:, :, self.args.closed_way:].contiguous().view(self.args.closed_way*self.args.query,x[0].shape[-3],x[0].shape[-2],x[0].shape[-1])
                support_txt =   x[1][support_idx.view(support_idx.shape).cpu().numpy()][:, :, :self.args.closed_way].reshape(-1)#.view(-1)#.view( *(support_idx.shape))[:, :, :self.args.closed_way].contiguous()#(1,5,5,640)
            
            # print(gt_label)
            # print('support_idx', gt_label[support_idx.contiguous().view(-1)].contiguous().view(support_idx.shape)[:, :, :self.args.closed_way])
            # print('support_idx', gt_label.view(-1).view(self.args.shot+self.args.query, self.args.way).numpy())
            # 

            
            # print(np.array(x[1])[[1,2,3,4,5,6]])

            kquery_txt  =   x[1][query_idx.view(query_idx.shape).cpu().numpy()][:, :, :self.args.closed_way].reshape(-1)#.contiguous().view(-1)
            uquery_txt  =   x[1][query_idx.view(query_idx.shape).cpu().numpy()][:, :, self.args.closed_way:].reshape(-1)#.contiguous().view(-1)
            # print('support_txt',support_txt)
            # print('kquery_txt',  kquery_txt)
            # print('uquery_txt',  uquery_txt)
            del x
            
            x_support, [loss_minus, loss_logit]  = self.return_imgtext_atten([support_embs, support_txt], calib_mask=False, return_minus_logits = True)
            torch.cuda.empty_cache()
            # print('process support...')
            x_kquery  = self.return_imgtext_atten([kquery_embs , kquery_txt ], calib_mask=False,return_feature= True)
            torch.cuda.empty_cache()
            x_uquery   = self.return_imgtext_atten([uquery_embs , uquery_txt ], calib_mask=False, return_feature = True)
            # print('feat',x_support[0].shape,'text',x_support[1].shape,'atten',x_support[2].shape)
            #[semantic_feature,semantic_feature_1],None
            torch.cuda.empty_cache()

            if self.training:
                if gt_label is not None:
                    klogits, ulogits, loss = self._forward(x_support, x_kquery, x_uquery,gt_label=gt_label)
                    
                    return klogits, ulogits, [loss,loss_minus,loss_logit]
                else: return klogits, ulogits, None
            else:
                if gt_label is not None:

                    klogits, ulogits, loss = self._forward(x_support, x_kquery, x_uquery,gt_label=gt_label)
                    return klogits, ulogits, loss

                # klogits, ulogits = self._forward(support_embs, kquery_embs, uquery_embs,test_true=test_true)
                # return klogits, ulogits
                # logits, map_info, distribute_info, gt_labels = self._forward(instance_embs, support_idx, query_idx,gt_label=gt_label,map=map,loss_mask_num=loss_mask_num)
                # return logits, map_info, distribute_info, gt_labels

    def _forward(self, x, support_idx, query_idx, support_text):
        raise NotImplementedError('Suppose to be implemented by subclass')
    def forward_loss(self, data):
        
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in data[1]]).to('cuda')
        # print('text shape',text_inputs.shape)
        image_features, atten = self.encoder.encode_image(data[0])  #(bsz, tgt_len, src_len)
        # print('imgfea shape', image_features.shape)
        
        text_features = self.encoder.encode_text(text_inputs)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.encoder.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(logits_per_image), device = 'cuda') 
        # print('labels', labels)
        loss_t = F.cross_entropy(logits_per_image, labels) 
        loss_i = F.cross_entropy(logits_per_image.T, labels) 
        clip_loss = (loss_t + loss_i) / 2. 

        # shape = [global_batch_size, global_batch_size]
        return clip_loss #logits_per_image, logits_per_text

    def split_instances(self):
        args = self.args
        if args.ori_shot is None:
            sample_num = args.shot
        else: sample_num = args.ori_shot 
        if self.training:
            return  (torch.Tensor(np.arange(args.way*sample_num)).long().view(1, sample_num, args.way), 
                     torch.Tensor(np.arange(args.way*sample_num, args.way *     (sample_num + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))
    
    def return_imgtext_atten(self,data, calib_mask= False, return_minus_logits = False,return_logits=False,return_feature=False):
        self.encoder.eval(); #self.proj_mask.eval(); self.proj_sample.eval();self.proj_text.eval()

        nb = data[0].shape[0]
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in data[1]]).to('cuda')#.view(self.arg.shot,self.args.closed_way,-1)
        x_, [atten,image_features] = self.encoder.visual(data[0])  #(bsz, tgt_len, src_len)
        del x_
        # print('image_features',image_features.shape)
        # print('atten',atten.shape)
        image_features = image_features.transpose(0, 1).contiguous().view(197, nb, -1)
        # print('image_features',image_features.shape)
        image_features= image_features.transpose(1,0);#[16, 197]
        image_features = self.proj_sample(image_features) # b, 197, 128
        
        atten = atten[:,0,:] 
        atten = atten / atten.norm(dim=-1, keepdim=True)
        image_masks = self.proj_mask(atten).view(nb,197)
        
        if calib_mask:
            calib_masks = [];other_masks = []; need_mask = []
            for ii,_ in enumerate(range(128)):
                # print()
                tops = torch.topk(image_masks[:,ii], 100,dim=-1)[0]
                temps = image_masks[:,ii] - torch.min(tops,dim=-1, keepdim=True)[0] +1e-7
                temps[temps>0] =1; temps[temps<0] =0
                # calib_masks.append((image_masks[:,ii]*temps))
                calib_masks.append(temps*image_masks[:,ii])
                need_mask.append((1-temps)*image_masks[:,ii])
                other_masks.append((1-temps)*(1-image_masks[:,ii]))
            calib_masks = torch.stack(calib_masks).permute(1,2,0)
            other_masks = torch.stack(other_masks).permute(1,2,0)
            need_mask = torch.stack(need_mask).permute(1,2,0)
            # print('other_masks',other_masks.shape)
            image_masks = calib_masks#.permute(0,2,1)
            # print(image_masks.shape) # 16 3 196
            # print('imgfea shape', image_features.shape)
        else: 
            image_masks = image_masks.unsqueeze(1)
            other_masks = 1-image_masks
            calib_masks = None
        
        if return_feature:
            semantic_feature= torch.bmm(image_masks,image_features).squeeze()
            semantic_feature = semantic_feature / semantic_feature.norm(dim=-1, keepdim=True)

            semantic_feature_1= torch.bmm(other_masks,image_features).squeeze() 
            semantic_feature_1 = semantic_feature_1 / semantic_feature_1.norm(dim=-1, keepdim=True)

            return [semantic_feature,semantic_feature_1],None

        else:
            
            text_features = self.encoder.encode_text(text_inputs)
            text_features =  self.proj_text(text_features).view( nb,128 )
            
            if return_minus_logits:
                # print('image_masks',image_masks.shape)
                # print('image_features',image_features.shape)
                semantic_feature= torch.bmm(image_masks,image_features).squeeze()
                semantic_feature = semantic_feature / semantic_feature.norm(dim=-1, keepdim=True)

                semantic_feature_1= torch.bmm(other_masks,image_features).squeeze() 
                semantic_feature_1 = semantic_feature_1 / semantic_feature_1.norm(dim=-1, keepdim=True)

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                sim_0 = F.cosine_similarity(  semantic_feature,text_features)
                sim_1 = F.cosine_similarity(semantic_feature_1,text_features)
                
                sims = torch.stack([sim_0,sim_1]).t()
                loss_sim = - sims[:,0].mean() + sims[:,1].mean()
                # label_dummy = torch.zeros(len(sims)).type(torch.LongTensor).cuda()
                # loss_minus = F.cross_entropy(sims[:,:2], label_dummy) + F.cross_entropy(sims[:,1:], label_dummy) + 0.1*F.cross_entropy(sims, label_dummy)

                logit_scale = self.encoder.logit_scale.exp()
                logits_per_image = logit_scale * semantic_feature @ text_features.t()
            
            else: loss_sim = None; logits_per_image=None
            torch.cuda.empty_cache()

            return [[semantic_feature,semantic_feature_1] ,text_features], [loss_sim,logits_per_image]
    
    
    def split_encode(self,embs):
        length = embs.shape[0]
        sp  = 1
        out,out_ = [],[]
        for i in range(length//sp):
            temp, temp_ = self.encoder(embs[i*sp:(i+1)*sp])
            out.append(temp); out_.append(temp_)
        out_embs = torch.cat(out,dim=0)
        out_embs_ = torch.cat(out_,dim=0)

        return [out_embs,out_embs_]

    def min_max_norm(self,data,scale = 1.0):
        return ((data-data.min(-1,keepdim=True)[0])/data.max(-1,keepdim=True)[0])*scale



    def gen_mask_single(self,atten):
        n,L = atten.shape; h = int(math.sqrt(L))
        atten = self.min_max_norm(atten,scale=255)
        atten = atten.reshape(n, h, h).float()
        # print(atten.shape)
        atten = F.interpolate(atten.unsqueeze(1), scale_factor=16, mode="bilinear").squeeze().detach()#.cpu().numpy()
        
        # print(atten.shape)
        atten=np.uint8(cv2.normalize(atten, None, 0, 255, cv2.NORM_MINMAX))
        atten = cv2.equalizeHist(atten)/256
        atten[atten<0.85] = 0
    def gen_mask(self,atten, max, min):
        n,L = atten.shape; h = int(math.sqrt(L))
        # print('n:',n,'L',L)
        atten = self.min_max_norm(atten,scale=255)
        atten = atten.reshape(n, h, h).float()
        # print(atten.shape)
        atten = F.interpolate(atten.unsqueeze(1), scale_factor=16, mode="bilinear").squeeze(1).detach().cpu().numpy()
        atten=np.uint8(atten)
        # print(atten.shape)
        masks = []
        # print('max, min',max, min)
        for item in atten:
            item = cv2.equalizeHist(item)/256
            # print(np.min(item))
            # print(np.max(item))
            # print(item.shape)
            # print(item)
            item[item<min] = 0
            # print(item)
            item[item>max] = 0
            # print(item)
            # 
            item = cv2.resize(item, (14,14))
            # print(item)
            item[item>0] = 1
            # print(item)
            masks.append(item)
        # print(np.array(masks).shape)
        # print(ddd)
        return torch.from_numpy(np.array(masks)).float().reshape(n, L).float()

