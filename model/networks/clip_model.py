
import clip
import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
import math
import cv2
class clip_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.encoder , preprocess  =  clip.load('/home/ywluo7/files/OSFS/clip/pretrain_model/ViT-B-16.pt', "cpu")                

    def forward(self, data):
        
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


    def return_imgtext_feature(self,data):
        text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in data[1]]).to('cuda')
        # print('text shape',text_inputs.shape)
        image_features, atten = self.encoder.encode_image(data[0])  #(bsz, tgt_len, src_len)
        # print('imgfea shape', image_features.shape)
        
        text_features = self.encoder.encode_text(text_inputs)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return image_features, text_features

    def reutrn_attention(self,data):
        self.encoder.eval()
        _, atten = self.encoder.encode_image(data)  #(bsz, tgt_len, src_len)
        atten = atten[0,0,1:]
        atten = atten / atten.norm(dim=-1, keepdim=True)

        return atten
    def reutrn_attention_masks(self,data):
        self.encoder.eval()
        _, atten = self.encoder.encode_image(data)  #(bsz, tgt_len, src_len)
        atten = atten[:,0,1:]
        atten = atten / atten.norm(dim=-1, keepdim=True)

        atten = self.gen_mask(atten)

        return atten
    def min_max_norm(self,data,scale = 1.0):
        return ((data-data.min(-1,keepdim=True)[0])/data.max(-1,keepdim=True)[0])*scale

    def torch_equalize(self,image):
        """Scale the data in the channel to implement equalize."""
        # im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = torch.histc(image, bins=256, min=0, max=255)#.type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]]) 
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = image
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, image.flatten().long())
            result = result.reshape_as(image)
        
        return result.type(torch.uint8)


        return image
    def gen_mask_single(self,atten):
        n,L = atten.shape; h = int(math.sqrt(L))
        atten = self.min_max_norm(atten,scale=255)
        atten = atten.reshape(n, h, h).float()
        print(atten.shape)
        atten = F.interpolate(atten.unsqueeze(1), scale_factor=16, mode="bilinear").squeeze().detach()#.cpu().numpy()
        
        print(atten.shape)
        atten=np.uint8(cv2.normalize(atten, None, 0, 255, cv2.NORM_MINMAX))
        atten = cv2.equalizeHist(atten)/256
        print(atten)
        atten[atten<0.85] = 0
    def gen_mask(self,atten):
        n,L = atten.shape; h = int(math.sqrt(L))
        atten = self.min_max_norm(atten,scale=255)
        atten = atten.reshape(n, h, h).float()
        # print(atten.shape)
        atten = F.interpolate(atten.unsqueeze(1), scale_factor=16, mode="bilinear").squeeze().detach().cpu().numpy()
        atten=np.uint8(atten)
        masks = []
        for item in atten:
            item = cv2.equalizeHist(item)/256
            item[item<0.85] = 0
            masks.append(item)
        return torch.from_numpy(np.array(masks)).reshape(n, L).float()
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto, _ = self.encoder.encode_image(data_shot) 
        proto = proto / proto.norm(dim=1, keepdim=True)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)

        query,_ = self.encoder.encode_image(data_query) 
        query = query / query.norm(dim=1, keepdim=True)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim