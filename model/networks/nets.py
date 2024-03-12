from multiprocessing import pool
import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
from model.algorithm.similar_mask import SMGBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                conv1x1(inplanes, planes , stride),
                nn.BatchNorm2d(planes),
            )
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class skin_sd(nn.Module):
    def __init__(self):
        super().__init__()
        # self.args = args
        from model.networks.res12 import ResNet
        self.encoder = ResNet(avg_pool=False)
        self.fea_encoder = ResNet(avg_pool=False)
        hdim = 640
        self.pool = nn.AvgPool2d(6, stride=1)  

    def forward(self, data):
        out_attr = self.encoder(data)
        out_en =self.fea_encoder(data)
        # out_en = self.en_mid_layer(out_en)
        out_en = self.pool(out_en).view(out_en.size(0), -1)
        return out_en, out_attr

from model.networks.vit import vit_small
class DinoFeaturizer(nn.Module):

    def __init__(self, cfg, dim=1):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.patch_size = self.args.patch_size
        # self.feat_type = self.cfg.feat_type
        self.model = vit_small(patch_size=self.patch_size , num_classes=0)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        # self.model.eval().cuda()
        # self.dropout = torch.nn.Dropout2d(p=.1)

        # if cfg.pretrained_weights is not None:
        #     state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
        #     state_dict = state_dict["teacher"]
        #     # remove `module.` prefix
        #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        #     # remove `backbone.` prefix induced by multicrop wrapper
        #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        #     # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
        #     # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

        #     msg = self.model.load_state_dict(state_dict, strict=False)
        #     print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        # else:
        #     print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        #     state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        #     self.model.load_state_dict(state_dict, strict=True)
        arch = "vit_small"
        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.feat_type = "atten_feat"
        self.dropout = torch.nn.Dropout2d(p=.1)


    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)
            w_featmap = img.shape[-2] // self.args.patch_size
            h_featmap = img.shape[-1] // self.args.patch_size


            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]
            # print('vit_fea',feat.shape)

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            # print(feat_h,feat_h)

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            elif self.feat_type == "atten":
                attentions = self.model.get_last_selfattention(img)
                # print('vit_fea',attentions.shape)

                nh = attentions.shape[1] # number of head
                nimg = img.shape[0]
                # we keep only the output patch attention
                attentions = attentions[:, :, 0, 1:].reshape(nimg,nh, -1)
                # print('vit_fea',attentions.shape) 

                min_a = torch.min(attentions,dim=-1)[0].unsqueeze(2)
                # print(min_a.shape)
                max_a = torch.max(attentions,dim=-1)[0].unsqueeze(2)
                attentions = (attentions - min_a) / (max_a - min_a)
                # print(torch.min(attentions))
                # print(torch.max(attentions))

                if self.args.threshold is not None:
                    # we keep only a certain percentage of the mass
                    val, idx = torch.sort(attentions)
                    val /= torch.sum(val, dim=1, keepdim=True)
                    cumval = torch.cumsum(val, dim=1)
                    th_attn = cumval > (1 - self.args.threshold)
                    idx2 = torch.argsort(idx)
                    for head in range(nh):
                        th_attn[head] = th_attn[head][idx2[head]]
                    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                    # interpolate
                    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=self.args.patch_size, mode="nearest")[0].cpu().numpy()

                image_feat = attentions.reshape(nimg, nh, w_featmap, h_featmap)
                # attentions = nn.functional.interpolate(attentions, scale_factor=self.args.patch_size, mode="nearest")[0].cpu().numpy()
                # print('vit_fea',image_feat.shape)
            elif self.feat_type == "atten_feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

                attentions = self.model.get_last_selfattention(img)
                # print('vit_fea',attentions.shape)
                nh = attentions.shape[1] # number of head
                nimg = img.shape[0]
                # we keep only the output patch attention
                attentions = attentions[:, :, 0, 1:].reshape(nimg,nh, -1)
                # print('vit_fea',attentions.shape) 
                # min_a = torch.min(attentions,dim=-1)[0].unsqueeze(2)
                # # print(min_a.shape)
                # max_a = torch.max(attentions,dim=-1)[0].unsqueeze(2)
                # attentions = (attentions - min_a) / (max_a - min_a) #+1e-6

            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))
            # print('img_fea',image_feat.shape)
            # print('attentions',attentions.shape)
            return [image_feat, attentions]











class cifar_attr_fea(nn.Module):

    def __init__(self):
        super().__init__()
        # self.args = args

        from model.networks.WRN28 import Wide_ResNet
        self.encoder = Wide_ResNet(16, 8, 0.4,pool=False)  
        self.fea_encoder = Wide_ResNet(16, 8, 0.4,pool=True) 
        hdim = 512

        self.cluster_layer = BasicBlock(hdim, 128)
    def forward(self, data):
        out_attr = self.encoder(data)
        out_attr = self.cluster_layer(out_attr)


        out_en =self.fea_encoder(data)
        # out_en = self.en_mid_layer(out_en)
        return out_en, out_attr

class Skin_attr_fea(nn.Module):

    def __init__(self):
        super().__init__()
        # self.args = args
        from model.networks.res12 import ResNet
        self.encoder = ResNet(avg_pool=False)
        self.fea_encoder = ResNet(avg_pool=False)
        hdim = 640

        self.cluster_layer = BasicBlock(hdim, 128)
        self.maxpool = nn.MaxPool2d(2) 
        self.pool = nn.AvgPool2d(14, stride=1)  

    def forward(self, data):
        out_attr = self.encoder(data)
        out_attr = self.cluster_layer(out_attr)
        out_attr = self.maxpool(out_attr)


        out_en =self.fea_encoder(data)
        # out_en = self.en_mid_layer(out_en)
        out_en = self.pool(out_en).view(out_en.size(0), -1)
        return out_en, out_attr

class Skin_res12(nn.Module):

    def __init__(self,pool=True):
        super().__init__()
        # self.args = args
        from model.networks.res12 import ResNet
        self.fea_encoder = ResNet(avg_pool=False) # hdim = 640

        # self.maxpool = nn.MaxPool2d(2) 
        self.pool = pool
        if pool:
            self.avgpool = nn.AvgPool2d(6, stride=1)  

    def forward(self, data):
        out_en =self.fea_encoder(data)
        # print(out_en.shape)
        if self.pool:
            out_en = self.avgpool(out_en).view(out_en.size(0), -1)
        # out_en = self.en_mid_layer(out_en)
        return out_en

