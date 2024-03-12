import os.path as osp
import PIL
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# IMAGE_PATH = '/home/ywluo7/Downloads/Data/SD-260/'#osp.join(ROOT_PATH1, 'data/cub')
# SPLIT_PATH = '/home/ywluo7/Downloads/Data/SD-260-split/'


IMAGE_PATH = '/home/ywluo7/Downloads/Data/SD-260//'#osp.join(ROOT_PATH1, 'data/cub')
SPLIT_PATH = '/home/ywluo7/Downloads/Data/SD-260-split/new_split//'

# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)
def identity(x):
    return x
class skin_SD260(Dataset):

    def __init__(self, setname, args=None, augment=False):
        txt_path = osp.join(SPLIT_PATH, setname + '.txt')
        # class_ = [x.strip() for x in open(txt_path, 'r').readlines()]#[1:]

        self.data, self.onehot_label  = self.parse_csv(txt_path,IMAGE_PATH)
        self.num_class = np.unique(np.array(self.onehot_label )).shape[0]
        self.num_data = len(self.data)
        image_size = args.imgsize
        
        if augment and setname == 'train':
            transforms_list = [
                  transforms.Resize(256),
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize(np.array([0.03914857,0.03031521,0.02787402]),
                                     np.array([0.01623609,0.01514057,0.01478142]))
                ]
        else:
            transforms_list = [
                  transforms.Resize(256),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize(np.array([0.04070394,0.02978333,0.0276212 ]),
                                     np.array([0.01360902,0.01229781,0.01219843]))
                ]
        self.transform = transforms.Compose(transforms_list)
        
        

        ## train
        # [298.58612, 231.21413, 212.59515] [123.83269, 115.477135, 112.73791]
        # [0.03914857,0.03031521,0.02787402] [0.01623609,0.01514057,0.01478142]

        # test
        # [0.04070394,0.02978333,0.0276212 ] [0.01360902,0.01229781,0.01219843]

    def parse_csv(self, txt_path,img_path):
        data, label, self.wnids = [], [], []
        lb = -1
        # text_feature = self.load_text_feature()
        # text_dict = {}
        lines = [x.strip() for x in open(txt_path, 'r').readlines()]#[1:]
        for l in lines:
            lb += 1
            # text_dict[str(lb)] = text_feature[l]
            images_list = os.listdir(img_path+l+'/')
            for item in images_list:
                path = osp.join(img_path,l,item)
                data.append(path)
                label.append(lb)

        return data, label  #, text_dict
    def load_text_feature(self):
        with open(SPLIT_PATH+'class_name_feature.pickle','rb') as f:
            text_feature = pickle.load(f)
        return text_feature


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.onehot_label [i]
        # print(data)
        image = self.transform(Image.open(data).convert('RGB'))
        return image, label            

