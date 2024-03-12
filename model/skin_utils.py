import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np
import datetime
def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.way = args.closed_way + args.open_way
    args.num_classes =  args.closed_way + args.open_way
    args.eval_way =   args.closed_way + args.open_way
    save_path1 = '-'.join([args.dataset, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.closed_way, args.shot, args.query)])
    save_path2 = '_'.join([ 'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler)])    
    if args.use_euclidean:
        save_path1 += '-DIS'
    else:
        save_path1 += '-SIM'
    if args.fix_BN:
        save_path2 += '-FBN'
    time_path =datetime.datetime.now().strftime('%m%d_%H_%M')
    if args.test == True:
        path_test = './checkpoints/test/'
        if not os.path.exists(os.path.join(path_test, time_path[:4])):
            os.mkdir(os.path.join(path_test, time_path[:4]))
        args.save_path = os.path.join(path_test, time_path[:4])
    else:     
        if not os.path.exists(os.path.join(args.save_dir, save_path1)):
            os.mkdir(os.path.join(args.save_dir, save_path1))
        args.save_path = os.path.join(args.save_dir, save_path1, save_path2, time_path)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--max_epoch', type=int, default=50)
    # parser.add_argument('--episodes_per_epoch', type=int, default=70)
    parser.add_argument('--imgsize', type=int, default=224)

    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=50)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--num_test_episodes', type=int, default=600)
    
    parser.add_argument('--model_class', type=str, default='Relaclip_base', 
                        choices=['Fsl_skin','MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 'SemiFEAT', 'SemiProtoFEAT','Relaclip']) # None for MatchNet or ProtoNet
    parser.add_argument('--use_euclidean', action='store_true', default=True)    
    parser.add_argument('--backbone_class', type=str, default='clip',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN','Res12_skin','Res12_SD', 'Res12_skin','dino','clip'])
    parser.add_argument('--dataset', type=str, default='SD260',
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB','Skin','dog','Skin_SD260'])
    
    parser.add_argument('--closed_way',      type=int, default=5)
    parser.add_argument('--closed_eval_way', type=int, default=5)
    parser.add_argument('--open_way',        type=int, default=5)
    parser.add_argument('--open_eval_way',   type=int, default=5)

    parser.add_argument('--patch_size', type=int, default=16)       
    parser.add_argument('--mask_num', type=int, default=5)
    parser.add_argument('--threshold',  default=None)
    parser.add_argument('--feapatch_size', type=int, default=1)
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--pos_ratio', type=int, default=1)
    parser.add_argument('--masks', type=list, default=[1.0,0.8,0.5,0])


    

    parser.add_argument('--ori_shot',      type=int, default=None)
    parser.add_argument('--ori_eval_shot', type=int, default=None)
    parser.add_argument('--shot',          type=int, default=14)
    parser.add_argument('--eval_shot',     type=int, default=14)
    parser.add_argument('--query',      type=int, default=1)
    parser.add_argument('--eval_query', type=int, default=1)

    parser.add_argument('--balance', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=32)  # the temperature in the  
     
    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=3)  
    # parser.add_argument('--lr_mul', type=float, default=10)   
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='7')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0,1,2')
    parser.add_argument('--init_weights', type=str,  default= None)#"/home/ywluo7/files/open_set/Feat_global/saves/init/Res12-pre.pth")
    # parser.add_argument('--clip_weights', type=str,  default='/home/ywluo7/files/OSFS/os/saves/new_clip/max_acc_sim.pth')

    parser.add_argument('--clip_weights', type=str,  default="/home/ywluo7/files/OSFS/os//checkpoints/skin_sd/0307/SD260-clip-05w05s10q-DIS/lr0.0001mul10_step/0316_09_01/max_auroc.pth")
    ## customer parameters
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--out_planes', type=int, default=640)
    parser.add_argument('--kl_margin', type=float, default=0.5)
    parser.add_argument('--ood_num',  type=int, default=5)
    

    parser.add_argument('--use_ood', type=bool, default=True)
    parser.add_argument('--use_set', type=bool, default=False)
    parser.add_argument('--expand', type=bool, default=True)
    parser.add_argument('--use_topo', type=bool, default=False)

    parser.add_argument('--test', type=bool, default=True)

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/skin_sd/0307/')

    
    return parser
