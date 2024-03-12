import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler_clip, RandomSampler, ClassSampler
from model.models.protonet import ProtoNet
from model.models.matchnet import MatchNet
from model.models.feat import FEAT
# from model.models.featstar import FEATSTAR
from model.models.deepset import DeepSet
from model.models.fsl_skin import Fsl_skin
from model.models.fsl_exp import Fsl_exp
from model.models.Snatcher import Snatcher
from model.models.Relation import Rela
from model.models.RelaClip import Relaclip
from model.models.RelaClip_base import Relaclip_base



class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])
                
                yield ( torch.cat(_, dim=0) for _ in output_batch )
            except StopIteration:
                done = True
        return

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'Skin':
        from model.dataloader.skin_dataset import SkinDATA as Dataset   
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'dog':
        from model.dataloader.stanford_dog import dogs as Dataset
    elif args.dataset == 'Skin_SD260':
        from model.dataloader.skin_SD260 import skin_SD260 as Dataset
    elif args.dataset == 'SD260':
        from model.dataloader.SD260 import skin_SD260 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    print('num_device:',num_device)
    # num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers
    num_episodes =  args.episodes_per_epoch
    # num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    trainset = Dataset('train', args, augment=args.augment)

    
    args.num_class = trainset.num_class
    if args.ori_shot is not None:
        args.smple_num = args.ori_shot
    else:args.smple_num = args.shot
    train_sampler = CategoriesSampler_clip(trainset.onehot_label,
                                      num_episodes,
                                      max(args.way, args.num_classes),
                                      args.smple_num + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    # if args.multi_gpu and num_device > 1:
    #     train_loader = MultiGPUDataloader(train_loader, num_device)
    #     args.way = args.way * num_device

    if args.ori_eval_shot is not None:
        args.eval_smple_num = args.ori_eval_shot
    else:args.eval_smple_num = args.eval_shot

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler_clip(valset.onehot_label,
                            args.num_eval_episodes,
                            args.eval_way, args.eval_smple_num + args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    
    testset = Dataset('val', args)
    test_sampler = CategoriesSampler_clip(testset.onehot_label,
                            args.num_test_episodes,
                            args.eval_way, args.eval_smple_num + args.eval_query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True) 

    # visset = Dataset('train30', args)
    # vis_loader = DataLoader(dataset=visset,
    #                             batch_size=16,
    #                               num_workers=num_workers,
    #                             #   batch_sampler=train_sampler,
    #                               pin_memory=True)

    return train_loader, val_loader, test_loader#, args

def get_dataloader_dino(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'Skin':
        from model.dataloader.skin_dataset import SkinDATA as Dataset   
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'dog':
        from model.dataloader.stanford_dog import dogs as Dataset
    elif args.dataset == 'Skin_SD260':
        from model.dataloader.skin_SD260 import skin_SD260 as Dataset
    elif args.dataset == 'SD260':
        from model.dataloader.SD260 import skin_SD260 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    print('num_device:',num_device)
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    # num_episodes =  args.episodes_per_epoch
    trainset = Dataset('train', args, augment=args.augment)
    text_feature = trainset.text_feature

    
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,
                                      max(args.way, args.num_classes),
                                      args.shot + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    # if args.multi_gpu and num_device > 1:
    #     train_loader = MultiGPUDataloader(train_loader, num_device)
    #     args.way = args.way * num_device

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            args.eval_way, args.eval_shot + args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                            args.num_test_episodes,
                            args.eval_way, args.eval_shot + args.eval_query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True) 

    visset = Dataset('train30', args)
    vis_loader = DataLoader(dataset=visset,
                                batch_size=16,
                                  num_workers=num_workers,
                                #   batch_sampler=train_sampler,
                                  pin_memory=True)

    return train_loader, val_loader, test_loader, text_feature#,vis_loader

def prepare_model(args):
    model = eval(args.model_class)(args)
    # load pre-trained model (no FC weights)
    if args.backbone_class =='clip':
        if args.clip_weights is not None  :
            print('Loading weight from pre-trained clip feature:', args.clip_weights)
            model_dict = model.state_dict()        
            pretrained_dict = torch.load(args.clip_weights)['params']
            # resumed_state = pretrained_dict.keys()
            # print(resumed_state)
            # # print(model_dict.keys())
            # print('---------------------------------------------------')
            # print( model_dict.keys())
            new_pretrained_dict = {}
            for  k, v in pretrained_dict.items():
                if 'module' in k:
                    temp_k = k.split('.'); temp_k.pop(1)
                    k ='.'.join(temp_k)
                if k in model_dict:# and 'slf_attn' not in k:
                    new_pretrained_dict[k] = v
            

            for k in ['head.weight', 'head.bias']:
                if k in pretrained_dict:
                    print(f"removing key {k} from pretrained checkpoint")
                    del pretrained_dict[k]
            
            
        # pretrained_dict = {'encoder.model.'+k: v for k, v in pretrained_dict.items()}
        # 
        # print(new_pretrained_dict.keys())

            model_dict.update(new_pretrained_dict)
            model.load_state_dict(model_dict)

            del pretrained_dict
            del new_pretrained_dict
            del model_dict
            
            print('Loaded the pre-trained clip weight ready')
    elif args.dino_weights is not None:
        print('Loading weight from pre-trained dino feature')
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.dino_weights,map_location=torch.device('cpu'))["model"]#['params']
        # print(model_dict.keys())
        
        # print( pretrained_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in pretrained_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del pretrained_dict[k]
        
        pretrained_dict = {'encoder.model.'+k: v for k, v in pretrained_dict.items()}
        # 

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del pretrained_dict
        print('Loaded the pre-trained dino weight ready')
    elif args.init_weights is not None:
        print('Loading weight from pre-trained feature')
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights,map_location=torch.device('cpu'))#['params']
        print(  [k for k,v in model_dict.items()])
        print('...................................')
        print([k for k,v in model_dict.items()])
        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        new_pretrained_dict = {}
        for  k, v in pretrained_dict.items():
            if 'module' in k:
                temp_k = k.split('.'); temp_k.pop(1)
                k ='.'.join(temp_k)
            if k in model_dict:# and 'slf_attn' not in k:
                new_pretrained_dict[k] = v
        # pretrained_dict = { k: v for k, v in pretrained_dict.items() if k in model_dict}
        # del pretrained_dict
        # print(new_pretrained_dict.keys())
        model_dict.update(new_pretrained_dict)
        model.load_state_dict(model_dict)
        del pretrained_dict
        del new_pretrained_dict
    elif args.init_attr is not None:
        print('Loading weight from attr & feature')
        model_dict = model.state_dict()
        pre_attr =  torch.load(args.init_attr,map_location=torch.device('cpu'))['params']
        pre_fea =  torch.load(args.init_fea,map_location=torch.device('cpu'))['params']

        new_pretrained_fea_dict = {}
        for k, v in pre_fea.items():
            if 'encoder.fea_'+k in model_dict:
                new_pretrained_fea_dict['encoder.fea_'+k] = v
            elif k in model_dict:
                new_pretrained_fea_dict[k] = v

        new_pretrained_attr_dict = {}
        for k, v in pre_attr.items():
            if 'encoder.'+k in model_dict:
                new_pretrained_attr_dict['encoder.'+k] = v
            elif k in model_dict:
                new_pretrained_attr_dict[k] = v
        print('Feature_part:')
        print(new_pretrained_fea_dict.keys())
        print('Attribute_part:')
        print(new_pretrained_attr_dict.keys())
        model_dict.update(new_pretrained_fea_dict)
        model_dict.update(new_pretrained_attr_dict)

        model.load_state_dict(model_dict)
        del new_pretrained_fea_dict, pre_fea
        del new_pretrained_attr_dict,pre_attr
        
    

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.multi_gpu:
        model.encoder.visual = nn.DataParallel(model.encoder.visual, dim=0)
        # model.encoder.transformer = nn.DataParallel(model.encoder.transformer)
        para_model = model.to(device)
    else:
        para_model = model.to(device)

    return model, para_model

def prepare_optimizer(model, args):
    # top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]   
    # 
    top_para = []; top_para_key = []; mid_para = []; mid_para_key = []; mid_para_ = []; mid_para_key_ = [];
    for k,v in model.named_parameters():
        if 'encoder' not in k and 'mask' not in k and  'proj' in k:
            mid_para.append(v)
            mid_para_key.append(k)
        if 'encoder' not in k and 'mask' in k:
            mid_para_.append(v)
            mid_para_key_.append(k)
            
        if 'encoder' not in k and 'proj' not in k:
            top_para.append(v)
            top_para_key.append(k)
    
 
    print('top para:', top_para_key)
    print('mid para_:', mid_para_key_)
    print('mid para:', mid_para_key)
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )    
    elif args.backbone_class == 'Res12':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )         
    elif args.backbone_class == 'dino':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )     
    elif args.backbone_class == 'clip':
        if args.model_class == 'Relaclip_base'  :
            for param in model.encoder.parameters():
                param.requires_grad = False
                
            # for param in model.proj_mask.parameters():
            #     param.requires_grad = False
            # for param in model.proj_sample.parameters():
            #     param.requires_grad = False
            # for param in model.proj_text.parameters():
            #     param.requires_grad = False
            optimizer = optim.Adam(
                [
                {'params': model.proj_mask.parameters()},
                #  {'params': mid_para_key, 'lr':  args.lr   },
                #  {'params': mid_para_, 'lr': args.lr*0.0   },
                {'params': model.proj_sample.parameters(), 'lr': args.lr  },
                {'params': model.proj_text.parameters(), 'lr': args.lr  },
                {'params': top_para, 'lr': args.lr * args.lr_mul}
                ],
                lr=args.lr*0.1,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
        else: 

            top_para = [];top_para_key =[]; 
            if 'encoder' not in k:
                top_para.append(v)
                top_para_key.append(k)
            
            optimizer = optim.Adam(
                [
                {'params': model.encoder.parameters()},
                #  {'params': mid_para, 'lr':  args.lr*0.0   },
                #  {'params': mid_para_, 'lr': args.lr*0.0   },
                # {'params': model.sample_proj, 'lr': args.lr  },
                {'params': top_para, 'lr': args.lr * args.lr_mul },
                # {'params': top_para, 'lr': args.lr * args.lr_mul}
                ],
                lr=args.lr*0.01,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
    else:
        # optimizer = optim.SGD(
        #     [{'params': model.encoder.parameters(), 'lr': args.lr * 0.1},
        #     {'params': model.mid_layer.parameters(), 'lr': args.lr  },
        #     {'params': model.attr_attn.parameters(), 'lr': args.lr  },
        #      {'params': top_para, 'lr': args.lr * args.lr_mul}],
        #     lr=args.lr,
        #     momentum=args.mom,
        #     nesterov=True,
        #     weight_decay=args.weight_decay
        # )  
  

        optimizer = optim.SGD(
            # [{'params': model.encoder.module.encoder.parameters(), 'lr': args.lr *0.2},
            # {'params': model.encoder.module.cluster_layer.parameters(), 'lr': args.lr },
            [{'params': model.encoder.module.fea_encoder.parameters(), 'lr': args.lr}],
            # {'params': model.mid_layer.parameters(), 'lr': args.lr*1  },
            # {'params': model.attr_attn.parameters(), 'lr': args.lr* args.lr_mul  },
            # {'params': top_para, 'lr': args.lr * args.lr_mul},
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )   

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
