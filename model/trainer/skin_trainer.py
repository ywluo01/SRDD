import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.skin_helper import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from sklearn.metrics import roc_auc_score, f1_score
from model.algorithm.similar_mask import spectral_clustering
from model.algorithm.FM_update import Cluster_loss,Multiclass_loss

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd

def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score
def cal_evaluate_auroc(klogits,ulogits,way):
    klogits__ = F.normalize(klogits,p=2,dim=0)  
    ulogits__ = F.normalize(ulogits,p=2,dim=0)  
    if way ==1:
        known_prob =   F.softmax(klogits, 1)[:,:way]#.unsqueeze(1)
        unknown_prob = F.softmax(ulogits, 1)[:,:way]#.unsqueeze(1)
        klogits_ = F.softmax(klogits__, 1)[:,:way]
        ulogits_ = F.softmax(ulogits__, 1)[:,:way]
        kdist = -(klogits[:,:way])
        udist = -(ulogits[:,:way])
    elif way>1:
        known_prob =   F.softmax(klogits, 1)[:,:way].max(1)[0]
        unknown_prob = F.softmax(ulogits, 1)[:,:way].max(1)[0]
        klogits_ = F.softmax(klogits__, 1)[:,:way].max(1)[0]
        ulogits_ = F.softmax(ulogits__, 1)[:,:way].max(1)[0]
        kdist = -(klogits[:,:way].max(1)[0])
        udist = -(ulogits[:,:way].max(1)[0])
    else: 
        known_prob =   klogits
        unknown_prob = ulogits
        klogits_ = klogits
        ulogits_ = ulogits
        kdist = -(klogits)
        udist = -(ulogits)

    known_scores = (known_prob).cpu().detach().numpy()
    unknown_scores = (unknown_prob).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc = calc_auroc(known_scores, unknown_scores)

    """ Distance """
    known_scores = (klogits_).cpu().detach().numpy()
    unknown_scores = (ulogits_).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc_ = calc_auroc(known_scores, unknown_scores)


    """ Distance """
    kdist = kdist.cpu().detach().numpy()
    udist = udist.cpu().detach().numpy()
    dist_auroc = calc_auroc(kdist, udist)



    return max(auroc,auroc_), dist_auroc
def cal_evaluate_auroc_orii(klogits,ulogits,way):
    if way ==1:
        known_prob =   F.softmax(klogits, 1)[:,:way]#.unsqueeze(1)
        unknown_prob = F.softmax(ulogits, 1)[:,:way]#.unsqueeze(1)
    elif way>1:
        known_prob =   F.softmax(klogits, 1)[:,:way].max(1)[0]
        unknown_prob = F.softmax(ulogits, 1)[:,:way].max(1)[0]
    else: 
        known_prob =   klogits
        unknown_prob = ulogits

    known_scores = (known_prob).cpu().detach().numpy()
    unknown_scores = (unknown_prob).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc = calc_auroc(known_scores, unknown_scores)

    """ Distance """
    klogits__ = F.normalize(klogits,p=2,dim=0)  
    ulogits__ = F.normalize(ulogits,p=2,dim=0)  
    klogits_ = F.softmax(klogits__, 1)[:,:way].max(1)[0]
    ulogits_ = F.softmax(ulogits__, 1)[:,:way].max(1)[0]

    known_scores = (klogits_).cpu().detach().numpy()
    unknown_scores = (ulogits_).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc_ = calc_auroc(known_scores, unknown_scores)


    """ Distance """
    kdist = -(klogits[:,:way].max(1)[0])
    udist = -(ulogits[:,:way].max(1)[0])
    kdist = kdist.cpu().detach().numpy()
    udist = udist.cpu().detach().numpy()
    dist_auroc = calc_auroc(kdist, udist)



    return max(auroc,auroc_), dist_auroc

def cal_custom_auroc(klogits,ulogits,fea_open,fea_u_open,way):
    aa = F.softmax(fea_open, 1)
    bb = F.softmax(fea_u_open, 1)
    kood_prob = F.softmax(fea_open, 1)[:,1].cpu().detach().numpy() *3  #.max(1)[0]
    uood_prob = F.softmax(fea_u_open, 1)[:,1].cpu().detach().numpy() *3 #.max(1)[0]
    ood_auroc = calc_auroc(kood_prob, uood_prob)

    known_prob = F.softmax(klogits, 1)[:,:way].max(1)[0]
    unknown_prob = F.softmax(ulogits, 1)[:,:way].max(1)[0]

    known_scores = (known_prob).cpu().detach().numpy()
    unknown_scores = (unknown_prob).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc = calc_auroc(known_scores*kood_prob, unknown_scores*uood_prob)

    """ Distance """
    klogits__ = F.normalize(klogits,p=2,dim=0)  
    ulogits__ = F.normalize(ulogits,p=2,dim=0)  
    klogits_ = F.softmax(klogits__, 1)[:,:way].max(1)[0]
    ulogits_ = F.softmax(ulogits__, 1)[:,:way].max(1)[0]

    known_scores = (klogits_).cpu().detach().numpy()
    unknown_scores = (ulogits_).cpu().detach().numpy()
    known_scores = 1 - known_scores #-->0
    unknown_scores = 1 - unknown_scores
    auroc_ = calc_auroc(known_scores*kood_prob, unknown_scores*uood_prob)

    """ Distance """
    klogits__ = F.normalize(klogits,p=2,dim=0)  
    ulogits__ = F.normalize(ulogits,p=2,dim=0)  


    kdist_ = -(klogits__[:,:way].max(1)[0]).cpu().detach().numpy()
    udist_= -(ulogits__[:,:way].max(1)[0]).cpu().detach().numpy()

    dist_auroc_ = calc_auroc(kdist_, udist_)


    """ Distance """
    kdist = -(klogits[:,:way].max(1)[0])
    udist = -(ulogits[:,:way].max(1)[0])
    kdist = kdist.cpu().detach().numpy()
    udist = udist.cpu().detach().numpy()
    dist_auroc = calc_auroc(kdist*kood_prob, udist*uood_prob)



    return ood_auroc,max(auroc,auroc_), dist_auroc,dist_auroc_




def cal_F1(logits, label,way):
    pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
    if way ==2:
        return f1_score(label.cpu().detach().numpy(), pred)
    else:
        return f1_score(label.cpu().detach().numpy(), pred, average='macro')


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.para_model, args)

    def prepare_label(self):
        args = self.args
        # prepare one-hot label
        label = torch.arange(args.closed_way, dtype=torch.int16).repeat(args.query)
        if self.args.ori_shot == 1:
            label_aux = torch.arange(args.closed_way, dtype=torch.int8).repeat(args.ori_shot+2)
        else: label_aux = torch.arange(args.closed_way, dtype=torch.int8).repeat(args.shot)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
        return label, label_aux
    
    def train(self):
        args = self.args
        self.para_model.train()
        if self.args.fix_BN:
            self.para_model.encoder.eval(); 
        label, label_aux = self.prepare_label()
        print('Begin the training process.........')
        # self.evaluate_test()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.para_model.train()
            if self.args.fix_BN:
                self.para_model.encoder.eval()
            start_tm = time.time()
            tl1  = Averager(); cont_loss = Averager(); indic_loss = Averager(); plc_loss = Averager();
            Sim_loss , Minus_loss,Fea_loss =Averager(),Averager(), Averager()
            # Fea_acc, Fea_auroc, Fea_distauroc, Fea_loss = Averager(),Averager(),Averager(),Averager(); Typo_loss = Averager();
            Se_acc, Se_auroc, Se_distauroc, SeF1= Averager(),Averager(),Averager(),Averager();
           
            for ii,batch in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    if self.args.ori_shot == 1:
                        data = batch[0]
                        text_label, gt_label =  batch[1]
                    else:
                        data = batch[0].cuda()
                        text_label, gt_label =  batch[1]
                        # print('no data aug')
                   

                data_tm = time.time();    self.dt.add(data_tm - start_tm)
                [[klogits, ulogits],[poslogit,neglogit],[se_klogits, se_ulogits],[kquery_indicate,uquery_indicate]],  slogits, [[con_loss,loss_indic,topy], loss_sim,loss_logit] \
                    =  self.para_model([data,np.array(text_label)],gt_label)
                # [slogits,klogits,pos_logit,neglogit], ulogits, [[con_loss,loss_indicator,topy_loss], loss_sim] 
                del data
                cliplabels = torch.arange(len(loss_logit), device = 'cuda') 
                loss_t = F.cross_entropy(loss_logit, cliplabels) ; loss_i = F.cross_entropy(loss_logit.T, cliplabels) 
                clip_loss = (loss_t + loss_i) / 2.      

                loss =F.cross_entropy(klogits, label)+ 0.1 * F.cross_entropy(slogits, label_aux) # + 0.1*F.cross_entropy(klogits, label) 
                total_loss = loss + 0.1*(clip_loss + loss_sim)
                if self.args.use_ood:
                    if args.use_con:
                        total_loss = total_loss+ 0.01*con_loss[0]
                        cont_loss.add(con_loss[0].item())
                    
                    [genauroc, gendist_auroc]= cal_evaluate_auroc(kquery_indicate,uquery_indicate,0)
                    
                Sim_loss.add(loss_sim.item()); Fea_loss.add(loss.item()); 
              

                if epoch>0 and self.args.use_indic:
                    total_loss = total_loss + 0.01* loss_indic
                    indic_loss.add(loss_indic.item()) 


                if self.args.use_ood and epoch >10:
                    dummpyoutputs=klogits.clone()
                    for i in range(len(dummpyoutputs)):
                        nowlabel=label[i]
                        dummpyoutputs[i][nowlabel]=dummpyoutputs[i][self.args.closed_way]
                    loss_place = F.cross_entropy(dummpyoutputs[:,:self.args.closed_way], label)
                    total_loss += 0.01*loss_place

                    plc_loss.add(loss_place.item())
                
                forward_tm = time.time() ;  self.ft.add(forward_tm - data_tm)


                post_klogit = 0.3*F.normalize(klogits[:,:self.args.closed_way],dim=-1)  + 0.7* F.normalize(se_klogits[:,:self.args.closed_way],dim=1)
                post_ulogit = 0.3*F.normalize(ulogits[:,:self.args.closed_way],dim=-1)  + 0.7* F.normalize(se_ulogits[:,:self.args.closed_way],dim=1)
                """ auroc evalue """
                # print(klogits)
                [sacc, sauroc, sdist_auroc]= self.cal_evalue_ori(se_klogits,se_ulogits,label,args.closed_way)
                Se_acc.add(sacc); Se_auroc.add(sauroc); Se_distauroc.add(sdist_auroc)
                se_F1 = cal_F1(klogits[:,:self.args.closed_way], label,self.args.closed_way); SeF1.add(se_F1)
                # [acc, auroc, dist_auroc]= self.cal_evalue_ori(klogits[:,:self.args.closed_way],ulogits[:,:self.args.closed_way],label,args.closed_way)
                tl1.add(total_loss.item()); 
                

                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.cuda.empty_cache()
                backward_tm = time.time();  self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time();  self.ot.add(optimizer_tm - backward_tm)   
                start_tm = time.time()

                if ii % 15 == 19:

                    print('step {}, total_loss={:.4f}, loss={:.4f}, place_loss={:.4f}, sim_loss={:.4f}'.format(ii,tl1.item(),Fea_loss.item(), plc_loss.item(),Sim_loss.item()))
                    print('Sacc={:.4f}, sfauroc={:.4f}, fdistauroc={:.4f}'.format(Se_acc.item(), Se_auroc.item(),Se_distauroc.item()))
                   
            self.lr_scheduler.step()
            
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )
            print('epoch {}, total_loss={:.4f}, loss={:.4f}, place_loss={:.4f}, sim_loss={:.4f}, typo_loss={:.4f}'.format(epoch,tl1.item(),Fea_loss.item(), plc_loss.item(),Sim_loss.item(), Typo_loss.item()))
            print('facc={:.4f}, cont_loss={:.4f},indic_loss={:.4f}, fauroc={:.4f}, fdistauroc={:.4f}'.format(Fea_acc.item(), cont_loss.item(),indic_loss.item(), Fea_auroc.item(), Fea_distauroc.item()))
            print('Sacc={:.4f}, sfauroc={:.4f}, fdistauroc={:.4f},  SF1={:.4f}'.format(Se_acc.item(), Se_auroc.item(),Se_distauroc.item(),SeF1.item()))
            

            self.try_evaluate(epoch, test_true=True)
            torch.cuda.empty_cache()

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        # self.save_model('epoch-last')

    def evaluate(self, data_loader,test_true=False):
        args = self.args
        self.para_model.eval()
        record = np.zeros((args.num_eval_episodes, 4)) # loss and acc, auroc
        label, _ = self.prepare_label();  label = label.long().cuda()

        
        with torch.no_grad():
        
            Se_acc, Se_auroc, Se_distauroc, SeF1= Averager(),Averager(),Averager(),Averager();
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                    text_label, gt_label =  batch[1]
                    # data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch
                
                
                [klogits, ulogits],[se_klogits, se_ulogits],[kquery_indicate,uquery_indicate]= self.para_model([data,np.array(text_label)],gt_label)
                loss = F.cross_entropy(klogits, label)

                [sacc, sauroc, sdist_auroc]= self.cal_evalue_ori(se_klogits,se_ulogits,label,args.closed_way)
                Se_acc.add(sacc); Se_auroc.add(sauroc); Se_distauroc.add(sdist_auroc)
                se_F1 = cal_F1(klogits[:,:self.args.closed_way], label,self.args.closed_way); SeF1.add(se_F1)                


                record[i-1, 0] = loss.item(); record[i-1, 1] = sacc
                record[i-1, 2] = sauroc;        record[i-1, 3] = sdist_auroc


        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0]); va, vap = compute_confidence_interval(record[:,1])
        vauroc, vaurocp = compute_confidence_interval(record[:,2]); vdauroc, vdaurocp = compute_confidence_interval(record[:,3])
        if vauroc >vdauroc: record_auroc = vauroc; record_aurocp = vaurocp
        else: record_auroc = vdauroc; record_aurocp = vdaurocp

        # train mode
        self.para_model.train()
        if self.args.fix_BN:
            self.para_model.encoder.eval();   #self.para_model.proj_sample.eval(); self.para_model.proj_text.eval();# self.para_model.proj_mask.eval(); 

        print('Val Sacc={:.4f}, sfauroc={:.4f}, fdistauroc={:.4f}, SF1={:.4f}'.format(Se_acc.item(), Se_auroc.item(),Se_distauroc.item(),SeF1.item()))

        return vl, [va, vap]

   
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val_acc={:.4f} + {:.4f},val_auroc={:.4f}+{:.4f}\n'.format(
                self.trlog['max_acc_epoch'], self.trlog['max_acc'], self.trlog['max_acc_interval'], self.trlog['max_auroc'],  self.trlog['max_auroc_interval']))

            f.write('Test acc={:.4f} + {:.4f}, Test auroc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'], self.trlog['test_acc_interval'],
                self.trlog['test_auroc'],  self.trlog['test_auroc_interval']))        
    

    def cal_evalue_ori(self,klogits,ulogits,label,way):

        acc = count_acc(klogits, label)
        auroc, dist_auroc = cal_evaluate_auroc(klogits,ulogits,way)        

        return acc, auroc, dist_auroc

