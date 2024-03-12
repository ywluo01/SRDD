import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))
        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc_epoch'] = 0.0;   self.trlog['max_acc'] = 0.0;   self.trlog['max_acc_interval'] = 0.0
        self.trlog['max_auroc'] = 0.0;           self.trlog['max_auroc_interval'] = 0.0
        self.trlog['max_open_acc'] = 0.0
        

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader, test_true):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader, test_true):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch,test_true):
        args = self.args
        if self.train_epoch % args.eval_interval == 0 and epoch>0:
            # vl, va, vap, auroc, dist_auroc ,diffauroc,open_acc,avg_attr_acc, attrauroc= self.evaluate(self.val_loader,Ground_true=Ground_true,global_dist=global_dist,test_true=test_true)
            vl, [va, vap],[record_auroc, record_aurocp]= self.evaluate(self.val_loader,test_true=test_true)

            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            self.logger.add_scalar('val_auroc', float(record_auroc),  self.train_epoch)
            # self.logger.add_scalar('attr_close_acc', float(Attr_close_acc),  self.train_epoch)
            # self.logger.add_scalar('attr_auroc', float(record_attr_auroc),  self.train_epoch)
            # self.logger.add_scalar('open_acc', float(Open_acc),  self.train_epoch)

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

            if record_auroc>= self.trlog['max_auroc']:
                self.trlog['max_auroc'] = record_auroc
                self.trlog['max_auroc_interval'] = record_aurocp
                self.save_model('max_auroc')
            # if record_attr_auroc>= self.trlog['max_open_acc']:
            #     self.trlog['max_open_acc'] = record_attr_auroc
            #     self.save_model('max_open_acc')
            
            print('best ep {}, best val_acc={:.4f} + {:.4f},val_auroc={:.4f}+{:.4f}'.format(
                self.trlog['max_acc_epoch'], self.trlog['max_acc'], self.trlog['max_acc_interval'], self.trlog['max_auroc'],  self.trlog['max_auroc_interval']))

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
