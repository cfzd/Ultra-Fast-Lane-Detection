from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU
from utils.dist_utils import DistSummaryWriter

import torch


def get_optimizer(net,cfg):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def get_loss_dict(cfg):

    if cfg.use_aux:
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'aux_loss', 'relation_dis'],
            'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), torch.nn.CrossEntropyLoss(), ParsingRelationDis()],
            'weight': [1.0, cfg.sim_loss_w, 1.0, cfg.shp_loss_w],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('seg_out', 'seg_label'), ('cls_out',)]
        }
    else:
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis'],
            'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), ParsingRelationDis()],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',)]
        }

    return loss_dict

def get_metric_dict(cfg):

    if cfg.use_aux:
        metric_dict = {
            'name': ['top1', 'top2', 'top3', 'iou'],
            'op': [MultiLabelAcc(), AccTopk(cfg.griding_num, 2), AccTopk(cfg.griding_num, 3), Metric_mIoU(cfg.num_lanes+1)],
            'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('seg_out', 'seg_label')]
        }
    else:
        metric_dict = {
            'name': ['top1', 'top2', 'top3'],
            'op': [MultiLabelAcc(), AccTopk(cfg.griding_num, 2), AccTopk(cfg.griding_num, 3)],
            'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label')]
        }

    
    return metric_dict


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)
            
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

        