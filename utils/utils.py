# -*- coding:utf-8 -*-
# @Time: 2023-9-7 17:04
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: utils.py
# @ProjectName: PolypNet

import torch
import numpy as np
import logging
from thop import profile
from thop import clever_format


def eval_mae(y_pred, y):
    """
    metric MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    clamp_ 操作就可以将梯度约束在[ -grad_clip, grad_clip] 的区间之内。大于grad_clip的梯度，将被修改等于grad_clip
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for para_group in optimizer.param_groups:
        para_group['lr'] *= decay


def adjust_lr_v2(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for para_group in optimizer.param_groups:
        para_group['lr'] = init_lr * decay


def adjust_lr_v3(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30, min_lr=1e-5):
    decay = decay_rate ** (epoch // decay_epoch)
    if init_lr * decay > min_lr:
        for para_group in optimizer.param_groups:
            para_group['lr'] = init_lr * decay
    else:
        for para_group in optimizer.param_groups:
            para_group['lr'] = min_lr


def adjust_lr_with_warmup(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30, min_lr=1e-5, warmup_epoch=10):
    """use warmup"""
    leap = init_lr - min_lr
    # leap_base_pase = leap/(warmup_epoch-1)
    leap_base_pase = leap / warmup_epoch
    if epoch <= warmup_epoch:  # do warmup
        for para_group in optimizer.param_groups:
            para_group['lr'] = min_lr + (epoch-1) * leap_base_pase
    else:
        current_decay_rate = decay_rate ** ((epoch - warmup_epoch - 1) // decay_epoch)
        if init_lr * current_decay_rate > min_lr:
            for para_group in optimizer.param_groups:
                para_group['lr'] = init_lr * current_decay_rate
        else:
            for para_group in optimizer.param_groups:
                para_group['lr'] = min_lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.loss = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def get_logger(filename, verbosity=1, name=None):
    """
        logger = get_logger('/path/to/exp/exp.log')

        logger.info('start training!')
        for epoch in range(MAX_EPOCH):
            ...
            loss = ...
            acc = ...
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , MAX_EPOCH, loss, acc ))
            ...

        logger.info('finish training!')
    """

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def init_logger(save_path=None):
    logging.basicConfig(filename=save_path + '/train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logger = logging.getLogger()
    return logger
