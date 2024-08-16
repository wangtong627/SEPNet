# -*- coding:utf-8 -*-
# @Time: 2023-9-7 17:22
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: loss_func.py
# @ProjectName: PolypNet
import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 生成权重，边缘是其他部分权重的5倍，相当于加强
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)  # 这里对pred取sigmoid，计算图中logits不需要再做sigmoid
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def region_loss(pred, mask):
    pred = torch.sigmoid(pred)  # 这里对pred取sigmoid，计算图中logits不需要再做sigmoid

    # 该部分损失更加强调前景区域或者背景区域内部的预测一致性
    numerator_fore = (mask - mask * pred).sum([2, 3])
    denominator_fore = mask.sum([2, 3]) + 1e-8

    numerator_back = ((1 - mask) * pred).sum([2, 3])
    denominator_back = (1 - mask).sum([2, 3]) + 1e-8

    fore = numerator_fore / denominator_fore
    back = numerator_back / denominator_back
    return (fore + back).mean()


class GroupLoss(nn.Module):
    def __init__(self):
        super(GroupLoss, self).__init__()
        # print('you are using group loss!')
        self.eps = 1e-8

    def wbce_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 生成权重，边缘是其他部分权重的5倍，相当于加强
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        return wbce

    def wiou_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 生成权重，边缘是其他部分权重的5倍，相当于加强
        pred = torch.sigmoid(pred)  # 这里对pred取sigmoid，计算图中logits不需要再做sigmoid
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return wiou

    def region_loss(self, pred, mask):
        pred = torch.sigmoid(pred)  # 这里对pred取sigmoid，计算图中logits不需要再做sigmoid

        # 该部分损失更加强调前景区域或者背景区域内部的预测一致性
        numerator_fore = (mask - mask * pred).sum([2, 3])
        denominator_fore = mask.sum([2, 3]) + 1e-8

        numerator_back = ((1 - mask) * pred).sum([2, 3])
        denominator_back = (1 - mask).sum([2, 3]) + 1e-8

        fore = numerator_fore / denominator_fore
        back = numerator_back / denominator_back
        return fore + back

    def forward(self, pred, mask):
        weit = self.wbce_loss(pred, mask)
        wiou = self.wiou_loss(pred, mask)
        region = self.region_loss(pred, mask)
        return (weit + wiou + region).mean()


if __name__ == '__main__':
    image = torch.randn(1, 1, 255, 255)
    mask = torch.randn(1, 1, 255, 255)
    loss_g = GroupLoss()
    loss_res = loss_g(image, mask)
    print(loss_res)
