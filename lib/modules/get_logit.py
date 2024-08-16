# -*- coding:utf-8 -*-
# @Time: 2023-9-5 19:53
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: get_logit.py
# @ProjectName: PolypNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.cbr_block import BasicBlock


class GetLogitsWithConv1(nn.Module):
    def __init__(self, channel):
        super(GetLogitsWithConv1, self).__init__()
        self.getlogits = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, feat):
        logits = self.getlogits(feat)
        return logits


class GetLogitsWithConv3(nn.Module):
    def __init__(self, channel):
        super(GetLogitsWithConv3, self).__init__()
        self.getlogits = nn.Sequential(BasicBlock(channel, channel, kernel_size=3, stride=1, padding=1, activate=True),
                                       nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, feat):
        logits = self.getlogits(feat)
        return logits


if __name__ == '__main__':
    feat = torch.rand(1, 64, 32, 32)
    logits = GetLogitsWithConv1(64)(feat)
    print(logits.shape)
