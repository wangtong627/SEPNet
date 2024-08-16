# -*- coding:utf-8 -*-
# @Time: 2023-9-4 20:47
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: crc_module.py
# @ProjectName: PolypNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.cbr_block import *


class DynamicFuse(nn.Module):

    def __init__(self, channels=64, channel_reduction=4):
        super(DynamicFuse, self).__init__()
        inter_channels = int(channels // channel_reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ll, x_hl):
        xa = x_ll + x_hl
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x_ll * wei + x_hl * (1 - wei)
        return xo


class DynamicFuse_GN(nn.Module):

    def __init__(self, channels=64, channel_reduction=4):
        super(DynamicFuse_GN, self).__init__()
        inter_channels = int(channels // channel_reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.GroupNorm(inter_channels // 4, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            nn.GroupNorm(channels // 4, channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.GroupNorm(inter_channels // 4, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            # nn.GroupNorm(channels // 4, channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ll, x_hl):
        xa = x_ll + x_hl
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x_ll * wei + x_hl * (1 - wei)
        return xo


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DynamicFocusAndMining(nn.Module):
    def __init__(self, channels, channel_reduction):
        super(DynamicFocusAndMining, self).__init__()
        self.conv_low = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_high = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_focus = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1, activate=True)
        self.conv_mining = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1, activate=True)
        self.conv_res = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1, activate=True)
        self.df_focus = DynamicFuse(channels, channel_reduction[0])
        self.df_res = DynamicFuse(channels, channel_reduction[1])
        self.se = SELayer(channels, 8)

    def forward(self, x_ll, x_hl):
        if x_ll.shape[2:] != x_hl.shape[2:]:
            x_hl = F.interpolate(x_hl, size=(x_ll.shape[2:]), mode='bilinear', align_corners=False)
        f_ll = self.conv_low(x_ll)
        f_hl = self.conv_high(x_hl)
        f_focus = self.conv_focus(self.df_focus(f_ll, f_hl))
        f_mining = self.conv_mining(f_ll - f_hl)
        result = self.conv_res(self.df_res(f_focus, f_mining))
        result = F.relu(self.se(result) + result)
        return result


if __name__ == '__main__':
    input_tensor = torch.rand(1, 16, 64, 64)
    input_tensor2 = torch.rand(1, 16, 64, 64)
    module = DynamicFocusAndMining(16, [4, 4])
    res = module(input_tensor, input_tensor2)
    print(res.shape)
