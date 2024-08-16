# -*- coding:utf-8 -*-
# @Time: 2023-9-4 16:22
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: cbr_block.py
# @ProjectName: PolypNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=False, activate=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.activate = activate

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.activate:
            out = self.relu(out)
        return out


class BasicBlock_GN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=False, activate=False):
        super(BasicBlock_GN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.gn = nn.GroupNorm(out_planes // 4, out_planes)
        # self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.activate = activate

    def forward(self, x):
        out = self.conv(x)
        out = self.gn(out)
        if self.activate:
            out = self.relu(out)
        return out


if __name__ == '__main__':
    input = torch.rand(1, 4, 64, 64)
    conv_m = BasicBlock_GN(in_planes=4, out_planes=16, kernel_size=3, stride=1, padding=1)
    output = conv_m(input)
    print('output:', output.shape)

    '''flops, params'''
    from thop import profile
    from thop import clever_format

    macs, params = profile(conv_m, inputs=(input,))
    print('macs:', macs, 'params:', params)
    # macs: 2621440.0 params: 608.0
