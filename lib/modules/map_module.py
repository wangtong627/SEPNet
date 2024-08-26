# -*- coding:utf-8 -*-
# @Time: 2023-9-4 16:19
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: map_module.py
# @ProjectName: PolypNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.cbr_block import *


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


class PGS_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate, se_channel_reduction=16):
        super(PGS_Block, self).__init__()
        self.left_branch = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
                                         nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
                                         nn.BatchNorm2d(in_channel),
                                         nn.ReLU(inplace=True)
                                         )
        self.right_branch = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
                                          nn.Conv2d(in_channel, in_channel, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
                                          nn.BatchNorm2d(in_channel),
                                          nn.ReLU(inplace=True)
                                          )
        self.channel_smooth = BasicBlock(in_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.se = SELayer(out_channel, se_channel_reduction)
        self.dilation_conv = BasicBlock(out_channel, out_channel, 3, padding=dilation_rate, dilation=dilation_rate, activate=True)

    def forward(self, x):
        left_feat = self.left_branch(x)
        right_feat = self.right_branch(x)
        sm_feat = self.channel_smooth(torch.cat((left_feat, right_feat), dim=1))
        se_feat = self.se(sm_feat)
        out_feat = self.dilation_conv(se_feat)
        return out_feat


class RFB_PGS_Modified_v2_S_BN_2LayerCR(nn.Module):
    """
    Use 2 layers for channels reduction
    """

    def __init__(self, in_channel, out_channel, reduction_scale=4, se_channel_reduction=16):
        super(RFB_PGS_Modified_v2_S_BN_2LayerCR, self).__init__()
        mid_channel = out_channel // reduction_scale

        # ---- channel reduction ----
        self.channel_reduction_1 = nn.Sequential(BasicBlock(in_channel, 2 * mid_channel, kernel_size=1, stride=1, padding=0),
                                                 BasicBlock(2 * mid_channel, mid_channel, kernel_size=1, stride=1, padding=0))
        self.channel_reduction_2 = nn.Sequential(BasicBlock(in_channel, 2 * mid_channel, kernel_size=1, stride=1, padding=0),
                                                 BasicBlock(2 * mid_channel, mid_channel, kernel_size=1, stride=1, padding=0))
        self.channel_reduction_3 = nn.Sequential(BasicBlock(in_channel, 2 * mid_channel, kernel_size=1, stride=1, padding=0),
                                                 BasicBlock(2 * mid_channel, mid_channel, kernel_size=1, stride=1, padding=0))
        self.channel_reduction_4 = nn.Sequential(BasicBlock(in_channel, 2 * mid_channel, kernel_size=1, stride=1, padding=0),
                                                 BasicBlock(2 * mid_channel, mid_channel, kernel_size=1, stride=1, padding=0))

        # ---- get feat ----
        self.get_feat_y1 = PGS_Block(mid_channel, mid_channel, kernel_size=3, dilation_rate=1, se_channel_reduction=se_channel_reduction)
        self.get_feat_y2 = PGS_Block(mid_channel, mid_channel, kernel_size=3, dilation_rate=2, se_channel_reduction=se_channel_reduction)
        self.get_feat_y3 = PGS_Block(mid_channel, mid_channel, kernel_size=5, dilation_rate=3, se_channel_reduction=se_channel_reduction)
        self.get_feat_y4 = PGS_Block(mid_channel, mid_channel, kernel_size=7, dilation_rate=4, se_channel_reduction=se_channel_reduction)

        # ---- get concat feat ----
        self.get_feat_cat = BasicBlock(in_planes=4 * mid_channel, out_planes=out_channel, kernel_size=3, stride=1, padding=1)

        # ---- get se feat ----
        self.se = SELayer(out_channel, se_channel_reduction)

        # ---- get res feat ----
        self.get_feat_res = BasicBlock(in_planes=in_channel, out_planes=out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---- get x ----
        x1, x2, x3, x4 = self.channel_reduction_1(x), self.channel_reduction_2(x), self.channel_reduction_3(x), self.channel_reduction_4(x)
        # ---- get y ----
        y1 = self.get_feat_y1(x1 + x2)
        y2 = self.get_feat_y2(y1 + x2 + x3)
        y3 = self.get_feat_y3(y2 + x3 + x4)
        y4 = self.get_feat_y4(y3 + x4)
        # ---- get concat y ----
        feat_cat = self.get_feat_cat(torch.cat((y1, y2, y3, y4), dim=1))
        feat_se = self.se(feat_cat)
        feat_y = F.relu(feat_se + self.get_feat_res(x))
        return feat_y


class RFB_LowLevelEnhance_BN_2LayerCR(nn.Module):
    """
    Use 2 layers for channels reduction
    """

    def __init__(self, in_channel, out_channel, se_channel_reduction=16):
        super(RFB_LowLevelEnhance_BN_2LayerCR, self).__init__()

        # ---- channel reduction ----
        self.channel_reduction = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            BasicBlock(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

        # ---- get se feat ----
        self.se = SELayer(out_channel, se_channel_reduction)

        # ---- get res feat ----
        self.get_feat_res = BasicBlock(in_planes=in_channel, out_planes=out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---- get x ----
        x1 = self.channel_reduction(x)
        # ---- get y ----
        feat_se = self.se(x1)
        feat_y = F.relu(feat_se + self.get_feat_res(x))
        return feat_y


if __name__ == '__main__':
    input_tensor = torch.rand(1, 32, 64, 64)

    # module = RF_v2_NPD(32, 128, shuffles=4)
    # output = module(input_tensor)
    # print(output.shape)  # torch.Size([1, 32, 64, 64])

    # module_2 = RFB_SINet(32, 128)
    # output2 = module_2(input_tensor)
    # print(output2.shape)  # torch.Size([1, 32, 64, 64]).

    # module_3 = RF_v2_GroupVersion_NPD(32, 128, shuffles=[4, 8])
    # output3 = module_3(input_tensor)
    # print(output3.shape)  # torch.Size([1, 32, 64, 64])
    #
    # module_4 = RF_v1_GroupVersion_NPD(32, 128, shuffles=[4, 8])
    # output4 = module_4(input_tensor)
    # print(output4.shape)  # torch.Size([1, 32, 64, 64])

    module_RFB_PGS_Modified = RFB_PGS_Modified_v2_S_BN_2LayerCR(32, 128)
    out_feat_5 = module_RFB_PGS_Modified(input_tensor)
    print(out_feat_5.shape)

    # flops, params
    from thop import profile
    from thop import clever_format

    # macs, params = profile(module, inputs=(input_tensor,))
    # macs, params = clever_format([macs, params], "%.3f")
    # macs2, params2 = profile(module_2, inputs=(input_tensor,))
    # macs2, params2 = clever_format([macs2, params2], "%.3f")
    # macs3, params3 = profile(module_3, inputs=(input_tensor,))
    # macs3, params3 = clever_format([macs3, params3], "%.3f")
    # macs4, params4 = profile(module_4, inputs=(input_tensor,))
    # macs4, params4 = clever_format([macs4, params4], "%.3f")
    # print('[LP_Block_v2_NPD] macs:', macs, 'params:', params)
    # print('[RFB_SINet] macs2:', macs2, 'params2:', params2)
    # print('[LP_Block_v2_G_NPD] macs:', macs3, 'params:', params3)
    # print('[LP_Block_v1_G_NPD] macs:', macs4, 'params:', params4)

    macs5, params5 = profile(module_RFB_PGS_Modified, inputs=(input_tensor,))
    macs5, params5 = clever_format([macs5, params5], "%.3f")
    print('[RFB_PGS_Modified_v3] macs5:', macs5, 'params5:', params5)

'''
    With dw and pw:
    [LP_Block_v2] macs: 791.020M params: 192.160K
    [RFB_SINet] macs2: 6.356G params2: 1.548M (***default***)
    [LP_Block_v2_G] macs: 1.654G params: 401.984K
    [LP_Block_v1_G] macs: 2.329G params: 566.976K # use 3*3 (default)
    [LP_Block_v1_G] macs: 1.590G params: 386.624K # use 1*1
    
    No Point-wise Conv and Depth-wise Conv:
    [LP_Block_v2_NPD] macs: 921.174M params: 224.032K
    [RFB_SINet] macs2: 6.356G params2: 1.548M
    [LP_Block_v2_G_NPD] macs: 2.041G params: 496.448K
    [LP_Block_v1_G_NPD] macs: 2.505G params: 610.240K # use 3*3
    [LP_Block_v1_G_NPD] macs: 1.767G params: 429.888K # use 1*1
    
    
    [RFB_Res2Net_Modified, reduction=4] macs5: 887.620M params5: 215.360K
    [RFB_Res2Net_Modified, reduction=2] macs5: 2.232G params5: 542.848K
    
    [RFB_Res2Net_Modified_v2, reduction=4] macs5: 925.893M params5: 224.640K
    [RFB_Res2Net_Modified_v2, reduction=2] macs5: 2.384G params5: 579.840K
    
    [RFB_PGS_Modified, reduction=4] macs5: 1.242G params5: 304.384K
    [RFB_PGS_Modified, reduction=2] macs5: 3.646G params5: 891.648K
    
    [RFB_PGS_Modified_v2 reduction=4] macs5: 1.408G params5: 344.832K
    [RFB_PGS_Modified_v2 reduction=2] macs5: 4.305G params5: 1.053M
    
    [RFB_PGS_v3_S_WithRes_Block reduction=4] macs5: 828.902M params5: 203.648K
    
    
    [RFB_PGS_v3_S_WithRes_Block_GroupVersion reduction=4] macs5: 1.777G params5: 436.224K
    [RFB_PGS_v3_S_NoRes_Block_GroupVersion reduction=4] macs5: 1.689G params5: 415.232K
'''
