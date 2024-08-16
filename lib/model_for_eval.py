# -*- coding:utf-8 -*-
# @Time: 2024-2-22 20:55
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: model_for_eval.py
# @ProjectName: PolypNet

import torch

from lib.modules.encoder import *
from lib.modules.map_module import *
from lib.modules.crc_module import *
from lib.modules.get_logit import *
# from lib.modules.cr_block import *  # revised at 2024-2-21
import numpy as np


class SEPNet_EvalModel(nn.Module):
    """
    The SEPNet network architecture for inference (vallina + MAP + CRC):
    due to the plug-and-play design of the PSE module, does not need to load the parameters of the PSE module, thereby saving computational resources.

    Notes:
    In the code, `rf_block`, `df_block`, and `sc_block` correspond respectively to the MAP module, CRC module, and PSE module as mentioned in the paper.
    Our paper link (https://ieeexplore.ieee.org/document/10608167)
    """

    def __init__(self,
                 backbone_name=None,
                 pretrained_backbone_path=None,
                 mid_channels=128,
                 rf_reduction_scale=4,
                 rf_se_channel_reduction=16,
                 df_channel_reduction=None,
                 sc_hidden_scale=32,
                 sc_ksize=None,
                 sc_gas_ksize=31,
                 sc_gas_ksize_scale=None,
                 sc_gas_sigma=None,
                 sc_middle_channle_scale=1):
        super().__init__()

        # ---- get backbone feat ----
        if sc_ksize is None:
            sc_ksize = [3, 3, 3]
        if sc_gas_ksize_scale is None:
            sc_gas_ksize_scale = [1, 1, 1]
        self.encoder_feat = Encoder(backbone_name, pretrained_backbone_path)

        # ---- set backbone in_channels ----
        if backbone_name == 'pvt-v2-b0':
            backbone_in_planes = [32, 64, 160, 256]
        elif backbone_name == 'pvt-v2-b1' or backbone_name == 'pvt-v2-b2' or backbone_name == 'pvt-v2-b4':
            backbone_in_planes = [64, 128, 320, 512]
        elif backbone_name == 'res2net50' or backbone_name == 'resnet50':
            backbone_in_planes = [256, 512, 1024, 2048]
        elif backbone_name == 'efficientnet-b1':
            backbone_in_planes = [24, 40, 112, 320]
        elif backbone_name == 'efficientnet-b4':
            backbone_in_planes = [32, 56, 160, 448]
        else:
            raise Exception('backbone in_channels is not callable !')

        # ---- get rf-feat ----
        self.rf1_block = RFB_LowLevelEnhance_BN_2LayerCR(backbone_in_planes[0], mid_channels, rf_se_channel_reduction)
        self.rf2_block = RFB_PGS_Modified_v2_S_BN_2LayerCR(backbone_in_planes[1], mid_channels, rf_reduction_scale, rf_se_channel_reduction)
        self.rf3_block = RFB_PGS_Modified_v2_S_BN_2LayerCR(backbone_in_planes[2], mid_channels, rf_reduction_scale, rf_se_channel_reduction)
        self.rf4_block = RFB_PGS_Modified_v2_S_BN_2LayerCR(backbone_in_planes[3], mid_channels, rf_reduction_scale, rf_se_channel_reduction)

        # ---- get df-feat ----
        if df_channel_reduction is None:
            df_channel_reduction = [4, 4]
        self.df1_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)
        self.df2_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)
        self.df3_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)

        # ---- get dff-feat ----
        if df_channel_reduction is None:
            df_channel_reduction = [4, 4]
        # self.dff1_block = DynamicFocusAndMining_GN(channels=mid_channels, channel_reduction=df_channel_reduction)
        self.dff2_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)
        self.dff3_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)

        # ---- get logit ----
        self.get_logit1 = GetLogitsWithConv3(channel=mid_channels)
        self.get_logit2 = GetLogitsWithConv1(channel=mid_channels)
        self.get_logit3 = GetLogitsWithConv1(channel=mid_channels)
        self.get_logit4 = GetLogitsWithConv1(channel=mid_channels)

        # ---- get deep_logit ----
        # self.get_deep_logit1 = GetLogitsWithConv1(channel=mid_channels)
        self.get_deep_logit2 = GetLogitsWithConv1(channel=mid_channels)
        self.get_deep_logit3 = GetLogitsWithConv1(channel=mid_channels)

    def forward(self, x):
        # encoder
        feat = self.encoder_feat(x)
        e1, e2, e3, e4 = feat[0], feat[1], feat[2], feat[3]

        # channel reduction
        cr_e1, cr_e2, cr_e3, cr_e4 = self.rf1_block(e1), self.rf2_block(e2), self.rf3_block(e3), self.rf4_block(e4)

        # decoder
        ff_feat3 = self.df3_block(cr_e3, cr_e4)
        ff_feat2 = self.df2_block(cr_e2, ff_feat3)
        ff_feat1 = self.df1_block(cr_e1, ff_feat2)

        # ---- get logit ----
        logit1 = self.get_logit1(ff_feat1)
        logit2 = self.get_logit2(ff_feat2)
        logit3 = self.get_logit3(ff_feat3)
        logit4 = self.get_logit4(cr_e4)

        return logit1, logit2, logit3, logit4


if __name__ == '__main__':
    x = torch.rand(1, 3, 352, 352)
    model = SEPNet_EvalModel(backbone_name='pvt-v2-b2',
                             pretrained_backbone_path=None,
                             mid_channels=64,
                             rf_reduction_scale=4,
                             rf_se_channel_reduction=16,
                             df_channel_reduction=[4, 4],
                             sc_ksize=[3, 3, 3],
                             sc_hidden_scale=32,
                             sc_gas_ksize=1,
                             sc_gas_ksize_scale=[15, 15, 15],
                             sc_gas_sigma=None,
                             sc_middle_channle_scale=2)
    y = model(x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print('[pvt-v2-b2] macs:', macs, 'params:', params)
