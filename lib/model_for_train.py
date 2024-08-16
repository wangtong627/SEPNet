# -*- coding:utf-8 -*-
# @Time: 2023-9-11 14:41
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: revisedfull_model_for train.py
# @ProjectName: PolypNet
"""
revised from BaseModelForTrainWithRF_v2
"""
from lib.modules.encoder import *
from lib.modules.map_module import *
from lib.modules.crc_module import *
from lib.modules.get_logit import *
from lib.modules.pse_module import *
# from lib.modules.cr_block import *  # revised at 2024-2-21
import numpy as np


# paper model
class SEPNet_TrainModel(nn.Module):
    """
    The SEPNet network architecture for training (vallina + MAP + CRC + PSE):
    See SEPNet_EvalModel at 'model_for_eval.py'

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
                 sc_middle_channle_scale=1
                 ):
        super(SEPNet_TrainModel, self).__init__()
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

        # ---- get feat_sc ----
        self.sc2_block = SConvNR3LayersWith2opAndRes_BN_WithChannelReduction(in_channels=backbone_in_planes[1],
                                                                             mid_channels=sc_middle_channle_scale * mid_channels,

                                                                             out_channels=mid_channels,
                                                                             kernel_size=sc_ksize[1],
                                                                             hidden_scale=sc_hidden_scale,
                                                                             ksize=int(sc_gas_ksize * sc_gas_ksize_scale[1]),
                                                                             sigma=sc_gas_sigma,
                                                                             stride=1,
                                                                             padding=sc_ksize[1] // 2
                                                                             )
        self.sc3_block = SConvNR3LayersWith2opAndRes_BN_WithChannelReduction(in_channels=backbone_in_planes[2],
                                                                             mid_channels=sc_middle_channle_scale * mid_channels,

                                                                             out_channels=mid_channels,
                                                                             kernel_size=sc_ksize[2],
                                                                             hidden_scale=sc_hidden_scale,
                                                                             ksize=int(sc_gas_ksize * sc_gas_ksize_scale[2]),
                                                                             sigma=sc_gas_sigma,
                                                                             stride=1,
                                                                             padding=sc_ksize[2] // 2
                                                                             )

        # ---- get dff-feat ----
        if df_channel_reduction is None:
            df_channel_reduction = [4, 4]

        self.dff2_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)
        self.dff3_block = DynamicFocusAndMining(channels=mid_channels, channel_reduction=df_channel_reduction)

        # ---- get logit ----
        self.get_logit1 = GetLogitsWithConv3(channel=mid_channels)
        self.get_logit2 = GetLogitsWithConv1(channel=mid_channels)
        self.get_logit3 = GetLogitsWithConv1(channel=mid_channels)
        self.get_logit4 = GetLogitsWithConv1(channel=mid_channels)

        # ---- get deep_logit ----

        self.get_deep_logit2 = GetLogitsWithConv1(channel=mid_channels)
        self.get_deep_logit3 = GetLogitsWithConv1(channel=mid_channels)

    def forward(self, x):
        # ---- encoder ----
        feat = self.encoder_feat(x)
        e1, e2, e3, e4 = feat[0], feat[1], feat[2], feat[3]

        # ---- rf ----
        rf_e1, rf_e2, rf_e3, rf_e4 = self.rf1_block(e1), self.rf2_block(e2), self.rf3_block(e3), self.rf4_block(e4)

        # ---- df ----
        df_feat3 = self.df3_block(rf_e3, rf_e4)
        df_feat2 = self.df2_block(rf_e2, df_feat3)
        df_feat1 = self.df1_block(rf_e1, df_feat2)

        # ---- get shallow_logit ----
        logit1 = self.get_logit1(df_feat1)
        logit2 = self.get_logit2(df_feat2)
        logit3 = self.get_logit3(df_feat3)
        logit4 = self.get_logit4(rf_e4)

        # ---- get sc_feat and deep_logit ----
        sc_e3 = self.sc3_block(e3, logit4)
        dff_feat3 = self.dff3_block(df_feat3, sc_e3)
        deep_logit3 = self.get_deep_logit3(dff_feat3)

        sc_e2 = self.sc2_block(e2, logit3)
        dff_feat2 = self.dff2_block(df_feat2, sc_e2)
        deep_logit2 = self.get_deep_logit2(dff_feat2)

        return logit1, logit2, logit3, logit4, deep_logit2, deep_logit3


if __name__ == '__main__':
    x = torch.rand(1, 3, 352, 352)

    model = SEPNet_TrainModel(backbone_name='pvt-v2-b2',
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
                              sc_middle_channle_scale=2
                              )
    print(model)

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print('[pvt-v2-b2] macs:', macs, 'params:', params)
