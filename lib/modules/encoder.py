# -*- coding:utf-8 -*-
# @Time: 2023-9-5 21:19
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: encoder.py
# @ProjectName: PolypNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbones.resnet import resnet50
from lib.backbones.res2net import res2net50_v1b_26w_4s
from lib.backbones.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b4
from lib.backbones.efficientnet import EfficientNet as efficient_net


def select_encoder(backbone_name):
    if backbone_name == 'pvt-v2-b0':
        backbone = pvt_v2_b0()
        print('>>> Using PVT-v2-b0 as backbone.')
    elif backbone_name == 'pvt-v2-b1':
        backbone = pvt_v2_b1()
        print('>>> Using PVT-v2-b1 as backbone.')
    elif backbone_name == 'pvt-v2-b2':
        backbone = pvt_v2_b2()
        print('>>> Using PVT-v2-b2 as backbone.')
    elif backbone_name == 'pvt-v2-b4':
        backbone = pvt_v2_b4()
        print('>>> Using PVT-v2-b4 as backbone.')
    elif backbone_name == 'res2net50':
        backbone = res2net50_v1b_26w_4s()
        print('>>> Using Res2Net50 as backbone.')
    elif backbone_name == 'resnet50':
        backbone = resnet50()
        print('>>> Using ResNet50 as backbone.')
    elif backbone_name == 'efficientnet-b1':
        backbone = efficient_net.from_name('efficientnet-b1')
        print('>>> Using efficientnet-b1 as backbone.')
    elif backbone_name == 'efficientnet-b4':
        backbone = efficient_net.from_name('efficientnet-b4')
        print('>>> Using efficientnet-b4 as backbone.')
    else:
        raise Exception('backbone is not callable !')
    return backbone


def set_backbone_flags(backbone_name):
    if backbone_name == 'pvt-v2-b0' or backbone_name == 'pvt-v2-b1' or backbone_name == 'pvt-v2-b2' or backbone_name == 'pvt-v2-b4':
        backbone_flag = 'PVT'
    elif backbone_name == 'res2net50' or backbone_name == 'resnet50':
        backbone_flag = 'ResNet'
    elif backbone_name == 'efficientnet-b1' or backbone_name == 'efficientnet-b4':
        backbone_flag = 'Efficientnet'
    else:
        raise Exception('backbone_flags is not callable !')
    return backbone_flag


class Encoder(nn.Module):
    def __init__(self, backbone_name=None, pretrained_backbone_path=None):
        super(Encoder, self).__init__()
        # ---- select encoder ----
        self.backbone = select_encoder(backbone_name)
        # ---- load pretrained params ----
        self.init_backbone_params(backbone_name, pretrained_backbone_path)
        # ---- set backbone in_channels and flags ----
        self.backbone_flag = set_backbone_flags(backbone_name)

    def init_backbone_params(self, backbone_name, pretrained_backbone_path):
        # ---- load pretrained params ----
        if pretrained_backbone_path is not None:
            backbone_load_path = pretrained_backbone_path

            backbone_pretrained_dict = torch.load(backbone_load_path)  # 读取参数
            backbone_pretrained_dict = {k: v for k, v in backbone_pretrained_dict.items() if k in self.backbone.state_dict()}  # 过滤参数
            self.backbone.load_state_dict(backbone_pretrained_dict)  # 加载参数
            print(">>> Successfully load pretrained {} params!".format(backbone_name))
        # ---- don't load pretrained params ----
        else:
            # if backbone_name == 'pvt-v2-b0':
            #     backbone_load_path = r'E:\model\backbone\pvt_v2_b0.pth'
            # elif backbone_name == 'pvt-v2-b1':
            #     backbone_load_path = r'E:\model\backbone\pvt_v2_b1.pth'
            # elif backbone_name == 'pvt-v2-b2':
            #     backbone_load_path = r'E:\model\backbone\pvt_v2_b2.pth'
            # elif backbone_name == 'pvt-v2-b4':
            #     backbone_load_path = r'E:\model\backbone\pvt_v2_b4.pth'
            # elif backbone_name == 'res2net50':
            #     backbone_load_path = r'E:\model\backbone\res2net50_v1b_26w_4s-3cf99910.pth'
            # elif backbone_name == 'resnet50':
            #     backbone_load_path = r'E:\model\backbone\resnet50-0676ba61.pth'
            # elif backbone_name == 'efficientnet-b1':
            #     backbone_load_path = r'E:\model\backbone\efficientnet-b1-f1951068.pth'
            # elif backbone_name == 'efficientnet-b4':
            #     backbone_load_path = r'E:\model\backbone\efficientnet-b4-6ed6700e.pth'
            # else:
            #     raise Exception('pretrained_backbone_path is not callable !')
            print(">>> Backbone do not use pretrained params!")

    def forward(self, x):
        if self.backbone_flag == 'PVT':
            out = self.backbone(x)
            e1 = out[0]  # torch.Size([1, 64, 96, 96])
            e2 = out[1]  # torch.Size([1, 128, 48, 48])
            e3 = out[2]  # torch.Size([1, 320, 24, 24])
            e4 = out[3]  # torch.Size([1, 512, 12, 12])
        elif self.backbone_flag == 'ResNet':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)  # bs, 64, 96, 96
            e1 = self.backbone.layer1(x)  # bs, 256, 96, 96
            e2 = self.backbone.layer2(e1)  # bs, 512, 48, 48
            e3 = self.backbone.layer3(e2)  # bs, 1024, 24, 24
            e4 = self.backbone.layer4(e3)  # bs, 2048, 12, 12
        elif self.backbone_flag == 'Efficientnet':
            endpoints = self.backbone.extract_endpoints(x)
            e1 = endpoints['reduction_2']
            e2 = endpoints['reduction_3']
            e3 = endpoints['reduction_4']
            e4 = endpoints['reduction_5']

        else:
            raise Exception('self.backbone_flag is not callable!')
        return e1, e2, e3, e4


if __name__ == '__main__':
    x = torch.rand(1, 3, 384, 384)
    model = Encoder('efficientnet-b1')
    y = model(x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, )

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print('[efficientnet-b1] macs:', macs, 'params:', params)
    """
    [resnet50] macs: 12.142G params: 23.508M
    [res2net50] macs: 13.363G params: 23.669M
    [pvt-v2-b0] macs: 1.565G params: 3.410M
    [pvt-v2-b1] macs: 5.999G params: 13.496M
    [pvt-v2-b2] macs: 11.458G params: 24.850M
    [pvt-v2-b4] macs: 28.859G params: 62.043M
    [efficientnet-b1] macs: 109.799M params: 62.048K ====> in_channels: [24, 40, 112, 320]
    [efficientnet-b4] macs: 200.024M params: 125.200K ====> in_channels: [32, 56, 160, 448]
    """
