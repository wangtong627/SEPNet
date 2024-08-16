# -*- coding:utf-8 -*-
# @Time: 2023-9-19 19:47
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: pse_module.py
# @ProjectName: PolypNet

from torch.autograd import Variable, Function
from lib.modules.cbr_block import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2)
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask
        grad_guide = grad_output.clone().unsqueeze(1) * kernel
        grad_guide = grad_guide.sum(dim=2)
        softmax = torch.softmax(guide_feature, 1)
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])  # 1, c_in, iH, iW
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])  # r * c_out, c_in, kH, kW
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    # x -> bs, in_channels, iH, iW
    # kenel ->  bs, (r*in*out), kW, kH
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])  # bs * r * c_out, c_in, kH,  kW
    px = x.view(1, -1, x.size()[2], x.size()[3])  # 1, bs * c_in, iH, iW
    po = F.conv2d(px, pk, **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))  # 1, bs * c_in, h, w
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow  # 如果有参数加载参数，否则默认（use_slow）
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


def _get_kernel(kernlen=5, nsig=1.1):
    # sigma = 0.3*((ksize-1)*0.5-1)+0.8
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param: in_
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class HAGuide(nn.Module):
    def __init__(self, ksize=5, sigma=None):
        super(HAGuide, self).__init__()
        if sigma is None:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        gaussian_kernel = np.float32(_get_kernel(ksize, sigma))  # 给出高斯卷积核和标准差，初始化高斯核
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]  # [N,C,H,W]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))
        self.padding = ksize // 2

    def forward(self, logit):
        pred = logit.sigmoid()
        soft_attention = F.conv2d(pred, self.gaussian_kernel, padding=self.padding)
        soft_attention = min_max_norm(soft_attention)  # normalization
        holistic_soft_attention = soft_attention.max(pred)
        reverse_holistic_soft_attention = 1 - holistic_soft_attention
        guide_map = torch.cat((holistic_soft_attention, reverse_holistic_soft_attention), dim=1)
        return guide_map


class SConv2d_v3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConv2d_v3, self).__init__()
        self.region_num = region_num

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, hidden_scale * region_num, kernel_size=1),  # revised at 2023.9.2
            nn.Sigmoid(),
            # nn.ReLU(),  # revised at 2023.9.5
            nn.Conv2d(hidden_scale * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
            # 分组卷积，实现对应区域不同的kernel
        )
        # 根据上一层的背景进行预测对应的卷积核
        # self.conv_guide = nn.Conv2d(1, region_num, kernel_size=kernel_size, **kwargs)
        self.get_guide = HAGuide(ksize=ksize, sigma=sigma)  # TODO

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, feat, pred):
        kernel = self.conv_kernel(feat)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # bs, (r*in*out), kH, kW
        output = self.corr(feat, kernel, **self.kwargs)  # bs, r(2)*c_out, iW, iH
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # bs, r(2), c_out, iH, iW
        # pred = F.interpolate(pred, size=feat.shape[2:], mode='bilinear', align_corners=False)
        guide_feature = self.get_guide(pred)  # bs, r(2), iH, iW
        output = self.asign_index(output, guide_feature)
        return output


class SConvNR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR, self).__init__()
        self.sconv = SConv2d_v3(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                region_num=region_num,
                                hidden_scale=hidden_scale,
                                ksize=ksize, sigma=sigma,
                                **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        if x.shape[2:] != y.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.sconv(x, y)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SConvNR_GN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR_GN, self).__init__()
        self.sconv = SConv2d_v3(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                region_num=region_num,
                                hidden_scale=hidden_scale,
                                ksize=ksize, sigma=sigma,
                                **kwargs)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        if x.shape[2:] != y.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.sconv(x, y)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SConvNRWithRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNRWithRes, self).__init__()
        self.sconv_nr = SConvNR(in_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)

    def forward(self, x, y):
        feat_scov_nr = self.sconv_nr(x, y)
        if self.reduction:
            x = self.conv_in_out(x)
        feat_out = F.relu(feat_scov_nr + x)
        return feat_out


class SConvNR3LayersWith2op(nn.Module):
    """used for revised model, remarked as SC_0"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2op, self).__init__()
        self.sconv1 = SConvNR(in_channels, mid_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv2 = SConvNR(mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv3 = SConvNR(out_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)

    def forward(self, x, y):
        x = self.sconv1(x, y)
        x = self.sconv2(x, y)
        x = self.sconv3(x, y)
        return x


class SConvNR3LayersWith2op_GN(nn.Module):
    """used for revised model, remarked as SC_0"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2op_GN, self).__init__()
        self.sconv1 = SConvNR_GN(in_channels, mid_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv2 = SConvNR_GN(mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv3 = SConvNR_GN(out_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)

    def forward(self, x, y):
        x = self.sconv1(x, y)
        x = self.sconv2(x, y)
        x = self.sconv3(x, y)
        return x


class SConvNR3LayersWith2opAndRes(nn.Module):
    """used for revised model, remarked as SC_1"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2opAndRes, self).__init__()
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)
        self.sc_3layer = SConvNR3LayersWith2op(in_channels, mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma,
                                               **kwargs)

    def forward(self, x, y):
        feat_sconv = self.sc_3layer(x, y)
        if self.reduction:
            x = self.conv_in_out(x)
        x = F.relu(feat_sconv + x)
        return x


class SConvNR3LayersWith2opAndRes_GN(nn.Module):
    """used for revised model, remarked as SC_1_GN"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2opAndRes_GN, self).__init__()
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock_GN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)
        self.sc_3layer = SConvNR3LayersWith2op_GN(in_channels, mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize,
                                                  sigma,
                                                  **kwargs)

    def forward(self, x, y):
        feat_sconv = self.sc_3layer(x, y)
        if self.reduction:
            x = self.conv_in_out(x)
        x = F.relu(feat_sconv + x)
        return x


class SConvNR3LayersWith2opAndRes_GN_WithChannelReduction(nn.Module):
    """used for revised model, remarked as SC_1_GN_CR"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2opAndRes_GN_WithChannelReduction, self).__init__()
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock_GN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)
        self.sc_3layer = SConvNR3LayersWith2op_GN(out_channels, mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize,
                                                  sigma,
                                                  **kwargs)
        self.channel_reduction = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                         out_channels=2 * out_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0),
                                               nn.GroupNorm(2 * out_channels // 4, 2 * out_channels),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(in_channels=2 * out_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0),
                                               nn.GroupNorm(out_channels // 4, out_channels),
                                               nn.ReLU(inplace=True)
                                               )

    def forward(self, x, y):
        x_ = self.channel_reduction(x)
        feat_sconv = self.sc_3layer(x_, y)
        if self.reduction:
            x = self.conv_in_out(x)
        x = F.relu(feat_sconv + x)
        return x


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


class SConvNR3LayersWith2opAndRes_BN_WithChannelReduction(nn.Module):
    """used for revised model, remarked as SC_1_GN_CR"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNR3LayersWith2opAndRes_BN_WithChannelReduction, self).__init__()
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)
        self.sc_3layer = SConvNR3LayersWith2op(out_channels, mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize,
                                               sigma,
                                               **kwargs)
        self.channel_reduction = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                         out_channels=2 * out_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0),
                                               nn.BatchNorm2d(2 * out_channels),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(in_channels=2 * out_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=1,  # 可以改成3 * 3
                                                         stride=1,
                                                         padding=0),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(inplace=True)
                                               )
        # self.se = SELayer(out_channels, 16)

    def forward(self, x, y):
        x_ = self.channel_reduction(x)
        feat_sconv = self.sc_3layer(x_, y)
        # feat_sconv = self.se(feat_sconv)
        if self.reduction:
            x = self.conv_in_out(x)
        x = F.relu(feat_sconv + x)
        return x


class SConvNRWithRes3LayersWith2op(nn.Module):
    """used for revised model, remarked as SC_2"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNRWithRes3LayersWith2op, self).__init__()
        self.sconv1 = SConvNRWithRes(in_channels, mid_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv2 = SConvNRWithRes(mid_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)
        self.sconv3 = SConvNRWithRes(out_channels, out_channels, kernel_size, region_num, hidden_scale, ksize, sigma, **kwargs)

    def forward(self, x, y):
        x = self.sconv1(x, y)
        x = self.sconv2(x, y)
        x = self.sconv3(x, y)
        return x


class SConvNRWithRes3LayersWith2opAndRes(nn.Module):
    """used for revised model, remarked as SC_3"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, region_num=2, hidden_scale=8, ksize=15, sigma=None, **kwargs):
        super(SConvNRWithRes3LayersWith2opAndRes, self).__init__()
        self.reduction = False
        if in_channels != out_channels:
            self.reduction = True
            self.conv_in_out = BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, activate=True)
        self.sc_3layer = SConvNRWithRes3LayersWith2op(in_channels, mid_channels, out_channels, kernel_size, region_num, hidden_scale,
                                                      ksize, sigma, **kwargs)

    def forward(self, x, y):
        feat_sconv = self.sc_3layer(x, y)
        if self.reduction:
            x = self.conv_in_out(x)
        x = F.relu(feat_sconv + x)
        return x


if __name__ == '__main__':
    # feat = torch.rand(1, 3, 32, 32)
    logit = torch.rand(1, 1, 32, 32)
    module = HAGuide()
    res = module(logit)
    print(res.shape)
