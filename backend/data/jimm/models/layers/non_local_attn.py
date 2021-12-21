"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
from jittor import nn
import jittor.nn as F
from jittor import init

from .conv_bn_act import ConvBnAct
from .helpers import make_divisible
from .adaptive_avgmax_pool import adaptive_max_pool2d
from .drop import Dropout2d
from .norm_act import BatchNorm2d


class NonLocalAttn(nn.Module):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(self, in_channels, use_scale=True,  rd_ratio=1/8, rd_channels=None, rd_divisor=8, **kwargs):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.norm = BatchNorm2d(in_channels)
        self.reset_parameters()

    def execute(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, C, H, W = t.shape
        t = t.view(B, C, -1).permute(0, 2, 1)
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).permute(0, 2, 1)

        att = F.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = F.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


class BilinearAttnTransform(nn.Module):

    def __init__(self, in_channels, block_size, groups, act_layer=nn.ReLU, norm_layer=BatchNorm2d):
        super(BilinearAttnTransform, self).__init__()

        self.conv1 = ConvBnAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1))
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size))
        self.conv2 = ConvBnAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        assert block_size == block_size1
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * init.eye((t, t), dtype=x.dtype)
        x = x.view(B * C, block_size, block_size, t, t)
        x = jt.concat(jt.split(x, 1, dim=1), dim=3)
        x = jt.concat(jt.split(x, 1, dim=2), dim=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def execute(self, x):
        assert x.shape[-1] % self.block_size == 0 and x.shape[-2] % self.block_size == 0
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = adaptive_max_pool2d(out, (self.block_size, 1))
        cp = adaptive_max_pool2d(out, (1, self.block_size))
        p = self.conv_p(rp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size)
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size)
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y


class BatNonLocalAttn(nn.Module):
    """ BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
            self, in_channels, block_size=7, groups=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
            drop_rate=0.2, act_layer=nn.ReLU, norm_layer=BatchNorm2d, **_):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.conv1 = ConvBnAct(in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.ba = BilinearAttnTransform(rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer)
        self.conv2 = ConvBnAct(rd_channels, in_channels, 1,  act_layer=act_layer, norm_layer=norm_layer)
        self.dropout = Dropout2d(p=drop_rate)

    def execute(self, x):
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x
