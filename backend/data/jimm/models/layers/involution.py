"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor.nn as nn
import jittor.nn as F
from .conv_bn_act import ConvBnAct
from .create_conv2d import create_conv2d
from .norm_act import BatchNorm2d


class Involution(nn.Module):

    def __init__(
            self,
            channels,
            kernel_size=3,
            stride=1,
            group_size=16,
            rd_ratio=4,
            norm_layer=BatchNorm2d,
            act_layer=nn.ReLU,
    ):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.group_size = group_size
        self.groups = self.channels // self.group_size
        self.conv1 = ConvBnAct(
            in_channels=channels,
            out_channels=channels // rd_ratio,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer)
        self.conv2 = self.conv = create_conv2d(
            in_channels=channels // rd_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1)
        self.avgpool = nn.AvgPool2d(stride, stride) if stride == 2 else nn.Identity()
        # self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def execute(self, x):
        weight = self.conv2(self.conv1(self.avgpool(x)))
        B, C, H, W = weight.shape
        KK = int(self.kernel_size ** 2)
        weight = weight.view(B, self.groups, KK, H, W).unsqueeze(2)
        out = F.unfold(x, self.kernel_size, 1, (self.kernel_size - 1) // 2, self.stride).view(B, self.groups,
                                                                                              self.group_size, KK, H, W)
        out = (weight * out).sum(dim=3).view(B, self.channels, H, W)
        return out
