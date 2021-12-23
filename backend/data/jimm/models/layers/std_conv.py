"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import numpy as np
import jittor as jt
import jittor.nn as F
from jittor import nn

from .padding import get_padding, get_padding_value, pad_same


def get_weight(module):
    std, mean = np.std(module.weight.data, axis=(1, 2, 3), keepdims=True), \
                np.mean(module.weight.data, axis=(1, 2, 3), keepdims=True)
    std, mean = jt.array(std), jt.array(mean)
    weight = (module.weight - mean) / (std + module.eps)
    return weight


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def get_weight(self):
        matsize = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        mean = self.weight.mean(dims=(1, 2, 3), keepdims=True)
        weight = self.weight - mean
        std = (weight.sqr().sum(dims=(1, 2, 3), keepdims=True) / (matsize - 1)).sqrt()
        weight /= (std + self.eps)
        return weight

    def execute(self, x):
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding='SAME', dilation=1,
            groups=1, bias=False, eps=1e-5):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def get_weight(self):
        matsize = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        mean = self.weight.mean(dims=(1, 2, 3), keepdims=True)
        weight = self.weight - mean
        std = (weight.sqr().sum(dims=(1, 2, 3), keepdims=True) / (matsize - 1)).sqrt()
        weight /= (std + self.eps)
        return weight

    def execute(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = jt.full((self.out_channels, 1, 1, 1), gain_init)
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def get_weight(self):
        matsize = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        mean = self.weight.mean(dims=(1, 2, 3), keepdims=True)
        weight = self.weight - mean
        std = (weight.sqr().sum(dims=(1, 2, 3), keepdims=True) / (matsize - 1)).sqrt()
        weight /= (std + self.eps)
        return weight

    def execute(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding='SAME',
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = jt.full((self.out_channels, 1, 1, 1), gain_init)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps

    def get_weight(self):
        matsize = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        mean = self.weight.mean(dims=(1, 2, 3), keepdims=True)
        weight = self.weight - mean
        std = (weight.sqr().sum(dims=(1, 2, 3), keepdims=True) / (matsize - 1)).sqrt()
        weight /= (std + self.eps)
        return weight

    def execute(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
