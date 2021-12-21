"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
from typing import Optional
from collections import OrderedDict
import jittor as jt
from jittor import nn

from .conv2d_same import create_conv2d_pad


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self._modules = OrderedDict()

    def add_module(self, name: str, module: Optional['nn.Module']):
        self._modules[name] = module
        self.__dict__[name] = module

    def values(self):
        return self._modules.values()

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def pop(self, key):
        v = self._modules.pop(key)
        self.__dict__.pop(key)
        return v


class MixedConv2d(ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def execute(self, x):
        x_split = jt.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = jt.concat(x_out, 1)
        return x
