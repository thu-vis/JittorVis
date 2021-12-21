"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
from .cond_conv2d import CondConv2d
from .conv2d_same import create_conv2d_pad


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    depthwise = kwargs.pop('depthwise', False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop('groups', 1)
    if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
        m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    else:
        m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m
