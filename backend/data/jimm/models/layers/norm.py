"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
import jittor.nn as nn
import jittor.nn as F


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def execute(self, x):
        return super().execute(x)


class LayerNorm2d(nn.LayerNorm):
    """ Layernorm for channels of '2d' spatial BCHW tensors """

    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1])

    def execute(self, x: jt.Var) -> jt.Var:
        return super().execute(x)
