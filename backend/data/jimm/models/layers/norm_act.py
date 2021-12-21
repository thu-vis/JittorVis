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

from .create_act import get_act_layer


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer()
        else:
            self.act = nn.Identity()

    def _forward_python(self, x):
        return super(BatchNormAct2d, self).execute(x)

    def execute(self, x):
        x = self._forward_python(x)
        x = self.act(x)
        return x


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True,
                 apply_act=True, act_layer=nn.ReLU, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            self.act = act_layer()
        else:
            self.act = nn.Identity()

    def execute(self, x):
        x = super(GroupNormAct, self).execute(x)
        x = self.act(x)
        return x


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, is_train=True, sync=True):
        super(BatchNorm2d, self).__init__()
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = nn.init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = nn.init.constant((num_features,), "float32", 0.0) if affine else 0.0
        self.running_mean = nn.init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = nn.init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        dims = [0] + list(range(2, x.ndim))
        x = x.unary(op='float64')
        if self.is_train:
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x * x, dims=dims)
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean - xmean * xmean).maximum(0.0)
            w = self.weight / jt.sqrt(xvar + self.eps)
            b = self.bias - xmean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

            self.running_mean.update(self.running_mean +
                                     (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var +
                                    (xvar.reshape((-1,)) - self.running_var) * self.momentum)
            return norm_x.unary(op='float32')
        else:
            w = self.weight / jt.sqrt(self.running_var + self.eps)
            b = self.bias - self.running_mean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            return norm_x.unary(op='float32')
