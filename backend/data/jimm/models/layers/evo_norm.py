"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""

import jittor as jt
import jittor.nn as nn


def variance(x, dim, unbiased, keepdim=False) -> jt.Var:
    if isinstance(dim, int):
        dim = (dim,)
    matsize = 1
    for i in dim:
        if i < 0:
            i = x.shape[i]
        matsize *= i
    out = (x - x.mean(dim=dim, keepdim=keepdim)).sqr().sum(dim=dim, keepdim=keepdim)
    if unbiased:
        matsize -= 1
    out /= matsize
    return out


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = jt.ones(param_shape)
        self.bias = jt.zeros(param_shape)
        if apply_act:
            self.v = jt.ones(param_shape)
        # self.register_buffer('running_var', jt.ones(1, num_features, 1, 1))
        self._running_var = jt.ones(1, num_features, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.ones_(self.weight)
        nn.init.constant_(self.weight, 0.)
        # nn.init.zeros_(self.bias)
        nn.init.constant_(self.bias, 0.)
        if self.apply_act:
            # nn.init.ones_(self.v)
            nn.init.constant_(self.v, 1.)

    def execute(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.is_training():
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self._running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) + self._running_var * (1 - self.momentum))
        else:
            var = self._running_var

        if self.apply_act:
            v = self.v.unary(op=x_type)
            d = x * v + (variance(x, dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().unary(op=x_type)
            d = d.max((var + self.eps).sqrt().unary(op=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups = groups
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = jt.ones(param_shape)
        self.bias = jt.zeros(param_shape)
        if apply_act:
            self.v = jt.ones(param_shape)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.ones_(self.weight)
        # nn.init.zeros_(self.bias)
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
        if self.apply_act:
            # nn.init.ones_(self.v)
            nn.init.constant_(self.v, 1.)

    def execute(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * (x * self.v).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (variance(x, dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias
