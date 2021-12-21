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


def swish_jit_fwd(x):
    return x * x.sigmoid()


def swish_jit_bwd(x, grad_output):
    x_sigmoid = x.sigmoid()
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


def mish_jit_fwd(x):
    return x * F.softplus(x).tanh()


def mish_jit_bwd(x, grad_output):
    x_sigmoid = x.sigmoid()
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output * (x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


def hard_sigmoid_jit_fwd(x):
    return ((x + 3).clamp(min_v=0, max_v=6)) / 6.


def hard_sigmoid_jit_bwd(x, grad_output):
    m = jt.ones_like(x) * ((x >= -3.) & (x <= 3.)) / 6.
    return grad_output * m


def hard_swish_jit_fwd(x):
    return x * ((x + 3).clamp(min_v=0, max_v=6) / 6.)


def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min_v=0, max_v=2)
