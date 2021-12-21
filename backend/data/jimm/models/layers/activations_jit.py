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


def swish_jit(x):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    x = x.unary('float64')
    x = x * x.sigmoid()
    x = x.unary('float32')
    return x


def mish_jit(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x * (F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self):
        super(SwishJit, self).__init__()

    def execute(self, x):
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self):
        super(MishJit, self).__init__()

    def execute(self, x):
        return mish_jit(x)


def hard_sigmoid_jit(x):
    return ((x + 3).clamp(min_v=0, max_v=6)) / 6.


class HardSigmoidJit(nn.Module):
    def __init__(self):
        super(HardSigmoidJit, self).__init__()

    def execute(self, x):
        return hard_sigmoid_jit(x)


def hard_swish_jit(x):
    return x * ((x + 3).clamp(min_v=0, max_v=6) / 6.)


class HardSwishJit(nn.Module):
    def __init__(self):
        super(HardSwishJit, self).__init__()

    def execute(self, x):
        return hard_swish_jit(x)


def hard_mish_jit(x):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min_v=0, max_v=2)


class HardMishJit(nn.Module):
    def __init__(self):
        super(HardMishJit, self).__init__()

    def execute(self, x):
        return hard_mish_jit(x)
