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


def swish(x):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    x = x.unary('float64')
    x = x * x.sigmoid()
    x = x.unary('float32')
    return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def execute(self, x):
        return swish(x)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x * F.softplus(x).tanh()


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """

    def __init__(self):
        super(Mish, self).__init__()

    def execute(self, x):
        return mish(x)


def sigmoid(x):
    return x.sigmoid()


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def execute(self, x):
        return x.sigmoid()


def tanh(x):
    return x.tanh()


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def execute(self, x):
        return x.tanh()


def hard_swish(x):
    inner = F.relu6(x + 3.) / 6.0
    return x * inner


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def execute(self, x):
        return hard_swish(x)


def hard_sigmoid(x):
    return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def execute(self, x):
        return hard_sigmoid(x)


def hard_mish(x):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min_v=0, max_v=2)


class HardMish(nn.Module):
    def __init__(self):
        super(HardMish, self).__init__()

    def execute(self, x):
        return hard_mish(x)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def execute(self, input: jt.Var) -> jt.Var:
        return F.prelu(input, self.weight)


def gelu(x: jt.Var) -> jt.Var:
    return F.gelu(x)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """

    def __init__(self):
        super(GELU, self).__init__()

    def execute(self, input: jt.Var) -> jt.Var:
        return F.gelu(input)
