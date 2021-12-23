"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
from jittor import nn, models
import jittor.nn as F
import math


class AdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
        else:
            raise TypeError(
                f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1:
            return x.reduce("mean", [2], keepdims=True)
        N, C, H = x.shape
        self.sh = math.floor(H / oh)
        self.ksh = H - (oh - 1) * self.sh
        h = (H - self.ksh) // self.sh + 1
        xx = x.reindex([N, C, h, self.ksh], [
            "i0",  # Nid
            "i1",  # Cid
            f"i2*{self.sh}+i3",  # Hid
        ])
        return xx.reduce("mean", [3])


def adaptive_avg_pool2d(x, output_size):
    if isinstance(output_size, int):
        oh = output_size
        ow = output_size
    elif isinstance(output_size, tuple) or isinstance(output_size, list):
        oh = x.shape[2] if output_size[0] is None else output_size[0]
        ow = x.shape[3] if output_size[1] is None else output_size[1]
    else:
        raise TypeError(
            f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
    if oh == 1 and ow == 1:
        return x.reduce("mean", [2, 3], keepdims=True)
    N, C, H, W = x.shape
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    h = (H - ksh) // sh + 1
    w = (W - ksw) // sw + 1
    xx = x.reindex([N, C, h, w, ksh, ksw], [
        "i0",  # Nid
        "i1",  # Cid
        f"i2*{sh}+i4",  # Hid
        f"i3*{sw}+i5",  # Wid
    ])
    return xx.reduce("mean", [4, 5])


def adaptive_max_pool2d(x, output_size):
    if isinstance(output_size, int):
        oh = output_size
        ow = output_size
    elif isinstance(output_size, tuple) or isinstance(output_size, list):
        oh = x.shape[2] if output_size[0] is None else output_size[0]
        ow = x.shape[3] if output_size[1] is None else output_size[1]
    else:
        raise TypeError(
            f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
    if oh == 1 and ow == 1:
        return x.reduce("max", [2, 3], keepdims=True)
    N, C, H, W = x.shape
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    h = (H - ksh) // sh + 1
    w = (W - ksw) // sw + 1
    xx = x.reindex([N, C, h, w, ksh, ksw], [
        "i0",  # Nid
        "i1",  # Cid
        f"i2*{sh}+i4",  # Hid
        f"i3*{sw}+i5",  # Wid
    ])
    return xx.reduce("max", [4, 5])


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = adaptive_avg_pool2d(x, output_size)
    x_max = adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = adaptive_avg_pool2d(x, output_size)
    x_max = adaptive_max_pool2d(x, output_size)
    return jt.concat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class AdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size

    def execute(self, x):
        adaptive_max_pool2d(x, self.output_size)


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def execute(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def execute(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def execute(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
            self.flatten = False
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def execute(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'
