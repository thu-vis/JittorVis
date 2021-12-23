"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
from typing import Union, Callable, Type

from .activations import *
from .activations_jit import *
from .activations_me import *
from .config import is_exportable, is_scriptable, is_no_jit

_has_silu = False
_has_hardswish = False
_has_hardsigmoid = False
_has_mish = False

# _ACT_FN_DEFAULT = dict(
#     silu=F.silu if _has_silu else swish,
#     swish=F.silu if _has_silu else swish,
#     mish=F.mish if _has_mish else mish,
#     relu=F.relu,
#     relu6=F.relu6,
#     leaky_relu=F.leaky_relu,
#     elu=F.elu,
#     celu=F.celu,
#     selu=F.selu,
#     gelu=gelu,
#     sigmoid=sigmoid,
#     tanh=tanh,
#     hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid,
#     hard_swish=F.hardswish if _has_hardswish else hard_swish,
#     hard_mish=hard_mish,
# )
#
# _ACT_FN_JIT = dict(
#     silu=F.silu if _has_silu else swish_jit,
#     swish=F.silu if _has_silu else swish_jit,
#     mish=F.mish if _has_mish else mish_jit,
#     hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid_jit,
#     hard_swish=F.hardswish if _has_hardswish else hard_swish_jit,
#     hard_mish=hard_mish_jit
# )
#
# _ACT_FN_ME = dict(
#     silu=F.silu if _has_silu else swish_me,
#     swish=F.silu if _has_silu else swish_me,
#     mish=F.mish if _has_mish else mish_me,
#     hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid_me,
#     hard_swish=F.hardswish if _has_hardswish else hard_swish_me,
#     hard_mish=hard_mish_me,
# )
#
# _ACT_FNS = (_ACT_FN_ME, _ACT_FN_JIT, _ACT_FN_DEFAULT)
# for a in _ACT_FNS:
#     a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
#     a.setdefault('hardswish', a.get('hard_swish'))

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=PReLU,
    gelu=GELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
    hard_mish=HardMish,
)

_ACT_LAYER_JIT = dict(
    silu=nn.SiLU if _has_silu else SwishJit,
    swish=nn.SiLU if _has_silu else SwishJit,
    mish=MishJit,
    hard_sigmoid=HardSigmoidJit,
    hard_swish=HardSwishJit,
    hard_mish=HardMishJit
)


# def get_act_fn(name: Union[Callable, str] = 'relu'):
#     """ Activation Function Factory
#     Fetching activation fns by name with this function allows export or torch script friendly
#     functions to be returned dynamically based on current config.
#     """
#     if not name:
#         return None
#     if isinstance(name, Callable):
#         return name
#     if not (is_no_jit() or is_exportable() or is_scriptable()):
#         if name in _ACT_FN_ME:
#             return _ACT_FN_ME[name]
#     if is_exportable() and name in ('silu', 'swish'):
#         return swish
#     if not (is_no_jit() or is_exportable()):
#         if name in _ACT_FN_JIT:
#             return _ACT_FN_JIT[name]
#     return _ACT_FN_DEFAULT[name]


def get_act_layer(name: Union[Type[nn.Module], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if isinstance(name, type):
        return name
    if is_exportable() and name in ('silu', 'swish'):
        return Swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name: Union[nn.Module, str], **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    return act_layer(**kwargs)
