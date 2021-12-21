"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
import jittor.nn as F
from jittor import nn


class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """

    def execute(self, input: jt.Var) -> jt.Var:
        x = F.matmul_transpose(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        return x
