"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
import jittor.nn as nn
from .norm_act import BatchNorm2d


class SplitBatchNorm2d(BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            BatchNorm2d(num_features, eps, momentum, affine) for _ in range(num_splits - 1)])

    def execute(self, input: jt.Var):
        if self.is_training():  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = input.split(split_size)
            x = [super().execute(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return jt.concat(x, dim=0)
        else:
            return super().execute(input)


def convert_splitbn_model(module, num_splits=2):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (nn.BatchNorm): input module
        num_splits: number of separate batchnorm layers to split input across
    """
    mod = module
    # if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
    #     return module
    if isinstance(module, nn.BatchNorm):
        mod = SplitBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            num_splits=num_splits)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight = module.weight.clone().stop_grad()
            mod.bias = module.bias.clone().stop_grad()
        for aux in mod.aux_bn:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            if module.affine:
                aux.weight = module.weight.clone().stop_grad()
                aux.bias = module.bias.clone().stop_grad()

    for name, child in module.__dict__.items():
        if child is not None and isinstance(child,nn.Module):
            mod.__setattr__(name, convert_splitbn_model(child, num_splits=num_splits))
    del module
    return mod
