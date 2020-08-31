import torch
import inspect
import json
import logging
import os
import signal
from pathlib import Path

import torch 
import numpy as np 

from libs.tools.checkpoint import *
from libs.tools.optim import *

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine)


def one_hot(tensor, depth, dim= -1, on_value=1.0,dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape),depth,dtype=dtype,device=tensor.device
    )
    tensor_onehot.scatter_(dim,tensor.unsqueeze(dim).long(),on_value)
    return tensor_onehot


def get_paddings_indicator(actual_num,max_num,axis = 0):
    '''Create boolean mask by actually number of a padded tensor.'''
    actual_num = torch.unsqueeze(actual_num,axis +1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis +1] = -1
    max_num = torch.arange(
        max_num,dtype=torch.int,device=actual_num.device).view(max_num_shape)

    paddings_indicator = actual_num.int() > max_num
    #paddings_indicator shape :[batch_size,max_num] 
    return paddings_indicator

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

