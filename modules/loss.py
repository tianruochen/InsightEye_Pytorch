#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :loss.py
# @Time     :2021/3/29 下午4:41
# @Author   :Chang Qing

import torch
import torch.nn as nn

LOSS_FACTORY = {
    #"cross_entry_loss": nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.5, 3.0, 1.0]).cuda())
    "cross_entry_loss": nn.CrossEntropyLoss()
}


def build_loss(loss_name):
    if loss_name not in LOSS_FACTORY:
        raise NotImplementedError
    return LOSS_FACTORY[loss_name]
