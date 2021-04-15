#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2021/4/6 下午12:00
# @Author   :Chang Qing
 

import torch
import torch.nn as nn
from modules.losses.cs_loss import CrossEntropyLoss
from modules.losses.focal_loss import FocalLoss
from modules.losses.class_balance_loss import ClassBalanceLoss

LOSS_FACTORY = {
    #"cross_entry_loss": nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.5, 3.0, 1.0]).cuda())
    "cross_entry_loss": CrossEntropyLoss,
    "focal_loss": FocalLoss,
    "class_balance_loss": ClassBalanceLoss
}


def build_loss(loss_name, class_num):
    if loss_name not in LOSS_FACTORY:
        raise NotImplementedError
    return LOSS_FACTORY[loss_name](class_num=class_num)
