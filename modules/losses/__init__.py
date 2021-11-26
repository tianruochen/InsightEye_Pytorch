#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2021/4/6 下午12:00
# @Author   :Chang Qing
 

import torch
from modules.losses.focal_loss import FocalLoss
from modules.losses.fixmatch_loss import FixMatchLoss
from modules.losses.class_balance_loss import ClassBalanceLoss
from modules.losses.cs_loss import CrossEntropyLoss, BCEWithLogitsLoss

LOSS_FACTORY = {
    #"cross_entry_loss": nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.5, 3.0, 1.0]).cuda())
    "bce_loss": torch.nn.BCELoss,
    "bce_with_logits_loss": BCEWithLogitsLoss,
    "cross_entry_loss": CrossEntropyLoss,
    "focal_loss": FocalLoss,
    "class_balance_loss": ClassBalanceLoss,
    "fixmatch_loss": FixMatchLoss
}


def build_loss(loss_name, class_num):
    if loss_name not in LOSS_FACTORY:
        raise NotImplementedError
    return LOSS_FACTORY[loss_name](class_num=class_num)
