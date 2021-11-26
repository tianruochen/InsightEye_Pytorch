#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :solver.py
# @Time     :2021/3/29 下午4:41
# @Author   :Chang Qing

import torch


def build_optimizer(params, cls_name, args):
    return getattr(torch.optim, cls_name)(params, **args)


def build_lr_scheduler(optimizer, cls_name, args):
    return getattr(torch.optim.lr_scheduler, cls_name)(optimizer, **args)
