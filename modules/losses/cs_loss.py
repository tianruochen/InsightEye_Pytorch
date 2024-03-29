#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :cs_loss.py
# @Time     :2021/4/6 下午8:34
# @Author   :Chang Qing
from typing import Optional

import torch.nn as nn
from torch import Tensor


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, class_num, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.class_num = class_num


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self, class_num, weight: Optional[Tensor] = None, size_average=None,reduce=None, reduction: str = 'mean'):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction)
        self.class_num = class_num