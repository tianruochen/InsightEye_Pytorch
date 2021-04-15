#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :focal_loss.py
# @Time     :2021/4/6 下午3:31
# @Author   :Chang Qing


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.size_average = size_average

        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha)

    def forward(self, logits, labels):
        # N = inputs.size(0)
        # C = inputs.size(1)
        P = F.softmax(logits)  # scores (N, class_num)

        class_mask = torch.zeros_like(logits)  # (N,C) filled 0
        ids = labels.view(-1, 1)  # label (N)
        class_mask.scatter_(dim=1, index=ids.data, value=1.0)  # (N, C) one_hot
        print(class_mask)

        if logits.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_probs = probs.log()

        batch_loss = -alpha * (torch.pow(1-probs, self.gamma)) * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



