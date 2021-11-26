#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :fixmatch_loss.py
# @Time     :2021/9/10 下午5:06
# @Author   :Chang Qing

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixMatchLoss(nn.Module):
    def __init__(self, class_num, lambda_u=1.0, threshold=0.95, p_temperature=1.0):
        super(FixMatchLoss, self).__init__()
        self.class_num = class_num
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.p_temperature = p_temperature

    def forward(self, logits_x, targets_x, logits_u_w, logits_u_s):
        loss_labeled = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach() / self.p_temperature, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        loss_unlabeled = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()
        loss = loss_labeled + self.lambda_u * loss_unlabeled

        return loss, loss_labeled, loss_unlabeled
