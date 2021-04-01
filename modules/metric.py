#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :metric.py
# @Time     :2021/3/29 下午4:41
# @Author   :Chang Qing

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def acc(outputs, targets, eps=1e-6):
    """
    :param outputs: N x cls_nums
    :param targets: cls_nums
    :param eps: 1e-6
    :return: right / all
    """
    assert outputs.shape[0] == targets.shape[0]

    cls_nums = outputs.shape[1]
    pred_targets = torch.argmax(outputs, dim=1)
    equal_nums = sum(torch.eq(targets, pred_targets).cpu().tolist())
    acc_for_cls = []
    for i in range(cls_nums):
        cls_i_nums = sum(torch.eq(targets, i).cpu().tolist())
        err_i_nums = sum(torch.bitwise_and(torch.eq(targets, i), torch.eq(pred_targets, i)).cpu().tolist())
        acc_for_cls.append(err_i_nums / (cls_i_nums + eps))
    avg_acc = equal_nums / (outputs.shape[0] + eps)
    # return avg_acc, acc_for_cls
    return avg_acc


def recall(outputs, targets, eps=1e-6):
    pass


def f1_score(ouptuts, targets, eps=1e-6):
    pass


def auc(outputs, targets):
    """
    :param outputs: N x cls_nums
    :param targets: N
    :return: auc
    """
    assert outputs.shape[0] == targets.shape[0]
    bat_nums = outputs.shape[0]
    cls_nums = outputs.shape[1]
    onehot_labels = []
    for i in range(bat_nums):
        onehot_label = [0] * cls_nums
        onehot_label[targets[i]] = 1
        onehot_labels.append(onehot_label)
    # print(onehot_labels)
    # print(outputs.cpu().tolist())
    return roc_auc_score(np.array(onehot_labels), np.array(outputs.cpu().tolist()))


# METRICS_FACTORY = {
#     "acc": acc,
#     "recall": recall,
#     "f1_score": f1_score,
#     "auc": auc
# }
#
#
# def build_metrics(metrics_list):
#     metrics = []
#     for name in metrics_list:
#         if name in METRICS_FACTORY:
#             metrics.append(METRICS_FACTORY[name])
#     return metrics


class Metrics:
    def __init__(self, cls_nums):
        self.eps = 1e-7
        self.cls_nums = cls_nums
        self.reset()

    def update(self, pred, label, loss):
        batch_count = pred.shape[0]
        self.img_nums += batch_count
        self.lose_sum += loss
        pred = F.softmax(pred)
        self.pd_label.extend(pred.cpu().tolist())
        batch_matched = 0

        for i in range(batch_count):
            temp_one_hot = [0] * self.cls_nums
            temp_one_hot[label[i]] = 1
            self.gt_label.append(temp_one_hot)
            self.total_num_for_class[label[i]] += 1
            if torch.argmax(pred[i]) == label[i]:
                self.right_num_for_class[label[i]] += 1
                batch_matched += 1
        batch_accuracy = batch_matched / (batch_count + self.eps)
        return batch_accuracy, batch_matched

    def reset(self):
        self.img_nums = 0
        self.lose_sum = 0
        self.gt_label = []
        self.pd_label = []
        self.error_pd = {}
        self.right_num_for_class = [0] * self.cls_nums
        self.total_num_for_class = [0] * self.cls_nums

    def report(self):
        avg_loss = self.lose_sum / (self.img_nums + self.eps)
        avg_acc = sum(self.right_num_for_class) / (self.img_nums + self.eps)

        gt_label = np.array(self.gt_label)
        pd_label = np.array(self.pd_label)
        avg_auc = roc_auc_score(gt_label, pd_label)
        acc_for_class = [matched / (total + self.eps) for matched, total in zip(self.right_num_for_class, self.total_num_for_class)]
        return avg_loss, avg_acc, avg_auc, acc_for_class



METRICS_FACTORY = {
   "common": Metrics
}


def build_metrics(name, cla_nums):
    return METRICS_FACTORY[name](cla_nums)