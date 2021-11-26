#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :custom_collatefn.py
# @Time     :2021/9/10 下午3:29
# @Author   :Chang Qing
 
import torch

class PathLabelCollate(object):
    def __call__(self, batch):
        img_paths = []
        img_tensors = []
        label_tensors = []
        for img_tensor, label_tensor, img_path in batch:
            img_tensors.append(img_tensor)
            label_tensors.append(label_tensor)
            img_paths.append(img_path)
        # 这里一定要用stack() 而不是cat(), stack()会建立一个新的轴
        # 或者先img_tensors.unsqueeze(0), 再cat()
        img_tensors = torch.stack(img_tensors, 0)
        # print(label_tensors)
        label_tensors = torch.tensor(label_tensors)
        # print(img_tensors.shape, label_tensors.shape)
        return img_tensors, label_tensors, img_paths


class FixMatchCollate(object):
    def __call__(self, batch):
        img_paths = []
        labeled_img_tensors = []
        unlabeled_img_tensors = []
        label_tensors = []
        for (labeled_img_tensor, unlabeled_img_tensor), label_tensor, img_path in batch:
            labeled_img_tensors.append(labeled_img_tensor)
            unlabeled_img_tensors.append(unlabeled_img_tensor)
            label_tensors.append(label_tensor)
            img_paths.append(img_path)
        # 这里一定要用stack() 而不是cat(), stack()会建立一个新的轴
        # 或者先img_tensors.unsqueeze(0), 再cat()
        labeled_img_tensors = torch.stack(labeled_img_tensors, 0)
        unlabeled_img_tensors = torch.stack(unlabeled_img_tensors, 0)

        label_tensors = torch.tensor(label_tensors)
        # print(img_tensors.shape, label_tensors.shape)
        return (labeled_img_tensors, unlabeled_img_tensors), label_tensors, img_paths
