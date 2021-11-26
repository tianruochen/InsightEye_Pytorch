#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :semi_dataset.py
# @Time     :2021/9/8 下午5:58
# @Author   :Chang Qing

import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from modules.datasets.rand_augment import RandAugmentMC


class FixMatchDataset(Dataset):
    def __init__(self, image_path_file, with_labeled=True, train=True, transform=None, label_transform=None):
        super(FixMatchDataset).__init__()
        self.image_path_file = image_path_file
        self.train = train
        self.with_label = with_labeled
        self.transform = transform
        self.label_transform = label_transform
        self.image_path_list, self.image_label_list = self._parse_image_path_file()
        # print(len(self.image_path_list))
        self.num_classes = self._get_num_class()

    def _get_num_class(self):
        return len(set(self.image_label_list))

    def _parse_image_path_file(self):
        image_path_list = list()
        image_label_list = list()
        assert os.path.exists(self.image_path_file), "Error, image path file is not exist!"
        with open(self.image_path_file) as f:
            lines = f.readlines()
            if self.with_label:
                for line in lines:
                    if not line:
                        continue
                    path, label = line.strip().split("\t")[:2]
                    if not path or not label:
                        continue
                    image_path_list.append(path)
                    image_label_list.append(label)
            else:
                for line in lines:
                    if not line:
                        continue
                    line = line.strip().split("\t")[0]
                    image_path_list.append(line)
        return image_path_list[:100000], image_label_list[:100000]

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.with_label:
            label = self.image_label_list[index]
            if self.label_transform:
                label = self.label_transform(label)
        else:
            label = -1
        # 这里换成int 为什么会报错？
        return image, torch.tensor(int(label), dtype=torch.long), image_path

