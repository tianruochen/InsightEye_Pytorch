#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :dataloader.py.py
# @Time     :2021/3/19 下午4:36
# @Author   :Chang Qing

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from modules.datasets.dataset import build_dataset


class NormDataLoader:
    def __init__(self, img_fmt="rgb", dataset_name="path_label", train_data=None,
                 valid_data=None, batch_size=8, num_workers=4, pin_memory=False):
        self.img_fmt = img_fmt
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.train_dataset = build_dataset(dataset_name, self.train_data)
        self.valid_dataset = build_dataset(dataset_name, self.valid_data)

    @property
    def train_loader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    @property
    def valid_loader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class LmdbDataLoader:
    def __init__(self):
        pass



