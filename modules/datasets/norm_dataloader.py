#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :dataloader.py.py
# @Time     :2021/3/19 下午4:36
# @Author   :Chang Qing

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from modules.datasets.norm_dataset import build_dataset
from modules.datasets.custom_collate_fn import PathLabelCollate


class NormDataLoader:
    def __init__(self, img_fmt="rgb", dataset_name="path_label", train_data=None,
                 valid_data=None, batch_size=8, num_workers=4, pin_memory=False, task_type="multi_lalel"):
        self.img_fmt = img_fmt
        self.task_type = task_type
        self.train_data = train_data
        self.valid_data = valid_data
        if self.train_data:
            self.train_data["task_type"] = task_type
        if self.valid_data:
            self.valid_data["task_type"] = task_type
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.train_dataset = build_dataset(dataset_name, self.train_data) if self.train_data else None
        self.valid_dataset = build_dataset(dataset_name, self.valid_data) if self.valid_data else None
        self.collate_fn = PathLabelCollate()

    @property
    def train_dataloader(self):
        if not self.train_dataset:
            return None
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn
        )

    @property
    def valid_dataloader(self):
        if not self.valid_dataset:
            return None
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )


class LmdbDataLoader:
    def __init__(self):
        pass



