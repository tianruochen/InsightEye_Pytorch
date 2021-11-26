#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :semi_dataloader.py
# @Time     :2021/9/8 下午5:55
# @Author   :Chang Qing


from torch.utils.data import DataLoader
from modules.datasets.argument import build_transform
from modules.datasets.semi_dataset import FixMatchDataset
from modules.datasets.custom_collate_fn import PathLabelCollate, FixMatchCollate
from modules.datasets.custom_transform import FixMatchTransform

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SemiDataLoader:
    def __init__(self, img_fmt="rgb", dataset_name="path_label", mu=3, labeled_data=None, unlabeled_data=None,
                 valid_data=None, batch_size=8, num_workers=4, pin_memory=False, task_type="multi_lalel"):
        self.img_fmt = img_fmt
        self.mu = mu
        self.task_type = task_type
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.valid_data = valid_data
        # if self.labeled_data:
        #     self.labeled_data["task_type"] = task_type
        # if self.valid_data:
        #     self.valid_data["task_type"] = task_type
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.labeled_dataset, self.unlabeled_dataset, self.valid_dataset = self._build_datasets()

        # self.collate_fn = PathLabelCollate()


    def _build_datasets(self):
        labeled_dataset = None
        unlabeled_dataset = None
        valid_dataset = None
        if self.labeled_data:
            labeled_transform = build_transform(self.labeled_data.data_aug)
            labeled_dataset = FixMatchDataset(image_path_file=self.labeled_data.data_path, with_labeled=True,
                                              train=True, transform=labeled_transform)
        if self.unlabeled_data:
            unlabeled_transform = FixMatchTransform(self.labeled_data.data_aug, mean=MEAN, std=STD)
            unlabeled_dataset = FixMatchDataset(image_path_file=self.unlabeled_data.data_path, with_labeled=False,
                                              train=True, transform=unlabeled_transform)
        if self.valid_data:
            valid_transform = build_transform(self.valid_data.data_aug)
            valid_dataset = FixMatchDataset(image_path_file=self.valid_data.data_path, with_labeled=True, train=True,
                                            transform=valid_transform)
        return labeled_dataset, unlabeled_dataset, valid_dataset

    @property
    def labeled_dataloader(self):
        if not self.labeled_dataset:
            return None
        return DataLoader(
            dataset=self.labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=PathLabelCollate()
        )

    @property
    def unlabeled_dataloader(self):
        if not self.unlabeled_dataset:
            return None
        return DataLoader(
            dataset=self.unlabeled_dataset,
            batch_size=self.batch_size * self.mu,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=FixMatchCollate()
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
            collate_fn=PathLabelCollate()
        )
