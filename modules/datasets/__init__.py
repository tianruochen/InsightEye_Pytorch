#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2021/3/26 上午11:49
# @Author   :Chang Qing

from modules.datasets.semi_dataloader import SemiDataLoader
from modules.datasets.norm_dataloader import NormDataLoader, LmdbDataLoader

DATALOADER_FACTORY = {
    "normdataloader": NormDataLoader,
    "lmdbdataloader": LmdbDataLoader,
    "semidataloader": SemiDataLoader
}


def build_dataloader(type, args):
    loader = DATALOADER_FACTORY[type](**args)
    if type == "semidataloader":
        return loader.labeled_dataloader, loader.unlabeled_dataloader, loader.valid_dataloader
    return loader.train_dataloader, loader.valid_dataloader
