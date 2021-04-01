#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2021/3/26 上午11:49
# @Author   :Chang Qing
 
from modules.datasets.dataloader import NormDataLoader, LmdbDataLoader

DATALOADER_FACTORY = {
    "normdataloader": NormDataLoader,
    "lmdbdataloader": LmdbDataLoader
}


def build_dataloader(type, args):
    # print(type)
    # print(args)
    loader = DATALOADER_FACTORY[type](**args)
    return loader.train_loader, loader.valid_loader
