#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :config_util.py
# @Time     :2021/3/26 上午11:18
# @Author   :Chang Qing


import os
from typing import Any

import yaml
import logging

logger = logging.getLogger(__name__)

__all__ = ["parse_config", "merge_config", "print_config"]


class AttrDict(dict):
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getattr__(self, key):
        return self[key]


def recursive_convert(attr_dict):
    if not isinstance(attr_dict, dict):
        return attr_dict
    obj_dict = AttrDict()
    for key, value in attr_dict.items():
        obj_dict[key] = recursive_convert(value)
    return obj_dict


def parse_config(cfg_file):
    with open(cfg_file, "r") as f:
        # == AttrDict(yaml.load(f.read()))
        attr_dict_conf = AttrDict(yaml.load(f, Loader=yaml.Loader))
    obj_dict_conf = recursive_convert(attr_dict_conf)
    return obj_dict_conf


def merge_config(cfg, args_dict):
    loader_cfg_node = getattr(cfg, "loader")
    trainer_cfg_node = getattr(cfg, "solver")
    for key, value in args_dict.items():
        if value is None:
            continue
        try:
            if hasattr(loader_cfg_node, key):
                setattr(loader_cfg_node, key, value)
            if hasattr(trainer_cfg_node, key):
                setattr(trainer_cfg_node, key, value)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            logger.warning(e)
            pass
    if trainer_cfg_node.save_dir and not os.path.exists(trainer_cfg_node.save_dir):
        os.makedirs(trainer_cfg_node.save_dir, exist_ok=True)
    return cfg


def print_config(config):
    print(config)