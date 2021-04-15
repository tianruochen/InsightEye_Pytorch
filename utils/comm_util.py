#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :comm_util.py
# @Time     :2021/3/26 上午11:18
# @Author   :Chang Qing

import os
import time
import json
import yaml
import logging
import logging.config

import torch


class ResultsLog:

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)


def setup_logger(default_path=None, default_level=logging.INFO):
    """
    Set up logging configuration
    :param default_path: file to logging configuration
    :param default_level: logging level (default: logging.INFO)
    :return: root logger
    """
    if default_path and os.path.isfile(default_path):
        with open(default_path, "rt") as f:
            logger_conf = yaml.load(f, Loader=yaml.Loader)
        logging.config.dictConfig(logger_conf)
    else:
        logging.basicConfig(level=default_level, format="[%(asctime)15s][%(levelname)6s][%(filename)s]: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("root")


def setup_device(n_gpu_need):
    """
    check training gpu environment
    :param n_gpu_need: int
    :return:
    """
    logger = logging.getLogger("root")
    n_gpu_available = torch.cuda.device_count()
    gpu_list_ids = []
    if n_gpu_need == 0:
        logger.info("run models on CPU.")
    if n_gpu_available == 0 and n_gpu_need > 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
    elif n_gpu_need > n_gpu_available:
        n_gpu_need = n_gpu_available
        logger.warining(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_need, n_gpu_available))
    else:
        logging.info(f"run model on {n_gpu_need} gpu(s)")
        gpu_list_str = os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_list_ids = [int(i) for i in gpu_list_str.split(",")][:n_gpu_need]
    return n_gpu_need, gpu_list_ids


def get_time_str():
    timestamp = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return time_str
