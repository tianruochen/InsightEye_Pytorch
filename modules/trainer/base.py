#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :base.py
# @Time     :2021/3/31 下午5:10
# @Author   :Chang Qing

import os
import json
import math
import logging
import datetime

import torch
import random
import numpy as np
from modules.models import build_model


class Base:

    def __init__(self, basic, arch):
        self.basic = basic
        self.arch = arch

        # basic
        self.name = self.basic.name
        self.version = self.basic.version
        self.task = self.basic.task
        self.seed = self.basic.seed
        self.n_gpus = self.basic.n_gpus
        self.id2name_path = self.basic.id2name

        # set random seed
        if self.seed:
            self._fix_random_seed()

        self.logger = self._setup_logger()
        # set up device
        self.device, self.device_ids = self._setup_device(self.n_gpus)

        # build id to name mapping
        self.id2name = self._build_label_class(self.id2name)
        self.cls_num = len(self.id2name)

        # build model
        self.model = build_model(self.arch.type, self.arch.args)
        if len(self.device_ids) > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model.to(self.device)

        # load checkpoint
        if self.arch.get("resume", None):
            self._resume_checkpoint(self.arch.resume)
        elif self.arch.get("best_model", None):
            self._load_best_model(self.arch.best_model)

    def _fix_random_seed(self):
        random.seed(self.seed)
        torch.random.seed()
        np.random.seed(self.seed)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            # format="[%(asctime)12s] [%(levelname)7s] (%(filename)15s:%(lineno)3s): %(message)s",
            format="[%(asctime)12s] [%(levelname)s] : %(message)s",
            handlers=[
                logging.StreamHandler(),
            ]
        )
        return logging.getLogger(self.__class__.__name__)

    def _setup_device(self, n_gpu_need):
        """
            check training gpu environment
            :param n_gpu_need: int
            :return:
            """
        n_gpu_available = torch.cuda.device_count()
        gpu_list_ids = []

        if n_gpu_available == 0 and n_gpu_need > 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        elif n_gpu_need > n_gpu_available:
            n_gpu_need = n_gpu_available
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_need, n_gpu_available))
        if n_gpu_need == 0:
            self.logger.info("run models on CPU.")
        else:
            logging.info(f"run model on {n_gpu_need} gpu(s)")
            gpu_list_str = os.environ["CUDA_VISIBLE_DEVICES"]
            gpu_list_ids = [int(i) for i in gpu_list_str.split(",")][:n_gpu_need]
        device = torch.device("cuda" if n_gpu_need > 0 else "cpu")
        return device, gpu_list_ids

    def _build_label_class(self, label2name_path):
        id2lable = {}
        if os.path.exists(label2name_path):
            label2name = json.loads(open(label2name_path).read())
            if len(label2name.keys()) == self.cls_num:
                return label2name
            else:
                raise ValueError("classes num error")
        else:
            for i in range(self.cls_num):
                id2lable[str(i)] = "class_" + str(i)
            return id2lable

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.arch:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        # 	self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
        # 						'Optimizer parameters not being resumed.')
        # else:
        # 	self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.results_log = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch - 1))

    def _load_best_model(self, best_model_path):
        """
        load best model (only weight)
        :param best_model_path:
        :return:
        """
        self.model.load_state_dict(torch.load(best_model_path))
