#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :precdictor.py
# @Time     :2021/3/31 下午4:44
# @Author   :Chang Qing
 
import torch
from modules.trainer.base import Base

class Predictor(Base):

    def __init__(self, basic, env, arch):
        super(Predictor, self).__init__(basic, env, arch)
        # self.task = config.task
        # self.arch = config.arch
        # self.n_gpus = config.n_gpus
        # self.img_resize = config.image_resize
        # self.label2name = config.label2name
        # self.best_model = config.best_model


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