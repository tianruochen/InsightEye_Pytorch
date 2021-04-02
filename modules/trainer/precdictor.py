#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :precdictor.py
# @Time     :2021/3/31 下午4:44
# @Author   :Chang Qing
 
import torch
from modules.trainer.base import Base


class Predictor(Base):
    def __init__(self, config):
        self.config = config
        super(Predictor, self).__init__(config.basic, config.arch)
        self.loader = config.loader
        self.scheme = config.scheme

    def predict(self, image):


