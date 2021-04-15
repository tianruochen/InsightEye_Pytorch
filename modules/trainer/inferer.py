#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :precdictor.py
# @Time     :2021/3/31 下午4:44
# @Author   :Chang Qing

import os
import numpy as np
import torch.nn.functional as F
from urllib import request
from PIL import Image
from modules.trainer.base import Base
from modules.datasets.argument import build_transform
from utils.visualization import WriterTensorboardX


class Predictor(Base):
    def __init__(self, config):
        self.config = config
        super(Predictor, self).__init__(config.basic, config.arch)
        self.loader = config.loader
        self.scheme = config.scheme
        self.tfms = build_transform(self.loader.data_aug)
        self.writer_infer = None
        if self.scheme.tensorboardx:
            self.model_log_dir = os.path.dirname(
                os.path.dirname(self.arch.best_model if self.arch.best_model else self.arch.resume))
            writer_infer_dir = os.path.join(self.model_log_dir, "infer")
            self.writer_infer = WriterTensorboardX(writer_infer_dir, self.logger, True)

        self.model.eval()

    def _parse_input(self, input):
        input_list = [item for item in input.split(" ") if item]
        label_list = []
        image_list = []
        if len(input_list) > 1:
            image_list = input_list
        else:
            if not input.endswith(".txt"):
                image_list = [input]
            else:
                with open(input, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip().split("\t") == 2:
                            img_path, img_label = line.strip().split("\t")
                        elif line.strip().split("\t") == 1:
                            img_path = line.strip()
                            img_label = None
                        image_list.append(img_path)
                        label_list.append(img_label)

                    image_list, label_list = zip(*[item.strip().split("\t") for item in f.readlines() if
                                                   len(item.strip().split("\t")) == 2])
        if not label_list:
            label_list = [None] * len(image_list)
        return image_list, label_list
    # def _preprocess(self, input):
    #     image_list = self._parse_input(input)
    #     if self.scheme.use_loader and self.loader:
    #         infer_loader =

    def predict(self, input):
        print(input)
        image_list, label_list = self._parse_input(input)
        results = {}
        for i in range(len(image_list)):
            one_result = {}
            image_path = image_list[i]
            if "http" in image_list[i]:
                image_path = request.urlopen(image_path)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.tfms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            pd_score = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            pd_label = str(np.argmax(pd_score))
            print(label_list[i])
            gt_label = label_list[i]
            if gt_label != pd_label and self.writer_infer:
                self.writer_infer.add_image(
                    f"pred_error/label_{self.id2name[str(label_list[i])]}/pred_{self.id2name[str(pd_label)]}",
                    image_tensor.squeeze().cpu())
            one_result["gt_label"] = str(label_list[i]) if label_list[i] is not None else str(None)
            one_result["gt_class"] = self.id2name[str(label_list[i])] if label_list[i] is not None else str(None)
            one_result["scores"] = pd_score
            one_result["pd_label"] = pd_label
            one_result["pd_class"] = self.id2name[str(pd_label)]
            results[image_list[i]] = one_result
        return results
