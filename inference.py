#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inference.py
# @Time     :2021/3/26 上午11:08
# @Author   :Chang Qing

import os
import argparse

os.environ["CUDA_VISIBLE_DIVICES"] = "8,9"

from utils.config_util import parse_config
from utils.config_util import merge_config
from modules.trainer.precdictor import Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/infer_default.yaml",
                        help="the config file to inference")
    parser.add_argument("--image", type=str, default="", help="the image path to inference")
    parser.add_argument("--image_file", type=str, default="", help="the image paths file to inference")
    parser.add_argument("--n_gpus", type=int, default=1, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")

    args = parser.parse_args()
    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    predictor = Predictor(config)
    predictor.predict(args.image)


