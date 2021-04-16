#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inference.py
# @Time     :2021/3/26 上午11:08
# @Author   :Chang Qing
import json
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

from utils.config_util import parse_config
from utils.config_util import merge_config, merge_config_bak
from modules.trainer.inferer import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/screen_classification_infer.yaml",
                        help="the config file to inference")
    parser.add_argument("--input", type=str, default="/data/changqing/InsightEye_Pytorch/data/screen_classification_valid.txt", help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=1, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")

    args = parser.parse_args()
    config = parse_config(args.infer_config)
    config = merge_config_bak(config, vars(args))

    predictor = Predictor(config)
    results = predictor.predict(args.input)
    print(results)
    with open("infer_results.json", "w") as f:
        f.write(json.dumps(results, indent=4))
        print("done")


