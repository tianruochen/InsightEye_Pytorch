#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inference.py
# @Time     :2021/3/26 上午11:08
# @Author   :Chang Qing
import json
import os
import cv2
import random
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

from tqdm import tqdm
from utils.config_util import parse_config
from utils.config_util import merge_config, merge_config_bak
from modules.trainer.inferer import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/image_tag_multi_class_infer.yaml",
                        help="the config file to inference")
    parser.add_argument("--input", type=str, default="/data1/changqing/ZyImage_Data/annotations/test_annotations_20210713/images_paths_20210713.txt", help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=1, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")

    args = parser.parse_args()
    config = parse_config(args.infer_config)
    config = merge_config_bak(config, vars(args))

    # with open("data/url2label.json") as f:
    #     url2label = json.load(f)
    # urls = random.choices(list(url2label.keys()), k=10000)
    # with open("10000_urls.txt", "a") as f:
    #     for i in urls:
    #         f.write(i + "\n")
    predictor = Predictor(config)
    # result = predictor.predict(args.input)
    print(args.input)
    result = predictor.predict_without_label(args.input)
    # result = predictor.predict_for_multi_label(args.input)
    print(result)

    # print("loaded done")
    # for i in range(50):
    #     print(urls[i])
    #     results = predictor.predict(urls[i])
    #     # print(results)
    # tik = cv2.getTickCount()
    # for i in tqdm(range(50)):
    #     results = predictor.predict(urls[i])
    # tok = cv2.getTickCount()
    #
    # avg_cost = (tok - tik) / cv2.getTickFrequency() / 10000
    # print(avg_cost)    # 0.0677736789836


