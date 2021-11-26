#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inference.py
# @Time     :2021/3/26 上午11:08
# @Author   :Chang Qing
import json
import os
import cv2
import random
import pprint
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

from tqdm import tqdm
from utils.config_util import parse_config
from utils.config_util import merge_config
from modules.solver.inferer import Inferer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/imgtag_multi_class_infer.yaml",
                        help="the config file to inference")
    parser.add_argument("--input", type=str, default="", help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=0, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")

    args = parser.parse_args()

    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    # with open("data/url2label.json") as f:
    #     url2label = json.load(f)
    # urls = random.choices(list(url2label.keys()), k=10000)
    # with open("10000_urls.txt", "a") as f:
    #     for i in urls:
    #         f.write(i + "\n")
    predictor = Inferer(config)
    # result = predictor.predict(args.input)
    # args.input = "/data1/changqing/ZyImage_Data/images_new/人物_男性_全身照/0233.png /data1/changqing/ZyImage_Data/images_new/兴趣爱好_美食类_蛋糕/dangao_20210824_0059.jpg /data1/changqing/ZyImage_Data/images_new/动植物_动物_猫/227584142_1521579280.jpg"
    # args.input = "/data1/changqing/ZyImage_Data/annotations/imgtag_cls47_valid_shuffled.txt"
    #
    # args.input = "/data1/changqing/ZyImage_Data/images_new/游戏_手机游戏_和平精英/214534841_1411543924.jpg"
    # args.input = "/data1/changqing/ZyImage_Data/images_new/人物_男性_全身照/0233.png /data1/changqing/ZyImage_Data/images_new/兴趣爱好_美食类_蛋糕/dangao_20210824_0059.jpg /data1/changqing/ZyImage_Data/images_new/动植物_动物_猫/227584142_1521579280.jpg"
    args.input = "/data1/changqing/ZyImage_Data/annotations/imgtag_test_20210726.txt"
    print(args.input)
    # result = predictor.infer_single_img(args.input)
    result = predictor.inference_for_multi_class(args.input)
    pp = pprint.PrettyPrinter(indent=4)
    # result = predictor.predict_for_multi_label(args.input)
    pp.pprint(result)

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


