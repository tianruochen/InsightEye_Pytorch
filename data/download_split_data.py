#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :aplit_data.py
# @Time     :2021/3/29 下午5:49
# @Author   :Chang Qing
import json
import os
import argparse
import requests
import traceback
import random
from tqdm import tqdm


def download_images(urls, save_dir="/data1/changqing/InduceClick_Data/images/"):

    count = 9963
    os.makedirs(save_dir, exist_ok=True)
    for url in tqdm(urls):
        print(url)
        try:
            image_path = save_dir + "{:0>7d}".format(count) + ".jpg"
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(res.content)
                with open("/data1/changqing/InduceClick_Data/url2path.txt", "a") as f:
                    f.write(url + "\t" + image_path + "\n")
                count += 1
            else:
                continue
        except Exception as e:
            traceback.print_exc()
            pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Split train and test Data")
    # parser.add_argument("--ori_file", type=str, default="urls.txt", help="total file list")
    # parser.add_argument("--save_dir", type=str, default="")
    # args = parser.parse_args()
    #
    # ori_file = args.ori_file
    # with open(ori_file, "r") as f:
    #     urls = [line.strip() for line in f.readlines()]
    # # 根据urls下载图片，并保留url到图片路径的映射关系
    # print(len(urls))
    # url_set = set(urls)
    # print(len(url_set))
    # # download_images(urls[10000:])
    # print("done!")
    # with open("/data1/changqing/InduceClick_Data/url2path.json", "r") as f:
    #     url2path = json.loads(f.read())

    # url_labels.txt 存放的是url + "\t" + label
    # 获取各个类别的url， shuffle后85%作为训练数据，15%作为测试数据
    cla_paths = {}
    with open("path_label_checked.txt", "r") as f:
        path_label_list = [line.strip().split("\t") for line in f.readlines()]
        print(len(path_label_list))
        for path, label in path_label_list:
            if label not in cla_paths:
                cla_paths[label] = [path]
            else:
                cla_paths[label].append(path)
    train_list = []
    test_list = []
    for label, paths in cla_paths.items():
        print(f"label:{label} \t nums: {len(paths)}")
        nums = len(paths)
        random.shuffle(paths)
        # train_list += [path + "\t" + label + "\n" for path in paths[:int(min(nums, 5000) * 0.85)]]
        # test_list += [path + "\t" + label + "\n" for path in paths[int(min(nums, 5000) * 0.85): int(min(nums, 5000) * 0.85) + int(min(nums, 5000) * 0.15)]]

        train_list += [path + "\t" + label + "\n" for path in paths[:int(nums * 0.85)]]
        test_list += [path + "\t" + label + "\n" for path in paths[int(nums * 0.85):]]
    random.shuffle(train_list)
    random.shuffle(test_list)

    # train.txt 保存 path + "\t" + label
    with open("train_imbalance.txt", "w") as f:
        f.writelines(train_list)
    with open("valid_imbalance.txt", "w") as f:
        f.writelines(test_list)

    print("done")
