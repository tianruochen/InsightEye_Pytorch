#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :test_img.py
# @Time     :2021/3/30 下午9:28
# @Author   :Chang Qing

from PIL import Image
from tqdm import tqdm


def check(file):
    new_img_list = []
    with open(file, "r") as f:
        ori_img_list = [line for line in f.readlines()]
    for img_info in tqdm(ori_img_list):
        img_path = img_info.split("\t")[0]
        try:
            img = Image.open(img_path).convert("RGB")
            h ,w = img.size
            img.verify()
            new_img_list.append(img_info)
        except:
            continue
    with open("path_label_checked.txt", "w") as f:
        f.writelines(new_img_list)
    print(f"ori image numbers: {len(ori_img_list)}")
    print(f"new image numbers: {len(new_img_list)}")
    print("check done")


if __name__ == "__main__":
    file = "path_label.txt"
    check(file)
