#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :imgs_downloader_mp.py
# @Time     :2021/9/3 下午4:11
# @Author   :Chang Qing
 
import os
import time
import random
import requests
import argparse
import traceback

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

requests.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False
random.seed(666)

def download_img(item):
    pid, img_id, url = item.strip().split(" ")
    img_name = os.path.join(imgs_root, f"{pid}_{img_id}.jpg")
    if not os.path.exists(img_name):
        try:
            res = requests.get(url, timeout=1)
            if res.status_code != 200:
                raise Exception
            with open(img_name, "wb") as f:
                f.write(res.content)
        except Exception as e:
            print(pid, img_id, url)
            traceback.print_exc()


def build_url_list(url_files):
    url_list = []
    for url_file in url_files:
        with open(url_file) as f:
            lines = f.readlines()
            url_list.extend(lines)
    return url_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Images Check Script")
    parser.add_argument("--img_root", default="/data1/zhouxuzhi/zyimg_tag_20210720_20210725/img/", type=str,
                        help="the directory of images")
    parser.add_argument("--workers", default=10, type=int, help="the nums of process")
    args = parser.parse_args()

    imgs_root = args.img_root
    workers = args.workers

    os.makedirs(imgs_root, exist_ok=True)

    url_files = glob("/data1/zhouxuzhi/zyimg_tag_20210720_20210725/url/*.txt")
    print(url_files[:5])
    print(f"total files: {len(url_files)}")

    url_list = build_url_list(url_files)
    print(url_list[:5])
    print(f"total items: {len(url_list)}")

    random.shuffle(url_list)
    url_list = url_list[:180000]

    tik_time = time.time()
    # create multiprocess pool
    pool = Pool(workers)  # process num: 20

    # 如果check_img函数仅有1个参数，用map方法
    # pool.map(check_img, img_paths)
    # 如果check_img函数有不止1个参数，用apply_async方法
    # for img_path in tqdm(img_paths):
    #     pool.apply_async(check_img, (img_path, False))
    list(tqdm(iterable=(pool.imap(download_img, url_list)), total=len(url_list)))
    pool.close()
    pool.join()
    tok_time = time.time()
    print(tok_time - tik_time)


