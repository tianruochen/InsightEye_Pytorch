#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :data_trans.py
# @Time     :2021/4/13 下午7:58
# @Author   :Chang Qing
 

import os
import json

import uuid
import cv2
import base64
import numpy as np
import urllib
import socket
import requests


def download_data_from_url(url, temp_dir, data_basename):
    try:
        data_type = -1
        resp = requests.head(url)
        print(resp)
        if resp.headers.get('content-type').startswith('video'):
            print("video")
            temp_path = os.path.join(temp_dir, "temp_videos", data_basename)
            if os.path.splitext(temp_path)[-1] == "":
                temp_path = temp_path + ".mp4"
            data_type = 1  # video
        else:
            print("image")
            print(temp_dir)
            temp_path = os.path.join(temp_dir, "temp_images", data_basename)
            print(temp_path)
            if os.path.splitext(temp_path)[-1] == "":
                temp_path = temp_path + ".jpg"
            data_type = 0  # image
        content = urllib.request.urlopen(url, timeout=5).read()
        print(temp_path, data_type)
        with open(temp_path, 'wb') as f:
            f.write(content)
        return temp_path, data_type
    except urllib.URLError:
        return None, -2
    except socket.timeout:
        return None, -3
    except Exception:
        return None, -4


# image or video
def url2nparr(data_url, temp_dir, data_basename):
    data_basename = str(uuid.uuid1()) + data_basename
    try:
        # data_type: 0--image 1--video  <1--download error
        temp_path, data_type = download_data_from_url(data_url, temp_dir, data_basename)
        if temp_path is None:
            print("Download error!")
            return None, None
        else:
            return temp_path, data_type
        # req = urllib.request.urlopen(data_url, data)
        # # bytearray() 方法返回一个新字节数组。这个数组里的元素是可变的，并且每个元素的值范围: 0 <= x < 256
        # img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # # 从网络读取图像数据并转换成图片格式
        # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # return image
    except:
        return None, -1


# only image
def str2nparr(image_str, temp_dir, image_basename):
    image_basename = str(uuid.uuid1()) + image_basename
    temp_path = os.path.join(temp_dir, "temp_images", image_basename)
    if os.path.splitext(temp_path)[-1] == "":
        temp_path = temp_path + ".jpg"
    image_str = base64.b64decode(image_str)
    img_array = np.asarray(bytearray(image_str), dtype=np.uint8)
    # base64str -- > rgb image
    img_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    print(temp_path)
    cv2.imwrite(temp_path, img_rgb)
    return temp_path, 0


# only image
def npstr2nparr(np_str, temp_dir, image_basename):
    image_basename = str(uuid.uuid1()) + image_basename
    temp_path = os.path.join(temp_dir, "temp_images", image_basename)
    if os.path.splitext(temp_path)[-1] == "":
        temp_path = temp_path + ".jpg"
    info = json.loads(np_str)
    size = info['size']
    # frombuffer将data以流的形式读入转化成ndarray对象
    # 第一参数为stream,第二参数为返回值的数据类型
    img_rgb = np.frombuffer(base64.b64decode(info['image']), dtype=np.uint8).reshape(size)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, img_bgr)
    return temp_path, 0

