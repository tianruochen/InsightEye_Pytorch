#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :server_uitl.py
# @Time     :2021/4/13 下午7:57
# @Author   :Chang Qing
 


import os
import uuid
import time
import hmac
import hashlib
import base64
import urllib


import datetime
from flask import jsonify
from pymongo import MongoClient
from utils.data_trans import *


def error_resp(error_code, error_message):
    resp = jsonify(error_code=error_code, error_message=error_message)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def log_info(text):
    with open("skymagic_service_log.txt", "a") as f:
        f.write('%s' % datetime.datetime.now())
        f.write('    ')
        f.write(text)
        f.write('\n')
    return


def get_connection(db_params):

    host, port = db_params["host"], db_params["port"]
    db_name, tb_name = db_params["database"], db_params["table"]
    client = MongoClient(host, int(port))
    database = client[db_name]
    table = client[tb_name]
    return table


def write2db(db_params, info):
    collection = get_connection(db_params)
    if type(info) is dict:
        _id = collection.update_one({'_id': info['_id']}, {'$set': info}, upsert=True)
    elif type(info) is list:
        for _ in info:
            _id = collection.update_one({'_id': _['_id']}, {'$set': _}, upsert=True)
    return _id


def parse_and_save_data(data, temp_dir):
    """
    parse_and_save_data
    :param data: post data (json)
    :param temp_dir: (./eval_ouput)
    :return: data_path and data_type(image:0, video:1)
    """
    if "name" in data:
        data_basename = data.get("name")
    else:
        data_basename = "test"

    if 'url' in data:
        url = data.get('url')
        data_path, data_type = url2nparr(data.get('url'), temp_dir, data_basename)
        print(data_path, data_type)
        log_info('Get %s image' % url)
    elif 'image' in data:
        # log_info('Got image buffer')

        data_path, data_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'numpy' in data:
        # log_info('Got numpy string')
        data_path, data_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return data_path, data_type


def parse_and_save_bgsky(data, temp_dir):
    """
        parse_and_save_data
        :param data: post data (json)
        :param temp_dir: (./eval_ouput)
        :return: data_path and data_type(image:0, video:1)
        """
    if "bgsky_name" in data:
        data_basename = data.get("bgsky_name")
    else:
        data_basename = "bgsky_test"

    if 'bgsky_url' in data:
        url = data.get('bgsky_url')
        bgsky_path, bgsky_type = url2nparr(data.get('bgsky_url'), temp_dir, data_basename)
        print(bgsky_path, bgsky_type)
        log_info('Get %s sky background image' % url)
    elif 'bgsky_image' in data:
        # log_info('Got image buffer')
        bgsky_path, bgsky_type = str2nparr(data.get('image'), temp_dir, data_basename)
    elif 'bgsky_numpy' in data:
        # log_info('Got numpy string')
        bgsky_path, bgsky_type = npstr2nparr(data.get('numpy'), temp_dir, data_basename)
    else:
        return None, -1
    # bgsky_type: 0-image  1:video
    return bgsky_path, bgsky_type


def gen_signature(secret):
    timestamp = round(time.time() * 1000)
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign

def check_security(timestamp, sign, secrets):
    cur_timestamp = time.time() - timestamp
    # time out
    if cur_timestamp - timestamp > 60:
        return False, None
    for secret in secrets:
        # generate candidate sign
        secret_encode = secret.encode("utf-8")
        str_sign = "{}\n{}".format(timestamp, secret)
        str_sign_encode = str_sign.encode("utf-8")
        hmac_code = hmac.new(secret_encode, str_sign_encode, digestmod=hashlib.sha256).digest()
        candidate_sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        # match
        if candidate_sign == sign:
            return True, secret
    return False, None

