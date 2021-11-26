#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :flask_server.py.py
# @Time     :2021/4/13 下午7:52
# @Author   :Chang Qing

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
import time
import json
import argparse

from waitress import serve
from flask import Flask, request, jsonify

from modules.solver.inferer import Predictor
from utils.comm_util import get_time_str
from utils.config_util import parse_config
from utils.server_util import log_info
from utils.server_util import error_resp, check_security

app = Flask(__name__)
# db_param = json.load(open("./config/database_configs/mongo.json"))
# if db_param["durl"]:
#     client = MongoClient(db_param["durl"])
# else:
#     client = MongoClient(host=db_param["host"], port=int(db_param["port"]))
# app.config["collection"] = client[db_param["db"]][db_param["table"]]
app.config["secrets"] = json.load(open("./configs/database_config/secrets.json"))
app.config["port"] = 0


@app.route('/healthcheck')
def healthcheck():
    return error_resp(0, "working")


@app.route("/api/induce_click_infer", methods=["POST"])
def nduce_click_infer():
    if not request.method == "POST":
        return error_resp(1, "Request method error, only support [POST] method")

    data = json.loads(request.data)

    # check time stamp and sign
    # if "timestamp" not in data and "sign" not in data:
    #     return error_resp(2, "Param miss, Both \'timestamp\' and ;\'sign\' are necessary")
    #
    # # check secrets
    # secure, secret = check_security(data.get("timestamp"), data.get("sign"), app.config["secrets"])
    #
    # if not secure:
    #     return error_resp(3, "You need a right signature before post a request")

    # get image url
    start_time = str(get_time_str())
    start_tik = time.time()
    url = data.get("url")

    results = predictor.predict(url)
    end_time = str(get_time_str())
    end_tok = time.time()
    # write to db
    db_info = {
        'start_time': start_time,
        'end_time': end_time,
        'cost_time': str(end_tok - start_tik) + 's',
        'results': results
    }
    print(db_info)
    # write2db(db_info)
    # log_info('Write to db %s' % data.get('url'))

    resp = jsonify(error_code=0, data=db_info)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    log_info('infer %s done' % url)
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InduceClick Inference Flask Server")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/infer_default.yaml",
                        help="path of sky config(yaml file)")
    parser.add_argument("--port", type=int, default=6606, help="service port (default is 6606)")
    # parser.add_argument("--temp_dir", type=str, default="./eval_output/", help="tamp directory for post data")

    args = parser.parse_args()

    config = parse_config(args.infer_config)
    predictor = Predictor(config)

    if args.port:
        app.config["port"] = args.port
    serve(app, host="0.0.0.0", port=int(args.port), threads=3)
