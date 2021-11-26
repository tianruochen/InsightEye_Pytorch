#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :torch2onnx.py
# @Time     :2021/4/22 下午7:11
# @Author   :Chang Qing


import os

from modules.datasets.argument import build_transform

os.chdir("..")
import argparse

import cv2
import torch
import onnxruntime
import numpy as np

from PIL import Image
from modules.solver.inferer import Inferer
from utils.config_util import parse_config, merge_config


def transform_to_onnx(batch_size, config):
    predictor = Inferer(config)
    model = predictor.model
    # set_swish useful only for efficientnet !!!
    model.set_swish(memory_efficient=False)
    # modify following your model and task
    in_size_h = config.loader.data_aug.image_resize
    in_size_w = config.loader.data_aug.image_resize
    input_names = ["input"]
    output_names = ["output"]

    dynamic = False
    if batch_size < 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, in_size_h, in_size_w))
        onnx_file_name = "{}-1_3_{}_{}-dynamic.onnx".format(model.__class__.__name__, in_size_h, in_size_w)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        x = torch.randn((batch_size, 3, in_size_h, in_size_w)).to(predictor.device)
        onnx_file_name = "{}-{}_3_{}_{}-static.onnx".format(model.__class__.__name__, batch_size, in_size_h, in_size_w)
        dynamic_axes = None

    # export the onnx model
    torch.onnx.export(model, x, onnx_file_name,
                      verbose=False,
                      training=False,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    print("Onnx model export done")
    # return onnx model file name
    return onnx_file_name


def model_forward(session, img_path):
    # image = cv2.imread(img_path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)
    # input_h = session.get_inputs()[0].shape[2]
    # input_w = session.get_inputs()[0].shape[3]
    # input_h = input_w = 380
    # image = cv2.resize(image, (input_h, input_w), interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image / 255.0
    # image = image - np.array([0.485, 0.456, 0.406])
    # image = image / np.array([0.229, 0.224, 0.225])
    # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # image = np.expand_dims(image, 0)
    image = Image.open(img_path)
    tfms = build_transform(config.loader.data_aug)
    image = np.expand_dims(tfms(image).numpy(), 0)

    print("Shape of the model input: ", image.shape)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image})[0]
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch (EfficientNet) to Onnx and Run Demo")
    parser.add_argument("--infer_config", type=str, default="configs/model_config/infer_default.yaml",
                        help="the config file to inference")
    parser.add_argument("--input", type=str, default="/data/changqing/InsightEye_Pytorch/data/images/0000001.jpg",
                        help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=1, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")

    args = parser.parse_args()
    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    batch_size = 1
    # convert to onnx model
    onnx_path_demo = transform_to_onnx(batch_size, config)
    # onnx_path_demo = "/data/changqing/InsightEye_Pytorch/Efficietnet_b5-1_3_380_380-static.onnx"
    # # run demo image
    session = onnxruntime.InferenceSession(onnx_path_demo)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    output = model_forward(session, args.input)
    print(type(output), output)
    output = torch.nn.functional.softmax(torch.tensor(output), dim=-1)
    # [0.00681603979319334, 0.00011067902960348874, 0.00018222357903141528, 0.9928910732269287]
    # [[6.8160e-03, 1.1068e-04, 1.8222e-04, 9.9289e-01]]
    # [[8.3317e-03, 1.2117e-04, 2.1826e-04, 9.9133e-01]]
    print(type(output), output)



