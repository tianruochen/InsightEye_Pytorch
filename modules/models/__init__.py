#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2021/3/29 下午2:07
# @Author   :Chang Qing

from modules.models.resnet import get_resnet
from modules.models.vggnet import get_vggnet
from modules.models.xception import get_xception
from modules.models.mobilenetv2 import get_mobilenet
from modules.models.efficientnet import get_efficientnet

MODEL_FACTORY = {
    # "resnet": get_resnet,
    # "vggnet": get_vggnet,
    # "xception": get_xception,
    # "mobilenetv2": get_mobilenet,
    "efficentnet": get_efficientnet
}

TIMM_MODEL_ZOO = open("model_name.list").read().split("\n")

def build_model(name, args):
    name, level = name.split("_")
    if name in MODEL_FACTORY:
        return MODEL_FACTORY[name](level, args)
    else:
        assert name in TIMM_MODEL_ZOO
        import timm
        return timm.create_model(name, pretrained=True, **args)
