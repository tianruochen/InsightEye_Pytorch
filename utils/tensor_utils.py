#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :tensor_utils.py
# @Time     :2021/9/13 上午11:38
# @Author   :Chang Qing


def interleave(x, size):
    # x : [bs+7*2*bs 3 224 224]  size:15
    s = list(x.shape)  # [bs+7*2*bs 3 224 224]
    # [bs,15,3,224,224] --> [15,bs,3,224,224] --> [15*bs, 3,224, 224]
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


if __name__ == '__main__':
    import torch

    a = torch.ones(2, 3, 2, 2)
    b = torch.ones(2, 3, 2, 2) * 2
    c = torch.ones(2, 3, 2, 2) * 3
    en_input = interleave(torch.cat([a,b,c]),size=3)
    print(en_input.shape)
    print(en_input)

    de_input = de_interleave(en_input, size=3)
    print(de_input.shape)
    print(de_input)
