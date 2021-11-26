#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :custom_transform.py
# @Time     :2021/9/10 下午3:28
# @Author   :Chang Qing

from torchvision.transforms import transforms
from modules.datasets.rand_augment import RandAugmentMC


class FixMatchTransform:
    def __init__(self, data_aug, mean, std):
        image_resize = data_aug.image_resize
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(int(image_resize * 1.1), int(image_resize * 1.1))),
            transforms.RandomCrop(size=image_resize,
                                  padding=int(image_resize*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(int(image_resize * 1.1), int(image_resize * 1.1))),
            transforms.RandomCrop(size=image_resize,
                                  padding=int(image_resize*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)