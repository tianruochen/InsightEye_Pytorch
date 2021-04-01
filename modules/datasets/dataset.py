#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :dataset.py
# @Time     :2021/3/19 下午5:03
# @Author   :Chang Qing

import random
import torch
import numpy as np
import traceback

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PathLabel_Dataset(Dataset):

    def __init__(self, dataset_cfg):
        self.img_label_list = self._parse_data_file(dataset_cfg.data_file)
        # no return
        # random.shuffle(self.img_label_list)
        # self.img_label_list = self.img_label_list[:1000]
        # print(len(self.img_label_list))
        self.is_train = dataset_cfg.is_train
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        # self.tfms = self._build_tfms(**dataset_cfg.data_aug)
        self.tfms = transforms.Compose([
            transforms.Resize((int(380), int(380))),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.num_classes = self._get_num_class()

    def __getitem__(self, index):
        try:
            img_path, img_label = self.img_label_list[index]
            img = Image.open(img_path).convert("RGB")
            # print(img)
            img = self.tfms(img)
            # print(img)
            return img, torch.tensor(int(img_label), dtype=torch.int64)
        except:
            traceback.print_exc()
            self.__getitem__(index + 1)

    def __len__(self):
        return len(self.img_label_list) - 1

    def _get_num_class(self):
        return len(set([label for _, label in self.img_label_list]))

    def _parse_data_file(self, data_file):
        # print(data_file)
        with open(data_file, "r") as f:
            return [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

    def _build_tfms(self, image_resize=380, random_hflip=False, random_vflip=False, random_crop=True,
                    random_rotate=False, gaussianblur=False, random_erase=False, brightness=0.0, contrast=0.0, hue=0.0):
        transform_list = []
        if random_crop:
            transform_list.append(transforms.Resize((int(image_resize * 1.1), int(image_resize * 1.1))))
            transform_list.append(transforms.RandomCrop((int(image_resize), int(image_resize))))
        else:
            transform_list.append(transforms.Resize(int(image_resize), int(image_resize)))
        if random_hflip and random.random() > 0.5:
            transform_list.append(transforms.RandomHorizontalFlip())
        if random_vflip and random.random() > 0.5:
            transform_list.append(transforms.RandomVerticalFlip())
        if random_rotate and random.random() > 0.5:
            degree = random.randint(0, 90)
            transform_list.append(transforms.RandomRotation(degrees=(-degree, degree)))
        if gaussianblur and random.random() > 0.5:
            transform_list.append(transforms.GaussianBlur(kernel_size=3))
        if random_erase:
            transform_list.append(transforms.RandomErasing())
        if random.random() > 0.5:
            brightness = np.clip(brightness, a_min=0.0, a_max=0.5)
            contrast = np.clip(contrast, a_min=0.0, a_max=0.5)
            hue = np.clip(hue, a_min=0.0, a_max=0.5)
            transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, hue=hue))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(self.img_mean, self.img_std))
        tfms = transforms.Compose(transform_list)
        return tfms


class ImageFold_Dataset(Dataset):
    pass


DATASET_FACTORY = {
    "path_label": PathLabel_Dataset,
    "image_fold": ImageFold_Dataset
}


def build_dataset(dataset, args):
    return DATASET_FACTORY[dataset](args)

# class AlignCollate(object):
#
#     def __init__(self, mode, imgsize):
#         self.mode = mode
#         self.imgH, self.imgW = imgsize
#
#         assert self.mode in ["train", "val"], print("mode should be one of train or val]")
#         self.tfms = get_tfms(self.imgH, self.imgW, self.mode)
#
#     def __call__(self, batch_imgs_info):
#         imgs_data = []
#         imgs_path = []
#         imgs_label = []
#         imgs_defficty = []
#
#         for imginfo in batch_imgs_info:
#             [image, label_, deffict_degree] = imginfo
#             try:
#                 # PIL获得的图像是RGB格式的   通过img.size属性获得图片的（宽，高）
#                 # cv2获得的图像是BGR格式的   通过img.shpae属性获得图片的（高，宽）
#                 # 在经过tfms之前先将图片转换为ndarray
#                 # img = cv2.imread(image, flags=cv2.IMREAD_COLOR)
#                 # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 # img = cv2.resize(img,self.imgH, self.imgW)
#                 # if self.mode == "train":
#                 #     img = image_data_augmentation(img)
#                 # tfms 中的Resize等要求输入是PIL Image 不能是ndarray
#                 # img = self.tfms(Image.fromarray(img)).unsqueeze(0)
#                 img = self.tfms(Image.open(image).convert("RGB").resize((self.imgH,
#                                                                          self.imgW))).unsqueeze(0)
#                 imgs_data.append(img)
#                 imgs_label.append(torch.tensor([int(label_)]))
#                 imgs_path.append(image)
#                 imgs_defficty.append(torch.tensor([deffict_degree]))
#             except Exception as ex:
#                 # print(ex)
#                 # print(img)
#                 continue
#         imgs_defficty_tensors = torch.cat(imgs_defficty, 0)
#         imgs_data_tensors = torch.cat(imgs_data, 0)
#         imgs_label_tensors = torch.cat(imgs_label, 0)
#         return imgs_data_tensors, imgs_label_tensors, imgs_path, imgs_defficty_tensors
