#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :precdictor.py
# @Time     :2021/3/31 下午4:44
# @Author   :Chang Qing

import os
import json
import numpy as np
import torch.nn.functional as F
from urllib import request
from PIL import Image
from tqdm import tqdm
from modules.trainer.base import Base
from modules.datasets.argument import build_transform

from utils.analysis_utils import metrix_analysis
from utils.visualization import WriterTensorboardX


class Predictor(Base):
    def __init__(self, config):
        self.config = config
        super(Predictor, self).__init__(config.basic, config.arch)
        self.loader = config.loader
        self.scheme = config.scheme
        self.tfms = build_transform(self.loader.data_aug)
        self.writer_infer = None
        if self.scheme.tensorboardx:
            self.model_log_dir = os.path.dirname(
                os.path.dirname(self.arch.best_model if self.arch.best_model else self.arch.resume))
            writer_infer_dir = os.path.join(self.model_log_dir, "infer")
            self.writer_infer = WriterTensorboardX(writer_infer_dir, self.logger, True)

        self.model.eval()
        self.resutls_dir = os.path.join(os.path.dirname(os.path.dirname(self.arch.best_model)),
                                        self.scheme.results_dir_name)
        os.makedirs(self.resutls_dir, exist_ok=True)
        print(self.resutls_dir)

    def _parse_input(self, input):
        input_list = [item for item in input.split(" ") if item]
        label_list = []
        image_list = []
        with_label = True
        if len(input_list) > 1:
            image_list = input_list
        else:
            if not input.endswith(".txt"):
                image_list = [input]
            else:
                with open(input, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if len(line.strip().split("\t")) == 2:
                            img_path, img_label = line.strip().split("\t")
                        elif (line.strip().split("\t")) == 1:
                            img_path = line.strip()
                            img_label = None
                        else:
                            continue
                        image_list.append(img_path)
                        label_list.append(img_label)

                    # image_list, label_list = zip(*[item.strip().split("\t") for item in f.readlines() if
                    #                                len(item.strip().split("\t")) == 2])
        if not label_list:
            label_list = [None] * len(image_list)
            with_label = False
        return image_list, label_list, with_label

    # def _preprocess(self, input):
    #     image_list = self._parse_input(input)
    #     if self.scheme.use_loader and self.loader:
    #         infer_loader =

    def predict(self, input):
        # print(input)
        # with label: 是否带有标签的标记位
        image_list, label_list, with_label = self._parse_input(input)
        results_path_dict = {}
        results = {}
        scores_matrix = []
        labels_matrix = []
        for i in tqdm(range(len(image_list))):
            one_result = {}
            image_path = image_list[i]
            if "http" in image_list[i]:
                print(image_list[i])
                try:
                    image_path = request.urlopen(image_path, timeout=1).read()
                except:
                    return results

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.tfms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)

            # print("logits:", logits.cpu().tolist())
            pd_score = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

            pd_label = str(np.argmax(pd_score))
            # print(label_list[i])
            gt_label = label_list[i]

            if gt_label != pd_label and self.writer_infer:
                self.writer_infer.add_image(
                    f"pred_error/label_{self.id2name[str(label_list[i])]}/pred_{self.id2name[str(pd_label)]}",
                    image_tensor.squeeze().cpu())
            if gt_label:
                one_result["gt_label"] = str(label_list[i]) if label_list[i] is not None else str(None)
                one_result["gt_class"] = self.id2name[str(label_list[i])] if label_list[i] is not None else str(None)
            one_result["pd_label"] = pd_label
            one_result["pd_class"] = self.id2name[str(pd_label)]
            name_score_dict = {}
            for idx, name in self.id2name.items():
                name_score_dict[name] = pd_score[int(idx)]
            one_result["scores"] = name_score_dict

            results[image_list[i]] = one_result

            if with_label:
                temp_one_hot = [0] * len(pd_score)
                temp_one_hot[int(gt_label)] = 1
                labels_matrix.append(temp_one_hot)
                scores_matrix.append(pd_score)

        if with_label:
            labels_matrix = np.array(labels_matrix)
            scores_matrix = np.array(scores_matrix)
            labels_matrix_path = os.path.join(self.resutls_dir, "labels_matrix.npy")
            scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
            analysis_results_path = os.path.join(self.resutls_dir, "statistic_results.csv")

            np.save(labels_matrix_path, labels_matrix)
            np.save(scores_matrix_path, scores_matrix)
            # except_cls = ["机械交通_私人交通-摩托车", "时尚_时尚穿搭", "情感_情侣日常-秀恩爱", "其他"]
            metrix_analysis(scores_matrix, labels_matrix, self.id2name, to_save_path=analysis_results_path)
            results_path_dict["labels_matrix_path"] = labels_matrix_path
            results_path_dict["scores_matrix_path"] = scores_matrix_path
            results_path_dict["analysis_results_path"] = analysis_results_path

        predict_resutls_path = os.path.join(self.resutls_dir, "predict_results.json")
        json.dump(results, fp=open(predict_resutls_path, "w"), indent=4, ensure_ascii=False)
        results_path_dict["predict_resutls_path"] = predict_resutls_path
        return results_path_dict

    def predict_for_multi_label(self, input):
        # print(input)
        # with label: 是否带有标签的标记位
        image_list = []
        label_list = []
        with open(input, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip().split("\t")) >= 2:
                    item_list = line.strip().split("\t")
                    path = item_list[0]
                    labels = [int(i) for i in item_list[1:]]
                    image_list.append(path)
                    label_list.append(labels)

        results_path_dict = {}
        results = {}
        scores_matrix = []
        labels_matrix = []
        for i in tqdm(range(len(image_list))):
            one_result = {}
            image_path = image_list[i]

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.tfms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)

            # print("logits:", logits.cpu().tolist())
            pd_score = F.sigmoid(logits).squeeze(0).cpu().tolist()

            pd_label = str(np.argmax(pd_score))
            # print(label_list[i])
            gt_labels = label_list[i]

            one_result["gt_label"] = label_list[i] if label_list[i] is not None else []
            one_result["gt_class"] = [self.id2name[str(label_list[i])] if label_list[i] is not None else str(None)]
            one_result["pd_label"] = pd_label
            one_result["pd_class"] = self.id2name[str(pd_label)]
            name_score_dict = {}
            for idx, name in self.id2name.items():
                name_score_dict[name] = pd_score[int(idx)]

            one_result["scores"] = name_score_dict

            results[image_list[i]] = one_result
            temp_one_hot = [0] * len(pd_score)
            for gt_label in gt_labels:
                temp_one_hot[int(gt_label)] = 1
            labels_matrix.append(temp_one_hot)
            scores_matrix.append(pd_score)

        labels_matrix = np.array(labels_matrix)
        scores_matrix = np.array(scores_matrix)
        labels_matrix_path = os.path.join(self.resutls_dir, "labels_matrix.npy")
        scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
        analysis_results_path = os.path.join(self.resutls_dir, "statistic_results.csv")

        np.save(labels_matrix_path, labels_matrix)
        np.save(scores_matrix_path, scores_matrix)
        metrix_analysis(scores_matrix, labels_matrix, self.id2name, to_save_path=analysis_results_path)
        results_path_dict["labels_matrix_path"] = labels_matrix_path
        results_path_dict["scores_matrix_path"] = scores_matrix_path
        results_path_dict["analysis_results_path"] = analysis_results_path

        # # predict_resutls_path = os.path.join(self.resutls_dir, "predict_results.json")
        # json.dump(results, fp=open(predict_resutls_path, "w"), indent=4, ensure_ascii=False)
        # results_path_dict["predict_resutls_path"] = predict_resutls_path
        return results_path_dict



    def predict_without_label(self, input):
        # print(input)
        # with label: 是否带有标签的标记位
        # image_list, label_list, with_label = self._parse_input(input)
        images_list = [image_path for image_path in open(input).read().split("\n") if image_path]
        valid_images_list = []
        invalid_images_list = []
        results_path_dict = {}
        results = {}
        scores_matrix = []
        # labels_matrix = []
        print(f"total images {len(images_list)}")
        for i in tqdm(range(len(images_list))):
            one_result = {}
            image_path = images_list[i]
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.tfms(image).unsqueeze(0)
                image_tensor = image_tensor.to(self.device)
                logits = self.model(image_tensor)

                # print("logits:", logits.cpu().tolist())
                pd_score = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

                pd_label = str(np.argmax(pd_score))
            except:
                invalid_images_list.append(image_path)
                continue

            # print(label_list[i])
            valid_images_list.append(image_path)
            one_result["pd_label"] = pd_label
            one_result["pd_class"] = self.id2name[str(pd_label)]
            name_score_dict = {}
            for idx, name in self.id2name.items():
                name_score_dict[name] = pd_score[int(idx)]
            one_result["scores"] = name_score_dict

            results[image_path] = one_result
            scores_matrix.append(pd_score)

        scores_matrix = np.array(scores_matrix)
        scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
        valid_images_path = os.path.join(self.resutls_dir, "valid_paths.txt")
        invalid_images_path = os.path.join(self.resutls_dir, "invalid_paths.txt")
        predict_resutls_path = os.path.join(self.resutls_dir, "predict_results.json")

        # except_cls = ["机械交通_私人交通-摩托车", "时尚_时尚穿搭", "情感_情侣日常-秀恩爱", "其他"]
        valid_images_list = [img_path + "\n" for img_path in valid_images_list]
        with open(valid_images_path, "w") as f:
            f.writelines(valid_images_list)

        invalid_images_list = [img_path + "\n" for img_path in invalid_images_list]
        with open(invalid_images_path, "w") as f:
            f.writelines(invalid_images_list)
        np.save(scores_matrix_path, scores_matrix)
        print(f"valid images {len(valid_images_list)}")
        print(f"invalid images {len(invalid_images_list)}")

        json.dump(results, fp=open(predict_resutls_path, "w"), indent=4, ensure_ascii=False)

        results_path_dict["scores_matrix_path"] = scores_matrix_path
        results_path_dict["predict_resutls_path"] = predict_resutls_path
        results_path_dict["valid_images_path"] = valid_images_path
        results_path_dict["invalid_images_path"] = invalid_images_path

        return results_path_dict