#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :precdictor.py
# @Time     :2021/3/31 下午4:44
# @Author   :Chang Qing

import os
import json
import time
import numpy as np
import traceback

import torch
import torch.nn.functional as F

from urllib import request
from PIL import Image
from tqdm import tqdm
from modules.solver.base import Base
from modules.datasets import build_dataloader
from modules.datasets.argument import build_transform

from utils.comm_util import AverageMeter
from utils.comm_util import save_to_json, save_to_txt
from utils.analysis_utils import metrix_analysis
from utils.visualization import WriterTensorboardX


def build_path_labels_dict(file_path):
    path2labels = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            item_list = line.strip().split("\t")
            path = item_list[0]

            if len(item_list) >= 2:
                labels = [int(i) for i in item_list[1:]]
                path2labels[path] = labels
            elif len(item_list) == 1:
                path2labels[path] = []
    return path2labels


# class MultiClassPredictor(Inferer):
#     def __init__(self, config):
#         super(Inferer, self).__init__(config.basic, config.arch)
#
#
# class MultiLabelPredictor(Inferer):
#     def __init__(self, config):
#         super(Inferer, self).__init__(config.basic, config.arch)


class Inferer(Base):
    def __init__(self, config):
        self.config = config
        super(Inferer, self).__init__(config.basic, config.arch)
        self.loader = config.loader
        self.solver = config.solver
        self.use_loader = self.solver.use_loader
        self.no_progress = self.solver.no_progress
        # if not self.use_loader:
        #     self.tfms = build_transform(self.loader.data_aug)

        # load checkpoint
        if self.arch.get("resume", None):
            self.logger.info("resume checkpoint...")
            self._resume_checkpoint(self.arch.resume)
        elif self.arch.get("best_model", None):
            self.logger.info("load best model.....")
            self._load_best_model(self.arch.best_model)

        self.writer_infer = None
        if self.solver.tensorboardx:
            self.model_log_dir = os.path.dirname(
                os.path.dirname(self.arch.best_model if self.arch.best_model else self.arch.resume))
            writer_infer_dir = os.path.join(self.model_log_dir, "infer")
            self.writer_infer = WriterTensorboardX(writer_infer_dir, self.logger, True)

        self.model.eval()
        self.resutls_dir = os.path.join(os.path.dirname(os.path.dirname(self.arch.best_model)),
                                        self.solver.results_dir_name)
        os.makedirs(self.resutls_dir, exist_ok=True)
        print(self.resutls_dir)

    def infer_single_img(self, img_path, report_logits=False, report_scores=False):
        result = dict()
        try:
            if "http" in img_path:
                img_path = request.urlopen(img_path, timeout=1).read()
            image = Image.open(img_path).convert("RGB")
            tfms = build_transform(self.loader.data_aug)
            image_tensor = tfms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)

            # print("logits:", logits.cpu().tolist())
            pd_scores = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            pd_logits = logits.squeeze().cpu().tolist()
            pd_score = round(max(pd_scores), 5)
            pd_label = str(np.argmax(pd_scores))
            pd_class = self.id2name[pd_label]
            result[img_path] = {
                "pd_label": pd_label,
                "pd_class": pd_class,
                "pd_score": pd_score
            }
            if report_logits:
                pd_logits = [round(logit, 5) for logit in pd_logits]
                result[img_path]["pd_logits"] = pd_logits
            if report_scores:
                pd_scores = [round(score, 5) for score in pd_scores]
                result[img_path]["pd_scores"] = pd_scores
        except:
            result[img_path] = dict()
            print(f"Input error! Please check!  {img_path}")
            traceback.print_exc()
        return result

    def infer_imgs_list(self, imgs_list, report_logits=False, report_scores=False):
        results = dict()
        for img_path in imgs_list:
            result = self.infer_single_img(img_path, report_logits=report_logits, report_scores=report_scores)
            results.update(result)
        return results

    def infer_txt_with_loader(self, file_path=None, save_matrix=False, analysis_results=False):
        results_path_dict = dict()
        self.config.loader.args["task_type"] = self.task_type
        self.config.loader.args["train_data"] = None
        if file_path:
            self.config.loader.args.valid_data.data_file = file_path
        else:
            file_path = self.config.loader.args.valid_data.data_file
        self.logger.info(f"build infer dataloader from {file_path}")
        path2labels = build_path_labels_dict(file_path)

        data_time = AverageMeter()
        batch_time = AverageMeter()
        start_time = time.time()
        valid_img_paths = []
        valid_gt_labels = []
        logits_matrix = []
        scores_matrix = []
        labels_matrix = []

        _, infer_loader = build_dataloader(self.config.loader.type, self.config.loader.args)

        if not self.no_progress:
            infer_loader = tqdm(infer_loader)

        with torch.no_grad():
            for batch_idx, (batch_imgs, _, batch_img_paths) in enumerate(infer_loader):
                data_time.update(time.time() - start_time)
                self.model.eval()

                batch_imgs = batch_imgs.to(self.device)
                # batch_labels = batch_labels.cpu().tolist()
                batch_logits = self.model(batch_imgs)
                if self.task_type == "multi_class":
                    batch_scores = F.softmax(batch_logits, dim=1).cpu().tolist()
                else:
                    batch_scores = F.sigmoid(batch_logits).cpu().tolist()

                batch_logits = batch_logits.cpu().tolist()
                valid_img_paths.extend(batch_img_paths)
                logits_matrix.extend(batch_logits)
                scores_matrix.extend(batch_scores)
                if not self.no_progress:
                    infer_loader.set_description(
                        "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s ".format(
                            batch=batch_idx + 1,
                            iter=len(infer_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                        ))
            if not self.no_progress:
                infer_loader.close()

        # build results
        self.logger.info("build results...")
        results = dict()
        assert len(valid_img_paths) == len(logits_matrix) == len(scores_matrix), "Error !! length not match"
        for i in tqdm(range(len(valid_img_paths))):
            one_result = {}
            img_path = valid_img_paths[i]
            gt_labels = path2labels[img_path]
            pd_score = scores_matrix[i]
            pd_label = int(np.argmax(pd_score))
            one_result["gt_label"] = []
            one_result["gt_class"] = [self.id2name[str(gt_label)] for gt_label in gt_labels]
            one_result["pd_label_max"] = pd_label
            one_result["pd_class_max"] = self.id2name[str(pd_label)]
            name_score_dict = {}
            for idx, name in self.id2name.items():
                name_score_dict[name] = pd_score[int(idx)]

            one_result["name_score_dict"] = name_score_dict

            results[img_path] = one_result
            temp_one_hot = [0] * len(pd_score)
            for gt_label in gt_labels:
                temp_one_hot[int(gt_label)] = 1
            labels_matrix.append(temp_one_hot)

        logits_matrix = np.array(logits_matrix)
        scores_matrix = np.array(scores_matrix)
        labels_matrix = np.array(labels_matrix)

        if save_matrix:
            labels_matrix_path = os.path.join(self.resutls_dir, "labels_matrix.npy")
            scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
            logits_matrix_path = os.path.join(self.resutls_dir, "logits_matrix.npy")
            valid_imgs_to_save = os.path.join(self.resutls_dir, "valid_imgpaths.txt")
            save_to_txt(valid_img_paths, valid_imgs_to_save)
            if not np.all(labels_matrix <= 0):
                np.save(labels_matrix_path, labels_matrix)
                results_path_dict["labels_matrix_path"] = labels_matrix_path
            np.save(scores_matrix_path, scores_matrix)
            np.save(logits_matrix_path, logits_matrix)
            results_path_dict["valid_imgs_to_save"] = valid_imgs_to_save
            results_path_dict["scores_matrix_path"] = scores_matrix_path
            results_path_dict["logits_matrix_path"] = logits_matrix_path

        analysis_results_path = os.path.join(self.resutls_dir, "statistic_results.csv")
        predict_resutls_path = os.path.join(self.resutls_dir, "predict_results.json")
        save_to_json(results, predict_resutls_path)
        # json.dump(results, fp=open(predict_resutls_path, "w"), indent=4, ensure_ascii=False)
        results_path_dict["predict_resutls_path"] = predict_resutls_path

        if analysis_results:
            metrix_analysis(scores_matrix, labels_matrix, self.id2name, to_save_path=analysis_results_path)
            results_path_dict["analysis_results_path"] = analysis_results_path
        return results_path_dict

    def infer_txt_without_loader(self, file_path, save_matrix=False):
        valid_img_paths = []
        valid_logits = []
        valid_scores = []
        label_list = []
        image_list = []
        with_label = True
        with open(file_path, "r") as f:
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
        return []

    def infer_txt_file(self, file_path, save_matrix=True):
        if self.use_loader:
            results = self.infer_txt_with_loader(file_path, save_matrix=save_matrix)
        else:
            results = self.infer_txt_without_loader(file_path, save_matrix=save_matrix, analysis_results=False)
        return results

    def inference_for_multi_class(self, input, save_logits=True, save_scores=False):
        img_list = [item for item in input.split(" ") if item]
        if len(img_list) > 1:  # 针对多张图像，多个命令行参数
            results = self.infer_imgs_list(img_list, report_logits=save_logits, report_scores=save_scores)
        elif len(img_list) == 1 and not input.endswith(".txt"):  # 针对单张图片，单个本地地址或者url
            results = self.infer_single_img(input, report_logits=save_logits, report_scores=save_scores)
        else:  # 针对大量图像，单个txt文件
            save_matrix = save_logits or save_scores
            results = self.infer_txt_file(input, save_matrix=save_matrix)
        return results

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
        logits_matrix = []
        for i in tqdm(range(len(image_list))):
            one_result = {}
            image_path = image_list[i]

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.tfms(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            pd_logits = logits.squeeze(0).cpu().tolist()
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
            one_result["logits"] = pd_logits

            results[image_list[i]] = one_result
            temp_one_hot = [0] * len(pd_score)
            for gt_label in gt_labels:
                temp_one_hot[int(gt_label)] = 1
            labels_matrix.append(temp_one_hot)
            scores_matrix.append(pd_score)
            logits_matrix.append(pd_logits)

        labels_matrix = np.array(labels_matrix)
        scores_matrix = np.array(scores_matrix)
        labels_matrix_path = os.path.join(self.resutls_dir, "labels_matrix.npy")
        scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
        logits_matrix_path = os.path.join(self.resutls_dir, "logits_matrix.npy")

        analysis_results_path = os.path.join(self.resutls_dir, "statistic_results.csv")

        np.save(labels_matrix_path, labels_matrix)
        np.save(scores_matrix_path, scores_matrix)
        np.save(logits_matrix_path, logits_matrix)
        # metrix_analysis(scores_matrix, labels_matrix, self.id2name, to_save_path=analysis_results_path)
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
        logits_matrix = []
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

                pd_logit = logits.squeeze(0).cpu().tolist()
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
            name_logit_dict = {}
            for idx, name in self.id2name.items():
                name_score_dict[name] = pd_score[int(idx)]
                name_logit_dict[name] = pd_logit[int(idx)]
            one_result["scores"] = name_score_dict
            one_result["logits"] = name_logit_dict

            results[image_path] = one_result
            scores_matrix.append(pd_score)
            logits_matrix.append(pd_logit)

        scores_matrix = np.array(scores_matrix)
        logits_matrix = np.array(logits_matrix)
        scores_matrix_path = os.path.join(self.resutls_dir, "scores_matrix.npy")
        logits_matrix_path = os.path.join(self.resutls_dir, "logits_matrix.npy")
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
        np.save(logits_matrix_path, logits_matrix)
        np.save(logits_matrix_path, logits_matrix)

        print(f"valid images {len(valid_images_list)}")
        print(f"invalid images {len(invalid_images_list)}")

        json.dump(results, fp=open(predict_resutls_path, "w"), indent=4, ensure_ascii=False)

        results_path_dict["scores_matrix_path"] = scores_matrix_path
        results_path_dict["predict_resutls_path"] = predict_resutls_path
        results_path_dict["valid_images_path"] = valid_images_path
        results_path_dict["invalid_images_path"] = invalid_images_path

        return results_path_dict
