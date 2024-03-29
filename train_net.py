#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :train_net.py
# @Time     :2021/3/26 上午11:06
# @Author   :Chang Qing
 
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"

from modules.solver.norm_trainer import NormTrainer
from modules.solver.semi_trainer import SemiTrainer
from utils.config_util import parse_config, merge_config, print_config


def parse_args():

    parser = argparse.ArgumentParser(description="Classification Network Training Script")

    parser.add_argument("--model_config", type=str, default="configs/model_config/imgtag_semi_supervised_train.yaml", help="path to config file")
    parser.add_argument("--batch_size", type=int, default=None, help="training batch size, None to use config setting")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="training learning rate, None to use config setting")
    parser.add_argument("--resume", type=str, default=None, help="path to pretrain weights")
    parser.add_argument("--n_gpu", type=int, default=True, help="default use gpu")
    parser.add_argument("--epoch", type=int, default=None, help="epoch number, 0 for read from config file")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="directory name to save train snapshoot, None to use config setting")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation epoch interval, 0 for no validation")
    parser.add_argument("--log_interval", type=int, default=1, help="mini batch interval to log")
    parser.add_argument("--fix_random_seed", type=bool, default=False,
                        help="If set True, set rand seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    config = parse_config(args.model_config)
    config = merge_config(config, vars(args))
    # print_config(config)
    if config.basic.task_type == "semi_supervised":
        trainer = SemiTrainer(config)
    else:
        trainer = NormTrainer(config)
    trainer.train()
