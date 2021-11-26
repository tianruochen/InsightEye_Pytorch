#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :base_trainer.py
# @Time     :2021/3/26 上午11:50
# @Author   :Chang Qing

import os
import json
import math
import logging
import datetime

import torch

from time import time
from collections import deque
from modules.models import build_model
from modules.datasets import build_dataloader
from modules.solver.optimer import build_optimizer, build_lr_scheduler
from modules.losses import build_loss
from modules.metric import build_metrics
from utils.comm_util import ResultsLog
from utils.visualization import WriterTensorboardX
from utils.summary_utils import summary_model


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.task = self.config.task
        self.task_type = self.config.task_type

        # Setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        # train_log/task_arch/datetime/ckpt/
        self.checkpoint_dir = os.path.join(os.path.abspath(self.config.solver.save_dir),
                                           self.config.task + "_" + self.config.arch.type, start_time,
                                           self.config.solver.ckpt_dir)
        # print(self.checkpoint_dir)
        # train_log/task_arch/datetime/log/
        self.models_log_dir = os.path.join(os.path.abspath(self.config.solver.save_dir),
                                           self.config.task + "_" + self.config.arch.type, start_time,
                                           self.config.solver.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_log_dir, exist_ok=True)

        # save config
        self.config_save_path = os.path.join(self.models_log_dir, 'config.json')
        self.results_log_path = os.path.join(self.models_log_dir, 'results_log.json')
        with open(self.config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        self.logger = self._setup_logger()
        self.device, self.device_ids = self._setup_device(self.config.solver.n_gpus)

        # print(self.device)
        # print(self.device_ids)
        # build dataloader
        self.config.loader.args["task_type"] = self.task_type
        self.train_loader, self.valid_loader = build_dataloader(self.config.loader.type, self.config.loader.args)
        assert self.train_loader.dataset.num_classes == self.valid_loader.dataset.num_classes, \
            f"train class num != valid class num:  {self.train_loader.dataset.num_classes} != " \
            f"{self.valid_loader.dataset.num_classes}"
        self.num_classes = self.train_loader.dataset.num_classes
        if not self.config.arch.args.get("num_classes", None):
            self.config.arch.args.num_classes = self.train_loader.dataset.num_classes
        else:
            assert self.config.arch.args.num_classes == self.num_classes, "arch.num_classes error"
        self.label2name = self._build_label_class(self.config.label2name)
        # print(self.label2name)

        # build model
        self.model = build_model(self.config.arch.type, self.config.arch.args)
        summary_model(self.model, input_size=(3, 380, 380), batch_size=1, device="cpu")

        if len(self.device_ids) > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model.to(self.device)

        # build loss
        if self.task_type == "multi_label":
            assert self.config.loss in ["bce_loss", "bce_with_logits_loss"]
        self.loss = build_loss(self.config.loss, self.num_classes)
        # build metrics
        self.metrics = build_metrics(self.config.metrics, self.num_classes)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = build_optimizer(trainable_params, self.config.solver.optimizer.type,
                                         self.config.solver.optimizer.args)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.config.solver.lr_scheduler.type,
                                               self.config.solver.lr_scheduler.args)

        # Setup GPU device if available, move model into configured device
        self.save_freq = config.solver.save_freq
        self.verbosity = config.solver.verbosity
        self.start_epoch = 1
        self.epochs = self.config.solver.epochs
        self.do_validation = self.valid_loader is not None
        self.max_iter = len(self.train_loader) * self.epochs
        self.init_lr = self.optimizer.param_groups[0]['lr']

        # configuration to monitor model performance and save best
        self.monitor = self.config.solver.monitor
        self.monitor_mode = self.config.solver.monitor_mode
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        # setup visualization writer instance
        self.results_log = ResultsLog()
        if self.config.solver.tensorboardx:
            writer_train_dir = os.path.join(self.models_log_dir, "train")
            writer_valid_dir = os.path.join(self.models_log_dir, "valid")
            self.writer_train = WriterTensorboardX(writer_train_dir, self.logger, self.config.solver.tensorboardx)
            self.writer_valid = WriterTensorboardX(writer_valid_dir, self.logger, self.config.solver.tensorboardx)

        # save deque
        self.save_max = self.config.solver.save_max
        self.save_deque = deque()
        # Resume
        self.resume = self.config.solver.resume
        if self.resume:
            self._resume_checkpoint(self.resume)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            # format="[%(asctime)15s] [%(levelname)7s] (%(filename)15s:%(lineno)3s): %(message)s",
            format="[%(asctime)12s] [%(levelname)s] : %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.models_log_dir, "train.log"))
            ]
        )
        return logging.getLogger(self.__class__.__name__)

    def _setup_device(self, n_gpu_need):
        """
            check training gpu environment
            :param n_gpu_need: int
            :return:
            """
        n_gpu_available = torch.cuda.device_count()
        gpu_list_ids = []

        if n_gpu_available == 0 and n_gpu_need > 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        elif n_gpu_need > n_gpu_available:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_need, n_gpu_available))
            n_gpu_need = n_gpu_available
        if n_gpu_need == 0:
            self.logger.info("run models on CPU.")
        else:
            logging.info(f"run model on {n_gpu_need} gpu(s)")
            # gpu_list_str = os.environ["CUDA_VISIBLE_DEVICES"]
            # gpu_list_ids = [int(i) for i in gpu_list_str.split(",")][:n_gpu_need]
            gpu_list_ids = [int(i) for i in range(n_gpu_need)]

        device = torch.device("cuda" if n_gpu_need > 0 else "cpu")
        return device, gpu_list_ids

    def _build_label_class(self, label2name_path):
        label2name = {}
        if os.path.exists(label2name_path):
            label2name = json.loads(open(label2name_path).read())
            if len(label2name.keys()) == self.num_classes:
                return label2name
            else:
                raise ValueError("classes num error")
        else:
            for i in range(self.num_classes):
                label2name[str(i)] = "class_" + str(i)
            return label2name

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("[EPOCH %d]" % (epoch))
            start_time = time()
            result = self._train_epoch(epoch)
            finish_time = time()
            self.logger.info(
                "Finish at {}, Runtime: {:.3f} [s]".format(datetime.datetime.now(), finish_time - start_time))

            # save logged informations into log dict
            # log = {}
            # print(result)
            # for key, value in result.items():
            #     if key == 'train_metrics':
            #         log.update({'train_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
            #     elif key == 'valid_metrics':
            #         log.update({'valid_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
            #     else:
            #         log[key] = value

            # print logged informations to the screen
            if self.results_log is not None:
                self.results_log.add_entry(result)
                if self.verbosity >= 1:
                    self.logger.info(f"=====================The results of epoch:  {epoch} ===================")
                    for key, value in sorted(list(result.items())):
                        self.logger.info('              {:25s}: {}'.format(str(key), value))
                    self.logger.info(f"=============================Report Done ================================")
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    if (self.monitor_mode == 'min' and result[self.monitor] < self.monitor_best) or \
                            (self.monitor_mode == 'max' and result[self.monitor] > self.monitor_best):
                        self.logger.info("Monitor improved from %f to %f" % (self.monitor_best, result[self.monitor]))
                        self.monitor_best = result[self.monitor]
                        best = True
                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor) \
                              + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.logger.warning(msg)

            # Save checkpoint
            self._save_checkpoint(epoch, save_best=best)
        logging.info("******************Training Done..*********************")
        self._save_results_log()

    def _save_results_log(self):
        with open(self.results_log_path, 'w') as f:
            f.write(json.dumps(self.results_log.entries, indent=4, sort_keys=True))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        # Construct savedict
        # arch = type(self.model).__name__  DataParallel
        state = {
            'arch': self.config.arch.type,
            'epoch': epoch,
            'results_log': self.results_log,
            'state_dict': self.model.state_dict(),
            # 'ema_state_dict': self.ema.state_dict() if self.use_ema else None
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            # 'config': self.config
        }

        monitor_best = round(self.monitor_best, 3)
        # Save checkpoint for each epoch
        if self.save_freq is not None:  # Use None mode to avoid over disk space with large models
            if epoch % self.save_freq == 0:
                filename = os.path.join(self.checkpoint_dir, f"{self.task}_{self.config.arch.type}_epoch{epoch}_{self.monitor}{monitor_best}.pth")
                torch.save(state, filename)
                self.logger.info("Saving checkpoint at {}".format(filename))


        # Save the best checkpoint
        if save_best:
            if len(self.save_deque) >= self.save_max:
                need_removed_checkpoint = self.save_deque.popleft()
                os.remove(need_removed_checkpoint)
            checkpoint_name = f"{self.task}_{self.config.arch.type}_epoch{epoch}_{self.monitor}{monitor_best}.pth"  #  arch + "_epoch" + str(epoch) + "_" + self.monitor
            best_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            torch.save(state, best_path)

            best_weight_path = os.path.join(self.checkpoint_dir, "model_best_weight.pth")
            torch.save(self.model.state_dict(), best_weight_path)
            self.save_deque.append(best_path)
            self.logger.info("Saving current best at {}".format(best_path))
        else:
            self.logger.info("Monitor is not improved from %f" % (self.monitor_best))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        # 	self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
        # 						'Optimizer parameters not being resumed.')
        # else:
        # 	self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_log = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch - 1))
