#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :trainer.py
# @Time     :2021/3/26 上午11:50
# @Author   :Chang Qing

import os
import json
import math
import logging
import datetime

import torch
import numpy as np

from time import time
from collections import deque

from modules.models import build_model
from modules.datasets import build_dataloader
from modules.solver.base import Base
from modules.solver.optimer import build_optimizer, build_lr_scheduler
from modules.losses import build_loss
from modules.metric import build_metrics
from utils.comm_util import ResultsLog
from utils.visualization import WriterTensorboardX
from utils.summary_utils import summary_model


# ------------------------------------------------------------------------------
#  Poly learning-rate Scheduler
# ------------------------------------------------------------------------------
def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    for g in optimizer.param_groups:
        g['lr'] = init_lr * (1 - curr_iter / max_iter) ** power


# ------------------------------------------------------------------------------
#   Class of Trainer
# ------------------------------------------------------------------------------
class Trainer(Base):

    def __init__(self, config):
        self.config = config
        super(Trainer, self).__init__(config.basic, config.arch)

        # Setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        # workshop/task_arch/datetime/ckpt/
        self.checkpoint_dir = os.path.join(os.path.abspath(self.config.solver.save_dir),
                                           self.task_name + "_" + self.task_type + "_" + self.config.arch.arch_type, start_time,
                                           self.config.solver.ckpt_dir)
        # print(self.checkpoint_dir)
        # workshop/task_arch/datetime/log/
        self.models_log_dir = os.path.join(os.path.abspath(self.config.solver.save_dir),
                                           self.task_name + "_" + self.task_type + "_" + self.config.arch.arch_type, start_time,
                                           self.config.solver.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_log_dir, exist_ok=True)

        # save config
        self.runtime_log_path = os.path.join(self.models_log_dir, 'runtime.log')
        self.logger.addHandler(logging.FileHandler(self.runtime_log_path))
        self.config_save_path = os.path.join(self.models_log_dir, 'config.json')
        self.results_log_path = os.path.join(self.models_log_dir, 'results_log.json')
        with open(self.config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        # self.logger = self._setup_logger()
        # self.device, self.device_ids = self._setup_device(self.config.solver.n_gpus)

        # print(self.device)
        # print(self.device_ids)
        # build dataloader
        # self.config.loader.args["task_type"] = self.task_type
        # self.train_loader, self.valid_loader = build_dataloader(self.config.loader.type, self.config.loader.args)
        # assert self.train_loader.dataset.num_classes == self.valid_loader.dataset.num_classes, \
        #     f"train class num != valid class num:  {self.train_loader.dataset.num_classes} != " \
        #     f"{self.valid_loader.dataset.num_classes}"
        # assert self.num_classes == self.train_loader.dataset.num_classes, \
        #     f"self.num_classes != train class num:  {self.num_classes} != " \
        #     f"{self.train_loader.dataset.num_classes}"

        # summary model
        summary_model(self.model, input_size=(3, 380, 380), batch_size=1, device="cuda")

        # build loss

        if self.task_type == "multi_label":
            assert self.config.loss in ["bce_loss", "bce_with_logits_loss"]
        self.loss = build_loss(self.config.loss, self.num_classes)

        # build metrics
        self.metrics = build_metrics(self.config.metrics, self.num_classes)

        # build optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = build_optimizer(trainable_params, self.config.solver.optimizer.type,
                                         self.config.solver.optimizer.args)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.config.solver.lr_scheduler.type,
                                               self.config.solver.lr_scheduler.args)

        # configuration to guide training
        self.save_freq = config.solver.save_freq
        self.verbosity = config.solver.verbosity
        self.start_epoch = 1
        self.epochs = self.config.solver.epochs
        # self.do_validation = self.valid_loader is not None
        # self.max_iter = len(self.train_loader) * self.epochs
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
        if self.arch.get("resume", None):
            self.logger.info("resume checkpoint...")
            self._resume_checkpoint(self.arch.resume)
        elif self.arch.get("best_model", None):
            self.logger.info("load best model.....")
            self._load_best_model(self.arch.best_model)

        # use_ema
        self.ema_model = None
        if self.use_ema and self.ema_decay:
            from modules.models.ema import ModelEMA
            self.logger.info("Use ema model ... ")
            self.ema_model = ModelEMA(self.model, self.ema_decay, self.device)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def poly_lr_scheduler(self, optimizer, init_lr, curr_iter, max_iter, power=0.9):
        for g in optimizer.param_groups:
            g['lr'] = init_lr * (1 - curr_iter / max_iter) ** power

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
            'results_log': str(self.results_log),
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.ema.state_dict() if self.use_ema else None,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            # 'config': self.config
        }

        monitor_best = round(self.monitor_best, 3)
        # Save checkpoint for each epoch
        if self.save_freq is not None:  # Use None mode to avoid over disk space with large models
            if epoch % self.save_freq == 0:
                filename = os.path.join(self.checkpoint_dir,
                                        f"{self.task_name}_{self.config.arch.type}_epoch{epoch}_{self.monitor}{monitor_best}.pth")
                torch.save(state, filename)
                self.logger.info("Saving checkpoint at {}".format(filename))

        # Save the best checkpoint
        if save_best:
            if len(self.save_deque) >= self.save_max:
                need_removed_checkpoint = self.save_deque.popleft()
                os.remove(need_removed_checkpoint)
            checkpoint_name = f"{self.task_name}_{self.config.arch.type}_epoch{epoch}_{self.monitor}{monitor_best}.pth"  # arch + "_epoch" + str(epoch) + "_" + self.monitor
            best_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            torch.save(state, best_path)

            # save best weights, if ues_ema, save ema weights
            if self.use_ema:
                model_to_save = self.ema_model.ema
                best_weights_path = os.path.join(self.checkpoint_dir, "best_weights_with_ema.pth")
            else:
                model_to_save = self.model
                best_weights_path = os.path.join(self.checkpoint_dir, "best_weights.pth")

            torch.save(model_to_save.state_dict(), best_weights_path)
            self.save_deque.append(best_path)
            self.logger.info("Saving current best at {}".format(best_path))
        else:
            self.logger.info("Monitor is not improved from %f" % (self.monitor_best))

    def _train_epoch(self, epoch):
        raise NotImplementedError

    # def _train_epoch(self, epoch):
    #     """
    #     Training logic for an epoch
    #
    #     :param epoch: Current training epoch.
    #     :return: A log that contains all information you want to save.
    #
    #     Note:
    #         If you have additional information to record, for example:
    #             > additional_log = {"x": x, "y": y}
    #         merge it with log before return. i.e.
    #             > log = {**log, **additional_log}
    #             > return log
    #
    #         The metrics in log must have the key 'metrics'.
    #     """
    #     self.logger.info(f"*****************************Training on epoch {epoch}...*****************************")
    #     self.model.train()
    #     self.metrics.reset()
    #     if self.writer_train:
    #         self.writer_train.set_step(epoch)
    #
    #     # Perform training
    #     # total_loss = 0
    #     # total_metrics = np.zeros(len(self.metrics))
    #     n_iter = len(self.train_loader)
    #     batch_count = len(self.train_loader)
    #     # (img_tensors, label_tensors, img_paths)
    #     for batch_idx, (data, target, _) in enumerate(self.train_loader):
    #         # print(data.shape, target.shape)
    #         curr_iter = batch_idx + (epoch - 1) * n_iter
    #         data, target = data.to(self.device), target.to(self.device)
    #         self.optimizer.zero_grad()
    #         output = self.model(data)  # (bs,1,384,384)
    #         loss = self.loss(output, target)
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         # update ema model
    #         if self.use_ema and self.ema_model:
    #             self.ema_model.update(self.model)
    #         self.model.zero_grad()
    #
    #
    #         batch_accuracy, batch_matched = self.metrics.update(output, target, loss.item())
    #         # total_loss += loss.item()
    #         # total_metrics += self._eval_metrics(output, target)
    #         if self.task_type == "multi_class":
    #             self.logger.info(
    #                 "Epoch:{:3d} training batch:{:4}/{:4} -- loss:{:.4f} lr:{:.5f} accuracy:{:.4f} specific：[{:3}/{:3}]".format(
    #                     epoch, batch_idx, batch_count, loss.item(),
    #                     self.optimizer.state_dict()['param_groups'][0]['lr'], batch_accuracy, batch_matched,
    #                     target.data.numel()))
    #         elif self.task_type == "multi_label":
    #             self.logger.info(
    #                 "Epoch:{:3d} training batch:{:4}/{:4} -- loss:{:.4f} lr:{:.5f} batch_top@1_acc:{:.4f} specific：[{:3}/{:3}]".format(
    #                     epoch, batch_idx, batch_count, loss.item(),
    #                     self.optimizer.state_dict()['param_groups'][0]['lr'], batch_accuracy, batch_matched,
    #                     target.shape[0]))
    #
    #         # 扩展output 与target的维度 用于tensorboard输出
    #         # (bs,h,w) -- > (bs, 3, h, w)
    #         # print("output shape: ", output.shape)
    #         # print("target shape: ", target.shape)
    #         # output = output.repeat([1, 3, 1, 1])
    #         # target = target.unsqueeze(1).repeat([1, 3, 1, 1])
    #         #
    #         # if (batch_idx == n_iter - 2) and (self.verbosity >= 2):
    #         #     self.writer_train.add_image('train/input', make_grid(data[:, :3, :, :].cpu(), nrow=4, normalize=False))
    #         #     self.writer_train.add_image('train/label', make_grid(target.cpu(), nrow=4, normalize=False))
    #         #     if type(output) == tuple or type(output) == list:
    #         #         self.writer_train.add_image('train/output', make_grid(output[0].cpu(), nrow=4, normalize=False))
    #         #     else:
    #         #         # self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))
    #         #         self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))
    #
    #         poly_lr_scheduler(self.optimizer, self.init_lr, curr_iter, self.max_iter, power=0.9)
    #
    #     # Record log
    #     avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report()
    #
    #     log = {
    #         'train_loss': avg_loss,
    #         'train_acc': avg_acc,
    #         "train_auc_for_class": auc_for_class,
    #         "train_auc": avg_auc,
    #         "train_acc_for_class": acc_for_class
    #     }
    #
    #     # Write training result to TensorboardX
    #     if self.writer_train and self.writer_valid:
    #         self.writer_train.add_scalar('train_loss', avg_loss)
    #         self.writer_train.add_scalar("train_acc", avg_acc)
    #         self.writer_train.add_scalar("train_auc", avg_auc)
    #         if acc_for_class:
    #             for class_i, class_acc in enumerate(acc_for_class):
    #                 self.writer_train.add_scalar('train_acc_for_/%s' % (self.id2name[str(class_i)]), class_acc)
    #         if auc_for_class:
    #             for class_i, class_auc in enumerate(auc_for_class):
    #                 self.writer_train.add_scalar('train_acc_for_/%s' % (self.id2name[str(class_i)]), class_auc)
    #
    #         if self.verbosity >= 2:
    #             for i in range(len(self.optimizer.param_groups)):
    #                 self.writer_train.add_scalar('lr/group%d' % i, self.optimizer.param_groups[i]['lr'])
    #
    #     # Perform validating
    #     if self.do_validation:
    #         self.logger.info(
    #             f"*****************************Validation on epoch {epoch}...*****************************")
    #         val_log = self._valid_epoch(epoch)
    #         log = {**log, **val_log}
    #
    #     # Learning rate scheduler
    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()
    #
    #     return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'valid_metrics'.
        """
        if self.use_ema:
            test_model = self.ema_model.ema
        else:
            test_model = self.model
        test_model.eval()
        self.metrics.reset()
        # total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        n_iter = len(self.valid_loader)
        if self.writer_valid:
            self.writer_valid.set_step(epoch)

        with torch.no_grad():
            # Validate
            for batch_idx, (data, target, _) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = test_model(data)
                loss = self.loss(output, target)
                self.metrics.update(output, target, loss.item())

                # total_val_loss += loss.item()
                # total_val_metrics += self._eval_metrics(output, target)

                # output = output.repeat([1, 3, 1, 1])
                # target = target.unsqueeze(1).repeat([1, 3, 1, 1])
                #
                # if (batch_idx == n_iter - 2) and (self.verbosity >= 2):
                #     self.writer_valid.add_image('valid/input',
                #                                 make_grid(data[:, :3, :, :].cpu(), nrow=4, normalize=True))
                #     self.writer_valid.add_image('valid/label', make_grid(target.cpu(), nrow=4, normalize=True))
                #     if type(output) == tuple or type(output) == list:
                #         self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))
                #     else:
                #         # self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))
                #         self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))

        # Record log
        # if the task_type == "multi_label", the ava_acc is top@1_acc
        avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report()

        val_log = {
            'valid_loss': avg_loss,
            'valid_acc': avg_acc,
            "valid_auc": avg_auc,
            "valid_acc_for_class": acc_for_class,
            "valid_auc_for_class": auc_for_class,

        }

        # Write validating result to TensorboardX
        if self.writer_train and self.writer_valid:
            self.writer_train.add_scalar('valid_loss', avg_loss)
            self.writer_train.add_scalar("valid_acc", avg_acc)
            self.writer_train.add_scalar("valid_auc", avg_auc)
            if acc_for_class:
                for class_i, class_acc in enumerate(acc_for_class):
                    self.writer_train.add_scalar('valid_acc_for_/%s' % (self.id2name[str(class_i)]), class_acc)
            if auc_for_class:
                for class_i, class_auc in enumerate(auc_for_class):
                    self.writer_train.add_scalar('train_acc_for_/%s' % (self.id2name[str(class_i)]), class_auc)
        return val_log
