#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :trainer.py
# @Time     :2021/3/26 上午11:50
# @Author   :Chang Qing

import logging
import datetime

import torch

from time import time
from modules.solver.trainer import Trainer
from modules.datasets import build_dataloader


# ------------------------------------------------------------------------------
#  Poly learning-rate Scheduler
# ------------------------------------------------------------------------------
# def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
#     for g in optimizer.param_groups:
#         g['lr'] = init_lr * (1 - curr_iter / max_iter) ** power


# ------------------------------------------------------------------------------
#   Class of Trainer
# ------------------------------------------------------------------------------
class NormTrainer(Trainer):

    def __init__(self, config):
        super(NormTrainer, self).__init__(config)
        # build dataloader
        self.config.loader.args["task_type"] = self.task_type
        self.train_loader, self.valid_loader = build_dataloader(self.config.loader.type, self.config.loader.args)
        assert self.train_loader.dataset.num_classes == self.valid_loader.dataset.num_classes, \
            f"train class num != valid class num:  {self.train_loader.dataset.num_classes} != " \
            f"{self.valid_loader.dataset.num_classes}"
        assert self.num_classes == self.train_loader.dataset.num_classes, \
            f"self.num_classes != train class num:  {self.num_classes} != " \
            f"{self.train_loader.dataset.num_classes}"

        self.do_validation = self.valid_loader is not None
        self.max_iter = len(self.train_loader) * self.epochs


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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.logger.info(f"*****************************Training on epoch {epoch}...*****************************")
        self.model.train()
        self.metrics.reset()
        if self.writer_train:
            self.writer_train.set_step(epoch)

        # Perform training
        # total_loss = 0
        # total_metrics = np.zeros(len(self.metrics))
        n_iter = len(self.train_loader)
        batch_count = len(self.train_loader)
        # (img_tensors, label_tensors, img_paths)
        for batch_idx, (data, target, _) in enumerate(self.train_loader):
            # print(data.shape, target.shape)
            curr_iter = batch_idx + (epoch - 1) * n_iter
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)  # (bs,1,384,384)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # update ema model
            if self.use_ema and self.ema_model:
                self.ema_model.update(self.model)
            self.model.zero_grad()

            batch_accuracy, batch_matched = self.metrics.update(output, target, loss.item())
            # total_loss += loss.item()
            # total_metrics += self._eval_metrics(output, target)
            if self.task_type == "multi_class":
                self.logger.info(
                    "Epoch:{:3d} training batch:{:4}/{:4} -- loss:{:.4f} lr:{:.5f} accuracy:{:.4f} specific：[{:3}/{:3}]".format(
                        epoch, batch_idx, batch_count, loss.item(),
                        self.optimizer.state_dict()['param_groups'][0]['lr'], batch_accuracy, batch_matched,
                        target.data.numel()))
            elif self.task_type == "multi_label":
                self.logger.info(
                    "Epoch:{:3d} training batch:{:4}/{:4} -- loss:{:.4f} lr:{:.5f} batch_top@1_acc:{:.4f} specific：[{:3}/{:3}]".format(
                        epoch, batch_idx, batch_count, loss.item(),
                        self.optimizer.state_dict()['param_groups'][0]['lr'], batch_accuracy, batch_matched,
                        target.shape[0]))

            # 扩展output 与target的维度 用于tensorboard输出
            # (bs,h,w) -- > (bs, 3, h, w)
            # print("output shape: ", output.shape)
            # print("target shape: ", target.shape)
            # output = output.repeat([1, 3, 1, 1])
            # target = target.unsqueeze(1).repeat([1, 3, 1, 1])
            #
            # if (batch_idx == n_iter - 2) and (self.verbosity >= 2):
            #     self.writer_train.add_image('train/input', make_grid(data[:, :3, :, :].cpu(), nrow=4, normalize=False))
            #     self.writer_train.add_image('train/label', make_grid(target.cpu(), nrow=4, normalize=False))
            #     if type(output) == tuple or type(output) == list:
            #         self.writer_train.add_image('train/output', make_grid(output[0].cpu(), nrow=4, normalize=False))
            #     else:
            #         # self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))
            #         self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))

            self.poly_lr_scheduler(self.optimizer, self.init_lr, curr_iter, self.max_iter, power=0.9)

        # Record log
        avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report()

        log = {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            "train_auc_for_class": auc_for_class,
            "train_auc": avg_auc,
            "train_acc_for_class": acc_for_class
        }

        # Write training result to TensorboardX
        if self.writer_train and self.writer_valid:
            self.writer_train.add_scalar('train_loss', avg_loss)
            self.writer_train.add_scalar("train_acc", avg_acc)
            self.writer_train.add_scalar("train_auc", avg_auc)
            if acc_for_class:
                for class_i, class_acc in enumerate(acc_for_class):
                    self.writer_train.add_scalar('train_acc_for_/%s' % (self.id2name[str(class_i)]), class_acc)
            if auc_for_class:
                for class_i, class_auc in enumerate(auc_for_class):
                    self.writer_train.add_scalar('train_acc_for_/%s' % (self.id2name[str(class_i)]), class_auc)

            if self.verbosity >= 2:
                for i in range(len(self.optimizer.param_groups)):
                    self.writer_train.add_scalar('lr/group%d' % i, self.optimizer.param_groups[i]['lr'])

        # Perform validating
        if self.do_validation:
            self.logger.info(
                f"*****************************Validation on epoch {epoch}...*****************************")
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        # Learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

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
