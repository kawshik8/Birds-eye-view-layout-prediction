# This file manages procedures like pretraining, training and evaluation.
import torch
import math
import logging as log
import os

from utils import save_model

class Trainer(object):
    def __init__(self, stage, model, task, args):
        """
        Setup training / evaluating
        """
        log.info("Setup trainer for %s" % task.name)
        self.args = args
        self.model = model
        self.task = task
        self.stage = stage

        self.training_infos = {"best_iter": -1, "best_performance": 0, "current_iter": 0}
        self.report_interval = args.report_interval
        if stage == "pretrain":
            self.total_iters = args.pretrain_total_iters
            self.val_interval = args.pretrain_val_interval
            self.ckpt_interval = args.pretrain_ckpt_interval
        elif stage == "finetune":
            self.total_iters = args.finetune_total_iters
            self.val_interval = args.finetune_val_interval
            self.ckpt_interval = args.finetune_ckpt_interval
        else:
            raise NotImplementedError  # unidentified stage

        self.optimizer, self.scheduler = self.model.config_stage(stage)
        return

    def eval(self, split):
        """
        Evaluate on eval or test data
        """
        log.info("Evaluate on  %s:%s data" % (self.task.name, split))
        self.model.eval()
        self.task.reset_scorers()
        with torch.no_grad():
            for batch, (idx,image,query) in enumerate(self.task.data_iterators[split]):
                
                batch_input = {"image":image,"query":query,"idx":idx}

                for k, v in batch_input.items():
                   batch_input[k] = batch_input[k].to(self.args.device)
                #batch_input = batch_input.to(self.args.device)
                batch_output = self.model(batch_input, self.task)
                self.task.update_scorers(batch_input, batch_output)
                if (batch + 1) % self.report_interval == 0:
                    log.info(
                        "eval batch %d / %d, current average result %s"
                        % (
                            batch + 1,
                            len(self.task.data_iterators[split]),
                            self.task.report_scorers().__str__(),
                        )
                    )

        scores = self.task.report_scorers(reset=True)
        log.info("Evalutation complete\nAverage result %s" % scores.__str__())
        return scores

    def train(self):
        """
        Train the model
        """
        log.info("Start training %s" % self.task.name)
        self.model.train()
        self.task.reset_scorers()
        #self.val_interval = len(self.task.data_iterators["train"])
        all_param = [param for group in self.optimizer.param_groups for param in group["params"]]
        for epoch in range(math.ceil(self.total_iters / len(self.task.data_iterators["train"]))):
            for batch, (idx,image,query) in enumerate(self.task.data_iterators["train"]):
                # print(idx,image.shape,query.shape)
                #if self.stage == "pretrain":
                batch_input = {"image":image,"query":query,"idx":idx}

                for k, v in batch_input.items():
                   batch_input[k] = batch_input[k].to(self.args.device)
                self.model.zero_grad()
                batch_output = self.model(batch_input, self.task)
                self.task.update_scorers(batch_input, batch_output)
                batch_output["loss"].backward()
                if self.args.clip != 0:
                    torch.nn.utils.clip_grad_norm_(all_param, self.args.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.training_infos["current_iter"] += 1
                if self.training_infos["current_iter"] % self.report_interval == 0:
                    log.info(
                        "train batch %d / %d (iter %d), current average result %s"
                        % (
                            batch + 1,
                            len(self.task.data_iterators["train"]),
                            self.training_infos["current_iter"],
                            self.task.report_scorers().__str__(),
                        )
                    )
                if (
                    self.val_interval > 0
                    and self.training_infos["current_iter"] % self.val_interval == 0
                ):
                    eval_scores = self.eval("val")
                    if eval_scores[self.task.eval_metric] > self.training_infos["best_performance"]:
                        self.training_infos["best_performance"] = eval_scores[self.task.eval_metric]
                        self.training_infos["best_iter"] = self.training_infos["current_iter"]
                        log.info("Best validation updated: %s" % self.training_infos)
                        save_model(
                            os.path.join(
                                self.args.exp_dir, "%s_%s_best.ckpt" % (self.stage, self.task.name),
                            ),
                            self.model,
                        )
                    self.model.train()
                if (
                    self.ckpt_interval > 0
                    and self.training_infos["current_iter"] % self.ckpt_interval == 0
                ):
                    save_model(
                        os.path.join(
                            self.args.exp_dir,
                            "%s_%s_%d.ckpt"
                            % (self.stage, self.task.name, self.training_infos["current_iter"]),
                        ),
                        self.model,
                    )
                if self.training_infos["current_iter"] == self.total_iters:
                    break
            if self.training_infos["current_iter"] == self.total_iters:
                break
        log.info("Training complete")
        if self.training_infos["best_iter"] > 0:
            log.info(
                "Best result found for %s: %s" % (self.task.name, self.training_infos.__str__())
            )
