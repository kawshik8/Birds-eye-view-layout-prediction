# This file manages procedures like pretraining, training and evaluation.
import torch
import math
import logging as log
import os
import torch.nn.functional as F
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

        if "loss" in self.task.eval_metric:
            best = float('inf')
        else:
            best = float('-inf')

        self.training_infos = {"best_iter": -1, "best_performance": best, "current_iter": 0}
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
            for batch, inputs in enumerate(self.task.data_iterators[split]):
              if batch < 5:
                if self.stage == "pretrain":
                    index, image, query = inputs
                    batch_input = {"image":image,"query":query,"idx":index}
                else:
                    index, image, bounding_box, classes, action, ego, road = inputs
                    batch_input = {"image":image,"idx":index, "bbox":bounding_box, "classes":classes, "action":action, "ego":ego, "road":road}


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
            for batch, inputs in enumerate(self.task.data_iterators["train"]):
            
                # print(idx,image.shape,query.shape)
                if self.stage == "pretrain":
                    index, image, query = inputs
                    batch_input = {"image":image,"query":query,"idx":index}
                else:
                    index, image, bounding_box, classes, action, ego, road = inputs
                    batch_input = {"image":image,"idx":index, "bbox":bounding_box, "classes":classes, "action":action, "ego":ego, "road":road}

                for k, v in batch_input.items():
                    batch_input[k] = batch_input[k].to(self.args.device)
                self.model.zero_grad()
                batch_output = self.model(batch_input, self.task)
                self.task.update_scorers(batch_input, batch_output)
                batch_output["loss"].backward()
                if self.args.clip != 0:
                    torch.nn.utils.clip_grad_norm_(all_param, self.args.clip)
                self.optimizer.step()
                
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
                    if eval_scores[self.task.eval_metric] < self.training_infos["best_performance"]:
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
                    self.scheduler.step(metrics=eval_scores["loss"],epoch=epoch)
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

class GANTrainer(object):
    def __init__(self, stage, model, task, args):
        """
        Setup training / evaluating
        """
        log.info("Setup trainer for %s" % task.name)
        self.args = args
        self.model = model
        self.task = task
        self.stage = stage

        if "loss" in self.task.eval_metric:
            best = float('inf')
        else:
            best = float('-inf')

        self.training_infos = {"best_iter": -1, "best_performance": best, "current_iter": 0}
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

        self.g_optimizer = self.model["generator"].config_stage(stage)
        self.d_optimizer = self.model["discriminator"].config_stage(stage)
        return

    def eval(self, split):
        """
        Evaluate on eval or test data
        """
        log.info("Evaluate on  %s:%s data" % (self.task.name, split))
        self.model["generator"].eval()
        self.model["discriminator"].eval()
        self.task.reset_scorers()
        with torch.no_grad():
            for batch, inputs in enumerate(self.task.data_iterators[split]):
              if batch < 5:
                if self.stage == "pretrain":
                    index, image, query = inputs
                    batch_input = {"image":image,"query":query,"idx":index}
                else:
                    index, image, bounding_box, classes, action, ego, road = inputs
                    batch_input = {"image":image,"idx":index, "bbox":bounding_box, "classes":classes, "action":action, "ego":ego, "road":road}


                for k, v in batch_input.items():
                   batch_input[k] = batch_input[k].to(self.args.device)
                #batch_input = batch_input.to(self.args.device)
                
                bs = self.args.batch_size

                batch_output = self.model["generator"](batch_input, self.task)
                gen_image = batch_output["road_map"]

                real_disc_inp = batch_input["road"]
                fake_disc_inp = gen_image.detach()

                if "patch" in self.args.disc_type:
                    b,c,h,w = real_disc_inp.shape
                    # real_disc_inp = real_disc_inp.view(b,-1)
                    # fake_disc_inp = fake_disc_inp.view(b,-1)
                    zeros = torch.zeros(bs,1,16,16).to(self.args.device)
                    ones = torch.ones(bs,1,16,16).to(self.args.device)

                else:
                    zeros = torch.zeros(bs,1).to(self.args.device)
                    ones = torch.ones(bs,1).to(self.args.device)

                real_disc_op = self.model["discriminator"](real_disc_inp)
                batch_output["real_DLoss"] = F.binary_cross_entropy(real_disc_op,ones)

                fake_disc_op = self.model["discriminator"](fake_disc_inp)
                batch_output["fake_DLoss"] = F.binary_cross_entropy(fake_disc_op,zeros)
                batch_output["DLoss"] = batch_output["real_DLoss"] + batch_output["fake_DLoss"]

                adv_output = self.model["discriminator"](gen_image)

                batch_output["GDiscLoss"] = F.binary_cross_entropy(adv_output,ones)
                batch_output["GLoss"] = batch_output["GDiscLoss"] + batch_output["GSupLoss"]

                batch_output["loss"] = batch_output["DLoss"] + batch_output["GLoss"]



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
        self.model["generator"].train()
        self.model["discriminator"].train()
        self.task.reset_scorers()
        #self.val_interval = len(self.task.data_iterators["train"])
        # all_param = [param for group in self.optimizer.param_groups for param in group["params"]]
        for epoch in range(math.ceil(self.total_iters / len(self.task.data_iterators["train"]))):
            for batch, inputs in enumerate(self.task.data_iterators["train"]):
            
                # print(idx,image.shape,query.shape)
                if self.stage == "pretrain":
                    index, image, query = inputs
                    batch_input = {"image":image,"query":query,"idx":index}
                else:
                    index, image, bounding_box, classes, action, ego, road = inputs
                    batch_input = {"image":image,"idx":index, "bbox":bounding_box, "classes":classes, "action":action, "ego":ego, "road":road}

                for k, v in batch_input.items():
                    batch_input[k] = batch_input[k].to(self.args.device)


                bs = self.args.batch_size

                self.model["discriminator"].zero_grad()

                batch_output = self.model["generator"](batch_input, self.task)
                gen_image = batch_output["road_map"]

                real_disc_inp = batch_input["road"]
                fake_disc_inp = gen_image.detach()

                if "patch" in self.args.disc_type:
                    b,c,h,w = real_disc_inp.shape
                    # real_disc_inp = real_disc_inp.view(b,-1)
                    # fake_disc_inp = fake_disc_inp.view(b,-1)
                    zeros = torch.zeros(bs,1,16,16).to(self.args.device)
                    ones = torch.ones(bs,1,16,16).to(self.args.device)

                else:
                    zeros = torch.zeros(bs,1).to(self.args.device)
                    ones = torch.ones(bs,1).to(self.args.device)

                real_disc_op = self.model["discriminator"](real_disc_inp)
                print(ones.shape, real_disc_op.shape)
                batch_output["real_DLoss"] = F.binary_cross_entropy(real_disc_op,ones)
                batch_output["real_DLoss"].backward()

                fake_disc_op = self.model["discriminator"](fake_disc_inp)
                batch_output["fake_DLoss"] = F.binary_cross_entropy(fake_disc_op,zeros)
                batch_output["fake_DLoss"].backward()
                batch_output["DLoss"] = batch_output["real_DLoss"] + batch_output["fake_DLoss"]
                self.d_optimizer.step()

                self.model["generator"].zero_grad()
                adv_output = self.model["discriminator"](gen_image)

                batch_output["GDiscLoss"] = F.binary_cross_entropy(adv_output,ones)
                batch_output["GLoss"] = batch_output["GDiscLoss"] + batch_output["GSupLoss"]

                
                batch_output["GLoss"].backward()
                self.g_optimizer.step()

                batch_output["loss"] = batch_output["DLoss"] + batch_output["GLoss"]
                
                self.task.update_scorers(batch_input, batch_output)
                # if self.args.clip != 0:
                #     torch.nn.utils.clip_grad_norm_(all_param, self.args.clip)
                
                
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
                    if eval_scores[self.task.eval_metric] < self.training_infos["best_performance"]:
                        self.training_infos["best_performance"] = eval_scores[self.task.eval_metric]
                        self.training_infos["best_iter"] = self.training_infos["current_iter"]
                        log.info("Best validation updated: %s" % self.training_infos)
                        save_model(
                            os.path.join(
                                self.args.exp_dir, "%s_%s_generator_best.ckpt" % (self.stage, self.task.name),
                            ),
                            self.model["generator"],
                        )
                        save_model(
                            os.path.join(
                                self.args.exp_dir, "%s_%s_discriminator_best.ckpt" % (self.stage, self.task.name),
                            ),
                            self.model["discriminator"],
                        )
                    self.model["generator"].train()
                    self.model["discriminator"].train()
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

