# This file defines all the configuations of the program
import argparse
import os

parser = argparse.ArgumentParser()

# General settings
# exp_name
parser.add_argument("--exp-name", type=str, default="debug", help="experiment name")
# device
parser.add_argument("--device", type=str, default="cuda:0", help="which device to run on")
# results_dir
parser.add_argument(
    "--results-dir",
    type=str,
    default="/scratch/hl3236/cv_results/",
    help="directory to save results and files",
)
# data_dir
parser.add_argument(
    "--data-dir", type=str, default="/scratch/hl3236/data/", help="directory of the data files",
)


# pretrain_task objective settings for models other than selfie
parser.add_argument(
    "--pretrain-obj",
    type=str,
    default="nce_loss",
    choices=["nce_loss", "crossentropy_loss", "multilabel_loss", "deepinfomax_loss"],
    help="pretrain task, '_un' is for unsupervised. 'none' means skip pretrain",
)

# Data settings
# pretrain_task and finetune_task
parser.add_argument(
    "--pretrain-task",
    type=str,
    default="cifar10_un",
    choices=["cifar10_un", "stl10_un", "mnist_un", "imagenet_un", "none"],
    help="pretrain task, '_un' is for unsupervised. 'none' means skip pretrain",
)

parser.add_argument(
    "--finetune-tasks",
    type=str,
    default="cifar10_lp5",
    help="""any non-empty subset from ['cifar10', 'mnist', 'imagenet'] x ['_lp5', '_lp10', '_lp20', '_lp100'] 
    (percent of labels available) x ["_res1", "_res2", "_res3", "_res4", "_res5"] 
    dont mention the layer in case of finetuning the whole layer
    (layers from the resnet to use for linear evaluation ) and 
    'stl10_fd' X ['0', ..., '9'] (fold number of supervised data), 
    seperated by comma (no space!), e.g. 'stl_10_fd0,cifar10_lp5'.
    or, choose 'none' to skip finetune&evaluation. """,
)

# view_type
parser.add_argument(
    "--view", type=str, default="normal", help="multiview or single view", choices = ["random_normal","random_multiview","random_singleview"]
)

# num_patches
parser.add_argument(
    "--num-patches", type=int, default=9, help="number of patches an image is broken into"
)
# num_queries
parser.add_argument(
    "--num-queries-percentage", type=float, default=0.25, help="number of patches an image to predict"
)
# num_workers
parser.add_argument("--num_workers", type=int, default=16, help="number of cpu workers in iterator")
# batch_size
parser.add_argument(
    "--batch-size", type=int, default=64, help="number of images per minibatch",
)
# cache_pos
parser.add_argument(
    "--dup-pos",
    type=int,
    default=0,
    help="number of duplicated positive images per image in minibatch",
)
# cache_neg
parser.add_argument(
    "--cache-neg",
    type=int,
    default=0,
    help="number of cached negative images per image in minibatch",
)

# Model settings
# model
parser.add_argument("--model", type=str, default="selfie", choices=["baseline","selfie","Allp","Exp","selfie1"])
# TODO: some settings about model extensions
# TODO: e.g. whether to use negative example from minibatch


# Training settings
# load_ckpt
parser.add_argument(
    "--load-ckpt",
    type=str,
    default="none",
    help="load parameters from a checkpoint, choose auto to resume interrupted experiment",
)
# clip
parser.add_argument("--clip", type=float, default=0.5, help="gradient clip")
# learning_rate
parser.add_argument(
    "--pretrain-learning-rate", type=float, default=1e-2, help="learning rate for pretraining"
)
parser.add_argument(
    "--finetune_learning_rate", type=float, default=1e-4, help="learning rate for finetuning"
)
# weight_decay
parser.add_argument(
    "--pretrain-weight-decay", type=float, default=1e-4, help="weight decay for pretraining"
)
parser.add_argument(
    "--finetune-weight-decay", type=float, default=1e-4, help="weight decay for finetuning"
)
# iters
parser.add_argument(
    "--pretrain-total-iters", type=int, default=100000, help="maximum iters for pretraining"
)
parser.add_argument(
    "--finetune-total-iters",
    type=int,
    default=10000,
    help="maximum iters for finetuning, set to 0 to skip finetune training",
)
parser.add_argument("--warmup-iters", type=int, default=100, help="lr warmup iters")
parser.add_argument(
    "--report-interval", type=int, default=250, help="number of iteratiopns between reports"
)
parser.add_argument("--finetune-val-interval", type=int, default=2000, help="validation interval")
parser.add_argument(
    "--pretrain-val-interval", type=int, default=2000, help="pretrain validation interval"
)
parser.add_argument(
    "--pretrain-ckpt-interval",
    type=int,
    default=0,
    help="pretrian mandatory saving interval, set to 0 to disable",
)
parser.add_argument(
    "--finetune-ckpt-interval",
    type=int,
    default=0,
    help="finetune mandatory saving interval, set to 0 to disable",
)
# transfer-paradigm
parser.add_argument(
    "--transfer-paradigm",
    type=str,
    default="frozen",
    choices=["frozen", "tunable", "bound"],
    help="""frozen: use fixed representation,
            tunable: finetune the whole model,
            (unimplemented) bound: parameters are tunable but decay towards pretrained model""",
)


def process_args(args):
    # TODO: some asserts, check the arguments
    args.num_queries = round(args.num_queries_percentage * args.num_patches)
    args.pretrain_task = list(filter(lambda task: task != "none", [args.pretrain_task]))
    args.finetune_tasks = list(filter(lambda task: task != "none", args.finetune_tasks.split(",")))
    args.exp_dir = os.path.join(args.results_dir, args.exp_name)
