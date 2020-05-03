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
    "--data-dir", type=str, default="./data/", help="directory of the data files",
)

# image_folder
parser.add_argument(
    "--image-folder", type=str, default="../../../data/data/", help="directory of the custom dataset files",
)

# pretrain_task objective settings for models other than selfie
parser.add_argument(
    "--image-pretrain-obj",
    type=str,
    default="none",#"pirl_nce_loss",
    choices=["pirl_nce_loss", "pirl_infonce_loss", "multilabel_loss","none"],
    help="pretrain image based task, '_un' is for unsupervised. 'none' means skip pretrain",
)

parser.add_argument(
    "--view-pretrain-obj",
    type=str,
    default="none",
    choices=["det_masked_autoencoder","var_masked_autoencoder","adv_masked_autoencoder", "nce_loss", "infonce_loss","none"],
    help="pretrain view based task, '_un' is for unsupervised. 'none' means skip pretrain",
)

parser.add_argument(
    "--finetune-obj",
    type=str,
    default="det_encoder",
    choices=["det_encoder","var_encoder","adv_encoder","none"],
    help="finetune task, 'none' means skip finetune",
)

parser.add_argument(
    "--obj-det-head",
    type=str,
    default="retinanet",
    choices=["retinanet","maskrcnn","none"],
    help="pretrain task, '_un' is for unsupervised. 'none' means skip pretrain",
)

parser.add_argument(
    "--road-map-loss",
    type=str,
    default="bce",
    choices=["bce","mse"],
    help="pretrain task, '_un' is for unsupervised. 'none' means skip pretrain",
)


parser.add_argument(
    "--gen-road-map",
    type=int,
    default=0,
    help="train road map generation as well?",
)

parser.add_argument(
    "--detect-objects",
    type=int,
    default=0,
    help="train object detection as well?",
)

parser.add_argument(
    "--dense-before-fuse",
    type=int,
    default=0,
    help="project encoder to dense before fusion?",
)

parser.add_argument(
    "--dense-after-fuse",
    type=int,
    default=0,
    help="project encoder to dense after fusion?",
)

parser.add_argument(
    "--fuse-type",
    type=str,
    default="concat",
    choices=["concat", "mean", "sum"],
    help="fuse 6 views in different ways",
)

parser.add_argument(
    "--view-fusion-strategy",
    type=str,
    default="conv",
    choices=["dense", "conv", "cross_attention"],
    help="fuse 6 views in different ways",
)

# self.blobs_strategy = "encoder_views"
#         assert self.blobs_strategy in ["decoder", "encoder_fused", "encoder_views"]
parser.add_argument(
    "--blobs-strategy",
    type = str,
    default="encoder_fused",
    choices=["decoder", "encoder_fused", "encoder_views"],
    help="fuse 6 views in different ways",
) 

# pretrain_task objective settings for models other than selfie
parser.add_argument(
    "--network-base",
    type=str,
    default="resnet18",
    choices=["resnet18", "resnet34", "resnet50", "resnet101","resnet152"],
    help="base model to train",
)

# Data settings
# pretrain_task and finetune_task
parser.add_argument(
    "--pretrain-task",
    type=str,
    default="custom_un",
    choices=["cifar10_un", "stl10_un", "mnist_un", "imagenet_un", "custom_un","none"],
    help="pretrain task, '_un' is for unsupervised. 'none' means skip pretrain",
)

parser.add_argument(
    "--use-memory-bank",
    type=int,
    default=1,
    help="train object detection as well?",
)

parser.add_argument(
    "--finetune-tasks",
    type=str,
    default="custom_sup",
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
    "--view", type=str, default="normal", help="multiview or single view", choices = ["random_normal","random_multiview","random_singleview","normal"]
)

# sampling_type
parser.add_argument(
    "--sampling-type", type=str, default="image", help="type of instance"
)

# num_patches
parser.add_argument(
    "--num-patches", type=int, default=9, help="number of patches an image is broken into"
)

# project_dim for pirl
parser.add_argument(
    "--project-dim", type=int, default=128, help="projection head dim for SSL approaches"
)

# z_dim for variational inference
parser.add_argument(
    "--latent-dim", type=int, default=128, help="z dim for variational inference"
)

# # z_dim for variational inference
# parser.add_argument(
#     "--latent-dim-type", type=str, default="dense", help="z dim type for variational inference", choices=
# )


# num_patches
parser.add_argument(
    "--lambda-pirl", type=float, default=0.5, help="lambda weights for pirl loss function"
)

# beta for exponential moving average
parser.add_argument(
    "--beta-ema", type=float, default=0.5, help="beta for exp moving average"
)

# alpha for 
parser.add_argument(
    "--num-negatives", type=int, default=1000, help="number of negatives to be used for contrastive learning"
)

# num_queries
parser.add_argument(
    "--num-queries-percentage", type=float, default=0.25, help="number of patches an image to predict"
)
# num_workers
parser.add_argument("--num_workers", type=int, default=16, help="number of cpu workers in iterator")

# vocab_size
parser.add_argument(
    "--vocab-size", type=int, default=64, help="number of images in dataset",
)

# batch_size
parser.add_argument(
    "--batch-size", type=int, default=8, help="number of images per minibatch",
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

# Training settings
# load_ckpt
parser.add_argument(
    "--imagessl-load-ckpt",
    type=str,
    default="none",
    help="load parameters from a checkpoint, choose auto to resume interrupted experiment",
)
parser.add_argument(
    "--viewssl-load-ckpt",
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
    "--finetune-learning-rate", type=float, default=1e-2, help="learning rate for finetuning"
)
# weight_decay
parser.add_argument(
    "--pretrain-weight-decay", type=float, default=1e-5, help="weight decay for pretraining"
)
parser.add_argument(
    "--finetune-weight-decay", type=float, default=1e-5, help="weight decay for finetuning"
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
    default="tunable",
    choices=["frozen", "tunable", "bound"],
    help="""frozen: use fixed representation,
            tunable: finetune the whole model,
            (unimplemented) bound: parameters are tunable but decay towards pretrained model""",
)


def process_args(args):
    # TODO: some asserts, check the arguments
    if args.dense_before_fuse:
        args.view_fusion_strategy = "dbf_" + args.view_fusion_strategy
    elif args.dense_after_fuse:
        args.view_fusion_strategy = "daf_" + args.view_fusion_strategy

    args.view_fusion_strategy += "_" + args.fuse_type

    args.gen_road_map = (args.gen_road_map==1)
    args.detect_objects = (args.detect_objects==1)
    args.num_queries = round(args.num_queries_percentage * args.num_patches)
    args.pretrain_task = list(filter(lambda task: task != "none", [args.pretrain_task]))

    for i,task in enumerate(args.pretrain_task):
        if "custom" in task:
            args.pretrain_task[i]+= "_" + args.sampling_type

    args.finetune_tasks = list(filter(lambda task: task != "none", args.finetune_tasks.split(",")))
    args.exp_dir = os.path.join(args.results_dir, args.exp_name)
