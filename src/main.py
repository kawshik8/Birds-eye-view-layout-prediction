# Main program
import os
import logging as log

from args import parser, process_args
from tasks import get_task
from models import get_model
from trainer import Trainer
from utils import config_logging, load_model, save_model


def main(args):

    # preparation
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    config_logging(os.path.join(args.exp_dir, "%s.log" % args.exp_name))
    log.info("Experiment %s" % (args.exp_name))
    log.info("Receive config %s" % (args.__str__()))
    log.info("Start creating tasks")
    pretrain_task = [get_task(taskname, args) for taskname in args.pretrain_task]
    finetune_tasks = [get_task(taskname, args) for taskname in args.finetune_tasks]
    log.info("Start loading data")
    for task in pretrain_task:
        task.load_data()
    for task in finetune_tasks:
        task.load_data()
    log.info("Start creating models")
    model = get_model(args.model, args)
    log.info("Loaded %s model" % (args.model))
    model.to(args.device)
    if args.load_ckpt!="none":
        args.load_ckpt = os.path.join(args.exp_dir, args.load_ckpt)
    #if args.load_ckpt != "none":
    #    load_model(model, pretrain_complete_ckpt)

    # pretrain
    if len(pretrain_task):
        pretrain = Trainer("pretrain", model, pretrain_task[0], args)
        pretrain.train()
        pretrain_complete_ckpt = os.path.join(
            args.exp_dir, "pretrain_%s_complete.pth" % pretrain_task[0].name
        )
        save_model(pretrain_complete_ckpt, model)
    else:
        pretrain_complete_ckpt = args.load_ckpt

    # finetune and test
    for task in finetune_tasks:
        if pretrain_complete_ckpt != "none":
            load_model(pretrain_complete_ckpt, model)
        finetune = Trainer("finetune", model, task, args)
        finetune.train()

        finetune.eval("test")

    # evaluate
    # TODO: evaluate result on test split, write prediction for leaderboard submission (for dataset
    # without test labels)
    log.info("Done")
    return


if __name__ == "__main__":
    args = parser.parse_args()
    process_args(args)
    main(args)
