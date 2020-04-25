# Main program
import os
import logging as log

from args import parser, process_args
from tasks import get_task
from SSLmodels import get_model
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
    if args.image_pretrain_obj != "none":
        image_ssl_model = get_model("image_ssl", args)
        log.info("Loaded image ssl model")

    if args.view_pretrain_obj != "none":
        view_ssl_model = get_model("view_ssl", args)
        log.info("Loaded view ssl model")

    if args.finetune_obj != "none": 
        sup_model = get_model("sup", args)
        log.info("Loaded supervised model")

    #if args.load_ckpt != "none":
    #    load_model(model, pretrain_complete_ckpt)

    # pretrain
    if len(pretrain_task):
        if args.image_pretrain_obj != "none":
            image_ssl_model.to(args.device)
            pretrain = Trainer("pretrain", image_ssl_model, pretrain_task[0], args)
            pretrain.train()
            image_pretrain_complete_ckpt = os.path.join(
                args.exp_dir, "image_pretrain_%s_complete.pth" % pretrain_task[0].name
            )
            save_model(image_pretrain_complete_ckpt, image_ssl_model)
        else:
            if args.imagessl_load_ckpt:
                image_pretrain_complete_ckpt = args.imagessl_load_ckpt

        if args.view_pretrain_obj != "none":
            view_ssl_model.to(args.device)
            pretrain = Trainer("pretrain", view_ssl_model, pretrain_task[0], args)
            pretrain.train()
            view_pretrain_complete_ckpt = os.path.join(
                args.exp_dir, "view_pretrain_%s_complete.pth" % pretrain_task[0].name
            )
            save_model(viewssl_complete_ckpt, view_ssl_model)
        else:
            if args.viewssl_pretrained_load_ckpt:
                view_pretrain_complete_ckpt = args.viewssl_load_ckpt

    # finetune and test
    for task in finetune_tasks:
        sup_model.to(args.device)
        finetune = Trainer("finetune", sup_model, task, args)
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
