import logging as log
import torch

EPSILON = 1e-8


def config_logging(log_file):
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s [%(threadName)-12.12s] %(message)s",
        handlers=[log.FileHandler(log_file), log.StreamHandler()],
    )


def load_model(load_ckpt, model):
    """
    Load a model, for training, evaluation or prediction
    """
    model_state = torch.load(load_ckpt)
    model.load_state_dict(model_state)
    log.info("Load parameters from %s" % load_ckpt)


def save_model(save_ckpt, model):
    """
    Save the parameters of the model to a checkpoint
    """
    torch.save(model.state_dict(), save_ckpt)
    log.info("Save parameters for %s" % save_ckpt)
