# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import torch
from memotr.utils.utils import distributed_rank

from .memotr import build as build_memotr


def build_model(config: dict, num_classes=None):
    model = build_memotr(config=config, num_classes=num_classes)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
