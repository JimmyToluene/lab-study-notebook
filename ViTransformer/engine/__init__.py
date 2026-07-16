"""Engineered training/eval/inference plumbing for the ViT model in ``vit/``.

Public API used by train.py / test.py / infer.py.
"""

from .config import Config, ModelConfig, DataConfig, TrainConfig, AugConfig, WandbConfig
from .data import build_dataloaders, build_transform, get_labels, get_spec, DATASETS
from .model import build_model, build_model_from_config
from .checkpoint import save_checkpoint, load_model
from .loops import train_one_epoch, evaluate
from .mixup import Mixup, SoftTargetCrossEntropy
from .optim import build_optimizer, build_scheduler
from .tracking import Tracker
from .utils import set_seed, get_device, describe_device

__all__ = [
    "Config", "ModelConfig", "DataConfig", "TrainConfig", "AugConfig", "WandbConfig",
    "build_dataloaders", "build_transform", "get_labels", "get_spec", "DATASETS",
    "build_model", "build_model_from_config",
    "save_checkpoint", "load_model",
    "train_one_epoch", "evaluate",
    "Mixup", "SoftTargetCrossEntropy",
    "build_optimizer", "build_scheduler",
    "Tracker",
    "set_seed", "get_device", "describe_device",
]
