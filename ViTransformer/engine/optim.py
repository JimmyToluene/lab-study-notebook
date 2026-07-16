"""Optimizer and LR-schedule construction (the DeiT-style setup)."""

from __future__ import annotations

import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import Config


def build_optimizer(model, cfg: Config) -> AdamW:
    """
    AdamW with weight decay applied only to matrix weights.

    Norm/bias/embedding params (anything 1-D, plus the CLS token and positional embedding)
    are excluded from weight decay
    Follow the standard ViT convention.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "cls_token" in name or name.endswith(".pe"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=cfg.train.lr)


def build_scheduler(optimizer, cfg: Config):
    """Linear warmup then cosine decay to ``min_lr``. Steps once per epoch.

    Returns None when ``train.scheduler`` is "none" (constant LR).
    """
    sched = cfg.train.scheduler
    if sched == "none":
        return None
    if sched != "cosine":
        raise ValueError(f"Unknown scheduler '{sched}'. Supported: none, cosine.")

    epochs = cfg.train.epochs
    warmup = cfg.train.warmup_epochs
    base_lr = cfg.train.lr
    min_lr = cfg.train.min_lr

    def lr_lambda(epoch):  # epoch is 0-indexed; LambdaLR multiplies base_lr by this
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    return LambdaLR(optimizer, lr_lambda)
