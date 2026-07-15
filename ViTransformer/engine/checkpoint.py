"""Minimal checkpoint I/O.

Deliberately bare: one ``torch.save`` of the weights plus the config that
produced them. No optimizer state, no best/last tracking, no resume -- those
are the "full-featured" extras we chose to skip. Persisting the config is the
one non-negotiable: it is what lets test.py / infer.py rebuild the exact
architecture from a checkpoint file with no config flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .config import Config
from .model import build_model_from_config


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: Config) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)


def load_model(path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, Config]:
    """Rebuild the model from a checkpoint and load its weights (eval mode)."""
    # weights_only=False: the payload contains the config dict, not just tensors.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
