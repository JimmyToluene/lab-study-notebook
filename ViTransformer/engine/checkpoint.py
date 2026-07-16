"""
Checkpoint I/O.

A checkpoint is a single ``torch.save`` payload: {"model": state_dict, "config": cfg dict}.
Persisting the config lets load_model() rebuild the exact architecture from the file alone.
Does not save optimizer state, so training cannot resume mid-run.
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
