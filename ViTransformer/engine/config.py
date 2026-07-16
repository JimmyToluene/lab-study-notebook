"""
Typed configuration loaded from the YAML files in ``configs/``.

This module is the single source of truth for config-file structure.
Downstream code (data, model, training loop) receives a ``Config``object;
nothing else should parse the YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    patch_size: Tuple[int, int]
    use_pe: bool = True   # learned positional embedding (the CLS token is always kept)
    r_ffnn: int = 4       # FFN expansion ratio inside each encoder block
    drop_path: float = 0.0  # max stochastic-depth rate (linearly scaled across layers)


@dataclass
class DataConfig:
    dataset: str
    img_size: Tuple[int, int]
    batch_size: int
    root: str = "./../datasets"   # matches the original simple_train_test.py layout
    num_workers: int = 2
    # Fraction of the *training* set held out as validation (monitored during
    # training). The test split is never touched until test.py. 0 disables val.
    val_split: float = 0.1
    val_split_seed: int = 0       # fixed so the train/val partition is reproducible


@dataclass
class TrainConfig:
    lr: float
    epochs: int
    seed: int = 0
    weight_decay: float = 0.0        # AdamW decoupled weight decay (DeiT uses 0.05)
    label_smoothing: float = 0.0     # only used when Mixup is off
    scheduler: str = "none"          # "none" | "cosine"
    warmup_epochs: int = 0           # linear warmup before cosine decay
    min_lr: float = 1e-5             # cosine floor
    grad_clip: float = 0.0           # max grad norm; 0 disables
    patience: int = 0                # early-stop patience in epochs; 0 disables


@dataclass
class AugConfig:
    # ---- transform-level (per-image) ----
    normalize: bool = False          # standardize with the dataset's mean/std
    hflip: bool = False              # random horizontal flip
    crop_padding: int = 0            # RandomCrop(img_size, padding); 0 disables
    rand_augment: bool = False
    randaug_num_ops: int = 2
    randaug_magnitude: int = 9
    random_erase: float = 0.0        # probability; 0 disables
    # ---- batch-level (per-batch, in the training loop) ----
    mixup: float = 0.0               # Beta alpha for Mixup; 0 disables
    cutmix: float = 0.0              # Beta alpha for CutMix; 0 disables
    mixup_prob: float = 1.0          # prob of applying mixup/cutmix to a batch
    mixup_switch_prob: float = 0.5   # prob of choosing CutMix over Mixup

    @property
    def mixup_active(self) -> bool:
        return (self.mixup > 0 or self.cutmix > 0) and self.mixup_prob > 0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "vitransformer"
    entity: str | None = None     # W&B team/user; None uses your default
    name: str | None = None       # run name; None falls back to the config name
    mode: str = "online"          # "online" | "offline" | "disabled"


@dataclass
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    aug: AugConfig = field(default_factory=AugConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        model = ModelConfig(**raw["model"])
        data = DataConfig(**raw["data"])
        train = TrainConfig(**raw["train"])
        aug = AugConfig(**raw.get("aug", {}))         # optional section
        wandb = WandbConfig(**raw.get("wandb", {}))   # optional section

        # YAML gives lists; the model code indexes these as fixed (row, col) pairs.
        model.patch_size = tuple(model.patch_size)
        data.img_size = tuple(data.img_size)

        return cls(name=raw["name"], model=model, data=data, train=train, aug=aug, wandb=wandb)

    def to_dict(self) -> dict:
        return asdict(self)
