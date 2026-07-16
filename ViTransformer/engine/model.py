"""Model factory: Config (+ dataset spec) -> a ready ViTClassifier.

Centralizing construction means train/test/infer can never disagree about how
a model is wired. ``n_channels`` and ``n_classes`` come from the dataset spec,
not the config, which is why they are threaded in here rather than read from
the YAML.
"""

from __future__ import annotations

import torch

from vit.ViT import ViTBackbone, ViTClassifier

from .config import Config
from .data import DatasetSpec, get_spec


def build_model(cfg: Config, spec: DatasetSpec, device: torch.device) -> ViTClassifier:
    m = cfg.model
    backbone = ViTBackbone(
        d_model=m.d_model,
        n_classes=spec.n_classes,
        img_size=cfg.data.img_size,
        patch_size=m.patch_size,
        n_channels=spec.n_channels,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        use_pe=m.use_pe,
        r_ffn=m.r_ffn,
        drop_path=m.drop_path,
    )
    model = ViTClassifier(backbone, spec.n_classes)
    return model.to(device)


def build_model_from_config(cfg: Config, device: torch.device) -> ViTClassifier:
    """Convenience wrapper that resolves the dataset spec from the config."""
    return build_model(cfg, get_spec(cfg.data.dataset), device)
