"""Dataset registry + transform construction.

Two things worth calling out:

1. Train and eval use *different* transforms: augmentation (RandAugment, crop,
   flip, erasing) is applied to training only; val/test/inference get a clean
   deterministic transform. This is why ``build_dataloaders`` builds the
   training set twice and splits by index -- the validation subset must see the
   eval transform even though it is carved from the training data.

2. ``n_channels`` / ``n_classes`` / ``mean`` / ``std`` belong to the dataset,
   not the config, so they live on ``DatasetSpec`` here.

To add a dataset: write a builder and a labels function, then register both in
``DATASETS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST

from .config import Config


@dataclass
class DatasetSpec:
    n_channels: int
    n_classes: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    # builder(root, img_size, transform, train) -> Dataset
    builder: Callable[[str, Tuple[int, int], T.Compose, bool], Dataset]
    # labels(root) -> human-readable name per class index, or None if the names
    # cannot be recovered (e.g. inference from a checkpoint with no dataset on
    # disk). Index i must match what the builder's Dataset assigns to class i.
    labels: Callable[[str], Optional[List[str]]]


def _build_eval_transform(cfg: Config, spec: DatasetSpec) -> T.Compose:
    """Deterministic preprocessing for val / test / inference."""
    steps = [T.Resize(cfg.data.img_size)]
    if spec.n_channels == 1:
        steps.append(T.Grayscale(num_output_channels=1))
    steps.append(T.ToTensor())
    if cfg.aug.normalize:
        steps.append(T.Normalize(spec.mean, spec.std))
    return T.Compose(steps)


def _build_train_transform(cfg: Config, spec: DatasetSpec) -> T.Compose:
    """Augmented preprocessing for training only (per the aug config)."""
    a = cfg.aug
    steps = [T.Resize(cfg.data.img_size)]

    # PIL-level geometric / policy augmentation
    if a.crop_padding > 0:
        steps.append(T.RandomCrop(cfg.data.img_size, padding=a.crop_padding))
    if a.hflip:
        steps.append(T.RandomHorizontalFlip())
    if a.rand_augment:
        steps.append(T.RandAugment(num_ops=a.randaug_num_ops, magnitude=a.randaug_magnitude))

    if spec.n_channels == 1:
        steps.append(T.Grayscale(num_output_channels=1))

    steps.append(T.ToTensor())
    if cfg.aug.normalize:
        steps.append(T.Normalize(spec.mean, spec.std))
    # Tensor-level erasing goes last.
    if a.random_erase > 0:
        steps.append(T.RandomErasing(p=a.random_erase))

    return T.Compose(steps)


def _build_mnist(root, img_size, transform, train):
    return MNIST(root=root, train=train, download=True, transform=transform)


def _mnist_labels(root):
    # MNIST targets are the digits themselves, so the index *is* the label.
    return [str(d) for d in range(10)]


def _build_tiny_imagenet(root, img_size, transform, train):
    # Expects <root>/tiny-imagenet-200/{train,val} in ImageFolder layout.
    # The official val/ split must be reorganized into per-class subdirs first
    # (see prepare_tiny_imagenet.py). We use the official val/ as the *test*
    # split; the validation split is carved from train/ by build_dataloaders.
    split = "train" if train else "val"
    path = Path(root) / "tiny-imagenet-200" / split
    if not path.is_dir():
        raise FileNotFoundError(
            f"tiny-imagenet split not found at '{path}'. Run "
            f"`python prepare_tiny_imagenet.py` to download and prepare it."
        )
    return ImageFolder(str(path), transform=transform)


def _tiny_imagenet_labels(root):
    base = Path(root) / "tiny-imagenet-200"

    # ImageFolder numbers classes by *sorted* dir name, so the order must come
    # from sorting -- wnids.txt is not stored in sorted order and using its file
    # order here would mislabel every prediction.
    train = base / "train"
    if train.is_dir():
        wnids = sorted(p.name for p in train.iterdir() if p.is_dir())
    elif (base / "wnids.txt").is_file():
        wnids = sorted((base / "wnids.txt").read_text().split())
    else:
        return None   # no dataset on disk; caller falls back to raw indices

    # words.txt is the full WordNet table (~82k rows), not just these 200:
    # <wnid>\t<comma-separated synonyms>. Keep the first synonym.
    words = base / "words.txt"
    if not words.is_file():
        return wnids
    names = {}
    for line in words.read_text().splitlines():
        wnid, _, synonyms = line.partition("\t")
        names[wnid] = synonyms.split(",")[0].strip()
    return [names.get(w, w) for w in wnids]


DATASETS = {
    "mnist": DatasetSpec(
        n_channels=1, n_classes=10,
        mean=(0.1307,), std=(0.3081,),
        builder=_build_mnist,
        labels=_mnist_labels,
    ),
    "tiny-imagenet": DatasetSpec(
        n_channels=3, n_classes=200,
        mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821),
        builder=_build_tiny_imagenet,
        labels=_tiny_imagenet_labels,
    ),
}


def get_spec(dataset: str) -> DatasetSpec:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {sorted(DATASETS)}")
    return DATASETS[dataset]


def get_labels(cfg: Config) -> Optional[List[str]]:
    """Class names indexed by model output index, or None if unavailable."""
    spec = get_spec(cfg.data.dataset)
    labels = spec.labels(cfg.data.root)
    if labels is not None and len(labels) != spec.n_classes:
        # A partial dataset dir would silently shift every name onto the wrong
        # index. Showing raw indices is better than showing confident lies.
        return None
    return labels


def build_transform(cfg: Config) -> T.Compose:
    """Eval-time transform (used by infer.py)."""
    return _build_eval_transform(cfg, get_spec(cfg.data.dataset))


def build_dataloaders(cfg: Config):
    """Return (train_loader, val_loader, test_loader, spec).

    ``val`` is carved from the *training* set with a fixed seed and given the
    *eval* transform (no augmentation); ``test`` is the dataset's held-out
    split. ``val_loader`` is None when ``data.val_split`` is 0.
    """
    spec = get_spec(cfg.data.dataset)
    train_tf = _build_train_transform(cfg, spec)
    eval_tf = _build_eval_transform(cfg, spec)

    test_set = spec.builder(cfg.data.root, cfg.data.img_size, eval_tf, False)

    val_split = cfg.data.val_split
    if not 0 <= val_split < 1:
        raise ValueError(f"data.val_split must be in [0, 1), got {val_split}")

    train_aug = spec.builder(cfg.data.root, cfg.data.img_size, train_tf, True)
    if val_split > 0:
        # Second view of the same data with the eval transform, so the val
        # subset is never augmented even though it comes from train.
        train_clean = spec.builder(cfg.data.root, cfg.data.img_size, eval_tf, True)
        n = len(train_aug)
        n_val = int(n * val_split)
        generator = torch.Generator().manual_seed(cfg.data.val_split_seed)
        perm = torch.randperm(n, generator=generator).tolist()
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_set = Subset(train_aug, train_idx)
        val_set = Subset(train_clean, val_idx)
    else:
        train_set, val_set = train_aug, None

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=cfg.data.batch_size, shuffle=shuffle,
            num_workers=cfg.data.num_workers,
        )

    train_loader = _loader(train_set, shuffle=True)
    val_loader = _loader(val_set, shuffle=False) if val_set is not None else None
    test_loader = _loader(test_set, shuffle=False)
    return train_loader, val_loader, test_loader, spec
