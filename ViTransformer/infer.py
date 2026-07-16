"""Run a trained checkpoint on one or more image files.

    python infer.py --ckpt runs/mnist/model.pt path/to/digit.png
    python infer.py --ckpt runs/mnist/model.pt a.png b.png --topk 3

Reuses the checkpoint's own preprocessing so inference matches training. The
input image is coerced to the model's expected channel count, so an ordinary
RGB photo works against an MNIST (grayscale) model and vice-versa.
"""

import argparse

import torch
from PIL import Image

from engine import (
    build_transform, describe_device, get_device, get_labels, get_spec, load_model,
)


def main():
    ap = argparse.ArgumentParser(description="Classify images with a ViT checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint saved by train.py.")
    ap.add_argument("images", nargs="+", help="One or more image files to classify.")
    ap.add_argument("--topk", type=int, default=5, help="How many top predictions to show.")
    args = ap.parse_args()

    device = get_device()
    print("Using device:", describe_device(device))

    model, cfg = load_model(args.ckpt, device)
    transform = build_transform(cfg)
    spec = get_spec(cfg.data.dataset)
    pil_mode = "L" if spec.n_channels == 1 else "RGB"
    k = min(args.topk, spec.n_classes)

    labels = get_labels(cfg)
    if labels is None:
        print(f"Class names unavailable for '{cfg.data.dataset}' under "
              f"'{cfg.data.root}'; showing raw class indices.")

    for path in args.images:
        img = Image.open(path).convert(pil_mode)
        x = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        confs, idxs = probs.topk(k)
        print(f"\n{path}")
        for conf, idx in zip(confs.tolist(), idxs.tolist()):
            name = f"{labels[idx]} " if labels else ""
            print(f"  {name}(class {idx:>3}): {100 * conf:5.2f}%")


if __name__ == "__main__":
    main()
