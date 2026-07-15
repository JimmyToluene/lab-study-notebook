"""Evaluate a trained checkpoint on its dataset's test split.

    python test.py --ckpt runs/mnist/model.pt

The checkpoint carries its own config, so no --config flag is needed: the
model architecture and the dataset are both reconstructed from the file.
"""

import argparse

from engine import build_dataloaders, describe_device, evaluate, get_device, load_model


def main():
    ap = argparse.ArgumentParser(description="Evaluate a ViT checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint saved by train.py.")
    args = ap.parse_args()

    device = get_device()
    print("Using device:", describe_device(device))

    model, cfg = load_model(args.ckpt, device)
    *_, test_loader, _ = build_dataloaders(cfg)

    _, acc = evaluate(model, test_loader, device)
    print(f"[{cfg.name}] test accuracy: {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
