"""Train a ViT from a YAML config (DeiT-style recipe, all config-driven).

    python train.py --config configs/mnist-baseline.yaml
    python train.py --config configs/tiny-imagenet-baseline.yaml --wandb

Monitors a held-out *validation* split each epoch (never the test set), saves
the best-val checkpoint plus the last, and optionally early-stops. Augmentation,
Mixup/CutMix, AdamW+weight-decay, label smoothing, cosine LR, stochastic depth
and grad clipping are all switched on from the config -- an empty/omitted aug
section and the defaults reproduce plain training.
"""

import argparse
from pathlib import Path

import torch.nn as nn

from engine import (
    Config, Mixup, SoftTargetCrossEntropy, Tracker, build_dataloaders,
    build_model, build_optimizer, build_scheduler, describe_device, evaluate,
    get_device, save_checkpoint, set_seed, train_one_epoch,
)


def main():
    ap = argparse.ArgumentParser(description="Train a ViT classifier.")
    ap.add_argument("--config", required=True, help="Path to a YAML config in configs/.")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: runs/<name>).")
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging (overrides config).")
    ap.add_argument("--wandb-project", default=None, help="Override wandb.project.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.wandb:
        cfg.wandb.enabled = True
    if args.wandb_project:
        cfg.wandb.project = args.wandb_project

    set_seed(cfg.train.seed)
    device = get_device()
    print("Using device:", describe_device(device))

    train_loader, val_loader, test_loader, spec = build_dataloaders(cfg)
    if val_loader is None:
        print("[warn] data.val_split is 0 -> no validation monitoring / best checkpoint.")
    model = build_model(cfg, spec, device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Mixup/CutMix produces soft targets -> needs a soft-target loss. Without it
    # we fall back to plain cross-entropy with (optional) label smoothing.
    if cfg.aug.mixup_active:
        mixup_fn = Mixup(cfg.aug.mixup, cfg.aug.cutmix, cfg.aug.mixup_prob,
                         cfg.aug.mixup_switch_prob, spec.n_classes, cfg.train.label_smoothing)
        train_criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # plain CE for comparable val/test loss

    out_dir = Path(args.out_dir or f"runs/{cfg.name}")
    best_acc, best_epoch, epochs_no_improve = -1.0, -1, 0

    with Tracker(cfg) as tracker:
        for epoch in range(cfg.train.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, train_criterion, device,
                mixup_fn=mixup_fn, grad_clip=cfg.train.grad_clip,
            )
            lr = optimizer.param_groups[0]["lr"]

            metrics = {"train/loss": train_loss, "lr": lr}
            msg = f"Epoch {epoch + 1}/{cfg.train.epochs}  train loss: {train_loss:.3f}  lr: {lr:.2e}"

            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, device, eval_criterion)
                metrics.update({"val/loss": val_loss, "val/acc": val_acc})
                msg += f"  val loss: {val_loss:.3f}  val acc: {100 * val_acc:.2f}%"
                if val_acc > best_acc:
                    best_acc, best_epoch, epochs_no_improve = val_acc, epoch + 1, 0
                    save_checkpoint(out_dir / "best.pt", model, cfg)
                    msg += "  *"
                else:
                    epochs_no_improve += 1

            print(msg)
            tracker.log(metrics, step=epoch + 1)

            if scheduler is not None:
                scheduler.step()

            if cfg.train.patience > 0 and val_loader is not None and epochs_no_improve >= cfg.train.patience:
                print(f"Early stopping: no val improvement for {cfg.train.patience} epochs.")
                break

        save_checkpoint(out_dir / "last.pt", model, cfg)

    if best_epoch > 0:
        print(f"Best val acc {100 * best_acc:.2f}% @ epoch {best_epoch}  ->  {out_dir / 'best.pt'}")
    print(f"Saved last checkpoint  ->  {out_dir / 'last.pt'}")
    ckpt = out_dir / ("best.pt" if best_epoch > 0 else "last.pt")
    print(f"Final test number:  python test.py --ckpt {ckpt}")


if __name__ == "__main__":
    main()
