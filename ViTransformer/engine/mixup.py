"""Mixup / CutMix batch augmentation and the matching soft-target loss.

Mixup and CutMix blend pairs of samples (and their labels), which turns the
hard integer labels into soft probability vectors. That is why they come with
their own loss: ``SoftTargetCrossEntropy`` consumes those soft targets. Label
smoothing is folded in here (applied to the one-hot targets before mixing), so
when Mixup is active the plain ``label_smoothing`` on CrossEntropyLoss is not
used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against soft (probability-vector) targets."""
    def forward(self, logits, target):  # target: (B, C)
        loss = torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()


def _smoothed_one_hot(target, num_classes, smoothing):
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y = torch.full((target.size(0), num_classes), off, device=target.device)
    return y.scatter_(1, target.unsqueeze(1), on)


def _rand_bbox(h, w, lam):
    """A random box whose area is (1 - lam) of the image (CutMix)."""
    ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


class Mixup:
    """timm-style Mixup+CutMix. Call as ``x, soft_target = mixup(x, labels)``."""

    def __init__(self, mixup_alpha, cutmix_alpha, prob, switch_prob,
                 num_classes, label_smoothing=0.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def __call__(self, x, target):
        soft = _smoothed_one_hot(target, self.num_classes, self.label_smoothing)

        # Per-batch coin flip: sometimes pass through un-mixed.
        if torch.rand(1).item() > self.prob:
            return x, soft

        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or torch.rand(1).item() < self.switch_prob
        )
        perm = torch.randperm(x.size(0), device=x.device)

        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            h, w = x.shape[-2:]
            y1, y2, x1, x2 = _rand_bbox(h, w, lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            # Correct lam to the *actual* pasted area.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha)) if self.mixup_alpha > 0 else 1.0
            x = lam * x + (1.0 - lam) * x[perm]

        target = lam * soft + (1.0 - lam) * soft[perm]
        return x, target
