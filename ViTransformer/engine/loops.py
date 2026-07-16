"""The train / evaluate loops, lifted out of the flat script.

Making ``evaluate`` a standalone function is the whole reason test.py can
exist as its own entrypoint: the eval logic no longer lives inside the epoch
loop.
"""

from __future__ import annotations

import torch


def train_one_epoch(model, loader, optimizer, criterion, device,
                    mixup_fn=None, grad_clip=0.0) -> float:
    """
    Run one pass over ``loader``; return the mean per-batch loss.

    When ``mixup_fn`` is given it turns (inputs, labels) into (mixed inputs,soft targets)
    and ``criterion`` must be a soft-target loss. ``grad_clip`` > 0 clips the gradient norm before the optimizer step.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        targets = labels
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Evaluate over ``loader``; return ``(mean_loss, accuracy)``.

    ``mean_loss`` is None when no ``criterion`` is passed (e.g. test.py only cares about accuracy);
    accuracy is a fraction in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion is not None:
            loss_sum += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    mean_loss = loss_sum / len(loader) if criterion is not None else None
    return mean_loss, correct / total
