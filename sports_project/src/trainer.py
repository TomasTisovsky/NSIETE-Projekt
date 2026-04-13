from __future__ import annotations

import numpy as np
import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device: torch.device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    total_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        preds = logits.argmax(dim=1)

        running_loss += loss.item() * batch_size
        running_correct += (preds == targets).sum().item()
        total_samples += batch_size

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            avg_loss = running_loss / max(total_samples, 1)
            avg_acc = running_correct / max(total_samples, 1)
            print(
                f"  batch {batch_idx:>4}/{total_batches} | "
                f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}",
                flush=True,
            )

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = running_correct / max(total_samples, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device: torch.device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == targets).sum().item()
        total_samples += batch_size

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu())

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = running_correct / max(total_samples, 1)

    if all_probs:
        probs_array = torch.cat(all_probs, dim=0).numpy()
    else:
        probs_array = np.empty((0, 0), dtype=np.float32)

    return epoch_loss, epoch_acc, all_targets, all_preds, probs_array