"""Loss helpers used during training."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import LABEL_SMOOTH


def event_loss_fn(logits: torch.Tensor, targets: torch.Tensor, mask_bool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = logits.size(-1)
    logits_flat = logits.reshape(-1, classes)
    targets_flat = targets.reshape(-1)
    valid = (targets_flat != 0) & mask_bool.reshape(-1)
    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero
    logits_sel = logits_flat[valid]
    targets_sel = targets_flat[valid]
    loss = F.cross_entropy(logits_sel, targets_sel, label_smoothing=LABEL_SMOOTH)
    with torch.no_grad():
        acc = (logits_sel.argmax(dim=-1) == targets_sel).float().mean()
    return loss, acc


def huber_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - target)[mask]
    abs_diff = diff.abs()
    sq = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    loss = 0.5 * (sq ** 2) + delta * (abs_diff - sq)
    return loss.mean()


def sampled_softmax_logits(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return query @ candidates.t()
