"""Training loop utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import (
    CLIP_NORM,
    LAMBDA_EVENT,
    LAMBDA_MASKP,
    LAMBDA_STATS,
    LAMBDA_TEAMGRL,
    MASK_PLAYER_FRAC,
    WARMUP_FRAC,
)
from .dataloading import CUDAPrefetcher
from .losses import event_loss_fn, huber_loss_masked, sampled_softmax_logits
from .model import StyleModel, grad_reverse
from .utils import log


@dataclass
class BatchLoss:
    event_loss: float
    event_acc: float
    stats_loss: float
    maskp_loss: float
    team_loss: float
    total: float


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "ln" in name.lower() or "layernorm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]
    try:
        return torch.optim.AdamW(groups, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float = WARMUP_FRAC):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, (total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def forward_one(
    model: StyleModel,
    batch: Dict[str, torch.Tensor],
    num_norm,
    k_stats: int,
    is_player_batch: bool,
    do_mask_player: bool,
    team_adv: bool,
    lambdas,
) -> Tuple[BatchLoss, torch.Tensor]:
    mean, std = num_norm
    device = next(model.parameters()).device
    batch_size = batch["mask"].size(0)

    if is_player_batch and do_mask_player:
        mask_player_vec = (torch.rand((batch_size,), device=device) < MASK_PLAYER_FRAC)
    else:
        mask_player_vec = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    out = model(batch, mean, std, mask_player_vec=mask_player_vec, causal=True)
    logits_event = out["logits_event"]
    pooled_h = out["pooled_h"]
    Ep = out["Ep"]
    mask_bool = out["mask_bool"]
    targets = batch["Y"].to(device)

    events_lambda, stats_lambda, mask_lambda, team_lambda = lambdas

    loss_event, acc = event_loss_fn(logits_event, targets, mask_bool)

    loss_stats = torch.tensor(0.0, device=device)
    if is_player_batch and (model.stats_head is not None) and ("stats_vec" in batch):
        unmasked = (~mask_player_vec)
        if unmasked.any():
            target_vec = batch["stats_vec"][unmasked].to(device)
            target_mask = batch["stats_mask"][unmasked].to(device)
            pred = model.stats_head(pooled_h[unmasked], Ep[unmasked])
            loss_stats = huber_loss_masked(pred, target_vec, target_mask, delta=1.0)

    loss_maskp = torch.tensor(0.0, device=device)
    if is_player_batch and mask_player_vec.any():
        H_players = batch["H_player"].to(device)
        cand_ids, inv = torch.unique(H_players, return_inverse=True)
        query = model.maskp_head(pooled_h[mask_player_vec])
        cand_mat = model.fusion.E_player(cand_ids)
        logits = sampled_softmax_logits(query, cand_mat)
        targets_mask = inv[mask_player_vec]
        loss_maskp = torch.nn.functional.cross_entropy(logits, targets_mask)

    loss_team = torch.tensor(0.0, device=device)
    if is_player_batch and team_adv:
        H_teams = batch["H_team"].to(device)
        cand_t, inv_t = torch.unique(H_teams, return_inverse=True)
        Ep_rev = grad_reverse(Ep, 1.0)
        team_emb = model.fusion.E_team(cand_t)
        logits_t = sampled_softmax_logits(Ep_rev, team_emb)
        loss_team = torch.nn.functional.cross_entropy(logits_t, inv_t)

    total = (
        events_lambda * loss_event
        + stats_lambda * loss_stats
        + mask_lambda * loss_maskp
        + team_lambda * loss_team
    )

    return (
        BatchLoss(
            event_loss=float(loss_event.item()),
            event_acc=float(acc.item()),
            stats_loss=float(loss_stats.item()),
            maskp_loss=float(loss_maskp.item()),
            team_loss=float(loss_team.item()),
            total=float(total.item()),
        ),
        total,
    )


def run_epoch(
    model: StyleModel,
    loaders: Tuple[Optional[DataLoader], Optional[DataLoader]],
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_norm,
    k_stats: int,
    train: bool,
    epoch: int,
    desc: str,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    lambdas = (LAMBDA_EVENT, LAMBDA_STATS, LAMBDA_MASKP, LAMBDA_TEAMGRL),
) -> Dict[str, float]:
    pos_loader, ply_loader = loaders
    device = next(model.parameters()).device

    pos_pref = CUDAPrefetcher(pos_loader, device) if pos_loader else None
    ply_pref = CUDAPrefetcher(ply_loader, device) if ply_loader else None
    if pos_pref:
        pos_pref.reset()
    if ply_pref:
        ply_pref.reset()

    model.train() if train else model.eval()

    n_steps = max(len(pos_loader) if pos_loader else 0, len(ply_loader) if ply_loader else 0)
    progress = range(n_steps)

    sums = {"event": 0.0, "acc": 0.0, "stats": 0.0, "maskp": 0.0, "team": 0.0, "total": 0.0}
    count = 0

    for _ in progress:
        step_losses = []
        totals = []

        bpos = pos_pref.next() if pos_pref else None
        bply = ply_pref.next() if ply_pref else None

        if bpos is not None:
            with torch.set_grad_enabled(train):
                if train and use_amp and scaler is not None:
                    with autocast():
                        bl, tot = forward_one(
                            model,
                            bpos,
                            num_norm,
                            k_stats,
                            is_player_batch=False,
                            do_mask_player=False,
                            team_adv=False,
                            lambdas=lambdas,
                        )
                else:
                    bl, tot = forward_one(
                        model,
                        bpos,
                        num_norm,
                        k_stats,
                        is_player_batch=False,
                        do_mask_player=False,
                        team_adv=False,
                        lambdas=lambdas,
                    )
                step_losses.append(bl)
                totals.append(tot)

        if bply is not None:
            with torch.set_grad_enabled(train):
                if train and use_amp and scaler is not None:
                    with autocast():
                        bl, tot = forward_one(
                            model,
                            bply,
                            num_norm,
                            k_stats,
                            is_player_batch=True,
                            do_mask_player=True,
                            team_adv=True,
                            lambdas=lambdas,
                        )
                else:
                    bl, tot = forward_one(
                        model,
                        bply,
                        num_norm,
                        k_stats,
                        is_player_batch=True,
                        do_mask_player=True,
                        team_adv=True,
                        lambdas=lambdas,
                    )
                step_losses.append(bl)
                totals.append(tot)

        if train and totals:
            loss_sum = torch.stack(totals).sum()
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(loss_sum).backward()
                if CLIP_NORM:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_sum.backward()
                if CLIP_NORM:
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        for bl in step_losses:
            sums["event"] += bl.event_loss
            sums["acc"] += bl.event_acc
            sums["stats"] += bl.stats_loss
            sums["maskp"] += bl.maskp_loss
            sums["team"] += bl.team_loss
            sums["total"] += bl.total
            count += 1

    if count == 0:
        return {"L_event": 0.0, "Acc": 0.0, "L_stats": 0.0, "L_maskp": 0.0, "L_team": 0.0, "L_total": 0.0}
    return {
        "L_event": sums["event"] / count,
        "Acc": sums["acc"] / count,
        "L_stats": sums["stats"] / count,
        "L_maskp": sums["maskp"] / count,
        "L_team": sums["team"] / count,
        "L_total": sums["total"] / count,
    }
