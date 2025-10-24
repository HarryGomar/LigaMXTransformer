"""High level training orchestration."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from . import artifacts
from .config import (
    BATCH_PLY,
    BATCH_POS,
    CLIP_NORM,
    D_MODEL,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    EXPORT_TOKEN_WINDOWS,
    LR,
    MASK_PLAYER_FRAC,
    NUM_WORKERS,
    PATHS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    SEED,
    SHOT_WINDOW_RADIUS,
    WEIGHT_DECAY,
)
from .dataloading import collate_fn, loader_kwargs
from .datasets import (
    ACTION_FAMILY_VSZ,
    CAT_FACTORS,
    NUM_FEATURES,
    QUALIFIERS,
    VOCAB_SIZES,
    build_splits,
    build_stats_map,
    compute_num_norm,
    load_datasets,
)
from .engine import BatchLoss, build_optimizer, build_scheduler, run_epoch
from .export import export_behavioral_embeddings, export_possession_embeddings, export_table_embeddings
from .model import StyleModel
from .utils import ensure_directories, log, save_json


def _gpu_cc() -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def _configure_torch() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        from torch.backends.cuda import sdp_kernel  # type: ignore

        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(True)
    except Exception:
        pass


def _amp_dtype() -> torch.dtype:
    cc = _gpu_cc()
    if torch.cuda.is_available() and cc and cc[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _loader(dataset, batch_size: int, shuffle: bool) -> Optional[DataLoader]:
    if dataset is None:
        return None
    kwargs = loader_kwargs(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(**kwargs)


def run() -> None:
    ensure_directories()
    _configure_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[setup] Device: {device} | Torch: {torch.__version__}")
    if device.type == "cuda":
        cc = _gpu_cc()
        log(f"[setup] CUDA capability: {cc[0]}.{cc[1]}" if cc else "[setup] CUDA capability: unknown")

    pos_ds, ply_ds, stats_df, stat_cols = load_datasets()
    pos_splits, ply_splits = build_splits(pos_ds, ply_ds)

    pos_tr = pos_splits[0] if pos_splits else None
    pos_va = pos_splits[1] if pos_splits else None
    pos_te = pos_splits[2] if pos_splits else None
    ply_tr = ply_splits[0] if ply_splits else None
    ply_va = ply_splits[1] if ply_splits else None
    ply_te = ply_splits[2] if ply_splits else None

    num_mean, num_std = compute_num_norm(pos_tr, ply_tr)
    num_mean_t = torch.tensor(num_mean, device=device)
    num_std_t = torch.tensor(num_std, device=device)

    stats_map, mu, sigma = build_stats_map(ply_tr, stats_df, stat_cols)
    if ply_ds is not None:
        ply_ds.set_stats_map(stats_map)

    pos_train_loader = _loader(pos_tr, BATCH_POS, True)
    pos_val_loader = _loader(pos_va, BATCH_POS, False)
    pos_test_loader = _loader(pos_te, BATCH_POS, False)

    ply_train_loader = _loader(ply_tr, BATCH_PLY, True)
    ply_val_loader = _loader(ply_va, BATCH_PLY, False)
    ply_test_loader = _loader(ply_te, BATCH_PLY, False)

    n_player_season = (max(artifacts.player_season_id2key().keys()) + 1) if artifacts.player_season_id2key() else 2
    ts_map = artifacts.team_season_map()
    if ts_map and ts_map.get("id2key"):
        n_team_season = max(int(k) for k in ts_map["id2key"].keys()) + 1
    else:
        n_team_season = 2
    n_poshint = VOCAB_SIZES.get("position_hint", 4)

    model = StyleModel(n_player_season, n_team_season, n_poshint).to(device)
    if len(stat_cols) > 0:
        model.set_stats_head(len(stat_cols))

    can_compile = device.type == "cuda" and (_gpu_cc() and _gpu_cc()[0] >= 7)
    if can_compile or device.type == "cpu":
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore[attr-defined]
            log("[compile] torch.compile enabled.")
        except Exception as exc:
            log(f"[compile] skipped (fallback to eager): {exc}")
    else:
        log("[compile] skipped: GPU compute capability < 7.0. Using eager mode.")

    steps_per_epoch = max(len(pos_train_loader) if pos_train_loader else 0, len(ply_train_loader) if ply_train_loader else 0)
    total_steps = max(1, EPOCHS * steps_per_epoch)

    optimizer = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, total_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    no_improve = 0
    best_path = PATHS.out_root / "best.pt"
    last_path = PATHS.out_root / "last.pt"

    meta = {
        "dataset_sizes": {
            "pos_train": len(pos_tr.indices) if pos_tr else 0,
            "pos_val": len(pos_va.indices) if pos_va else 0,
            "pos_test": len(pos_te.indices) if pos_te else 0,
            "ply_train": len(ply_tr.indices) if ply_tr else 0,
            "ply_val": len(ply_va.indices) if ply_va else 0,
            "ply_test": len(ply_te.indices) if ply_te else 0,
        },
        "dims": {
            "k_num": len(NUM_FEATURES),
            "q_qual": len(QUALIFIERS),
            "n_player_season": n_player_season,
            "n_team_season": n_team_season,
            "n_poshint": n_poshint,
        },
        "cat_factors": CAT_FACTORS,
        "qualifiers": QUALIFIERS,
        "num_features": NUM_FEATURES,
    }
    save_json(PATHS.out_root / "meta.json", meta)

    history: List[Dict[str, float]] = []

    amp_dtype = _amp_dtype()

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()
        tr_metrics = run_epoch(
            model,
            (pos_train_loader, ply_train_loader),
            optimizer,
            scheduler,
            (num_mean_t, num_std_t),
            len(stat_cols),
            train=True,
            epoch=epoch,
            desc=f"train {epoch}/{EPOCHS}",
            scaler=scaler,
            use_amp=True,
        )
        va_metrics = run_epoch(
            model,
            (pos_val_loader, ply_val_loader),
            optimizer=None,
            scheduler=None,
            num_norm=(num_mean_t, num_std_t),
            k_stats=len(stat_cols),
            train=False,
            epoch=epoch,
            desc=f"valid {epoch}/{EPOCHS}",
            scaler=None,
            use_amp=True,
        )

        log(
            f"[epoch {epoch}] train: L_evt={tr_metrics['L_event']:.4f} Acc={tr_metrics['Acc']:.3f} "
            f"L_st={tr_metrics['L_stats']:.4f} L_mp={tr_metrics['L_maskp']:.4f} "
            f"L_tm={tr_metrics['L_team']:.4f} L_tot={tr_metrics['L_total']:.4f}"
        )
        log(
            f"[epoch {epoch}]   val: L_evt={va_metrics['L_event']:.4f} Acc={va_metrics['Acc']:.3f} "
            f"L_st={va_metrics['L_stats']:.4f} L_mp={va_metrics['L_maskp']:.4f} "
            f"L_tm={va_metrics['L_team']:.4f} L_tot={va_metrics['L_total']:.4f}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "D_MODEL": D_MODEL,
                    "cat_factors": CAT_FACTORS,
                    "qualifiers": QUALIFIERS,
                    "num_features": NUM_FEATURES,
                },
                "optim": optimizer.state_dict(),
            },
            last_path,
        )

        record = {
            "epoch": epoch,
            **{k: float(v) for k, v in tr_metrics.items()},
            **{f"val_{k}": float(v) for k, v in va_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        save_json(PATHS.out_root / "history.json", {"history": history})

        if va_metrics["L_total"] < best_val - 1e-4:
            best_val = va_metrics["L_total"]
            torch.save({"model": model.state_dict()}, best_path)
            log(f"[ckpt] new best val L_tot={best_val:.4f} → saved to {best_path.name}")
            no_improve = 0
        else:
            no_improve += 1
            log(f"[early-stop] no improvement streak: {no_improve}/{EARLY_STOP_PATIENCE}")
            if no_improve >= EARLY_STOP_PATIENCE:
                log(f"[early-stop] stopping training at epoch {epoch}.")
                break

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
        log(f"[load] best checkpoint loaded from {best_path.name}")
    else:
        log("[warn] best checkpoint not found; using last weights.")

    test_metrics = run_epoch(
        model,
        (pos_test_loader, ply_test_loader),
        optimizer=None,
        scheduler=None,
        num_norm=(num_mean_t, num_std_t),
        k_stats=len(stat_cols),
        train=False,
        epoch=0,
        desc="test",
        scaler=None,
        use_amp=True,
    )
    log(
        f"[test] L_evt={test_metrics['L_event']:.4f} Acc={test_metrics['Acc']:.3f} "
        f"L_st={test_metrics['L_stats']:.4f} L_mp={test_metrics['L_maskp']:.4f} "
        f"L_tm={test_metrics['L_team']:.4f} L_tot={test_metrics['L_total']:.4f}"
    )
    save_json(PATHS.out_root / "test_metrics.json", test_metrics)

    export_table_embeddings(model)
    export_behavioral_embeddings(model, ply_ds if ply_ds is not None else None, (num_mean_t, num_std_t), batch_size=64)
    export_possession_embeddings(
        model,
        pos_ds if pos_ds is not None else None,
        (num_mean_t, num_std_t),
        batch_size=64,
        export_token_windows=EXPORT_TOKEN_WINDOWS,
        shot_radius=SHOT_WINDOW_RADIUS,
    )

    log("[done] training + exports complete.")
    log(f" - {PATHS.export_root / 'pos_embeddings.npz'}")
    log(f" - {PATHS.export_root / 'player_identity_embeddings.npy'} (+ .tsv)")
    log(f" - {PATHS.export_root / 'player_style_embeddings.npy'} (+ .tsv)")
    log(f" - {PATHS.export_root / 'pos_token_windows.npz'} (if enabled)")
