"""Dataset definitions and artefact loading for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from . import artifacts
from .config import PATHS, SEED
from .utils import load_json, save_json, split_indices

CAT_FACTORS = artifacts.categorical_factors()
QUALIFIERS = artifacts.qualifiers()
NUM_FEATURES = artifacts.numeric_features()
VOCABS = artifacts.vocabs()
VOCAB_SIZES = artifacts.vocab_sizes()
ACTION_FAMILY_VSZ = artifacts.action_family_vocab_size()
PS_ID2KEY = artifacts.player_season_id2key()


class BaseSeqDataset(Dataset):
    """Base dataset that owns the arrays loaded from NPZ files."""

    def __init__(self, npz_path: Path, kind: str):
        super().__init__()
        self.kind = kind
        self.npz_path = npz_path

        with np.load(npz_path, allow_pickle=False) as z:
            for factor in CAT_FACTORS:
                if factor not in z.files:
                    raise RuntimeError(f"Missing factor array '{factor}' in {npz_path.name}")

            self.H_team = z["H_team"].astype(np.int64)
            self.H_season = z["H_season"].astype(np.int64)
            self.H_seqtype = z["H_seqtype"].astype(np.int64)
            self.H_poshint = z["H_poshint"].astype(np.int64) if "H_poshint" in z.files else None
            self.H_player = z["H_player"].astype(np.int64) if "H_player" in z.files else None
            self.H_match = z["H_match"].astype(np.int64) if "H_match" in z.files else None
            self.H_pos_idx = z["H_pos_idx"].astype(np.int64) if "H_pos_idx" in z.files else None

            mask = z["mask"]
            self.mask = mask.astype(np.bool_)
            self.X_num = z["X_num"].astype(np.float32)
            self.X_qual = z["X_qual"].astype(np.float32)
            self.Y_next = z["Y_next_action_family"].astype(np.int64)
            self.cats: Dict[str, np.ndarray] = {factor: z[factor].astype(np.int64) for factor in CAT_FACTORS}

            self.has_actor_ps = "actor_ps" in z.files
            if self.has_actor_ps:
                self.actor_ps = z["actor_ps"].astype(np.int64)

        self.N = self.mask.shape[0]
        self.T = self.mask.shape[1]

    def __len__(self) -> int:  # type: ignore[override]
        return self.N

    def __getitem__(self, index: int) -> Dict[str, Any]:  # type: ignore[override]
        item = {
            "mask": torch.from_numpy(self.mask[index]),
            "nums": torch.from_numpy(self.X_num[index]),
            "quals": torch.from_numpy(self.X_qual[index]),
            "cats": {factor: torch.from_numpy(self.cats[factor][index]) for factor in CAT_FACTORS},
            "Y": torch.from_numpy(self.Y_next[index]),
            "H_team": int(self.H_team[index]),
            "H_season": int(self.H_season[index]),
            "H_seqtype": int(self.H_seqtype[index]),
            "H_poshint": int(self.H_poshint[index]) if self.H_poshint is not None else -1,
            "H_player": int(self.H_player[index]) if self.H_player is not None else -1,
        }
        if getattr(self, "has_actor_ps", False):
            item["actor_ps"] = torch.from_numpy(self.actor_ps[index])
        return item


class PossessionSeqDataset(BaseSeqDataset):
    def __init__(self, npz_path: Path):
        super().__init__(npz_path, kind="possession")


class PlayerSeqDataset(BaseSeqDataset):
    def __init__(self, npz_path: Path, stat_cols: List[str]):
        super().__init__(npz_path, kind="player")
        self.stat_cols = stat_cols
        self.player_stats_map: Dict[int, np.ndarray] = {}

    def set_stats_map(self, mapping: Dict[int, np.ndarray]) -> None:
        self.player_stats_map = mapping

    def __getitem__(self, index: int) -> Dict[str, Any]:  # type: ignore[override]
        item = super().__getitem__(index)
        if getattr(self, "has_actor_ps", False):
            item["actor_ps"] = torch.from_numpy(self.actor_ps[index])

        hid = int(item["H_player"])
        if hid in self.player_stats_map:
            vec = torch.from_numpy(self.player_stats_map[hid]).float()
            item["stats_vec"] = vec
            item["stats_mask"] = torch.isfinite(vec)
        else:
            vec = torch.full((len(self.stat_cols),), float("nan"), dtype=torch.float32)
            item["stats_vec"] = vec
            item["stats_mask"] = torch.zeros_like(vec, dtype=torch.bool)
        return item


def load_datasets():
    pos_npz = PATHS.sequence_dir / "possession_sequences.npz"
    ply_npz = PATHS.sequence_dir / "player_sequences.npz"

    if not pos_npz.exists() and not ply_npz.exists():
        raise RuntimeError("No sequences found. Run the preprocessing step first.")

    stats_path = PATHS.stats_dir / "player_season_stats.parquet"
    if stats_path.exists():
        stats_df = pd.read_parquet(stats_path)
        drop_cols = {"player_id", "season_id"}
        stat_cols = [col for col in stats_df.columns if col not in drop_cols and np.issubdtype(stats_df[col].dtype, np.number)]
    else:
        stats_df = pd.DataFrame(columns=["player_id", "season_id"])
        stat_cols: List[str] = []

    pos_ds = PossessionSeqDataset(pos_npz) if pos_npz.exists() else None
    ply_ds = PlayerSeqDataset(ply_npz, stat_cols) if ply_npz.exists() else None
    return pos_ds, ply_ds, stats_df, stat_cols


def build_splits(pos_ds, ply_ds):
    pos_splits = None
    if pos_ds is not None:
        tr, va, te = split_indices(len(pos_ds), seed=SEED)
        pos_splits = (Subset(pos_ds, tr), Subset(pos_ds, va), Subset(pos_ds, te))

    ply_splits = None
    if ply_ds is not None:
        tr, va, te = split_indices(len(ply_ds), seed=SEED)
        ply_splits = (Subset(ply_ds, tr), Subset(ply_ds, va), Subset(ply_ds, te))
    return pos_splits, ply_splits


def _compute_num_norm_from_datasets(train_datasets: Iterable[BaseSeqDataset]) -> Tuple[np.ndarray, np.ndarray]:
    iterator = list(train_datasets)
    if not iterator:
        raise RuntimeError("No training datasets provided to compute numeric normalisation.")

    K = iterator[0].X_num.shape[2]
    sum_ = np.zeros((K,), dtype=np.float64)
    sqsum = np.zeros((K,), dtype=np.float64)
    count = np.zeros((K,), dtype=np.float64)

    for ds in iterator:
        X = ds.X_num
        M = ds.mask
        mask = M.astype(bool)
        valid = X[mask]
        if valid.size == 0:
            continue
        sum_ += valid.sum(axis=0, dtype=np.float64)
        sqsum += (valid ** 2).sum(axis=0, dtype=np.float64)
        count += np.array([mask.sum()] * K, dtype=np.float64)

    mean = sum_ / np.maximum(count, 1.0)
    var = (sqsum / np.maximum(count, 1.0)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_num_norm(train_pos: Optional[Subset], train_ply: Optional[Subset]):
    train_sets: List[BaseSeqDataset] = []
    if train_pos is not None:
        train_sets.append(train_pos.dataset)
    if train_ply is not None:
        train_sets.append(train_ply.dataset)
    if not train_sets:
        raise RuntimeError("No training datasets available to compute numeric normalisation.")

    mean, std = _compute_num_norm_from_datasets(train_sets)
    save_json(PATHS.out_root / "num_norm.json", {"mean": mean.tolist(), "std": std.tolist()})
    return mean, std


def compute_stats_norm(stats_df: pd.DataFrame, stat_cols: List[str], train_player_ids: List[Tuple[int, int]]):
    df_tr = stats_df.merge(
        pd.DataFrame(train_player_ids, columns=["player_id", "season_id"]),
        on=["player_id", "season_id"],
        how="inner",
    )
    mu = df_tr[stat_cols].mean(numeric_only=True)
    sigma = df_tr[stat_cols].std(numeric_only=True).replace(0.0, 1.0)
    return mu, sigma


def build_stats_map(ply_train: Optional[Subset], stats_df: pd.DataFrame, stat_cols: List[str]):
    if (ply_train is None) or (len(stat_cols) == 0) or (not PS_ID2KEY):
        return {}, None, None

    hplayers = {int(ply_train.dataset.H_player[idx]) for idx in ply_train.indices}

    train_ps: List[Tuple[int, int]] = []
    for hid in hplayers:
        if hid in PS_ID2KEY:
            train_ps.append(PS_ID2KEY[hid])

    if not train_ps:
        return {}, None, None

    mu, sigma = compute_stats_norm(stats_df, stat_cols, train_ps)
    save_json(PATHS.out_root / "stats_norm.json", {"mu": mu.to_dict(), "sigma": sigma.to_dict()})

    mapping: Dict[int, np.ndarray] = {}
    df = stats_df.copy()
    for col in stat_cols:
        if col in mu.index:
            denom = sigma[col] if (col in sigma.index and sigma[col] != 0) else 1.0
            df[col] = (df[col] - mu[col]) / denom
    grouped = df.set_index(["player_id", "season_id"])
    for hid, (pid, sid) in PS_ID2KEY.items():
        if (pid, sid) in grouped.index:
            vec = grouped.loc[(pid, sid), stat_cols].astype(float).values
            mapping[hid] = vec.astype(np.float32)
    return mapping, mu, sigma
