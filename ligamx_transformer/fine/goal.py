from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ----------------------------
# Paths / Config
# ----------------------------
BASE_DIR  = Path(".")
STEP1_DIR   = BASE_DIR / "artifacts" / "data" 
STEP2_DIR   = BASE_DIR / "artifacts" / "train"
EXPORTS     = BASE_DIR / "exports"

SEQ_DIR     = STEP1_DIR / "sequences"
MAPS_DIR    = STEP1_DIR / "maps"

# Inputs from your exports
POS_WIN_NPZ       = EXPORTS / "pos_token_windows.npz"      
PLY_STYLE_NPY     = EXPORTS / "player_style_embeddings.npy"
PLY_STYLE_TSV     = EXPORTS / "player_style_embeddings.tsv"
PLY_IDENTITY_NPY  = EXPORTS / "player_identity_embeddings.npy"  # Ep table

# Core sequences (labels & shooter)
POS_ALL_NPZ       = SEQ_DIR / "possession_sequences.npz"
POS_TOKENS_PQT    = STEP1_DIR / "pos_tokens.parquet" 

# Output dir
OUT_DIR   = BASE_DIR / "artifacts" / "goal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training knobs
SEED            = 73
BATCH_SIZE      = 512
EPOCHS          = 25
LR              = 5e-4
WD              = 1e-2
DROPOUT         = 0.10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS     = 0  # Windows-safe, bump on Linux
PIN_MEMORY      = True

# Loss weights
LAMBDA_BCE      = 1.0
LAMBDA_BPR      = 0.5
NEG_SWAPS       = 8    # negatives per shot for ranking
POS_WEIGHT_AUTO = True # compute from class balance
POS_WEIGHT_FIX  = 6.0  # used if auto = False

# Shooter vector config
USE_IDENTITY    = True  # concat identity Ep
USE_STYLE       = True  # concat behavioral style z
NORM_STYLE      = True
NORM_IDENTITY   = True

# Model choice: "bilinear" or "film"
MODEL_KIND      = "bilinear"

# Calibration
DO_PLATT_CALIB  = True

# Repro
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Small utils
# ----------------------------
def log(msg: str): print(msg, flush=True)

def load_ps_id2key() -> Dict[int, Tuple[int,int]]:
    """player_season_index (hid) -> (player_id, season_id)"""
    mpath = MAPS_DIR / "player_season_map.json"
    if not mpath.exists(): return {}
    d = json.loads(mpath.read_text(encoding="utf-8"))
    out = {}
    for k, pair in d.get("id2key", {}).items():
        out[int(k)] = (int(pair[0]), int(pair[1]))
    return out

def robust_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.clamp(-20, 20))

# ----------------------------
# Load windows + labels
# ----------------------------
class ShotTable:
    """
    Binds shot-centered hidden vectors, labels, and shooter IDs via (segment_row, center_t).
    """
    # <-- CHANGED: Added pos_tokens_pqt to constructor
    def __init__(self, pos_win_npz: Path, pos_all_npz: Path, pos_tokens_pqt: Path):
        with np.load(pos_win_npz, allow_pickle=False) as w:
            self.h_windows   = w["h_windows"]      # [M, L, d]
            self.masks       = w["masks"].astype(bool)  # [M, L]
            self.segment_row = w["segment_row"]    # [M]
            self.center_t    = w["center_t"]       # [M]
            self.window_radius = int(w["window_radius"])

        # center index in window
        self.center_idx = self.window_radius
        self.M, self.L, self.d = self.h_windows.shape

        # Load labels & shooter from full NPZ
        with np.load(pos_all_npz, allow_pickle=False) as z:
            self.actor_ps = z["actor_ps"].astype(np.int64) if "actor_ps" in z.files else None
            self.mask = z["mask"].astype(bool) if "mask" in z.files else None
            
            # --- START: MODIFIED LOGIC ---
            # Try to load Y_is_goal. If it's not there, rebuild it from the parquet file.
            if "Y_is_goal" in z.files:
                log("[data] Found 'Y_is_goal' in .npz, loading from cache.")
                self.Y_is_goal = z["Y_is_goal"].astype(np.int64)
            else:
                log("[data] 'Y_is_goal' not found in .npz. Rebuilding from 'pos_tokens.parquet'...")
                if not pos_tokens_pqt.exists():
                    raise FileNotFoundError(f"Cannot rebuild goal labels. Missing: {pos_tokens_pqt}")
                if self.mask is None and self.actor_ps is None:
                    raise RuntimeError("Cannot get (N, T) shape from .npz (missing 'mask' or 'actor_ps'). Re-run Step-1.")
                
                # Get shape from mask or actor_ps
                base_arr = self.mask if self.mask is not None else self.actor_ps
                N, T = base_arr.shape
                
                # Default to -1 (not a shot)
                self.Y_is_goal = np.full((N, T), -1, dtype=np.int64) 

                # Load tokens file
                log(f"[data] Loading {pos_tokens_pqt}...")
                df_tokens = pd.read_parquet(pos_tokens_pqt, columns=["segment_row", "t", "action_family", "shot_outcome"])
                
                # Filter for shots
                df_shots = df_tokens[df_tokens["action_family"] == "shot"].copy()
                
                # Create goal flag: 1 if goal, 0 if shot but not goal
                df_shots["is_goal"] = df_shots["shot_outcome"].astype(str).str.lower().str.contains("goal", na=False).astype(int)
                
                # Get indices and values
                rows = df_shots["segment_row"].values
                cols = df_shots["t"].values
                vals = df_shots["is_goal"].values

                # Fill the array, ensuring indices are within bounds
                valid_idx = (rows < N) & (cols < T)
                self.Y_is_goal[rows[valid_idx], cols[valid_idx]] = vals[valid_idx]
                log(f"[data] Rebuilt Y_is_goal: found {len(df_shots)} shots, {vals.sum()} goals.")
            # --- END: MODIFIED LOGIC ---


        # Index validity mask (ensure indices in range & non-pad)
        N, T = self.Y_is_goal.shape
        in_bounds = (self.segment_row >= 0) & (self.segment_row < N) & \
                    (self.center_t    >= 0) & (self.center_t    < T)

        # If mask available, keep only valid tokens
        if self.mask is not None:
            valid_tok = self.mask[self.segment_row, self.center_t]
        else:
            valid_tok = np.ones_like(self.segment_row, dtype=bool)

        self.valid = in_bounds & valid_tok

    def __len__(self): return int(self.valid.sum())

    def get_slice(self, idx_valid: np.ndarray):
        """Return arrays sliced to a valid index subset"""
        return (self.h_windows[idx_valid, self.center_idx, :],
                self.Y_is_goal[self.segment_row[idx_valid], self.center_t[idx_valid]],
                (self.actor_ps[self.segment_row[idx_valid], self.center_t[idx_valid]] if self.actor_ps is not None else None),
                self.segment_row[idx_valid],
                self.center_t[idx_valid])

# ----------------------------
# Shooter embeddings
# ----------------------------
class ShooterEmbBank:
    """Concatenate style & identity embeddings, with mapping from HID -> row index."""
    def __init__(self, style_npy: Path, style_tsv: Path, identity_npy: Path,
                 use_style=True, use_identity=True, norm_style=True, norm_identity=True):
        self.use_style    = use_style
        self.use_identity = use_identity
        self.norm_style   = norm_style
        self.norm_id      = norm_identity

        self.style = None
        self.style_map: Dict[int,int] = {}
        self.identity = None

        if use_style and style_npy.exists() and style_tsv.exists():
            self.style = np.load(style_npy)  # [K, d_style]
            # TSV: "player_season_index ..."; the .npy rows follow sorted HID order used in exporter.
            df = pd.read_csv(style_tsv, sep="\t")
            # The exporter writes: first column is 'player_season_index'
            hid_col = "player_season_index"
            if hid_col not in df.columns:
                # <-- CHANGED: More robust error check
                if df.columns[0] == "player_season_hid":
                    hid_col = "player_season_hid"
                elif "player_season_hid" in df.columns:
                     hid_col = "player_season_hid"
                else:
                    raise RuntimeError(f"{style_tsv} missing '{hid_col}' or 'player_season_hid' column")
            # Build HID -> row
            ids_sorted = df[hid_col].astype(int).tolist()
            self.style_map = {hid: i for i, hid in enumerate(ids_sorted)}

        if use_identity and identity_npy.exists():
            self.identity = np.load(identity_npy)  # [Hmax, d_id]

        # dims
        d = 0
        if self.use_style and self.style is not None:
            d += self.style.shape[1]
        if self.use_identity and self.identity is not None:
            d += self.identity.shape[1]
        self.d_vec = d

    def get(self, hid: int) -> np.ndarray:
        parts = []
        if self.use_style and self.style is not None:
            i = self.style_map.get(int(hid), None)
            if i is None:
                v = np.zeros((self.style.shape[1],), dtype=np.float32)
            else:
                v = self.style[i].astype(np.float32)
            if self.norm_style:
                n = np.linalg.norm(v) + 1e-8
                v = v / n
            parts.append(v)
        if self.use_identity and self.identity is not None:
            if 0 <= hid < self.identity.shape[0]:
                v = self.identity[int(hid)].astype(np.float32)
            else:
                v = np.zeros((self.identity.shape[1],), dtype=np.float32)
            if self.norm_id:
                n = np.linalg.norm(v) + 1e-8
                v = v / n
            parts.append(v)
        if not parts:
            raise RuntimeError("No shooter embeddings available (enable USE_STYLE and/or USE_IDENTITY).")
        return np.concatenate(parts, axis=0)

# ----------------------------
# Dataset
# ----------------------------
class ShotDataset(Dataset):
    """
    Each item = one real shot:
    - h_shot: [d]
    - shooter_hid: int
    - y: {0,1}
    """
    def __init__(self, table: ShotTable, bank: ShooterEmbBank,
                 keep_unknown_shooter=False):
        valid_idx = np.where(table.valid)[0]
        h, y, shooter, seg, tt = table.get_slice(valid_idx)

        # Keep only shot entries that have labels; y should be 0/1 at shots
        y_mask = (y == 0) | (y == 1)
        self.h = h[y_mask].astype(np.float32)                # [M', d]
        self.y = y[y_mask].astype(np.int64)                  # [M']
        if shooter is None:
            if not keep_unknown_shooter:
                raise RuntimeError("actor_ps missing from possession_sequences.npz; required for shooter conditioning.")
            self.shooter = np.full_like(self.y, -1)
        else:
            self.shooter = shooter[y_mask].astype(np.int64)

        self.seg = seg[y_mask]
        self.tt  = tt[y_mask]
        self.bank = bank
        self.d_h = self.h.shape[1]
        self.d_z = bank.d_vec

        # class balance
        self.pos_rate = float(self.y.mean()) if len(self.y) else 0.1

        # Precompute shooter vectors for speed
        self._z_cache: Dict[int, np.ndarray] = {}

        # Build list of unique HIDs we have in labels
        self.unique_hids: np.ndarray = np.unique(self.shooter[self.shooter >= 0])

    def __len__(self): return self.h.shape[0]

    def get_z(self, hid: int) -> np.ndarray:
        if hid not in self._z_cache:
            self._z_cache[hid] = self.bank.get(int(hid))
        return self._z_cache[hid]

    def __getitem__(self, i: int):
        h = torch.from_numpy(self.h[i])
        y = int(self.y[i])
        hid = int(self.shooter[i])
        z = torch.from_numpy(self.get_z(hid))
        return h, z, y, hid

# ----------------------------
# Negatives sampler for ranking
# ----------------------------
def sample_negatives(hid_true: int, all_hids: np.ndarray, k: int) -> np.ndarray:
    if len(all_hids) == 0:
        return np.zeros((0,), dtype=np.int64)
    # simple uniform w/o replacement, excluding true
    pool = all_hids[all_hids != hid_true]
    if len(pool) == 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= len(pool):
        return np.random.choice(pool, size=len(pool), replace=False)
    return np.random.choice(pool, size=k, replace=False)

# ----------------------------
# Models
# ----------------------------
class BilinearScorer(nn.Module):
    def __init__(self, d_h: int, d_z: int, p_drop: float = DROPOUT):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_h, d_z) * (2.0 / (d_h + d_z))**0.5)
        self.b = nn.Parameter(torch.zeros(1))
        self.dp = nn.Dropout(p_drop)
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: [B, d_h], z: [B, d_z]
        h_ = self.dp(h)
        s = (h_ @ self.W * z).sum(dim=-1) + self.b  # [B]
        return s

class FiLMScorer(nn.Module):
    def __init__(self, d_h: int, d_z: int, hidden: int = 256, p_drop: float = DROPOUT):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(d_z, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d_h)  # gamma, beta
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_h, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 1)
        )
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gb = self.cond(z)                  # [B, 2d]
        gamma, beta = gb.chunk(2, dim=-1)      # each [B, d]
        h_ = gamma * h + beta
        return self.mlp(h_).squeeze(-1)            # [B]

def make_model(kind: str, d_h: int, d_z: int) -> nn.Module:
    if kind.lower() == "bilinear":
        return BilinearScorer(d_h, d_z)
    elif kind.lower() == "film":
        return FiLMScorer(d_h, d_z)
    else:
        raise ValueError("MODEL_KIND must be 'bilinear' or 'film'.")

# ----------------------------
# Training & Eval
# ----------------------------
@dataclass
class TrainCfg:
    lr: float = LR
    wd: float = WD
    epochs: int = EPOCHS
    neg_swaps: int = NEG_SWAPS
    lambda_bce: float = LAMBDA_BCE
    lambda_bpr: float = LAMBDA_BPR
    pos_weight: Optional[float] = None

def bpr_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    # s_pos: [B], s_neg: [B, K]
    if s_neg.numel() == 0:
        return torch.tensor(0.0, device=s_pos.device)
    diff = s_pos.unsqueeze(-1) - s_neg  # [B, K]
    return -F.logsigmoid(diff).mean()

def eval_metrics(y_true: np.ndarray, p: np.ndarray, name: str):
    out = {}
    if len(np.unique(y_true)) > 1:
        out["pr_auc"]  = float(average_precision_score(y_true, p))
        out["roc_auc"] = float(roc_auc_score(y_true, p))
    else:
        out["pr_auc"] = out["roc_auc"] = float("nan")
    out["brier"]  = float(brier_score_loss(y_true, p))
    # ECE
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    ece = 0.0; n = len(p)
    for b in range(10):
        mask = (idx == b)
        if mask.sum() == 0: continue
        prob = p[mask].mean()
        acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(prob - acc)
    out["ece"] = float(ece)
    log(f"[{name}] PR-AUC={out['pr_auc']:.4f} | ROC-AUC={out['roc_auc']:.4f} | Brier={out['brier']:.4f} | ECE={out['ece']:.4f}")
    return out

def train_epoch(model, opt, loader, cfg: TrainCfg, all_hids: np.ndarray):
    model.train()
    total = 0.0
    for (h, z, y, hid) in tqdm(loader, desc="train", leave=False):
        h   = h.to(DEVICE, non_blocking=True).float()
        z   = z.to(DEVICE, non_blocking=True).float()
        y   = y.to(DEVICE, non_blocking=True).float()
        hid = hid.cpu().numpy()  # for negative sampling

        # Scores for true shooter
        s_pos = model(h, z)  # [B]

        # BCE
        bce = F.binary_cross_entropy_with_logits(s_pos, y, pos_weight=(torch.tensor(cfg.pos_weight, device=s_pos.device) if cfg.pos_weight else None))

        # Rank negatives (sample in numpy then fetch z_neg)
        K = cfg.neg_swaps
        s_neg_all = []
        if K > 0:
            # Build a big batch of negatives: for each i, draw K hids
            neg_lists = [sample_negatives(int(hid_i), all_hids, K) for hid_i in hid]
            # Flatten and embed
            neg_hids_flat = np.concatenate(neg_lists) if len(neg_lists) else np.array([], dtype=np.int64)
            if neg_hids_flat.size > 0:
                # Reuse z for each neg shooter; pair with same h row-wise
                z_negs = []
                # Map once per unique HID
                unique_neg = np.unique(neg_hids_flat)
                # Build a cache for this minibatch
                z_cache = {}
                for nh in unique_neg:
                    z_cache[int(nh)] = loader.dataset.dataset.bank.get(int(nh))                # Create tensor per row
                for row_i, negs in enumerate(neg_lists):
                    if len(negs) == 0:
                        s_neg_all.append(torch.empty((0,), device=DEVICE))
                        continue
                    z_neg = torch.from_numpy(np.stack([z_cache[int(nh)] for nh in negs], axis=0)).to(DEVICE).float()  # [K, d_z]
                    h_row = h[row_i].unsqueeze(0).expand(z_neg.size(0), -1)  # [K, d_h]
                    s_neg = model(h_row, z_neg)  # [K]
                    s_neg_all.append(s_neg)
                # Stack to [B, K] (pad if some rows had 0 negs)
                maxK = max((len(x) for x in s_neg_all), default=0)
                if maxK > 0:
                    s_neg_mat = torch.full((len(s_neg_all), maxK), -10.0, device=DEVICE)
                    for i_row, svec in enumerate(s_neg_all):
                        if svec.numel() > 0:
                            s_neg_mat[i_row, :svec.numel()] = svec
                else:
                    s_neg_mat = torch.empty((len(s_neg_all), 0), device=DEVICE)
            else:
                s_neg_mat = torch.empty((h.size(0), 0), device=DEVICE)
        else:
            s_neg_mat = torch.empty((h.size(0), 0), device=DEVICE)

        # BPR
        bpr = bpr_loss(s_pos, s_neg_mat)

        loss = cfg.lambda_bce * bce + cfg.lambda_bpr * bpr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def infer_scores(model, loader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    p_hat  = []
    for (h, z, y, _) in tqdm(loader, desc="infer", leave=False):
        h = h.to(DEVICE).float()
        z = z.to(DEVICE).float()
        s = model(h, z)
        p = robust_sigmoid(s).detach().cpu().numpy()
        p_hat.append(p)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(p_hat)

# ----------------------------
# Shooter swap inference
# ----------------------------
@torch.no_grad()
def score_shooter_swap(model, h_shot: np.ndarray, candidate_hids: List[int], bank: ShooterEmbBank, topk: int = 10):
    """
    Given one shot hidden h_shot (d,), returns top-k shooters with highest P(goal).
    """
    h = torch.from_numpy(h_shot).to(DEVICE).float().unsqueeze(0)  # [1,d]
    rows = []
    for hid in candidate_hids:
        z = torch.from_numpy(bank.get(int(hid))).to(DEVICE).float().unsqueeze(0)
        s = model(h, z)
        p = float(robust_sigmoid(s).cpu().item())
        rows.append((int(hid), p))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:topk]

# ----------------------------
# Main
# ----------------------------
def main():
    log(f"[setup] Device: {DEVICE}")
    log(f"[setup] Loading artifacts from: {STEP1_DIR}") # <-- ADDED
    # Load shot table
    # <-- CHANGED: Pass the new POS_TOKENS_PQT path
    table = ShotTable(POS_WIN_NPZ, POS_ALL_NPZ, POS_TOKENS_PQT)
    keep = table.valid
    n_all = keep.sum()
    log(f"[data] shots found in windows: total={table.M}, valid_center={n_all}")

    # Shooter bank
    bank = ShooterEmbBank(
        PLY_STYLE_NPY, PLY_STYLE_TSV, PLY_IDENTITY_NPY,
        use_style=USE_STYLE, use_identity=USE_IDENTITY,
        norm_style=NORM_STYLE, norm_identity=NORM_IDENTITY
    )
    log(f"[emb] shooter vec dim = {bank.d_vec} (style={USE_STYLE}, identity={USE_IDENTITY})")

    # Dataset
    ds = ShotDataset(table, bank)
    log(f"[data] dataset size (shots with labels): {len(ds)} | pos rate ≈ {ds.pos_rate:.3f}")

    # Splits: time-based is ideal; for simplicity random split (set seed). Adjust as needed.
    n_total = len(ds)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    
    # <-- ADDED: Ensure splits are non-negative
    if n_train < 0: n_train = 0
    if n_val < 0: n_val = 0
    if n_test < 0: n_test = 0
    # Adjust in case of rounding
    n_test = n_total - n_train - n_val
    if n_test < 0:
        n_val += n_test # decrease val to make n_test 0
        n_test = 0
        
    if n_total == 0:
        raise RuntimeError("Dataset is empty. Cannot train.")
    if n_train == 0 or (n_val == 0 and n_test == 0):
        log("[warn] Small dataset: using all data for training.")
        n_train = n_total
        n_val = 0
        n_test = 0
        ds_tr = ds
        ds_va = Subset(ds, []) # Empty subset
        ds_te = Subset(ds, []) # Empty subset
    else:
        g = torch.Generator().manual_seed(SEED)
        ds_tr, ds_va, ds_te = random_split(ds, [n_train, n_val, n_test], generator=g)

    log(f"[data] splits: train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}") # <-- ADDED

    # Dataloaders
    def mk_loader(dset, shuffle=False):
        if len(dset) == 0: return None # <-- ADDED: Handle empty dataloaders
        return DataLoader(dset, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    tr_loader = mk_loader(ds_tr, True)
    va_loader = mk_loader(ds_va, False)
    te_loader = mk_loader(ds_te, False)
    
    if tr_loader is None:
        raise RuntimeError("Training loader is empty. Cannot train.")

    # Model
    model = make_model(MODEL_KIND, ds.d_h, ds.d_z).to(DEVICE)
    log(f"[model] {MODEL_KIND} | params={sum(p.numel() for p in model.parameters()):,}")

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Pos weight
    if POS_WEIGHT_AUTO:
        pos_weight = (1.0 - ds.pos_rate) / max(1e-6, ds.pos_rate)
    else:
        pos_weight = POS_WEIGHT_FIX
    cfg = TrainCfg(pos_weight=float(pos_weight))
    log(f"[train] pos_weight={cfg.pos_weight:.3f}   |   BPR K={cfg.neg_swaps}   |   λ_BCE={cfg.lambda_bce} λ_BPR={cfg.lambda_bpr}")

    # Train
    best_val = float("inf"); best_state = None
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, opt, tr_loader, cfg, ds.unique_hids)
        
        # <-- CHANGED: Handle empty validation set
        if va_loader:
            yv, pv = infer_scores(model, va_loader)
            # Use Brier as early-stop metric (well-calibrated focus)
            va_brier = brier_score_loss(yv, pv)
            log(f"[epoch {epoch}] train_loss={tr_loss:.4f} | val_brier={va_brier:.4f}")
            if va_brier < best_val - 1e-5:
                best_val = va_brier
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            log(f"[epoch {epoch}] train_loss={tr_loss:.4f} | val_brier=N/A (no val set)")
            # <-- ADDED: Save last epoch if no validation
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, OUT_DIR / f"goal_scorer_{MODEL_KIND}.pt")
        log(f"[ckpt] saved best scorer → {OUT_DIR / f'goal_scorer_{MODEL_KIND}.pt'}")

    # Eval (pre-calibration)
    # <-- CHANGED: Handle empty loaders
    y_tr, p_tr = infer_scores(model, tr_loader) if tr_loader else (np.array([]), np.array([]))
    y_va, p_va = infer_scores(model, va_loader) if va_loader else (np.array([]), np.array([]))
    y_te, p_te = infer_scores(model, te_loader) if te_loader else (np.array([]), np.array([]))
    
    m_tr = eval_metrics(y_tr, p_tr, "train/raw") if len(y_tr) > 0 else {}
    m_va = eval_metrics(y_va, p_va, "val/raw") if len(y_va) > 0 else {}
    m_te = eval_metrics(y_te, p_te, "test/raw") if len(y_te) > 0 else {}


    # Optional: Platt calibration on val, apply to test
    # <-- CHANGED: Check if calibration is possible
    if DO_PLATT_CALIB and len(y_va) > 0 and len(y_te) > 0:
        log("[calib] Performing Platt calibration on validation set...")
        lr = LogisticRegression(max_iter=500, solver="lbfgs")
        lr.fit(p_va.reshape(-1,1), y_va.astype(int))
        p_te_cal = lr.predict_proba(p_te.reshape(-1,1))[:,1]
        m_te_cal = eval_metrics(y_te, p_te_cal, "test/calibrated")
    else:
        if len(y_va) == 0: log("[calib] Skipping calibration (no validation data).")
        if len(y_te) == 0: log("[calib] Skipping calibration (no test data).")
        p_te_cal = p_te
        m_te_cal = m_te

    # Save artifacts
    np.savez_compressed(
        OUT_DIR / "scores_raw_and_calibrated.npz",
        y_train=y_tr, p_train=p_tr,
        y_val=y_va, p_val=p_va,
        y_test=y_te, p_test_raw=p_te, p_test_cal=p_te_cal
    )
    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "val": m_va, "test_raw": m_te, "test_cal": m_te_cal}, f, indent=2)

    log("[done] training & evaluation complete.")
    log(f" - metrics: {OUT_DIR / 'metrics.json'}")
    log(f" - scores:  {OUT_DIR / 'scores_raw_and_calibrated.npz'}")

    # --------------- Example: shooter swap ---------------
    # Take one validation shot and rank 10 candidate shooters
    if len(ds_va) > 0:
        h_ex, _, _, hid_ex = ds_va[0]  # (h, z, y, hid)
        h_np = h_ex.numpy()
        # Candidate pool: all known HIDs (or restrict to a team/role list if you have those maps)
        cands = ds.unique_hids.tolist()
        if len(cands) > 0:
            top = score_shooter_swap(model, h_np, cands, bank, topk=10)
            # Persist a small table
            top_df = pd.DataFrame(top, columns=["player_season_hid", "p_goal"])
            top_df.to_csv(OUT_DIR / "example_swap_top10.csv", index=False)
            log(f"[swap-demo] wrote {OUT_DIR / 'example_swap_top10.csv'}")
        else:
            log("[swap-demo] Skipping (no unique HIDs found).")

if __name__ == "__main__":
    main()