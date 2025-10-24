from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
# ----------------------------
# --- Configuration
# ----------------------------
# --- Paths (Must match your goal.py setup)
BASE_DIR = Path(".")
STEP1_DIR = BASE_DIR / "artifacts" / "data"
EXPORTS = BASE_DIR / "exports"

SEQ_DIR = STEP1_DIR / "sequences"
MAPS_DIR = STEP1_DIR / "maps"
STATS_DIR = STEP1_DIR / "stats" # <-- ADDED

# --- Inputs
# Model
MODEL_KIND = "bilinear" # <-- !! Make sure this matches the MODEL_KIND you trained
MODEL_PATH = BASE_DIR / "artifacts" / "goal" / f"goal_scorer_{MODEL_KIND}.pt"

# Data
POS_WIN_NPZ = EXPORTS / "pos_token_windows.npz"
POS_ALL_NPZ = SEQ_DIR / "possession_sequences.npz"
POS_TOKENS_PQT = STEP1_DIR / "pos_tokens.parquet"

# Embeddings
PLY_STYLE_NPY = EXPORTS / "player_style_embeddings.npy"
PLY_STYLE_TSV = EXPORTS / "player_style_embeddings.tsv"
PLY_IDENTITY_NPY = EXPORTS / "player_identity_embeddings.npy"

# Metadata
PLAYER_META_PATH = STATS_DIR / "player_season_meta.parquet" # <-- ADDED
PLAYER_MAP_PATH = MAPS_DIR / "player_season_map.json"      # <-- ADDED

# --- Config (Must match your goal.py setup)
SEED = 73
BATCH_SIZE = 1024 # Can be larger for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True

# Shooter vector config
USE_IDENTITY = True
USE_STYLE = True
NORM_STYLE = True
NORM_IDENTITY = True

# --- Analysis Config
MIN_SHOTS_IN_TEST_SET = 10 # Min shots for a player to be in the ranking

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# --- Utility Functions (Copied from goal.py)
# ----------------------------

def log(msg: str): print(msg, flush=True)

def robust_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.clamp(-20, 20))

# ----------------------------
# --- Data/Model Classes (Copied from goal.py)
# ----------------------------

class ShotTable:
    def __init__(self, pos_win_npz: Path, pos_all_npz: Path, pos_tokens_pqt: Path):
        with np.load(pos_win_npz, allow_pickle=False) as w:
            self.h_windows   = w["h_windows"]
            self.masks       = w["masks"].astype(bool)
            self.segment_row = w["segment_row"]
            self.center_t    = w["center_t"]
            self.window_radius = int(w["window_radius"])

        self.center_idx = self.window_radius
        self.M, self.L, self.d = self.h_windows.shape

        with np.load(pos_all_npz, allow_pickle=False) as z:
            self.actor_ps = z["actor_ps"].astype(np.int64) if "actor_ps" in z.files else None
            self.mask = z["mask"].astype(bool) if "mask" in z.files else None
            
            if "Y_is_goal" in z.files:
                self.Y_is_goal = z["Y_is_goal"].astype(np.int64)
            else:
                log("[info] 'Y_is_goal' not found, rebuilding...")
                if not pos_tokens_pqt.exists():
                    raise FileNotFoundError(f"Cannot rebuild goal labels. Missing: {pos_tokens_pqt}")
                base_arr = self.mask if self.mask is not None else self.actor_ps
                N, T = base_arr.shape
                self.Y_is_goal = np.full((N, T), -1, dtype=np.int64) 
                df_tokens = pd.read_parquet(pos_tokens_pqt, columns=["segment_row", "t", "action_family", "shot_outcome"])
                df_shots = df_tokens[df_tokens["action_family"] == "shot"].copy()
                df_shots["is_goal"] = df_shots["shot_outcome"].astype(str).str.lower().str.contains("goal", na=False).astype(int)
                rows, cols, vals = df_shots["segment_row"].values, df_shots["t"].values, df_shots["is_goal"].values
                valid_idx = (rows < N) & (cols < T)
                self.Y_is_goal[rows[valid_idx], cols[valid_idx]] = vals[valid_idx]

        N, T = self.Y_is_goal.shape
        in_bounds = (self.segment_row >= 0) & (self.segment_row < N) & \
                    (self.center_t    >= 0) & (self.center_t    < T)
        if self.mask is not None:
            valid_tok = self.mask[self.segment_row, self.center_t]
        else:
            valid_tok = np.ones_like(self.segment_row, dtype=bool)
        self.valid = in_bounds & valid_tok

    def get_slice(self, idx_valid: np.ndarray):
        return (self.h_windows[idx_valid, self.center_idx, :],
                self.Y_is_goal[self.segment_row[idx_valid], self.center_t[idx_valid]],
                (self.actor_ps[self.segment_row[idx_valid], self.center_t[idx_valid]] if self.actor_ps is not None else None),
                self.segment_row[idx_valid],
                self.center_t[idx_valid])

class ShooterEmbBank:
    def __init__(self, style_npy: Path, style_tsv: Path, identity_npy: Path,
                 use_style=True, use_identity=True, norm_style=True, norm_identity=True):
        self.use_style, self.use_identity = use_style, use_identity
        self.norm_style, self.norm_id = norm_style, norm_identity
        self.style, self.identity = None, None
        self.style_map: Dict[int,int] = {}

        if use_style and style_npy.exists() and style_tsv.exists():
            self.style = np.load(style_npy)
            df = pd.read_csv(style_tsv, sep="\t")
            hid_col = "player_season_index"
            if hid_col not in df.columns:
                hid_col = "player_season_hid" if "player_season_hid" in df.columns else df.columns[0]
            ids_sorted = df[hid_col].astype(int).tolist()
            self.style_map = {hid: i for i, hid in enumerate(ids_sorted)}

        if use_identity and identity_npy.exists():
            self.identity = np.load(identity_npy)

        d = 0
        if self.use_style and self.style is not None: d += self.style.shape[1]
        if self.use_identity and self.identity is not None: d += self.identity.shape[1]
        self.d_vec = d

    def get(self, hid: int) -> np.ndarray:
        parts = []
        if self.use_style and self.style is not None:
            i = self.style_map.get(int(hid), None)
            v = np.zeros((self.style.shape[1],), dtype=np.float32) if i is None else self.style[i].astype(np.float32)
            if self.norm_style: v = v / (np.linalg.norm(v) + 1e-8)
            parts.append(v)
        if self.use_identity and self.identity is not None:
            v = self.identity[int(hid)].astype(np.float32) if 0 <= hid < self.identity.shape[0] else np.zeros((self.identity.shape[1],), dtype=np.float32)
            if self.norm_id: v = v / (np.linalg.norm(v) + 1e-8)
            parts.append(v)
        if not parts: raise RuntimeError("No shooter embeddings available.")
        return np.concatenate(parts, axis=0)

class ShotDataset(torch.utils.data.Dataset):
    def __init__(self, table: ShotTable, bank: ShooterEmbBank):
        valid_idx = np.where(table.valid)[0]
        h, y, shooter, _, _ = table.get_slice(valid_idx)
        y_mask = (y == 0) | (y == 1)
        self.h = h[y_mask].astype(np.float32)
        self.y = y[y_mask].astype(np.int64)
        if shooter is None: raise RuntimeError("actor_ps missing from possession_sequences.npz.")
        self.shooter = shooter[y_mask].astype(np.int64)
        self.bank = bank
        self.d_h = self.h.shape[1]
        self.d_z = bank.d_vec
        self._z_cache: Dict[int, np.ndarray] = {}

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

class BilinearScorer(nn.Module):
    def __init__(self, d_h: int, d_z: int, p_drop: float = 0.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_h, d_z) * (2.0 / (d_h + d_z))**0.5)
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = (h @ self.W * z).sum(dim=-1) + self.b
        return s

class FiLMScorer(nn.Module):
    def __init__(self, d_h: int, d_z: int, hidden: int = 256, p_drop: float = 0.0):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(d_z, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_h))
        self.mlp = nn.Sequential(nn.Linear(d_h, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gb = self.cond(z)
        gamma, beta = gb.chunk(2, dim=-1)
        h_ = gamma * h + beta
        return self.mlp(h_).squeeze(-1)

def make_model(kind: str, d_h: int, d_z: int) -> nn.Module:
    if kind.lower() == "bilinear":
        return BilinearScorer(d_h, d_z)
    elif kind.lower() == "film":
        return FiLMScorer(d_h, d_z)
    else:
        raise ValueError("MODEL_KIND must be 'bilinear' or 'film'.")

# ----------------------------
# --- NEW: Analysis Functions
# ----------------------------

@torch.no_grad()
def infer_with_hids(model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference and returns labels, probs, and HIDs."""
    model.eval()
    y_true, p_hat, hids = [], [], []
    
    for (h, z, y, hid) in tqdm(loader, desc="Inferring"):
        h = h.to(DEVICE).float()
        z = z.to(DEVICE).float()
        s = model(h, z)
        
        p_hat.append(robust_sigmoid(s).detach().cpu().numpy())
        y_true.append(y.numpy())
        hids.append(hid.numpy())
        
    return np.concatenate(y_true), np.concatenate(p_hat), np.concatenate(hids)

def load_metadata_maps() -> Dict[int, str]:
    """Loads metadata and maps to create a hid -> name lookup."""
    if not PLAYER_META_PATH.exists():
        log(f"[warn] Missing {PLAYER_META_PATH}. Player names will be 'Unknown'.")
        return {}
    if not PLAYER_MAP_PATH.exists():
        log(f"[warn] Missing {PLAYER_MAP_PATH}. Cannot map HIDs.")
        return {}

    # Load HID -> (PID, SID)
    d = json.loads(PLAYER_MAP_PATH.read_text(encoding="utf-8"))
    id2key = {}
    for k, pair in d.get("id2key", {}).items():
        id2key[int(k)] = (int(pair[0]), int(pair[1]))
    
    # Load (PID, SID) -> Name
    df_meta = pd.read_parquet(PLAYER_META_PATH, columns=["player_id", "season_id", "player_name"])
    df_meta = df_meta.dropna(subset=["player_id", "season_id"])
    df_meta["player_id"] = df_meta["player_id"].astype(int)
    df_meta["season_id"] = df_meta["season_id"].astype(int)
    
    # Create (PID, SID) -> Name map
    pid_sid_to_name = df_meta.set_index(["player_id", "season_id"])["player_name"].to_dict()
    
    # Combine them: HID -> Name
    hid_to_name = {}
    for hid, (pid, sid) in id2key.items():
        hid_to_name[hid] = pid_sid_to_name.get((pid, sid), "Unknown")
        
    log(f"[info] Loaded {len(hid_to_name)} player names from metadata.")
    return hid_to_name

# ----------------------------
# --- Main Analysis
# ----------------------------

def main():
    log(f"[setup] Device: {DEVICE}")
    log(f"[setup] Loading trained model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        log(f"[error] Model file not found: {MODEL_PATH}")
        log("Please run the training script (goal.py) first.")
        return

    # 1. Load metadata
    hid_to_name_map = load_metadata_maps()

    # 2. Load data (tables, banks, dataset)
    log("[data] Loading shot table and embeddings...")
    table = ShotTable(POS_WIN_NPZ, POS_ALL_NPZ, POS_TOKENS_PQT)
    bank = ShooterEmbBank(PLY_STYLE_NPY, PLY_STYLE_TSV, PLY_IDENTITY_NPY,
                          use_style=USE_STYLE, use_identity=USE_IDENTITY,
                          norm_style=NORM_STYLE, norm_identity=NORM_IDENTITY)
    ds = ShotDataset(table, bank)

    # 3. Re-create the test split *exactly*
    log(f"[data] Re-creating test split with SEED={SEED}...")
    n_total = len(ds)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    if n_train < 0: n_train = 0
    if n_val < 0: n_val = 0
    if n_test < 0: n_test = 0
    n_test = n_total - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
        
    g = torch.Generator().manual_seed(SEED)
    _, _, ds_te = random_split(ds, [n_train, n_val, n_test], generator=g)

    log(f"[data] Test set size: {len(ds_te)} shots")
    if len(ds_te) == 0:
        log("[error] Test set is empty. Cannot perform analysis.")
        return

    te_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # 4. Load trained model
    log(f"[model] Loading {MODEL_KIND} model...")
    model = make_model(MODEL_KIND, ds.d_h, ds.d_z).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # 5. Run inference
    log("[model] Running inference on test set...")
    y_true, p_hat, hids = infer_with_hids(model, te_loader)
    
    # 6. Aggregate results
    log("[analysis] Aggregating results by player...")
    df = pd.DataFrame({
        'hid': hids,
        'p_goal': p_hat,
        'is_goal': y_true
    })

    # Group by player HID and get mean P(Goal) and number of shots
    df_agg = df.groupby('hid')['p_goal'].agg(
        p_goal_avg='mean',
        num_shots='count'
    ).reset_index()

    # 7. Filter, map names, and sort
    df_agg_filtered = df_agg[df_agg['num_shots'] >= MIN_SHOTS_IN_TEST_SET].copy()
    
    # Map HIDs to names
    df_agg_filtered['player_name'] = df_agg_filtered['hid'].map(hid_to_name_map).fillna('Unknown')
    
    # Sort by average P(Goal)
    df_final = df_agg_filtered.sort_values(by='p_goal_avg', ascending=False)
    
    # Re-order columns for clarity
    df_final = df_final[['player_name', 'p_goal_avg', 'num_shots', 'hid']]

    # 8. Save and display results
    output_csv = "top_scorers_by_model.csv"
    df_final.to_csv(output_csv, index=False, float_format="%.4f")
    
    log("---" * 10)
    log(f"Results saved to {output_csv}")
    log(f"Top 20 Players by Avg. Model-Predicted P(Goal) (min {MIN_SHOTS_IN_TEST_SET} test set shots)")
    log("---" * 10)
    print(df_final.head(20).to_string(index=False))
    log("---" * 10)

if __name__ == "__main__":
    main()