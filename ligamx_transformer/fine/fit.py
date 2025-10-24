from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

# ----------------------------
# --- Configuration
# ----------------------------

# --- Analysis Targets
# We will find any team names that *contain* these strings (case-insensitive)
TARGET_TEAM_NAMES = ["Pumas", "Necaxa", "Toluca"]

# --- Paths (Must match your goal.py setup)
BASE_DIR = Path(".")
STEP1_DIR = BASE_DIR / "artifacts" / "data"
EXPORTS = BASE_DIR / "exports"

SEQ_DIR = STEP1_DIR / "sequences"
MAPS_DIR = STEP1_DIR / "maps"
STATS_DIR = STEP1_DIR / "stats"

# --- Inputs
# Model
MODEL_KIND = "bilinear" 
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
PLAYER_META_PATH = STATS_DIR / "player_season_meta.parquet"
PLAYER_MAP_PATH = MAPS_DIR / "player_season_map.json"
TEAM_META_PATH = STATS_DIR / "team_season_meta.parquet"
TEAM_MAP_PATH = MAPS_DIR / "team_season_map.json"

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
MIN_SHOTS_FOR_TEAM = 20 # Min test set shots a team needs to be analyzed

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
# --- Data/Model Classes (Modified from goal.py)
# ----------------------------

class ShotTable:
    """
    Modified ShotTable to also load H_team from the .npz file.
    """
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
            
            # --- ADDED: Load H_team ---
            if "H_team" not in z.files:
                raise RuntimeError(f"H_team array not found in {pos_all_npz}. Please re-run Step 1.")
            self.H_team = z["H_team"].astype(np.int64)
            # ---
            
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
        """
        Returns h_window, y_label, shooter_hid, and the segment_row index.
        """
        return (self.h_windows[idx_valid, self.center_idx, :],
                self.Y_is_goal[self.segment_row[idx_valid], self.center_t[idx_valid]],
                (self.actor_ps[self.segment_row[idx_valid], self.center_t[idx_valid]] if self.actor_ps is not None else None),
                self.segment_row[idx_valid]) # <-- CHANGED: No longer need to return segment_row AND center_t


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
    """
    Modified ShotDataset to also store the team_season_id for each shot.
    """
    def __init__(self, table: ShotTable, bank: ShooterEmbBank):
        valid_idx = np.where(table.valid)[0]
        # <-- CHANGED: Unpack segment_row
        h, y, shooter, seg_rows = table.get_slice(valid_idx)
        
        y_mask = (y == 0) | (y == 1)
        self.h = h[y_mask].astype(np.float32)
        self.y = y[y_mask].astype(np.int64)
        if shooter is None: raise RuntimeError("actor_ps missing from possession_sequences.npz.")
        self.shooter = shooter[y_mask].astype(np.int64)
        
        # --- ADDED: Map segment_row to team_tsid ---
        seg_rows_of_shots = seg_rows[y_mask]
        self.team_tsid = table.H_team[seg_rows_of_shots]
        # ---
        
        self.bank = bank
        self.d_h = self.h.shape[1]
        self.d_z = bank.d_vec
        self._z_cache: Dict[int, np.ndarray] = {}
        
        if len(self.h) == 0:
            raise RuntimeError("ShotDataset is empty after filtering.")

    def __len__(self): return self.h.shape[0]

    def get_z(self, hid: int) -> np.ndarray:
        if hid not in self._z_cache:
            self._z_cache[hid] = self.bank.get(int(hid))
        return self._z_cache[hid]

    def __getitem__(self, i: int):
        # This is not used by the analysis, but required by the class
        h = torch.from_numpy(self.h[i])
        y = int(self.y[i])
        hid = int(self.shooter[i])
        z = torch.from_numpy(self.get_z(hid))
        tsid = int(self.team_tsid[i])
        return h, z, y, hid, tsid


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
# --- NEW: Metadata Loaders
# ----------------------------

def load_player_metadata_maps() -> Dict[int, str]:
    """Loads metadata and maps to create a hid -> name lookup."""
    if not PLAYER_META_PATH.exists() or not PLAYER_MAP_PATH.exists():
        log(f"[warn] Missing player metadata. Player names will be 'Unknown'.")
        return {}

    d = json.loads(PLAYER_MAP_PATH.read_text(encoding="utf-8"))
    id2key = {int(k): (int(pair[0]), int(pair[1])) for k, pair in d.get("id2key", {}).items()}
    
    df_meta = pd.read_parquet(PLAYER_META_PATH, columns=["player_id", "season_id", "player_name"])
    df_meta = df_meta.dropna(subset=["player_id", "season_id"])
    df_meta["player_id"] = df_meta["player_id"].astype(int)
    df_meta["season_id"] = df_meta["season_id"].astype(int)
    
    pid_sid_to_name = df_meta.set_index(["player_id", "season_id"])["player_name"].to_dict()
    
    hid_to_name = {hid: pid_sid_to_name.get(pid_sid, "Unknown") for hid, pid_sid in id2key.items()}
        
    log(f"[info] Loaded {len(hid_to_name)} player names from metadata.")
    return hid_to_name

def load_team_metadata_maps() -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """Loads metadata and maps to create tsid -> name and name -> list[tsid] lookups."""
    if not TEAM_META_PATH.exists() or not TEAM_MAP_PATH.exists():
        log(f"[warn] Missing team metadata. Team names will be 'Unknown'.")
        return {}, {}
        
    d = json.loads(TEAM_MAP_PATH.read_text(encoding="utf-8"))
    id2key = {int(k): (int(pair[0]), int(pair[1])) for k, pair in d.get("id2key", {}).items()}
    
    df_meta = pd.read_parquet(TEAM_META_PATH, columns=["team_id", "season_id", "team_name"])
    df_meta = df_meta.dropna(subset=["team_id", "season_id"])
    df_meta["team_id"] = df_meta["team_id"].astype(int)
    df_meta["season_id"] = df_meta["season_id"].astype(int)
    
    pid_sid_to_name = df_meta.set_index(["team_id", "season_id"])["team_name"].to_dict()
    
    tsid_to_name: Dict[int, str] = {}
    name_to_tsids: Dict[str, List[int]] = {}
    
    for tsid, pid_sid in id2key.items():
        name = pid_sid_to_name.get(pid_sid, f"Unknown (TSID: {tsid})")
        tsid_to_name[tsid] = name
        name_to_tsids.setdefault(name, []).append(tsid)

    log(f"[info] Loaded {len(tsid_to_name)} team names from metadata.")
    return tsid_to_name, name_to_tsids


# ----------------------------
# --- Main Analysis
# ----------------------------
@torch.no_grad()
def main():
    log(f"[setup] Device: {DEVICE}")
    log(f"[setup] Loading trained model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        log(f"[error] Model file not found: {MODEL_PATH}")
        log("Please run the training script (goal.py) first.")
        return

    # 1. Load metadata maps
    hid_to_name_map = load_player_metadata_maps()
    tsid_to_name_map, name_to_tsids_map = load_team_metadata_maps()

    # 2. Resolve Target Teams
    target_team_map: Dict[str, Set[int]] = {}
    all_known_team_names = name_to_tsids_map.keys()
    
    log("[info] Resolving target team names...")
    for target_name in TARGET_TEAM_NAMES:
        found_names = set()
        for known_name in all_known_team_names:
            if target_name.lower() in known_name.lower():
                found_names.add(known_name)
        
        if not found_names:
            log(f"[warn] No team name matching '{target_name}' found in metadata.")
            continue
        
        # Collect all season IDs for all matching names
        full_name = " / ".join(sorted(list(found_names)))
        all_tsids: Set[int] = set()
        for name in found_names:
            all_tsids.update(name_to_tsids_map[name])
            
        log(f"  > Found '{target_name}': Mapped to {full_name} (TSIDs: {all_tsids})")
        target_team_map[full_name] = all_tsids

    if not target_team_map:
        log("[error] No target teams were found. Exiting.")
        return

    # 3. Load data (tables, banks, dataset)
    log("[data] Loading shot table and embeddings...")
    table = ShotTable(POS_WIN_NPZ, POS_ALL_NPZ, POS_TOKENS_PQT)
    bank = ShooterEmbBank(PLY_STYLE_NPY, PLY_STYLE_TSV, PLY_IDENTITY_NPY,
                          use_style=USE_STYLE, use_identity=USE_IDENTITY,
                          norm_style=NORM_STYLE, norm_identity=NORM_IDENTITY)
    ds = ShotDataset(table, bank)

    # 4. Re-create the test split *exactly*
    log(f"[data] Re-creating test split with SEED={SEED}...")
    n_total = len(ds)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    if n_train < 0: n_train = 0
    if n_val < 0: n_val = 0
    if n_test < 0: n_test = 0
    n_test = n_total - n_train - n_val
    if n_test < 0: n_val += n_test; n_test = 0
        
    g = torch.Generator().manual_seed(SEED)
    _, _, ds_te = random_split(ds, [n_train, n_val, n_test], generator=g)

    log(f"[data] Full dataset size: {n_total}, Test set size: {len(ds_te)} shots")
    if len(ds_te) == 0:
        log("[error] Test set is empty. Cannot perform analysis.")
        return

    # 5. Load trained model
    log(f"[model] Loading {MODEL_KIND} model...")
    model = make_model(MODEL_KIND, ds.d_h, ds.d_z).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 6. Get all player embeddings (the "Shooter Bank")
    log("[analysis] Caching all player embeddings...")
    all_hids = sorted([hid for hid in hid_to_name_map.keys() if hid >= 0])
    all_z_vectors = torch.stack(
        [torch.from_numpy(bank.get(hid)) for hid in all_hids]
    ).to(DEVICE).float()
    # all_z_vectors shape: [Num_Players, d_z]
    
    # 7. Build DataFrame of test set shots to analyze
    log("[analysis] Filtering test set for target teams...")
    test_indices = ds_te.indices
    df_test_shots = pd.DataFrame({
        'h_index': test_indices, # Index into the full ds.h array
        'tsid': ds.team_tsid[test_indices]
    })
    
    # 8. Run Simulation
    log(f"[analysis] Running 'what-if' simulation for {len(all_hids)} players...")
    
    # This will be our final results table
    df_results = pd.DataFrame({
        'player_hid': all_hids,
        'player_name': [hid_to_name_map.get(hid, "Unknown") for hid in all_hids]
    }).set_index('player_hid')

    for team_name, team_tsids in target_team_map.items():
        
        # Find all test set shots for this team
        team_shot_rows = df_test_shots[df_test_shots['tsid'].isin(team_tsids)]
        num_team_shots = len(team_shot_rows)
        
        if num_team_shots < MIN_SHOTS_FOR_TEAM:
            log(f"\n[warn] Skipping '{team_name}': Only found {num_team_shots} shots in test set (min is {MIN_SHOTS_FOR_TEAM}).")
            df_results[team_name] = np.nan
            continue
            
        log(f"\n[analysis] Analyzing '{team_name}' ({num_team_shots} shots)...")
        
        # Get all h_vectors for this team's shots
        h_indices = team_shot_rows['h_index'].values
        h_vectors = torch.from_numpy(ds.h[h_indices]).to(DEVICE).float()
        # h_vectors shape: [N_Shots, d_h]
        
        N_shots, d_h = h_vectors.shape
        N_players, d_z = all_z_vectors.shape
        
        # --- The "Virtual Substitution" ---
        # We create a giant grid of all shots vs. all players
        
        # 1. Expand h_vectors: [N_Shots, 1, d_h] -> [N_Shots, N_Players, d_h]
        h_expanded = h_vectors.unsqueeze(1).expand(N_shots, N_players, d_h)
        
        # 2. Expand z_vectors: [N_Players, d_z] -> [1, N_Players, d_z] -> [N_Shots, N_Players, d_z]
        z_expanded = all_z_vectors.unsqueeze(0).expand(N_shots, N_players, d_z)
        
        # 3. Flatten for model: [N_Shots * N_Players, d_h/d_z]
        # We process in chunks to avoid OOM errors
        all_probs = []
        for h_chunk, z_chunk in zip(h_expanded.split(BATCH_SIZE), z_expanded.split(BATCH_SIZE)):
            # h_chunk shape: [BATCH_SIZE, N_Players, d_h]
            h_flat = h_chunk.reshape(-1, d_h)
            z_flat = z_chunk.reshape(-1, d_z)
            
            s_flat = model(h_flat, z_flat)
            p_flat = robust_sigmoid(s_flat)
            
            # Reshape back to [BATCH_SIZE, N_Players]
            p_chunk = p_flat.reshape(h_chunk.shape[0], N_players)
            all_probs.append(p_chunk)
            
        # 4. Concatenate all chunks: [N_Shots, N_Players]
        probs = torch.cat(all_probs, dim=0)
        
        # 5. Get the average P(Goal) for each player, *on this team's shots*
        # We average over the shots (dim=0)
        avg_p_goal_per_player = probs.mean(dim=0).cpu().numpy()
        
        # Add to our results table
        df_results[team_name] = avg_p_goal_per_player

    # 9. Save and display results
    output_csv = "team_fit_analysis.csv"
    df_results.to_csv(output_csv, float_format="%.5f")
    log("\n" + "---" * 15)
    log(f"ANALYSIS COMPLETE. Full results saved to: {output_csv}")
    log("---" * 15)

    for team_name in df_results.columns.drop('player_name'):
        print("\n")
        log(f"Top 20 'Best Fit' Players for: {team_name}")
        log("Sorted by avg. P(Goal) if they took all this team's shots")
        log("---" * 10)
        
        # Sort by this team's column and show top 20
        sorted_df = df_results[['player_name', team_name]].sort_values(by=team_name, ascending=False)
        print(sorted_df.head(20).to_string(index=True))
        log("---" * 10)

if __name__ == "__main__":
    main()