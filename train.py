# step2_train.py
from __future__ import annotations
from torch.utils.data._utils.collate import default_collate

import json
import math
import random
from dataclasses import dataclass

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

# ==============================
# Configuration
# ==============================
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEP1_DIR   = Path("./artifacts/data")
OUT_DIR     = Path("./artifacts/train")
EXPORT_DIR  = Path("./exports") 

VOCAB_DIR = STEP1_DIR / "vocabs"
SEQ_DIR   = STEP1_DIR / "sequences"
STATS_DIR = STEP1_DIR / "stats"
MAPS_DIR  = STEP1_DIR / "maps"

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL       = 384
N_LAYERS      = 12
N_HEADS       = 12
FFN_HIDDEN    = 4 * D_MODEL
DROPOUT       = 0.10

# Token encoder subspaces
D_CAT         = 32
D_NUM         = 64
D_QUAL        = 24

# Header embeddings
D_ID          = 96
D_POSHINT     = 32
D_SEQTYPE     = 8

# Aux tasks
MASK_PLAYER_FRAC = 0.30  # fraction of player batches masked for masked-ID task

# Loss weights
LAMBDA_EVENT   = 1.0
LAMBDA_STATS   = 0.30
LAMBDA_MASKP   = 0.30
LAMBDA_TEAMGRL = 0.05   # GRL uses lambda=1.0 internally; this scales the loss contribution

# Optim
LR            = 2e-4
WEIGHT_DECAY  = 1e-2
WARMUP_FRAC   = 0.06
CLIP_NORM     = 0.5

LABEL_SMOOTH  = 0.05
EPOCHS        = 15
BATCH_POS     = 96
BATCH_PLY     = 96
EARLY_STOP_PATIENCE = 8
SEED          = 73

COLLATE_INCLUDE_ACTOR_PS = False  # True in probes; False for training

# ----- Exports / diagnostics knobs -----
EXPORT_TOKEN_WINDOWS = True   
SHOT_WINDOW_RADIUS   = 4       # window half-size around each shot (total length = 2R+1)

# ----- DataLoader workers (Windows-safe) -----
IS_WINDOWS = (os.name == "nt")
NUM_WORKERS = 0
PIN_MEMORY  = True
PERSISTENT_WORKERS = (NUM_WORKERS > 0) and (not IS_WINDOWS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==============================
# Speed knobs / CUDA features
# ==============================
from torch.cuda.amp import autocast, GradScaler

# TF32 matmul (safe on Ampere+)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

# Matmul precision (torch>=2.0)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Flash / Efficient SDPA (torch>=2.0)
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(True)
    sdp_kernel.enable_mem_efficient_sdp(True)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass

def _gpu_cc():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None

AMP_DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and _gpu_cc() and _gpu_cc()[0] >= 8)
    else torch.float16
)

# ==============================
# Utilities
# ==============================
def log(msg: str):
    print(msg, flush=True)

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def split_indices(n: int, train=0.8, val=0.1, test=0.1, seed=SEED) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_tr = int(round(n * train))
    n_va = int(round(n * val))
    tr = idx[:n_tr]
    va = idx[n_tr:n_tr + n_va]
    te = idx[n_tr + n_va:]
    return tr, va, te

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out

# Prefetcher to overlap H2D with compute
class CUDAPrefetcher:
    """
    Overlaps CPU->GPU copy with compute using a dedicated CUDA stream.
    Works even with num_workers=0. Expects move_batch_to_device() to use non_blocking=True.
    """
    def __init__(self, loader: Optional[DataLoader], device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if (device.type == "cuda" and loader is not None) else None
        self.it = None
        self.next_batch = None

    def reset(self):
        if self.loader is None:
            self.it, self.next_batch = None, None
            return
        self.it = iter(self.loader)
        self._preload()

    def _preload(self):
        if self.loader is None:
            self.next_batch = None
            return
        try:
            nxt = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = nxt
        else:
            with torch.cuda.stream(self.stream):
                self.next_batch = move_batch_to_device(nxt, self.device)

    def next(self):
        if self.loader is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            # restart iterator
            self.reset()
            batch = self.next_batch
            if batch is None:
                return None
        self._preload()
        return batch

# ==============================
# Load vocabs and constants (from Step-1)
# ==============================
# Discover categorical factor files (excluding qualifiers/num_features)
CAT_FACTORS: List[str] = []
for p in VOCAB_DIR.glob("*.json"):
    name = p.stem
    if name in ("qualifiers", "num_features"):
        continue
    CAT_FACTORS.append(name)
CAT_FACTORS = sorted(CAT_FACTORS)  # deterministic

QUALIFIERS = load_json(VOCAB_DIR / "qualifiers.json")["qualifiers"]
NUM_FEATURES = load_json(VOCAB_DIR / "num_features.json")["num_features"]

# Load per-factor vocab sizes and action_family size for head
VOCABS: Dict[str, dict] = {f: load_json(VOCAB_DIR / f"{f}.json") for f in CAT_FACTORS}
VOCAB_SIZES: Dict[str, int] = {f: len(VOCABS[f]["token2id"]) for f in CAT_FACTORS}
ACTION_FAMILY_VSZ = VOCAB_SIZES.get("action_family", None)
if ACTION_FAMILY_VSZ is None:
    raise RuntimeError("action_family vocab not found; required for next-event head.")

# Load maps for player/team season
PS_MAP = load_json(MAPS_DIR / "player_season_map.json") if (MAPS_DIR / "player_season_map.json").exists() else None
TS_MAP = load_json(MAPS_DIR / "team_season_map.json") if (MAPS_DIR / "team_season_map.json").exists() else None

# Reverse map for id2key (player_id, season_id)
PS_ID2KEY: Dict[int, Tuple[int,int]] = {}
if PS_MAP:
    for i, pair in PS_MAP["id2key"].items():
        PS_ID2KEY[int(i)] = (int(pair[0]), int(pair[1]))

# ==============================
# Datasets (Windows-safe)
# ==============================
class BaseSeqDataset(Dataset):
    def __init__(self, npz_path: Path, kind: str):
        """
        kind: "possession" or "player"
        Loads all arrays eagerly and closes the NPZ file immediately.
        """
        super().__init__()
        self.kind = kind
        self._npz_path = npz_path  # keep path for export-time header reuse if needed

        with np.load(npz_path, allow_pickle=False) as z:
            # Validate presence of per-factor arrays
            for f in CAT_FACTORS:
                if f not in z.files:
                    raise RuntimeError(f"Missing factor array '{f}' in {npz_path.name}")

            # headers
            self.H_team    = z["H_team"].astype(np.int64)
            self.H_season  = z["H_season"].astype(np.int64)
            self.H_seqtype = z["H_seqtype"].astype(np.int64)
            self.H_poshint = z["H_poshint"].astype(np.int64) if "H_poshint" in z.files else None
            self.H_player  = z["H_player"].astype(np.int64)  if "H_player"  in z.files else None
            # << new: stable match/possession keys for POS dataset
            self.H_match   = z["H_match"].astype(np.int64)   if "H_match"   in z.files else None
            self.H_pos_idx = z["H_pos_idx"].astype(np.int64) if "H_pos_idx" in z.files else None
            self.has_player = self.H_player is not None

            # token-level
            mask = z["mask"]
            self.mask   = mask.astype(np.bool_)
            self.X_num  = z["X_num"].astype(np.float32)
            self.X_qual = z["X_qual"].astype(np.float32)
            self.Y_next = z["Y_next_action_family"].astype(np.int64)

            # cats per factor
            self.cats: Dict[str, np.ndarray] = {f: z[f].astype(np.int64) for f in CAT_FACTORS}

            # optional actor_ps
            self.has_actor_ps = ("actor_ps" in z.files)
            if self.has_actor_ps:
                self.actor_ps = z["actor_ps"].astype(np.int64)

        self.N = self.mask.shape[0]
        self.T = self.mask.shape[1]

    def __len__(self): return self.N

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = {
            "mask":   torch.from_numpy(self.mask[i]),
            "nums":   torch.from_numpy(self.X_num[i]),
            "quals":  torch.from_numpy(self.X_qual[i]),
            "cats":   {f: torch.from_numpy(self.cats[f][i]) for f in CAT_FACTORS},
            "Y":      torch.from_numpy(self.Y_next[i]),
            "H_team": int(self.H_team[i]),
            "H_season": int(self.H_season[i]),
            "H_seqtype": int(self.H_seqtype[i]),
        }
        item["H_poshint"] = int(self.H_poshint[i]) if self.H_poshint is not None else -1
        item["H_player"]  = int(self.H_player[i])  if self.H_player  is not None else -1
        if getattr(self, "has_actor_ps", False):
            item["actor_ps"] = torch.from_numpy(self.actor_ps[i])
        return item


class PossessionSeqDataset(BaseSeqDataset):
    def __init__(self, npz_path: Path):
        super().__init__(npz_path, kind="possession")

class PlayerSeqDataset(BaseSeqDataset):
    """
    CLEAN version: does NOT store a DataFrame.
    Only holds stat column names and a lightweight mapping H_player -> np.ndarray.
    """
    def __init__(self, npz_path: Path, stat_cols: List[str]):
        super().__init__(npz_path, kind="player")
        self.stat_cols = list(stat_cols)
        self.player_stats_map: Dict[int, np.ndarray] = {}

    def set_stats_map(self, mapping: Dict[int, np.ndarray]):
        self.player_stats_map = mapping

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = {
            "mask":  torch.from_numpy(self.mask[i]),
            "nums":  torch.from_numpy(self.X_num[i]),
            "quals": torch.from_numpy(self.X_qual[i]),
            "cats":  {f: torch.from_numpy(self.cats[f][i]) for f in CAT_FACTORS},
            "Y":     torch.from_numpy(self.Y_next[i]),
            "H_team":    torch.tensor(self.H_team[i],    dtype=torch.long),
            "H_season":  torch.tensor(self.H_season[i],  dtype=torch.long),
            "H_seqtype": torch.tensor(self.H_seqtype[i], dtype=torch.long),
            "H_poshint": torch.tensor(self.H_poshint[i] if self.H_poshint is not None else -1, dtype=torch.long),
            "H_player":  torch.tensor(self.H_player[i]  if self.H_player  is not None else -1, dtype=torch.long),
        }
        if getattr(self, "has_actor_ps", False):
            item["actor_ps"] = torch.from_numpy(self.actor_ps[i])  # [T]

        hid = int(item["H_player"].item()) if isinstance(item["H_player"], torch.Tensor) else int(item["H_player"])
        if hid in self.player_stats_map:
            vec = self.player_stats_map[hid]
            item["stats_vec"]  = torch.from_numpy(vec).float()
            item["stats_mask"] = torch.isfinite(item["stats_vec"])
        else:
            vec = torch.full((len(self.stat_cols),), float("nan"), dtype=torch.float32)
            item["stats_vec"]  = vec
            item["stats_mask"] = torch.zeros_like(vec, dtype=torch.bool)
        return item


# ==============================
# Normalization helpers
# ==============================
def _compute_num_norm_from_datasets(train_datasets: List[BaseSeqDataset]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std for numeric features across train sets (masked tokens only).
    Returns arrays of shape [K_num].
    """
    K = train_datasets[0].X_num.shape[2]
    sum_ = np.zeros((K,), dtype=np.float64)
    sqsum = np.zeros((K,), dtype=np.float64)
    count = np.zeros((K,), dtype=np.float64)
    for ds in train_datasets:
        X = ds.X_num  # [N,T,K]
        M = ds.mask   # [N,T] bool
        m = M.astype(bool)
        x_ = X[m]  # [n_valid, K]
        if x_.size == 0:
            continue
        sum_   += x_.sum(axis=0, dtype=np.float64)
        sqsum  += (x_ ** 2).sum(axis=0, dtype=np.float64)
        count  += np.array([m.sum()] * K, dtype=np.float64)
    mean = (sum_ / np.maximum(count, 1.0))
    var  = (sqsum / np.maximum(count, 1.0)) - (mean ** 2)
    std  = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_num_norm(x: torch.Tensor, mean, std) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        tmean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
        tstd  = torch.as_tensor(std,  device=x.device, dtype=x.dtype)
    else:
        tmean = mean.to(device=x.device, dtype=x.dtype)
        tstd  = std.to(device=x.device, dtype=x.dtype)
    return (x - tmean) / torch.clamp(tstd, min=1e-6)

def compute_stats_norm(stats_df: pd.DataFrame, stat_cols: List[str], train_player_ids: List[Tuple[int,int]]) -> Tuple[pd.Series, pd.Series]:
    df_tr = stats_df.merge(
        pd.DataFrame(train_player_ids, columns=["player_id","season_id"]),
        on=["player_id","season_id"], how="inner"
    )
    mu = df_tr[stat_cols].mean(numeric_only=True)
    sigma = df_tr[stat_cols].std(numeric_only=True).replace(0.0, 1.0)
    return mu, sigma

# ==============================
# Model: Token Encoder, Fusion, Transformer, Heads, GRL
# ==============================
class TokenEncoder(nn.Module):
    def __init__(self, vocab_sizes: Dict[str,int], k_num: int, q_qual: int,
                 d_cat=D_CAT, d_num=D_NUM, d_qual=D_QUAL):
        super().__init__()
        self.factors = list(vocab_sizes.keys())
        self.embs = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes[f], d_cat) for f in self.factors
        })
        self.num_proj = nn.Sequential(
            nn.Linear(k_num, 64), nn.ReLU(),
            nn.Linear(64, d_num)
        )
        self.qual_proj = nn.Linear(q_qual, d_qual)
        self.ln = nn.LayerNorm(d_cat + d_num + d_qual)

    def forward(self, cats: Dict[str, torch.Tensor], nums: torch.Tensor, quals: torch.Tensor):
        e_cats = None
        for f in self.factors:
            ef = self.embs[f](cats[f])
            e_cats = ef if e_cats is None else (e_cats + ef)

        e_num  = self.num_proj(nums)
        e_qual = self.qual_proj(quals)
        e      = torch.cat([e_cats, e_num, e_qual], dim=-1)
        return self.ln(e)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Fusion(nn.Module):
    def __init__(self, n_player_season: int, n_team_season: int, n_poshint: int,
                 d_base: int, d_model: int,
                 d_id=D_ID, d_poshint=D_POSHINT, d_seq=D_SEQTYPE):
        super().__init__()
        self.E_player = nn.Embedding(n_player_season, d_id)
        self.E_team   = nn.Embedding(n_team_season, d_id)
        self.E_pos    = nn.Embedding(max(n_poshint, 2), d_poshint)
        self.E_seq    = nn.Embedding(3, d_seq)  # 0-pad, 1=POS, 2=PLY

        in_dim = d_base + d_id + d_id + d_poshint + d_seq
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, e_base: torch.Tensor,
                H_player: Optional[torch.Tensor], H_team: torch.Tensor, H_poshint: torch.Tensor, H_seqtype: torch.Tensor,
                mask_player_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = e_base.device
        B, T, _ = e_base.shape

        H_team    = H_team.to(device)
        H_poshint = H_poshint.to(device)
        H_seqtype = H_seqtype.to(device)
        H_player  = H_player.to(device) if H_player is not None else None

        has_player = (H_player is not None) and bool((H_player >= 0).any().item())
        Ep = self.E_player(H_player) if has_player else torch.zeros((B, self.E_player.embedding_dim), device=device)
        Et = self.E_team(H_team)
        Epos = self.E_pos(torch.clamp(H_poshint, min=0))
        Es   = self.E_seq(H_seqtype)

        if mask_player_vec is not None and mask_player_vec.any():
            Ep = Ep * (~mask_player_vec).float().unsqueeze(-1)

        EpT   = Ep.unsqueeze(1).expand(B, T, -1)
        EtT   = Et.unsqueeze(1).expand(B, T, -1)
        EposT = Epos.unsqueeze(1).expand(B, T, -1)
        EsT   = Es.unsqueeze(1).expand(B, T, -1)

        Z = torch.cat([e_base, EpT, EtT, EposT, EsT], dim=-1)
        e_tok = self.mlp(Z)
        return e_tok, Ep, Et

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)

class Encoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ffn=FFN_HIDDEN, dropout=DROPOUT):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln  = nn.LayerNorm(d_model)
        self.pe  = SinusoidalPE(d_model)

    def forward(self, e_tok: torch.Tensor, causal_mask: bool, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]):
        device = self.ln.weight.device
        x = self.pe(e_tok.to(device))
        cm = None
        if causal_mask:
            T = x.size(1)
            cm = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        h = self.enc(x, mask=cm, src_key_padding_mask=key_padding_mask)
        return self.ln(h)

# Heads
class EventHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.out = nn.Linear(d_model, n_classes)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(h)  # [B,T,C]

class StatsHead(nn.Module):
    def __init__(self, d_model: int, d_id: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_id, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, pooled_h: torch.Tensor, Ep: torch.Tensor) -> torch.Tensor:
        z = torch.cat([pooled_h, Ep], dim=-1)
        return self.mlp(z)

class TeamAdvHead(nn.Module):
    def __init__(self, d_id: int, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_id, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, Ep: torch.Tensor) -> torch.Tensor:
        return self.mlp(Ep)

class MaskPlayerHead(nn.Module):
    def __init__(self, d_model: int, d_id: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_id)
    def forward(self, pooled_h: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled_h)

# ==============================
# Full Model wrapper
# ==============================
class StyleModel(nn.Module):
    def __init__(self, vocab_sizes: Dict[str,int], k_num: int, q_qual: int,
                 n_player_season: int, n_team_season: int, n_poshint: int):
        super().__init__()
        d_base = D_CAT + D_NUM + D_QUAL
        self.token_enc = TokenEncoder(vocab_sizes, k_num, q_qual)
        self.fusion    = Fusion(n_player_season, n_team_season, n_poshint, d_base, D_MODEL)
        self.encoder   = Encoder(D_MODEL, N_LAYERS, N_HEADS, FFN_HIDDEN, DROPOUT)
        self.event_head= EventHead(D_MODEL, ACTION_FAMILY_VSZ)
        self.stats_head= None  # set later when we know K_stats
        self.maskp_head= MaskPlayerHead(D_MODEL, D_ID)
        self.team_head = TeamAdvHead(D_ID)

    def set_stats_head(self, k_stats: int):
        self.stats_head = StatsHead(D_MODEL, D_ID, k_stats)
        device = next(self.parameters()).device
        self.stats_head.to(device)

    def _move_batch_to(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = {kk: (vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv)
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def forward(self, batch: Dict[str, torch.Tensor], num_mean, num_std,
                mask_player_vec: Optional[torch.Tensor] = None, causal=True):
        device = next(self.parameters()).device
        batch = self._move_batch_to(batch, device)

        cats = {f: batch["cats"][f] for f in CAT_FACTORS}                 # [B,T]
        nums = apply_num_norm(batch["nums"], num_mean, num_std)           # [B,T,K]
        quals= batch["quals"]                                             # [B,T,Q]
        mask = batch["mask"].bool()                                       # [B,T]

        H_player = batch.get("H_player", None)                            # [B] or missing
        H_team   = batch["H_team"]                                        # [B]
        H_poshint= batch["H_poshint"]                                     # [B]
        H_seqtype= batch["H_seqtype"]                                     # [B]

        if mask_player_vec is not None:
            mask_player_vec = mask_player_vec.to(device)

        e_base = self.token_enc(cats, nums, quals)                        # [B,T,d_base]
        e_tok, Ep, Et = self.fusion(e_base, H_player, H_team, H_poshint, H_seqtype, mask_player_vec)

        # key padding mask: True on PAD
        kpm = ~mask                                                       # [B,T]
        h = self.encoder(e_tok, causal_mask=causal, attn_mask=None, key_padding_mask=kpm)
        logits_event = self.event_head(h)                                 # [B,T,C]

        # pooled (mean over valid tokens)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return {
            "logits_event": logits_event,
            "pooled_h": pooled,
            "Ep": Ep,
            "Et": Et,
            "h_all": h,
            "mask_bool": mask
        }

# ==============================
# Training utilities
# ==============================
def build_optimizer(model: nn.Module):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.endswith(".bias") or "ln" in n.lower() or "layernorm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY, "lr": LR},
        {"params": no_decay, "weight_decay": 0.0, "lr": LR},
    ]
    try:
        opt = torch.optim.AdamW(groups, fused=True)
        return opt
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups)

def build_scheduler(optimizer, total_steps: int, warmup_frac: float = WARMUP_FRAC):
    warmup = max(1, int(total_steps*warmup_frac))
    def lr_lambda(step):
        if step < warmup: return step/max(1, warmup)
        prog = (step - warmup)/max(1,(total_steps - warmup))
        prog = min(max(prog, 0.0), 1.0)
        return 0.1 + 0.9*0.5*(1.0 + math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def event_loss_fn(logits: torch.Tensor, Y: torch.Tensor, mask_bool: torch.Tensor):
    C = logits.size(-1)
    logits_flat = logits.reshape(-1, C)
    Y_flat = Y.reshape(-1)
    valid = (Y_flat != 0) & mask_bool.reshape(-1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    logits_sel = logits_flat[valid]
    Y_sel = Y_flat[valid]
    loss = F.cross_entropy(logits_sel, Y_sel, label_smoothing=LABEL_SMOOTH)
    with torch.no_grad():
        acc = (logits_sel.argmax(dim=-1) == Y_sel).float().mean()
    return loss, acc

def huber_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta=1.0):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = pred - target
    diff = diff[mask]
    abs_diff = diff.abs()
    sq = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    loss = 0.5 * (sq ** 2) + delta * (abs_diff - sq)
    return loss.mean()

def sampled_softmax_logits(query: torch.Tensor, candidates: torch.Tensor):
    return query @ candidates.t()

# ==============================
# Data loading, normalization, splits
# ==============================
def load_datasets():
    pos_npz = SEQ_DIR / "possession_sequences.npz"
    ply_npz = SEQ_DIR / "player_sequences.npz"
    if not pos_npz.exists() and not ply_npz.exists():
        raise RuntimeError("No sequences found in STEP1_DIR/sequences. Run step1_preprocess first.")

    stats_path = STATS_DIR / "player_season_stats.parquet"
    if stats_path.exists():
        stats_df = pd.read_parquet(stats_path)
        drop_cols = {"player_id", "season_id"}
        stat_cols = [c for c in stats_df.columns if c not in drop_cols and np.issubdtype(stats_df[c].dtype, np.number)]
    else:
        stats_df = pd.DataFrame(columns=["player_id","season_id"])
        stat_cols = []

    pos_ds = PossessionSeqDataset(pos_npz) if pos_npz.exists() else None
    ply_ds = PlayerSeqDataset(ply_npz, stat_cols) if ply_npz.exists() else None
    return pos_ds, ply_ds, stats_df, stat_cols

def build_splits(pos_ds, ply_ds):
    pos_splits = None
    if pos_ds is not None:
        tr, va, te = split_indices(len(pos_ds))
        pos_splits = (Subset(pos_ds, tr), Subset(pos_ds, va), Subset(pos_ds, te))
    ply_splits = None
    if ply_ds is not None:
        tr, va, te = split_indices(len(ply_ds))
        ply_splits = (Subset(ply_ds, tr), Subset(ply_ds, va), Subset(ply_ds, te))
    return pos_splits, ply_splits

def compute_num_norm(train_pos: Optional[Subset], train_ply: Optional[Subset]) -> Tuple[np.ndarray, np.ndarray]:
    train_sets: List[BaseSeqDataset] = []
    if train_pos is not None:
        train_sets.append(train_pos.dataset)
    if train_ply is not None:
        train_sets.append(train_ply.dataset)
    if not train_sets:
        raise RuntimeError("No training datasets available to compute numeric normalization.")

    mean, std = _compute_num_norm_from_datasets(train_sets)
    save_json(OUT_DIR / "num_norm.json", {"mean": mean.tolist(), "std": std.tolist()})
    return mean, std


def build_stats_map(ply_train: Optional[Subset], stats_df: pd.DataFrame, stat_cols: List[str]):
    if (ply_train is None) or (len(stat_cols) == 0) or (PS_ID2KEY is None):
        return {}, None, None

    hplayers = set()
    for idx in ply_train.indices:
        hplayers.add(int(ply_train.dataset.H_player[idx]))
    train_ps = []
    for hid in hplayers:
        if hid in PS_ID2KEY:
            train_ps.append(PS_ID2KEY[hid])

    mu, sigma = compute_stats_norm(stats_df, stat_cols, train_ps)
    save_json(OUT_DIR / "stats_norm.json", {"mu": mu.to_dict(), "sigma": sigma.to_dict()})

    mapping: Dict[int, np.ndarray] = {}
    df = stats_df.copy()
    for c in stat_cols:
        if c in mu.index:
            denom = sigma[c] if (c in sigma.index and sigma[c] != 0) else 1.0
            df[c] = (df[c] - mu[c]) / denom
    grouped = df.set_index(["player_id","season_id"])
    for hid, (pid, sid) in PS_ID2KEY.items():
        if (pid, sid) in grouped.index:
            vec = grouped.loc[(pid, sid), stat_cols].astype(float).values
            mapping[hid] = vec.astype(np.float32)
    return mapping, mu, sigma

# ==============================
# Collate
# ==============================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_keys = {k: v for k, v in batch[0].items() if k not in ("cats", "actor_ps", "stats_vec", "stats_mask")}
    out = default_collate([{k: b[k] for k in base_keys} for b in batch])

    cats_batch = [b["cats"] for b in batch]
    out["cats"] = {f: torch.stack([cb[f] for cb in cats_batch], dim=0) for f in CAT_FACTORS}

    if "stats_vec" in batch[0]:
        out["stats_vec"]  = default_collate([b["stats_vec"]  for b in batch])
        out["stats_mask"] = default_collate([b["stats_mask"] for b in batch])

    if COLLATE_INCLUDE_ACTOR_PS and ("actor_ps" in batch[0]):
        out["actor_ps"] = torch.stack([b["actor_ps"] for b in batch], dim=0)
    return out

# ==============================
# Training / Evaluation loops
# ==============================
@dataclass
class BatchLoss:
    event_loss: float
    event_acc: float
    stats_loss: float
    maskp_loss: float
    team_loss: float
    total: float

def forward_one(model: StyleModel, batch: Dict[str, torch.Tensor], num_norm,
                k_stats: int, is_player_batch: bool, do_mask_player: bool, team_adv: bool):
    mean, std = num_norm
    B = batch["mask"].size(0)
    dev = next(model.parameters()).device

    if is_player_batch and do_mask_player:
        mask_player_vec = (torch.rand((B,), device=dev) < MASK_PLAYER_FRAC)
    else:
        mask_player_vec = torch.zeros((B,), dtype=torch.bool, device=dev)

    out = model(batch, mean, std, mask_player_vec=mask_player_vec, causal=True)
    logits_event = out["logits_event"]                 # [B,T,C]
    pooled_h     = out["pooled_h"]                     # [B,d_model]
    Ep           = out["Ep"]                           # [B,d_id]
    mask_bool    = out["mask_bool"]                    # [B,T]
    Y            = batch["Y"].to(dev)

    L_event, acc = event_loss_fn(logits_event, Y, mask_bool)

    L_stats = torch.tensor(0.0, device=dev)
    if is_player_batch and (model.stats_head is not None) and ("stats_vec" in batch):
        unmasked = (~mask_player_vec)
        if unmasked.any():
            target = batch["stats_vec"][unmasked].to(dev)   # [B',K]
            tmask  = batch["stats_mask"][unmasked].to(dev)  # [B',K]
            pred   = model.stats_head(pooled_h[unmasked], Ep[unmasked])
            L_stats = huber_loss_masked(pred, target, tmask, delta=1.0)

    L_maskp = torch.tensor(0.0, device=dev)
    if is_player_batch and mask_player_vec.any():
        H_players = batch["H_player"].to(dev)                     # [B]
        cand_ids, inv = torch.unique(H_players, return_inverse=True)
        query   = model.maskp_head(pooled_h[mask_player_vec])     # [B_m, d_id]
        cand_mat= model.fusion.E_player(cand_ids)                 # [M, d_id]
        logits  = sampled_softmax_logits(query, cand_mat)         # [B_m, M]
        targets = inv[mask_player_vec]
        L_maskp = F.cross_entropy(logits, targets)

    L_team = torch.tensor(0.0, device=dev)
    if is_player_batch and team_adv:
        H_teams = batch["H_team"].to(dev)
        cand_t, inv_t = torch.unique(H_teams, return_inverse=True)
        Ep_rev = grad_reverse(Ep, 1.0)
        team_emb = model.fusion.E_team(cand_t)
        logits_t = sampled_softmax_logits(Ep_rev, team_emb)
        L_team = F.cross_entropy(logits_t, inv_t)

    total = (LAMBDA_EVENT * L_event
             + LAMBDA_STATS * L_stats
             + LAMBDA_MASKP * L_maskp
             + LAMBDA_TEAMGRL * L_team)

    return BatchLoss(
        event_loss=float(L_event.item()),
        event_acc=float(acc.item()),
        stats_loss=float(L_stats.item()),
        maskp_loss=float(L_maskp.item()),
        team_loss=float(L_team.item()),
        total=float(total.item())
    ), total

def run_epoch(model: StyleModel,
              loaders: Tuple[Optional[DataLoader], Optional[DataLoader]],
              optimizer: Optional[torch.optim.Optimizer],
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
              num_norm,  # (mean_t, std_t)
              k_stats: int,
              train: bool,
              epoch: int,
              desc: str,
              scaler: Optional[GradScaler] = None,
              use_amp: bool = True):
    pos_loader, ply_loader = loaders
    pos_pref = CUDAPrefetcher(pos_loader, DEVICE) if pos_loader else None
    ply_pref = CUDAPrefetcher(ply_loader, DEVICE) if ply_loader else None
    if pos_pref: pos_pref.reset()
    if ply_pref: ply_pref.reset()

    model.train() if train else model.eval()

    n_steps = max(len(pos_loader) if pos_loader else 0, len(ply_loader) if ply_loader else 0)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, dynamic_ncols=True)

    sums = {"event":0.0, "acc":0.0, "stats":0.0, "maskp":0.0, "team":0.0, "total":0.0}
    counts = 0

    for _ in pbar:
        step_losses = []
        totals = []

        bpos = pos_pref.next() if pos_pref else None
        bply = ply_pref.next() if ply_pref else None

        if bpos is not None:
            with torch.set_grad_enabled(train):
                if train and use_amp and scaler is not None:
                    with autocast(dtype=AMP_DTYPE):
                        bl, tot = forward_one(model, bpos, num_norm, k_stats, is_player_batch=False, do_mask_player=False, team_adv=False)
                else:
                    bl, tot = forward_one(model, bpos, num_norm, k_stats, is_player_batch=False, do_mask_player=False, team_adv=False)
                step_losses.append(bl); totals.append(tot)

        if bply is not None:
            with torch.set_grad_enabled(train):
                if train and use_amp and scaler is not None:
                    with autocast(dtype=AMP_DTYPE):
                        bl, tot = forward_one(model, bply, num_norm, k_stats, is_player_batch=True, do_mask_player=True, team_adv=True)
                else:
                    bl, tot = forward_one(model, bply, num_norm, k_stats, is_player_batch=True, do_mask_player=True, team_adv=True)
                step_losses.append(bl); totals.append(tot)

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
            sums["acc"]   += bl.event_acc
            sums["stats"] += bl.stats_loss
            sums["maskp"] += bl.maskp_loss
            sums["team"]  += bl.team_loss
            sums["total"] += bl.total
            counts += 1

        if counts > 0:
            pbar.set_postfix({
                "L_evt": f"{sums['event']/counts:.3f}",
                "Acc":   f"{sums['acc']/counts:.3f}",
                "L_st":  f"{sums['stats']/counts:.3f}",
                "L_mp":  f"{sums['maskp']/counts:.3f}",
                "L_tm":  f"{sums['team']/counts:.3f}",
                "L_tot": f"{sums['total']/counts:.3f}",
                "lr":    f"{optimizer.param_groups[0]['lr']:.2e}" if train else "-",
            })

    if counts == 0:
        return {"L_event":0,"Acc":0,"L_stats":0,"L_maskp":0,"L_team":0,"L_total":0}
    return {
        "L_event": sums["event"]/counts,
        "Acc":     sums["acc"]/counts,
        "L_stats": sums["stats"]/counts,
        "L_maskp": sums["maskp"]/counts,
        "L_team":  sums["team"]/counts,
        "L_total": sums["total"]/counts
    }

# ==============================
# Export embeddings
# ==============================
def export_table_embeddings(model: StyleModel):
    """
    Exports the learned player-season and team-season table embeddings (Ep/Et).
    Files:
      - exports/player_identity_embeddings.npy
      - exports/player_identity_embeddings.tsv  (with player_id, season_id)
      - exports/team_season_table.npy
    """
    # Player-season table (Ep)
    if PS_ID2KEY:
        Ep = model.fusion.E_player.weight.detach().cpu().numpy()
        np.save(EXPORT_DIR / "player_identity_embeddings.npy", Ep)
        tsv = EXPORT_DIR / "player_identity_embeddings.tsv"
        with tsv.open("w", encoding="utf-8") as f:
            f.write("player_season_index\tplayer_id\tseason_id\tnorm\tvector_json\n")
            for i in range(Ep.shape[0]):
                pid, sid = PS_ID2KEY.get(i, (None, None))
                f.write(f"{i}\t{'' if pid is None else pid}\t{'' if sid is None else sid}\t{float(np.linalg.norm(Ep[i])):.6f}\t{json.dumps(Ep[i].tolist())}\n")

    # Team-season table (Et)
    if TS_MAP:
        Et = model.fusion.E_team.weight.detach().cpu().numpy()
        np.save(EXPORT_DIR / "team_season_table.npy", Et)
    log(f"[export] player_identity_embeddings.* and team_season_table.npy written.")

def export_behavioral_embeddings(model: StyleModel, ply_ds: PlayerSeqDataset,
                                 num_norm,
                                 batch_size: int = 64):
    """
    Pass over player dataset (eval) and export pooled_h per player-season (style embedding).
    Files:
      - exports/player_style_embeddings.npy
      - exports/player_style_embeddings.tsv
    """
    if ply_ds is None:
        return
    model.eval()

    kwargs = dict(
        dataset=ply_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=collate_fn
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(**kwargs)

    pooled_per_player: Dict[int, List[np.ndarray]] = {}
    with torch.no_grad():
        for b in tqdm(loader, desc="[export] player-style pass", leave=False):
            b = move_batch_to_device(b, DEVICE)
            out = model(b, num_norm[0], num_norm[1], mask_player_vec=None, causal=True)
            pooled = out["pooled_h"].detach().cpu().numpy()
            ids = b["H_player"].detach().cpu().numpy()
            for hid, vec in zip(ids, pooled):
                pooled_per_player.setdefault(int(hid), []).append(vec)

    ids_sorted = sorted(pooled_per_player.keys())
    if len(ids_sorted) == 0:
        log("[export] no player ids found in behavioral export.")
        return
    mat = np.stack([np.mean(np.stack(pooled_per_player[i], axis=0), axis=0) for i in ids_sorted], axis=0)
    np.save(EXPORT_DIR / "player_style_embeddings.npy", mat)

    tsv = EXPORT_DIR / "player_style_embeddings.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("player_season_index\tplayer_id\tseason_id\tnorm\tvector_json\n")
        for row_i, hid in enumerate(ids_sorted):
            pid, sid = PS_ID2KEY.get(hid, (None, None))
            f.write(f"{hid}\t{'' if pid is None else pid}\t{'' if sid is None else sid}\t{float(np.linalg.norm(mat[row_i])):.6f}\t{json.dumps(mat[row_i].tolist())}\n")
    log(f"[export] player_style_embeddings.npy/tsv written.")

def export_possession_embeddings(model: StyleModel,
                                 pos_ds: PossessionSeqDataset,
                                 num_norm,
                                 batch_size: int = 64,
                                 export_token_windows: bool = EXPORT_TOKEN_WINDOWS,
                                 shot_radius: int = SHOT_WINDOW_RADIUS):
    """
    Pass over possession dataset (eval) and export:
      - exports/pos_embeddings.npz with:
          Z_possession: [N_pos, d_model]
          segment_row:  [N_pos] = 0..N-1 (aligns with NPZ rows and pos_meta.segment_row)
          H_match:      [N_pos] if available
          H_pos_idx:    [N_pos] if available
      - (optional) exports/pos_token_windows.npz:
          h_windows:  [M, L, d_model]  where L=2*shot_radius+1 (padded when near edges)
          masks:      [M, L] bool
          segment_row:[M]
          center_t:   [M] token index of the shot
    """
    if pos_ds is None:
        return
    model.eval()

    # loader over the FULL dataset (no Subset), deterministic order == segment_row
    kwargs = dict(
        dataset=pos_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=collate_fn
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(**kwargs)

    N = len(pos_ds)
    d_model = D_MODEL
    Z = np.zeros((N, d_model), dtype=np.float32)

    # Stable keys (if present)
    H_match   = pos_ds.H_match if getattr(pos_ds, "H_match", None) is not None else None
    H_pos_idx = pos_ds.H_pos_idx if getattr(pos_ds, "H_pos_idx", None) is not None else None

    # Optional per-shot windows export setup
    do_windows = bool(export_token_windows)
    shot_token_id = None
    if do_windows and "action_family" in VOCABS:
        shot_token_id = VOCABS["action_family"]["token2id"].get("shot", None)
        if shot_token_id is None:
            log("[export] token windows requested but 'shot' not found in action_family vocab; skipping windows.")
            do_windows = False

    L = 2 * shot_radius + 1
    win_h_list: List[np.ndarray] = []
    win_m_list: List[np.ndarray] = []
    win_sr_list: List[int]       = []
    win_t_list: List[int]        = []

    seg_counter = 0
    with torch.no_grad():
        for b in tqdm(loader, desc="[export] possession pass", leave=False):
            b_dev = move_batch_to_device(b, DEVICE)
            out = model(b_dev, num_norm[0], num_norm[1], mask_player_vec=None, causal=True)
            pooled = out["pooled_h"].detach().cpu().numpy()         # [B,d]
            Bsz = pooled.shape[0]
            Z[seg_counter:seg_counter+Bsz, :] = pooled

            if do_windows:
                h_all    = out["h_all"].detach().cpu().numpy()      # [B,T,d]
                mask_bool= b["mask"].detach().cpu().numpy().astype(bool)  # [B,T]
                af_ids   = b["cats"]["action_family"].detach().cpu().numpy()  # [B,T]
                for i in range(Bsz):
                    valid_len = int(mask_bool[i].sum())
                    if valid_len == 0:
                        continue
                    shot_positions = np.where((af_ids[i, :valid_len] == shot_token_id))[0]
                    if shot_positions.size == 0:
                        continue
                    for t in shot_positions.tolist():
                        t0 = max(0, t - shot_radius)
                        t1 = min(valid_len - 1, t + shot_radius)
                        window = h_all[i, t0:t1 + 1, :]                                   # [w, d]
                        # Pad to fixed L
                        pad_left  = t - t0
                        pad_right = L - (t1 - t0 + 1) - pad_left
                        # Build padded window [L, d] and mask
                        w_full = np.zeros((L, d_model), dtype=np.float32)
                        m_full = np.zeros((L,), dtype=np.bool_)
                        start = max(0, shot_radius - pad_left)
                        w_full[start:start + (t1 - t0 + 1), :] = window
                        m_full[start:start + (t1 - t0 + 1)] = True
                        win_h_list.append(w_full)
                        win_m_list.append(m_full)
                        win_sr_list.append(seg_counter + i)
                        win_t_list.append(t)
            seg_counter += Bsz

    # Save main possession embedding file
    save_kwargs = {
        "Z_possession": Z,
        "segment_row": np.arange(N, dtype=np.int64)
    }
    if H_match is not None:
        save_kwargs["H_match"] = H_match
    if H_pos_idx is not None:
        save_kwargs["H_pos_idx"] = H_pos_idx
    np.savez_compressed(EXPORT_DIR / "pos_embeddings.npz", **save_kwargs)
    log(f"[export] pos_embeddings.npz written: Z_possession shape={Z.shape}")

    # Save optional windows
    if do_windows and len(win_h_list) > 0:
        H = np.stack(win_h_list, axis=0)
        M = np.stack(win_m_list, axis=0)
        SR= np.asarray(win_sr_list, dtype=np.int64)
        TT= np.asarray(win_t_list, dtype=np.int32)
        np.savez_compressed(EXPORT_DIR / "pos_token_windows.npz",
                            h_windows=H, masks=M, segment_row=SR, center_t=TT,
                            window_radius=np.int32(shot_radius))
        log(f"[export] pos_token_windows.npz written: h_windows={H.shape}, L={L}")

# ==============================
# Main
# ==============================
def main():
    log(f"[setup] Device: {DEVICE} | Torch: {torch.__version__}")
    if DEVICE.type == "cuda":
        cc = _gpu_cc()
        log(f"[setup] CUDA capability: {cc[0]}.{cc[1]}" if cc else "[setup] CUDA capability: unknown")
    pos_ds, ply_ds, stats_df, stat_cols = load_datasets()

    # Splits
    pos_splits, ply_splits = build_splits(pos_ds, ply_ds)
    pos_tr = pos_splits[0] if pos_splits else None
    pos_va = pos_splits[1] if pos_splits else None
    pos_te = pos_splits[2] if pos_splits else None
    ply_tr = ply_splits[0] if ply_splits else None
    ply_va = ply_splits[1] if ply_splits else None
    ply_te = ply_splits[2] if ply_splits else None

    # Build numeric normalization on TRAIN only
    num_mean, num_std = compute_num_norm(pos_tr, ply_tr)
    num_mean_t = torch.tensor(num_mean, device=DEVICE)
    num_std_t  = torch.tensor(num_std,  device=DEVICE)

    # Build stats z-norm and map H_player -> z-scored vector
    stats_map, mu, sigma = build_stats_map(ply_tr, stats_df, stat_cols)
    if ply_ds is not None:
        ply_ds.set_stats_map(stats_map)

    # DataLoaders (Windows-safe)
    def mk_loader(ds_subset, batch_size, shuffle):
        if ds_subset is None:
            return None
        kwargs = dict(
            dataset=ds_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=collate_fn
        )
        if NUM_WORKERS > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(**kwargs)

    pos_train_loader = mk_loader(pos_tr, BATCH_POS, True)
    pos_val_loader   = mk_loader(pos_va, BATCH_POS, False)
    pos_test_loader  = mk_loader(pos_te, BATCH_POS, False)

    ply_train_loader = mk_loader(ply_tr, BATCH_PLY, True)
    ply_val_loader   = mk_loader(ply_va, BATCH_PLY, False)
    ply_test_loader  = mk_loader(ply_te, BATCH_PLY, False)

    k_num = len(NUM_FEATURES)
    q_qual = len(QUALIFIERS)

    # Sizes for embeddings
    n_player_season = (max(PS_ID2KEY.keys()) + 1) if PS_ID2KEY else 2
    if TS_MAP and "id2key" in TS_MAP and len(TS_MAP["id2key"]) > 0:
        n_team_season = max(int(i) for i in TS_MAP["id2key"].keys()) + 1
    else:
        n_team_season = 2
    n_poshint       = VOCAB_SIZES.get("position_hint", 4)

    # Build model
    model = StyleModel(VOCAB_SIZES, k_num, q_qual, n_player_season, n_team_season, n_poshint).to(DEVICE)

    # torch.compile where safe
    can_compile_cuda = (DEVICE.type == "cuda" and _gpu_cc() and _gpu_cc()[0] >= 7)
    tried_compile = False
    if can_compile_cuda or DEVICE.type == "cpu":
        try:
            model = torch.compile(model, mode="max-autotune")
            tried_compile = True
            log("[compile] torch.compile enabled.")
        except Exception as e:
            log(f"[compile] skipped (fallback to eager): {e}")
    else:
        log("[compile] skipped: GPU compute capability < 7.0. Using eager mode.")

    k_stats = len(stat_cols)
    if k_stats > 0:
        model.set_stats_head(k_stats)
    log(f"[model] params={sum(p.numel() for p in model.parameters()):,} | d_model={D_MODEL} | layers={N_LAYERS}")
    log(f"[dims] K_num={k_num} | Q_qual={q_qual} | #cat_factors={len(CAT_FACTORS)}")

    # Optim/sched
    steps_per_epoch = max(
        len(pos_train_loader) if pos_train_loader else 0,
        len(ply_train_loader) if ply_train_loader else 0
    )
    total_steps = max(1, EPOCHS * steps_per_epoch)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, total_steps)

    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # Checkpoints
    best_val = float("inf")
    no_improve = 0
    best_path = OUT_DIR / "best.pt"
    last_path = OUT_DIR / "last.pt"

    # Meta
    meta = {
        "dataset_sizes": {
            "pos_train": len(pos_tr.indices) if pos_tr else 0,
            "pos_val":   len(pos_va.indices) if pos_va else 0,
            "pos_test":  len(pos_te.indices) if pos_te else 0,
            "ply_train": len(ply_tr.indices) if ply_tr else 0,
            "ply_val":   len(ply_va.indices) if ply_va else 0,
            "ply_test":  len(ply_te.indices) if ply_te else 0,
        },
        "dims": {
            "k_num": k_num, "q_qual": q_qual,
            "n_player_season": n_player_season,
            "n_team_season": n_team_season,
            "n_poshint": n_poshint
        },
        "cat_factors": CAT_FACTORS,
        "qualifiers": QUALIFIERS,
        "num_features": NUM_FEATURES
    }
    save_json(OUT_DIR / "meta.json", meta)

    history: List[dict] = []

    for epoch in range(1, EPOCHS + 1):
        tr_metrics = run_epoch(
            model,
            (pos_train_loader, ply_train_loader),
            optimizer, scheduler,
            (num_mean_t, num_std_t),
            k_stats,
            train=True,
            epoch=epoch,
            desc=f"train {epoch}/{EPOCHS}",
            scaler=scaler,
            use_amp=True
        )
        va_metrics = run_epoch(
            model,
            (pos_val_loader, ply_val_loader),
            optimizer=None, scheduler=None,
            num_norm=(num_mean_t, num_std_t),
            k_stats=k_stats,
            train=False,
            epoch=epoch,
            desc=f"valid {epoch}/{EPOCHS}",
            scaler=None,
            use_amp=True
        )

        log(f"[epoch {epoch}] "
            f"train: L_evt={tr_metrics['L_event']:.4f} Acc={tr_metrics['Acc']:.3f} "
            f"L_st={tr_metrics['L_stats']:.4f} L_mp={tr_metrics['L_maskp']:.4f} L_tm={tr_metrics['L_team']:.4f} L_tot={tr_metrics['L_total']:.4f}")
        log(f"[epoch {epoch}]   val: L_evt={va_metrics['L_event']:.4f} Acc={va_metrics['Acc']:.3f} "
            f"L_st={va_metrics['L_stats']:.4f} L_mp={va_metrics['L_maskp']:.4f} L_tm={va_metrics['L_team']:.4f} L_tot={va_metrics['L_total']:.4f}")

        torch.save({
            "model": model.state_dict(),
            "config": {
                "D_MODEL": D_MODEL, "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS, "FFN_HIDDEN": FFN_HIDDEN,
                "DROPOUT": DROPOUT, "D_CAT": D_CAT, "D_NUM": D_NUM, "D_QUAL": D_QUAL, "D_ID": D_ID,
                "D_POSHINT": D_POSHINT, "D_SEQTYPE": D_SEQTYPE,
                "VOCAB_SIZES": VOCAB_SIZES, "QUALIFIERS": QUALIFIERS, "NUM_FEATURES": NUM_FEATURES
            },
            "optim": optimizer.state_dict(),
        }, last_path)

        history.append({
            "epoch": epoch,
            **{k: float(v) for k, v in tr_metrics.items()},
            **{f"val_{k}": float(v) for k, v in va_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        save_json(OUT_DIR / "history.json", {"history": history})

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

    # Load best for evaluation/export
    if best_path.exists():
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        log(f"[load] best checkpoint loaded from {best_path.name}")
    else:
        log("[warn] best checkpoint not found; using last weights.")

    # Test evaluation
    test_metrics = run_epoch(
        model,
        (pos_test_loader, ply_test_loader),
        optimizer=None, scheduler=None,
        num_norm=(num_mean_t, num_std_t),
        k_stats=k_stats,
        train=False,
        epoch=0,
        desc="test",
        scaler=None,
        use_amp=True
    )
    log(f"[test] L_evt={test_metrics['L_event']:.4f} Acc={test_metrics['Acc']:.3f} "
        f"L_st={test_metrics['L_stats']:.4f} L_mp={test_metrics['L_maskp']:.4f} "
        f"L_tm={test_metrics['L_team']:.4f} L_tot={test_metrics['L_total']:.4f}")
    save_json(OUT_DIR / "test_metrics.json", test_metrics)

    # ======= Exports (viz-ready) =======
    export_table_embeddings(model)  # Ep (identity) + Et
    export_behavioral_embeddings(model, ply_ds if ply_ds is not None else None, (num_mean_t, num_std_t), batch_size=64)
    export_possession_embeddings(model, pos_ds if pos_ds is not None else None, (num_mean_t, num_std_t),
                                 batch_size=64, export_token_windows=EXPORT_TOKEN_WINDOWS, shot_radius=SHOT_WINDOW_RADIUS)

    log("[done] training + exports complete.\n"
        f" - {EXPORT_DIR / 'pos_embeddings.npz'}\n"
        f" - {EXPORT_DIR / 'player_identity_embeddings.npy'} (+ .tsv)\n"
        f" - {EXPORT_DIR / 'player_style_embeddings.npy'} (+ .tsv)\n"
        f" - {EXPORT_DIR / 'pos_token_windows.npz'} (if enabled)")

if __name__ == "__main__":
    main()
