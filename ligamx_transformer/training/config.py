"""Training configuration for the LigaMX Transformer project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Directory layout consumed by the training step."""

    step1_root: Path = Path("./artifacts/data")
    out_root: Path = Path("./artifacts/train")
    export_root: Path = Path("./exports")

    @property
    def vocab_dir(self) -> Path:
        return self.step1_root / "vocabs"

    @property
    def sequence_dir(self) -> Path:
        return self.step1_root / "sequences"

    @property
    def stats_dir(self) -> Path:
        return self.step1_root / "stats"

    @property
    def maps_dir(self) -> Path:
        return self.step1_root / "maps"


PATHS = PathConfig()

# Model architecture -------------------------------------------------------
D_MODEL = 384
N_LAYERS = 12
N_HEADS = 12
FFN_HIDDEN = 4 * D_MODEL
DROPOUT = 0.10

D_CAT = 32
D_NUM = 64
D_QUAL = 24

D_ID = 96
D_POSHINT = 32
D_SEQTYPE = 8

# Optimisation -------------------------------------------------------------
MASK_PLAYER_FRAC = 0.30

LAMBDA_EVENT = 1.0
LAMBDA_STATS = 0.30
LAMBDA_MASKP = 0.30
LAMBDA_TEAMGRL = 0.05

LR = 2e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.06
CLIP_NORM = 0.5

LABEL_SMOOTH = 0.05
EPOCHS = 15
BATCH_POS = 96
BATCH_PLY = 96
EARLY_STOP_PATIENCE = 8
SEED = 73

COLLATE_INCLUDE_ACTOR_PS = False

EXPORT_TOKEN_WINDOWS = True
SHOT_WINDOW_RADIUS = 4

# Data loader knobs --------------------------------------------------------
IS_WINDOWS = os.name == "nt"
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = (NUM_WORKERS > 0) and (not IS_WINDOWS)
