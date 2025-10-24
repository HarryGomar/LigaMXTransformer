"""Generic helpers shared across the training modules."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from .config import PATHS


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_directories() -> None:
    PATHS.out_root.mkdir(parents=True, exist_ok=True)
    PATHS.export_root.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def split_indices(n: int, train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 73) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_tr = int(round(n * train))
    n_va = int(round(n * val))
    tr = idx[:n_tr]
    va = idx[n_tr : n_tr + n_va]
    te = idx[n_tr + n_va :]
    return tr, va, te


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            out[key] = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


def flatten_dicts(dicts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for payload in dicts:
        result.update(payload)
    return result
