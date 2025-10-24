"""Helpers to access Step-1 artefacts required for training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PATHS
from .utils import load_json


def _list_vocab_files() -> List[Path]:
    if not PATHS.vocab_dir.exists():
        raise FileNotFoundError(
            f"Expected vocab directory at {PATHS.vocab_dir}. Run preprocessing step first."
        )
    return sorted(PATHS.vocab_dir.glob("*.json"))


@lru_cache(maxsize=1)
def categorical_factors() -> List[str]:
    files = _list_vocab_files()
    factors = []
    for path in files:
        name = path.stem
        if name in {"qualifiers", "num_features"}:
            continue
        factors.append(name)
    factors.sort()
    return factors


@lru_cache(maxsize=1)
def qualifiers() -> List[str]:
    payload = load_json(PATHS.vocab_dir / "qualifiers.json")
    return payload.get("qualifiers", [])


@lru_cache(maxsize=1)
def numeric_features() -> List[str]:
    payload = load_json(PATHS.vocab_dir / "num_features.json")
    return payload.get("num_features", [])


@lru_cache(maxsize=1)
def vocabs() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {factor: load_json(PATHS.vocab_dir / f"{factor}.json") for factor in categorical_factors()}


@lru_cache(maxsize=1)
def vocab_sizes() -> Dict[str, int]:
    return {factor: len(vocabs()[factor]["token2id"]) for factor in categorical_factors()}


@lru_cache(maxsize=1)
def action_family_vocab_size() -> int:
    if "action_family" not in vocab_sizes():
        raise RuntimeError("action_family vocab not found; required for next-event head.")
    return vocab_sizes()["action_family"]


@lru_cache(maxsize=1)
def player_season_map() -> Optional[Dict[str, Dict[str, Tuple[int, int]]]]:
    path = PATHS.maps_dir / "player_season_map.json"
    return load_json(path) if path.exists() else None


@lru_cache(maxsize=1)
def team_season_map() -> Optional[Dict[str, Dict[str, Tuple[int, int]]]]:
    path = PATHS.maps_dir / "team_season_map.json"
    return load_json(path) if path.exists() else None


@lru_cache(maxsize=1)
def player_season_id2key() -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
    payload = player_season_map()
    if not payload:
        return mapping
    for idx, pair in payload.get("id2key", {}).items():
        mapping[int(idx)] = (int(pair[0]), int(pair[1]))
    return mapping
