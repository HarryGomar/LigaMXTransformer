"""Data loading utilities for training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .config import COLLATE_INCLUDE_ACTOR_PS, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS
from .utils import move_batch_to_device


class CUDAPrefetcher:
    """Overlaps CPU to GPU copies with compute using a dedicated CUDA stream."""

    def __init__(self, loader: Optional[DataLoader], device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if (device.type == "cuda" and loader is not None) else None
        )
        self.iterator = None
        self.next_batch = None

    def reset(self) -> None:
        if self.loader is None:
            self.iterator = None
            self.next_batch = None
            return
        self.iterator = iter(self.loader)
        self._preload()

    def _preload(self) -> None:
        if self.loader is None:
            self.next_batch = None
            return
        try:
            nxt = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = nxt
        else:
            with torch.cuda.stream(self.stream):
                self.next_batch = move_batch_to_device(nxt, self.device)

    def next(self) -> Optional[Dict[str, Any]]:
        if self.loader is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            self.reset()
            batch = self.next_batch
            if batch is None:
                return None
        self._preload()
        return batch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_keys = {
        key: value for key, value in batch[0].items() if key not in {"cats", "actor_ps", "stats_vec", "stats_mask"}
    }
    out = default_collate([{key: sample[key] for key in base_keys} for sample in batch])

    cats_batch = [sample["cats"] for sample in batch]
    out["cats"] = {factor: torch.stack([cats[factor] for cats in cats_batch], dim=0) for factor in cats_batch[0]}

    if "stats_vec" in batch[0]:
        out["stats_vec"] = default_collate([sample["stats_vec"] for sample in batch])
        out["stats_mask"] = default_collate([sample["stats_mask"] for sample in batch])

    if COLLATE_INCLUDE_ACTOR_PS and ("actor_ps" in batch[0]):
        out["actor_ps"] = torch.stack([sample["actor_ps"] for sample in batch], dim=0)
    return out


def loader_kwargs(dataset, batch_size: int, shuffle: bool) -> Dict[str, Any]:
    if dataset is None:
        return {}
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "collate_fn": collate_fn,
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs
