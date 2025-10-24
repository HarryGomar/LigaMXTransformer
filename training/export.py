"""Model export helpers for downstream analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import artifacts
from .config import EXPORT_TOKEN_WINDOWS, PATHS, SHOT_WINDOW_RADIUS
from .dataloading import collate_fn, loader_kwargs
from .datasets import PlayerSeqDataset, PossessionSeqDataset
from .model import StyleModel
from .utils import log, move_batch_to_device


def export_table_embeddings(model: StyleModel) -> None:
    ps_map = artifacts.player_season_map()
    ts_map = artifacts.team_season_map()
    ps_id2key = artifacts.player_season_id2key()

    if ps_map:
        player_embeddings = model.fusion.E_player.weight.detach().cpu().numpy()
        np.save(PATHS.export_root / "player_identity_embeddings.npy", player_embeddings)
        with (PATHS.export_root / "player_identity_embeddings.tsv").open("w", encoding="utf-8") as handle:
            handle.write("player_season_index\tplayer_id\tseason_id\tnorm\tvector_json\n")
            for idx, vector in enumerate(player_embeddings):
                pid, sid = ps_id2key.get(idx, (None, None))
                handle.write(
                    f"{idx}\t{'' if pid is None else pid}\t{'' if sid is None else sid}\t"
                    f"{float(np.linalg.norm(vector)):.6f}\t{np.array2string(vector, separator=',', precision=6)}\n"
                )

    if ts_map:
        team_embeddings = model.fusion.E_team.weight.detach().cpu().numpy()
        np.save(PATHS.export_root / "team_season_table.npy", team_embeddings)

    log("[export] identity embeddings saved.")


def export_behavioral_embeddings(
    model: StyleModel,
    dataset: Optional[PlayerSeqDataset],
    num_norm: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 64,
) -> None:
    if dataset is None:
        return
    model.eval()

    kwargs = loader_kwargs(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(**kwargs)

    pooled_per_player: Dict[int, List[np.ndarray]] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="[export] player-style pass", leave=False):
            device_batch = move_batch_to_device(batch, next(model.parameters()).device)
            out = model(device_batch, num_norm[0], num_norm[1], mask_player_vec=None, causal=True)
            pooled = out["pooled_h"].detach().cpu().numpy()
            ids = device_batch["H_player"].detach().cpu().numpy()
            for hid, vec in zip(ids, pooled):
                pooled_per_player.setdefault(int(hid), []).append(vec)

    ids_sorted = sorted(pooled_per_player.keys())
    if not ids_sorted:
        log("[export] no player ids found for behavioural embeddings.")
        return

    matrix = np.stack([
        np.mean(np.stack(pooled_per_player[hid], axis=0), axis=0) for hid in ids_sorted
    ], axis=0)
    np.save(PATHS.export_root / "player_style_embeddings.npy", matrix)

    ps_id2key = artifacts.player_season_id2key()
    with (PATHS.export_root / "player_style_embeddings.tsv").open("w", encoding="utf-8") as handle:
        handle.write("player_season_index\tplayer_id\tseason_id\tnorm\tvector_json\n")
        for row_i, hid in enumerate(ids_sorted):
            pid, sid = ps_id2key.get(hid, (None, None))
            handle.write(
                f"{hid}\t{'' if pid is None else pid}\t{'' if sid is None else sid}\t"
                f"{float(np.linalg.norm(matrix[row_i])):.6f}\t{np.array2string(matrix[row_i], separator=',', precision=6)}\n"
            )
    log("[export] player_style_embeddings saved.")


def export_possession_embeddings(
    model: StyleModel,
    dataset: Optional[PossessionSeqDataset],
    num_norm: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 64,
    export_token_windows: bool = EXPORT_TOKEN_WINDOWS,
    shot_radius: int = SHOT_WINDOW_RADIUS,
) -> None:
    if dataset is None:
        return

    model.eval()

    kwargs = loader_kwargs(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(**kwargs)

    N = len(dataset)
    d_model = model.event_head.out.in_features
    pooled_matrix = np.zeros((N, d_model), dtype=np.float32)

    H_match = dataset.H_match if getattr(dataset, "H_match", None) is not None else None
    H_pos_idx = dataset.H_pos_idx if getattr(dataset, "H_pos_idx", None) is not None else None

    do_windows = bool(export_token_windows)
    shot_token_id = None
    vocab = artifacts.vocabs().get("action_family", {})
    if do_windows:
        shot_token_id = vocab.get("token2id", {}).get("shot")
        if shot_token_id is None:
            log("[export] token windows requested but 'shot' not found in vocab; skipping windows.")
            do_windows = False

    L = 2 * shot_radius + 1
    win_h_list: List[np.ndarray] = []
    win_m_list: List[np.ndarray] = []
    win_sr_list: List[int] = []
    win_t_list: List[int] = []

    seg_counter = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[export] possession pass", leave=False):
            device_batch = move_batch_to_device(batch, next(model.parameters()).device)
            out = model(device_batch, num_norm[0], num_norm[1], mask_player_vec=None, causal=True)
            pooled = out["pooled_h"].detach().cpu().numpy()
            B = pooled.shape[0]
            pooled_matrix[seg_counter : seg_counter + B, :] = pooled

            if do_windows and shot_token_id is not None:
                h_all = out["h_all"].detach().cpu().numpy()
                mask_bool = batch["mask"].detach().cpu().numpy().astype(bool)
                af_ids = batch["cats"]["action_family"].detach().cpu().numpy()
                for i in range(B):
                    valid_len = int(mask_bool[i].sum())
                    if valid_len == 0:
                        continue
                    shot_positions = np.where(af_ids[i, :valid_len] == shot_token_id)[0]
                    if shot_positions.size == 0:
                        continue
                    for t in shot_positions.tolist():
                        t0 = max(0, t - shot_radius)
                        t1 = min(valid_len - 1, t + shot_radius)
                        window = h_all[i, t0 : t1 + 1, :]
                        pad_left = t - t0
                        pad_right = L - (t1 - t0 + 1) - pad_left
                        window_full = np.zeros((L, d_model), dtype=np.float32)
                        mask_full = np.zeros((L,), dtype=np.bool_)
                        start = max(0, shot_radius - pad_left)
                        window_full[start : start + (t1 - t0 + 1), :] = window
                        mask_full[start : start + (t1 - t0 + 1)] = True
                        win_h_list.append(window_full)
                        win_m_list.append(mask_full)
                        win_sr_list.append(seg_counter + i)
                        win_t_list.append(t)
            seg_counter += B

    save_payload = {"Z_possession": pooled_matrix, "segment_row": np.arange(N, dtype=np.int64)}
    if H_match is not None:
        save_payload["H_match"] = H_match
    if H_pos_idx is not None:
        save_payload["H_pos_idx"] = H_pos_idx
    np.savez_compressed(PATHS.export_root / "pos_embeddings.npz", **save_payload)
    log("[export] pos_embeddings saved.")

    if do_windows and win_h_list:
        H = np.stack(win_h_list, axis=0)
        M = np.stack(win_m_list, axis=0)
        SR = np.asarray(win_sr_list, dtype=np.int64)
        TT = np.asarray(win_t_list, dtype=np.int32)
        np.savez_compressed(
            PATHS.export_root / "pos_token_windows.npz",
            h_windows=H,
            masks=M,
            segment_row=SR,
            center_t=TT,
            window_radius=np.int32(shot_radius),
        )
        log("[export] pos_token_windows saved.")
