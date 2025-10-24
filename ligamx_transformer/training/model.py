"""Model definition for LigaMX Transformer training."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D_CAT, D_ID, D_MODEL, D_NUM, D_POSHINT, D_QUAL, D_SEQTYPE, FFN_HIDDEN, N_HEADS, N_LAYERS
from .datasets import ACTION_FAMILY_VSZ, CAT_FACTORS, NUM_FEATURES, QUALIFIERS, VOCAB_SIZES


class TokenEncoder(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], k_num: int, q_qual: int,
                 d_cat: int = D_CAT, d_num: int = D_NUM, d_qual: int = D_QUAL):
        super().__init__()
        self.factors = list(vocab_sizes.keys())
        self.embs = nn.ModuleDict({factor: nn.Embedding(vocab_sizes[factor], d_cat) for factor in self.factors})
        self.num_proj = nn.Sequential(nn.Linear(k_num, 64), nn.ReLU(), nn.Linear(64, d_num))
        self.qual_proj = nn.Linear(q_qual, d_qual)
        self.norm = nn.LayerNorm(d_cat + d_num + d_qual)

    def forward(self, cats: Dict[str, torch.Tensor], nums: torch.Tensor, quals: torch.Tensor) -> torch.Tensor:
        cat_embed = None
        for factor in self.factors:
            emb = self.embs[factor](cats[factor])
            cat_embed = emb if cat_embed is None else (cat_embed + emb)
        e_num = self.num_proj(nums)
        e_qual = self.qual_proj(quals)
        fused = torch.cat([cat_embed, e_num, e_qual], dim=-1)
        return self.norm(fused)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


class Fusion(nn.Module):
    def __init__(self, n_player_season: int, n_team_season: int, n_poshint: int,
                 d_base: int, d_model: int,
                 d_id: int = D_ID, d_poshint: int = D_POSHINT, d_seq: int = D_SEQTYPE):
        super().__init__()
        self.E_player = nn.Embedding(n_player_season, d_id)
        self.E_team = nn.Embedding(n_team_season, d_id)
        self.E_pos = nn.Embedding(max(n_poshint, 2), d_poshint)
        self.E_seq = nn.Embedding(3, d_seq)

        in_dim = d_base + d_id + d_id + d_poshint + d_seq
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        e_base: torch.Tensor,
        H_player: Optional[torch.Tensor],
        H_team: torch.Tensor,
        H_poshint: torch.Tensor,
        H_seqtype: torch.Tensor,
        mask_player_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = e_base.device
        B, T, _ = e_base.shape

        H_team = H_team.to(device)
        H_poshint = H_poshint.to(device)
        H_seqtype = H_seqtype.to(device)
        H_player = H_player.to(device) if H_player is not None else None

        has_player = H_player is not None and bool((H_player >= 0).any().item())
        player_emb = self.E_player(H_player) if has_player else torch.zeros((B, self.E_player.embedding_dim), device=device)
        if mask_player_vec is not None and mask_player_vec.any():
            player_emb = player_emb * (~mask_player_vec).float().unsqueeze(-1)

        team_emb = self.E_team(H_team)
        pos_emb = self.E_pos(torch.clamp(H_poshint, min=0))
        seq_emb = self.E_seq(H_seqtype)

        player_tok = player_emb.unsqueeze(1).expand(B, T, -1)
        team_tok = team_emb.unsqueeze(1).expand(B, T, -1)
        pos_tok = pos_emb.unsqueeze(1).expand(B, T, -1)
        seq_tok = seq_emb.unsqueeze(1).expand(B, T, -1)

        fused = torch.cat([e_base, player_tok, team_tok, pos_tok, seq_tok], dim=-1)
        return self.mlp(fused), player_emb, team_emb


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 ffn: int = FFN_HIDDEN, dropout: float = 0.10):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.positional = SinusoidalPE(d_model)

    def forward(
        self,
        e_tok: torch.Tensor,
        causal_mask: bool,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = self.norm.weight.device
        x = self.positional(e_tok.to(device))
        mask = None
        if causal_mask:
            length = x.size(1)
            mask = torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        encoded = self.encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded)


class EventHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(h)


class StatsHead(nn.Module):
    def __init__(self, d_model: int, d_id: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model + d_id, 256), nn.ReLU(), nn.Linear(256, out_dim))

    def forward(self, pooled_h: torch.Tensor, player_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([pooled_h, player_emb], dim=-1))


class TeamAdvHead(nn.Module):
    def __init__(self, d_id: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_id, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, player_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(player_emb)


class MaskPlayerHead(nn.Module):
    def __init__(self, d_model: int, d_id: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_id)

    def forward(self, pooled_h: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled_h)


class StyleModel(nn.Module):
    def __init__(self, n_player_season: int, n_team_season: int, n_poshint: int):
        super().__init__()
        d_base = D_CAT + D_NUM + D_QUAL
        self.token_encoder = TokenEncoder(VOCAB_SIZES, len(NUM_FEATURES), len(QUALIFIERS))
        self.fusion = Fusion(n_player_season, n_team_season, n_poshint, d_base, D_MODEL)
        self.encoder = Encoder(D_MODEL, N_LAYERS, N_HEADS, FFN_HIDDEN)
        self.event_head = EventHead(D_MODEL, ACTION_FAMILY_VSZ)
        self.stats_head: Optional[StatsHead] = None
        self.maskp_head = MaskPlayerHead(D_MODEL, D_ID)
        self.team_head = TeamAdvHead(D_ID)

    def set_stats_head(self, out_dim: int) -> None:
        self.stats_head = StatsHead(D_MODEL, D_ID, out_dim).to(next(self.parameters()).device)

    def _move_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
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

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        num_mean: torch.Tensor,
        num_std: torch.Tensor,
        mask_player_vec: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        batch = self._move_batch(batch, device)

        cats = {factor: batch["cats"][factor] for factor in CAT_FACTORS}
        nums = apply_num_norm(batch["nums"], num_mean, num_std)
        quals = batch["quals"]
        mask = batch["mask"].bool()

        H_player = batch.get("H_player")
        H_team = batch["H_team"]
        H_poshint = batch["H_poshint"]
        H_seqtype = batch["H_seqtype"]

        if mask_player_vec is not None:
            mask_player_vec = mask_player_vec.to(device)

        token_repr = self.token_encoder(cats, nums, quals)
        fused, player_emb, team_emb = self.fusion(
            token_repr,
            H_player,
            H_team,
            H_poshint,
            H_seqtype,
            mask_player_vec,
        )

        key_padding_mask = ~mask
        encoded = self.encoder(fused, causal_mask=causal, attn_mask=None, key_padding_mask=key_padding_mask)
        logits_event = self.event_head(encoded)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom

        return {
            "logits_event": logits_event,
            "pooled_h": pooled,
            "Ep": player_emb,
            "Et": team_emb,
            "h_all": encoded,
            "mask_bool": mask,
        }


def apply_num_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return (x - mean) / torch.clamp(std, min=1e-6)
