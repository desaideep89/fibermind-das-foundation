"""Encoder building blocks shared by every objective.

Stage A (temporal): TemporalPatchEmbed, TransformerBlock, TemporalEncoder.
Stage B (cross-channel): CrossChannelBlock with local sliding-window attention
and global attention over pooled channels.

Token tensors are (B, C, N, D): batch, channel, time patch, feature. Any block
that needs another layout permutes internally and permutes back.

Numerics unchanged from the June 2026 ``das_mae.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatchEmbed(nn.Module):
    """Split each channel's time series into non-overlapping 1-D patches and embed them."""

    def __init__(self, patch_t: int = 16, embed_dim: int = 128):
        super().__init__()
        self.patch_t = patch_t
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_t, stride=patch_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, N, D) with N = T // patch_t."""
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.proj(x)
        N = x.shape[-1]
        x = x.transpose(1, 2)
        return x.reshape(B, C, N, self.embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    """Transformer over the time-patch sequence, run independently per channel."""

    def __init__(self, dim: int, depth: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, D = x.shape
        x = x.reshape(B * C, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.reshape(B, C, N, D)


def sliding_window_mask(C: int, k: int, device) -> torch.Tensor:
    """Additive mask (C, C): 0 within +/- k//2 channels, -inf outside."""
    idx = torch.arange(C, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    allow = dist <= (k // 2)
    mask = torch.zeros(C, C, device=device)
    mask[~allow] = float("-inf")
    return mask


class CrossChannelBlock(nn.Module):
    """Mix tokens across channels at each time patch.

    Input and output (B, C, N, D). Local sparse k-NN attention over neighbouring
    channels, then global attention over channels pooled to ``downsample_to``
    and interpolated back.

    Assumes a complete channel grid at every time patch it sees: position n of
    channel c must be simultaneous with position n of channel c+1. See decision
    0004 for why the masking strategy has to guarantee this.
    """

    def __init__(self, dim: int, heads: int, k: int = 8, downsample_to: int = 32, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.downsample_to = downsample_to
        self.local_norm = nn.LayerNorm(dim)
        self.local_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.global_norm = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, D = x.shape
        xc = x.permute(0, 2, 1, 3).reshape(B * N, C, D)

        mask = sliding_window_mask(C, self.k, x.device)
        xn = self.local_norm(xc)
        local_out, _ = self.local_attn(xn, xn, xn, attn_mask=mask, need_weights=False)
        xc = xc + local_out

        xt = xc.transpose(1, 2)
        xd = F.adaptive_avg_pool1d(xt, self.downsample_to).transpose(1, 2)
        gn = self.global_norm(xd)
        g, _ = self.global_attn(gn, gn, gn, need_weights=False)
        g = xd + g
        g = g.transpose(1, 2)
        g = F.interpolate(g, size=C, mode="linear", align_corners=False).transpose(1, 2)
        xc = xc + g

        xc = xc + self.ff(self.ff_norm(xc))
        return xc.reshape(B, N, C, D).permute(0, 2, 1, 3)
