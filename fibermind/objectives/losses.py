"""Losses as functions of (prediction, target, mask) and nothing else."""

from __future__ import annotations

import torch


def patch_targets(w: torch.Tensor, n_patches: int, patch_t: int, var_floor: float) -> torch.Tensor:
    """Turn ``w`` (B, 1, T, C) into per-patch normalised targets (B, C, N, patch_t).

    Each patch is standardised by its own mean and variance, with a variance
    floor so flat patches do not blow up.
    """
    B, _, T, C = w.shape
    x_in = w.squeeze(1).permute(0, 2, 1)
    target = x_in.reshape(B, C, n_patches, patch_t)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True).clamp(min=var_floor)
    return (target - mean) / var.sqrt()


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, patch_t: int
) -> torch.Tensor:
    """Mean squared error over masked patches only. ``mask`` is (B, C, N), 1 = masked."""
    m = mask.unsqueeze(-1)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * patch_t + 1e-6)


def amplitude_loss(amp_pred: torch.Tensor, log_rms: torch.Tensor) -> torch.Tensor:
    return ((amp_pred - log_rms) ** 2).mean()
