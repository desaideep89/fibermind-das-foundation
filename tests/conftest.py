"""Shared fixtures. Everything is synthetic and small so the suite runs on a
laptop CPU in well under two minutes (docs/ARCHITECTURE.md section 9)."""

import numpy as np
import pytest
import torch

SMALL = dict(
    win_t=128,
    win_c=16,
    patch_t=16,
    enc_dim=32,
    enc_depth=2,
    enc_heads=4,
    dec_dim=16,
    dec_depth=1,
    dec_heads=4,
    mask_ratio=0.5,
    dropout=0.0,
    var_floor=1e-3,
    amplitude_weight=0.0,
    use_cross_channel=True,
    k_neighbours=4,
    global_downsample=4,
)


@pytest.fixture
def small_cfg():
    return dict(SMALL)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_strain_rate(rng):
    """(T, C) strain rate: pink-ish noise plus a moving band event across channels."""
    T, C = 2048, 64
    x = rng.standard_normal((T, C)) * 1e-9
    t = np.arange(T)
    for c in range(20, 40):
        onset = 600 + 10 * (c - 20)
        x[:, c] += 5e-8 * np.exp(-((t - onset) ** 2) / (2 * 40.0**2)) * np.sin(2 * np.pi * 5.0 * t / 100.0)
    return x


@pytest.fixture
def batch(small_cfg):
    torch.manual_seed(0)
    B = 2
    w = torch.randn(B, 1, small_cfg["win_t"], small_cfg["win_c"])
    w = w / w.pow(2).mean(dim=2, keepdim=True).sqrt()
    log_rms = torch.randn(B, small_cfg["win_c"])
    return w, log_rms
