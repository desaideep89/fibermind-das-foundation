"""Shapes, the Stage B movement check that would have caught both historical
Stage B bugs, and checkpoint compatibility with the June 2026 attribute names."""

import torch

from fibermind.models import CrossChannelBlock, build_das_mae

LEGACY_KEYS_PREFIXES = (
    "patch_embed.",
    "pos_enc",
    "rms_embed.",
    "encoder.",
    "cross_channel.",
    "enc_norm.",
    "enc_to_dec.",
    "mask_token",
    "pos_dec",
    "decoder.",
    "dec_norm.",
    "pred_head.",
    "amp_head.",
)


def _stage_b_params(model):
    return [p for m in model.modules() if isinstance(m, CrossChannelBlock) for p in m.parameters()]


def test_forward_shapes(small_cfg, batch):
    model = build_das_mae(**small_cfg)
    w, log_rms = batch
    pred, mask, amp = model(w, log_rms)
    B, C, N = w.shape[0], small_cfg["win_c"], small_cfg["win_t"] // small_cfg["patch_t"]
    assert pred.shape == (B, C, N, small_cfg["patch_t"])
    assert mask.shape == (B, C, N)
    assert amp is None
    loss, struct, ampl = model.loss(w, log_rms, pred, mask, amp)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.equal(loss, struct)


def test_amplitude_head_when_enabled(small_cfg, batch):
    small_cfg["amplitude_weight"] = 0.1
    model = build_das_mae(**small_cfg)
    w, log_rms = batch
    pred, mask, amp = model(w, log_rms)
    assert amp.shape == (w.shape[0], small_cfg["win_c"])
    loss, struct, ampl = model.loss(w, log_rms, pred, mask, amp)
    assert torch.isclose(loss, struct + 0.1 * ampl)


def test_temporal_only_config_has_no_stage_b(small_cfg, batch):
    small_cfg["use_cross_channel"] = False
    model = build_das_mae(**small_cfg)
    assert not _stage_b_params(model)
    w, log_rms = batch
    pred, mask, _ = model(w, log_rms)
    assert pred.shape[:3] == mask.shape


def test_stage_b_parameters_move_during_training(small_cfg, batch):
    """Every cross-channel parameter tensor must change after a few optimiser steps.
    v0.2 of the August stack failed this (0 of 12 moved); it guards against a
    masking strategy silently bypassing Stage B."""
    torch.manual_seed(0)
    model = build_das_mae(**small_cfg)
    w, log_rms = batch
    before = [p.detach().clone() for p in _stage_b_params(model)]
    assert before, "config should include Stage B"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        pred, mask, amp = model(w, log_rms)
        loss, _, _ = model.loss(w, log_rms, pred, mask, amp)
        opt.zero_grad()
        loss.backward()
        opt.step()
    moved = [not torch.equal(a, b) for a, b in zip(before, _stage_b_params(model), strict=True)]
    assert all(moved), f"{sum(moved)}/{len(moved)} Stage B tensors moved"


def test_loss_decreases_on_fixed_batch(small_cfg, batch):
    torch.manual_seed(0)
    model = build_das_mae(**small_cfg)
    w, log_rms = batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(30):
        pred, mask, amp = model(w, log_rms)
        loss, _, _ = model.loss(w, log_rms, pred, mask, amp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < first


def test_state_dict_keeps_legacy_names(small_cfg):
    model = build_das_mae(**small_cfg)
    keys = list(model.state_dict().keys())
    assert keys, "empty state dict"
    for k in keys:
        assert k.startswith(LEGACY_KEYS_PREFIXES), k


def test_production_config_builds():
    """The config on configs/config.yaml must build and run at reduced batch."""
    model = build_das_mae(
        win_t=512,
        win_c=256,
        patch_t=16,
        enc_dim=128,
        enc_depth=4,
        enc_heads=4,
        dec_dim=64,
        dec_depth=2,
        dec_heads=4,
        mask_ratio=0.5,
        use_cross_channel=True,
        k_neighbours=8,
        global_downsample=32,
    )
    w = torch.randn(1, 1, 512, 256)
    log_rms = torch.randn(1, 256)
    with torch.no_grad():
        pred, mask, _ = model(w, log_rms)
    assert pred.shape == (1, 256, 32, 16)
