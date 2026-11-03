"""Masking returns consistent structures, and the simultaneity requirement of
the cross-channel block is documented (and currently unmet) here."""

import pytest
import torch

from fibermind.objectives import random_patch_mask, restore_with_mask_tokens


def test_random_patch_mask_shapes_and_counts():
    torch.manual_seed(0)
    B, C, N, D = 2, 4, 8, 6
    x = torch.randn(B, C, N, D)
    r = random_patch_mask(x, 0.5)
    assert r.x_keep.shape == (B, C, 4, D)
    assert r.mask.shape == (B, C, N)
    assert torch.all(r.mask.sum(-1) == 4)
    assert r.ids_restore.shape == (B, C, N)


def test_restore_puts_kept_tokens_back_in_place():
    torch.manual_seed(0)
    B, C, N, D = 2, 3, 8, 5
    x = torch.randn(B, C, N, D)
    r = random_patch_mask(x, 0.25)
    full = restore_with_mask_tokens(r.x_keep, torch.zeros(1, 1, 1, D), r.ids_restore, N)
    kept = r.mask == 0
    assert torch.allclose(full[kept], x[kept])
    assert torch.all(full[~kept] == 0)


def test_mask_ratio_keeps_at_least_one_token():
    x = torch.randn(1, 2, 4, 3)
    r = random_patch_mask(x, 0.99)
    assert r.x_keep.shape[2] == 1


@pytest.mark.xfail(
    strict=True,
    reason="Decision 0004: random per-channel masking does not keep a complete "
    "channel grid per time patch, so CrossChannelBlock mixes non-simultaneous tokens. "
    "Tube masking will make this pass.",
)
def test_stage_b_sees_simultaneous_tokens():
    """Position n of channel c must be the same time patch as position n of channel c+1."""
    torch.manual_seed(0)
    B, C, N, D = 1, 8, 16, 4
    x = torch.arange(N, dtype=torch.float32).view(1, 1, N, 1).expand(B, C, N, D).clone()
    r = random_patch_mask(x, 0.5)
    time_index = r.x_keep[..., 0]  # (B, C, n_keep): the original time patch each kept token came from
    assert torch.all(time_index == time_index[:, :1, :])
