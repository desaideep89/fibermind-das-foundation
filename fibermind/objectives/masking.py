"""Masking strategies.

Every strategy returns the same structure so models do not care which one
produced it:

    MaskResult(x_keep, mask, ids_restore)

- ``x_keep``: the kept tokens, (B, C, n_keep, D)
- ``mask``: 1 where a token was masked, 0 where kept, (B, C, N)
- ``ids_restore``: permutation that puts the concatenation [kept, mask tokens]
  back into original time order, (B, C, N)

Only ``random_patch_mask`` exists in this version. It is the June 2026
behaviour, unchanged. Decision 0004 adds ``tube_mask`` and makes it the only
strategy allowed when a cross-channel block is present; until then the
simultaneity assumption of CrossChannelBlock is NOT met by this strategy (see
tests/test_masking.py::test_stage_b_sees_simultaneous_tokens, marked xfail).
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class MaskResult(NamedTuple):
    x_keep: torch.Tensor
    mask: torch.Tensor
    ids_restore: torch.Tensor


def random_patch_mask(x: torch.Tensor, mask_ratio: float) -> MaskResult:
    """Mask a random subset of time patches independently per channel.

    ``x`` is (B, C, N, D). Keeps ``max(1, int(N * (1 - mask_ratio)))`` patches
    per channel, chosen independently for every (batch, channel).
    """
    B, C, N, D = x.shape
    n_keep = max(1, int(N * (1 - mask_ratio)))
    noise = torch.rand(B, C, N, device=x.device)
    ids_shuf = torch.argsort(noise, dim=-1)
    ids_rest = torch.argsort(ids_shuf, dim=-1)
    ids_keep = ids_shuf[:, :, :n_keep]
    x_keep = torch.gather(x, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, D))
    mask = torch.ones(B, C, N, device=x.device)
    mask[:, :, :n_keep] = 0
    mask = torch.gather(mask, 2, ids_rest)
    return MaskResult(x_keep, mask, ids_rest)


def restore_with_mask_tokens(
    x_keep: torch.Tensor, mask_token: torch.Tensor, ids_restore: torch.Tensor, n_patches: int
) -> torch.Tensor:
    """Append mask tokens to the kept tokens and un-shuffle to original order.

    ``x_keep`` (B, C, n_keep, D), ``mask_token`` broadcastable to (1, 1, 1, D),
    returns (B, C, N, D).
    """
    B, C, n_keep, D = x_keep.shape
    mt = mask_token.expand(B, C, n_patches - n_keep, D)
    full = torch.cat([x_keep, mt], dim=2)
    return torch.gather(full, 2, ids_restore.unsqueeze(-1).expand(-1, -1, -1, D))
