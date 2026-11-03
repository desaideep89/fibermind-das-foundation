"""DAS temporal MAE with optional Stage B cross-channel attention.

Pipeline (docs/ARCHITECTURE.md section 6):

    w_norm (B, 1, T, C) per-channel unit-RMS structure
    log_rms (B, C)      per-channel energy, injected as conditioning
    1. temporal patch embed per channel
    2. temporal encoder per channel (Stage A)
    3. cross-channel block (Stage B), applied once after the temporal encoder
    4. decode masked patches of w_norm

Optional amplitude head predicts log_rms (off by default).

Attribute names are kept identical to the June 2026 ``DASTemporalMAE`` so old
checkpoints load with ``load_state_dict`` unchanged. Numerics are identical;
the refactor pull request's equivalence check proves this.

Known limitation carried over, not fixed here: masking is random per channel,
which does not meet CrossChannelBlock's simultaneity assumption. Decision 0004
replaces it with tube masking in a separate change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fibermind.models.blocks import CrossChannelBlock, TemporalEncoder, TemporalPatchEmbed
from fibermind.objectives.losses import amplitude_loss, masked_reconstruction_loss, patch_targets
from fibermind.objectives.masking import random_patch_mask, restore_with_mask_tokens


class DASTemporalMAE(nn.Module):
    def __init__(
        self,
        win_t=512,
        win_c=256,
        patch_t=16,
        enc_dim=128,
        enc_depth=4,
        enc_heads=4,
        dec_dim=64,
        dec_depth=2,
        dec_heads=4,
        mask_ratio=0.75,
        dropout=0.0,
        var_floor=1e-3,
        amplitude_weight=0.0,
        use_cross_channel=True,
        k_neighbours=8,
        global_downsample=32,
    ):
        super().__init__()
        self.patch_t = patch_t
        self.mask_ratio = mask_ratio
        self.n_patches = win_t // patch_t
        self.var_floor = var_floor
        self.amplitude_weight = amplitude_weight
        self.use_cross_channel = use_cross_channel

        self.patch_embed = TemporalPatchEmbed(patch_t, enc_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, self.n_patches, enc_dim))
        self.rms_embed = nn.Sequential(nn.Linear(1, enc_dim), nn.GELU(), nn.Linear(enc_dim, enc_dim))

        self.encoder = TemporalEncoder(enc_dim, enc_depth, enc_heads, dropout)
        if use_cross_channel:
            self.cross_channel = CrossChannelBlock(
                enc_dim, enc_heads, k_neighbours, global_downsample, dropout
            )
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dec_dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, 1, self.n_patches, dec_dim))
        self.decoder = TemporalEncoder(dec_dim, dec_depth, dec_heads, dropout)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred_head = nn.Linear(dec_dim, patch_t)
        self.amp_head = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.GELU(), nn.Linear(enc_dim, 1))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_dec, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(self, w: torch.Tensor, log_rms: torch.Tensor) -> torch.Tensor:
        """Patch-embed and condition: (B, 1, T, C), (B, C) -> (B, C, N, D)."""
        x_in = w.squeeze(1).permute(0, 2, 1)
        tokens = self.patch_embed(x_in) + self.pos_enc
        rms_cond = self.rms_embed(log_rms.unsqueeze(-1))
        return tokens + rms_cond.unsqueeze(2)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Stage A then Stage B on an already-masked token tensor (B, C, n, D)."""
        tokens = self.encoder(tokens)
        if self.use_cross_channel:
            tokens = self.cross_channel(tokens)
        return self.enc_norm(tokens)

    def forward(self, w: torch.Tensor, log_rms: torch.Tensor):
        tokens = self.embed(w, log_rms)
        tokens, mask, ids_rest = random_patch_mask(tokens, self.mask_ratio)
        tokens = self.encode(tokens)

        amp_pred = None
        if self.amplitude_weight > 0:
            amp_pred = self.amp_head(tokens.mean(dim=2)).squeeze(-1)

        tokens = self.enc_to_dec(tokens)
        full = restore_with_mask_tokens(tokens, self.mask_token, ids_rest, self.n_patches)
        full = full + self.pos_dec
        full = self.decoder(full)
        full = self.dec_norm(full)
        pred = self.pred_head(full)
        return pred, mask, amp_pred

    def loss(self, w, log_rms, pred, mask, amp_pred=None):
        target = patch_targets(w, self.n_patches, self.patch_t, self.var_floor)
        struct = masked_reconstruction_loss(pred, target, mask, self.patch_t)
        if self.amplitude_weight > 0 and amp_pred is not None:
            amp = amplitude_loss(amp_pred, log_rms)
            return struct + self.amplitude_weight * amp, struct, amp
        return struct, struct, torch.tensor(0.0, device=w.device)


def build_das_mae(**kwargs) -> DASTemporalMAE:
    return DASTemporalMAE(**kwargs)
