"""
DAS Foundation MAE with per-channel RMS conditioning.

Input:
  w_norm  (B, 1, T, C)  per-channel unit-RMS dynamic strain (structure)
  log_rms (B, C)        per-channel log RMS (energy profile)

The temporal encoder reconstructs masked patches of w_norm.
log_rms is embedded and injected as per-channel conditioning.
Optional amplitude head predicts log_rms (Mode 2) - off by default.
"""

import torch
import torch.nn as nn


class TemporalPatchEmbed(nn.Module):
    def __init__(self, patch_t=16, embed_dim=128):
        super().__init__()
        self.patch_t = patch_t
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_t, stride=patch_t)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.proj(x)
        N = x.shape[-1]
        x = x.transpose(1, 2)
        return x.reshape(B, C, N, self.embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, N, D = x.shape
        x = x.reshape(B * C, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.reshape(B, C, N, D)


class DASTemporalMAE(nn.Module):
    def __init__(
        self,
        win_t=512, win_c=256, patch_t=16,
        enc_dim=128, enc_depth=4, enc_heads=4,
        dec_dim=64, dec_depth=2, dec_heads=4,
        mask_ratio=0.75, dropout=0.0, var_floor=1e-3,
        amplitude_weight=0.0,  # 0 = Mode 1 (structure only)
    ):
        super().__init__()
        self.patch_t = patch_t
        self.mask_ratio = mask_ratio
        self.n_patches = win_t // patch_t
        self.var_floor = var_floor
        self.amplitude_weight = amplitude_weight

        self.patch_embed = TemporalPatchEmbed(patch_t, enc_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, self.n_patches, enc_dim))

        # log_rms conditioning: scalar per channel -> enc_dim embedding
        self.rms_embed = nn.Sequential(
            nn.Linear(1, enc_dim), nn.GELU(), nn.Linear(enc_dim, enc_dim),
        )

        self.encoder = TemporalEncoder(enc_dim, enc_depth, enc_heads, dropout)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dec_dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, 1, self.n_patches, dec_dim))
        self.decoder = TemporalEncoder(dec_dim, dec_depth, dec_heads, dropout)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred_head = nn.Linear(dec_dim, patch_t)

        # Optional amplitude head (Mode 2)
        self.amp_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim), nn.GELU(), nn.Linear(enc_dim, 1),
        )

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

    def _mask(self, x):
        B, C, N, D = x.shape
        n_keep = max(1, int(N * (1 - self.mask_ratio)))
        noise = torch.rand(B, C, N, device=x.device)
        ids_shuf = torch.argsort(noise, dim=-1)
        ids_rest = torch.argsort(ids_shuf, dim=-1)
        ids_keep = ids_shuf[:, :, :n_keep]
        x_keep = torch.gather(x, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, D))
        mask = torch.ones(B, C, N, device=x.device)
        mask[:, :, :n_keep] = 0
        mask = torch.gather(mask, 2, ids_rest)
        return x_keep, mask, ids_rest

    def forward(self, w, log_rms):
        # w: (B, 1, T, C), log_rms: (B, C)
        B, _, T, C = w.shape
        x_in = w.squeeze(1).permute(0, 2, 1)  # (B, C, T)

        tokens = self.patch_embed(x_in) + self.pos_enc  # (B, C, N, D)

        # Inject per-channel RMS conditioning
        rms_cond = self.rms_embed(log_rms.unsqueeze(-1))  # (B, C, D)
        tokens = tokens + rms_cond.unsqueeze(2)

        tokens, mask, ids_rest = self._mask(tokens)
        tokens = self.encoder(tokens)
        tokens = self.enc_norm(tokens)

        # Amplitude prediction (Mode 2): from mean channel embedding
        amp_pred = None
        if self.amplitude_weight > 0:
            ch_emb = tokens.mean(dim=2)  # (B, C, D)
            amp_pred = self.amp_head(ch_emb).squeeze(-1)  # (B, C)

        tokens = self.enc_to_dec(tokens)
        B2, C2, n_keep, D = tokens.shape
        N = self.n_patches
        mt = self.mask_token.expand(B2, C2, N - n_keep, D)
        full = torch.cat([tokens, mt], dim=2)
        full = torch.gather(full, 2, ids_rest.unsqueeze(-1).expand(-1, -1, -1, D))
        full = full + self.pos_dec
        full = self.decoder(full)
        full = self.dec_norm(full)
        pred = self.pred_head(full)  # (B, C, N, patch_t)
        return pred, mask, amp_pred

    def loss(self, w, log_rms, pred, mask, amp_pred=None):
        B, _, T, C = w.shape
        x_in = w.squeeze(1).permute(0, 2, 1)
        target = x_in.reshape(B, C, self.n_patches, self.patch_t)
        # Per-patch normalisation with variance floor
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True).clamp(min=self.var_floor)
        target = (target - mean) / var.sqrt()
        m = mask.unsqueeze(-1)
        struct_loss = ((pred - target) ** 2 * m).sum() / (m.sum() * self.patch_t + 1e-6)

        if self.amplitude_weight > 0 and amp_pred is not None:
            amp_loss = ((amp_pred - log_rms) ** 2).mean()
            return struct_loss + self.amplitude_weight * amp_loss, struct_loss, amp_loss
        return struct_loss, struct_loss, torch.tensor(0.0, device=w.device)


def build_das_mae(**kwargs):
    return DASTemporalMAE(**kwargs)
