"""
DAS Foundation MAE - Stage B: temporal + per-time-patch cross-channel attention.

Pipeline:
  w_norm (B,1,T,C) per-channel unit-RMS structure
  log_rms (B,C) per-channel energy, injected as conditioning
  1. Temporal patch embed per channel
  2. Temporal encoder per channel (primary structure)
  3. Cross-channel block at each time patch (secondary structure / moveout):
       - local sparse k-NN attention (sliding-window mask)
       - global attention on spatially downsampled channels
  4. Decode masked patches of w_norm
Optional amplitude head predicts log_rms (Mode 2, off by default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, attn_mask=None):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         attn_mask=attn_mask, need_weights=False)
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


def sliding_window_mask(C, k, device):
    """Additive mask (C, C): 0 within +/- k//2, -inf outside."""
    idx = torch.arange(C, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    allow = dist <= (k // 2)
    mask = torch.zeros(C, C, device=device)
    mask[~allow] = float("-inf")
    return mask


class CrossChannelBlock(nn.Module):
    """
    Mixes across channels at each time patch.
    Input/output: (B, C, N, D)
    Local sparse k-NN attention + global downsampled attention.
    """
    def __init__(self, dim, heads, k=8, downsample_to=32, dropout=0.0):
        super().__init__()
        self.k = k
        self.downsample_to = downsample_to
        self.local_norm = nn.LayerNorm(dim)
        self.local_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.global_norm = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, N, D = x.shape
        # reshape so channels are the sequence, (B*N) the batch
        xc = x.permute(0, 2, 1, 3).reshape(B * N, C, D)  # (B*N, C, D)

        # local sparse attention
        mask = sliding_window_mask(C, self.k, x.device)
        xn = self.local_norm(xc)
        local_out, _ = self.local_attn(xn, xn, xn, attn_mask=mask, need_weights=False)
        xc = xc + local_out

        # global downsampled attention
        xt = xc.transpose(1, 2)  # (B*N, D, C)
        xd = F.adaptive_avg_pool1d(xt, self.downsample_to).transpose(1, 2)  # (B*N, ds, D)
        gn = self.global_norm(xd)
        g, _ = self.global_attn(gn, gn, gn, need_weights=False)
        g = xd + g
        g = g.transpose(1, 2)  # (B*N, D, ds)
        g = F.interpolate(g, size=C, mode="linear", align_corners=False).transpose(1, 2)  # (B*N, C, D)
        xc = xc + g

        xc = xc + self.ff(self.ff_norm(xc))
        return xc.reshape(B, N, C, D).permute(0, 2, 1, 3)  # (B, C, N, D)


class DASTemporalMAE(nn.Module):
    def __init__(
        self,
        win_t=512, win_c=256, patch_t=16,
        enc_dim=128, enc_depth=4, enc_heads=4,
        dec_dim=64, dec_depth=2, dec_heads=4,
        mask_ratio=0.75, dropout=0.0, var_floor=1e-3,
        amplitude_weight=0.0,
        use_cross_channel=True, k_neighbours=8, global_downsample=32,
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
            self.cross_channel = CrossChannelBlock(enc_dim, enc_heads, k_neighbours, global_downsample, dropout)
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
        B, _, T, C = w.shape
        x_in = w.squeeze(1).permute(0, 2, 1)
        tokens = self.patch_embed(x_in) + self.pos_enc
        rms_cond = self.rms_embed(log_rms.unsqueeze(-1))
        tokens = tokens + rms_cond.unsqueeze(2)

        tokens, mask, ids_rest = self._mask(tokens)
        tokens = self.encoder(tokens)
        if self.use_cross_channel:
            tokens = self.cross_channel(tokens)
        tokens = self.enc_norm(tokens)

        amp_pred = None
        if self.amplitude_weight > 0:
            amp_pred = self.amp_head(tokens.mean(dim=2)).squeeze(-1)

        tokens = self.enc_to_dec(tokens)
        B2, C2, n_keep, D = tokens.shape
        N = self.n_patches
        mt = self.mask_token.expand(B2, C2, N - n_keep, D)
        full = torch.cat([tokens, mt], dim=2)
        full = torch.gather(full, 2, ids_rest.unsqueeze(-1).expand(-1, -1, -1, D))
        full = full + self.pos_dec
        full = self.decoder(full)
        full = self.dec_norm(full)
        pred = self.pred_head(full)
        return pred, mask, amp_pred

    def loss(self, w, log_rms, pred, mask, amp_pred=None):
        B, _, T, C = w.shape
        x_in = w.squeeze(1).permute(0, 2, 1)
        target = x_in.reshape(B, C, self.n_patches, self.patch_t)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True).clamp(min=self.var_floor)
        target = (target - mean) / var.sqrt()
        m = mask.unsqueeze(-1)
        struct = ((pred - target) ** 2 * m).sum() / (m.sum() * self.patch_t + 1e-6)
        if self.amplitude_weight > 0 and amp_pred is not None:
            amp = ((amp_pred - log_rms) ** 2).mean()
            return struct + self.amplitude_weight * amp, struct, amp
        return struct, struct, torch.tensor(0.0, device=w.device)


def build_das_mae(**kwargs):
    return DASTemporalMAE(**kwargs)
