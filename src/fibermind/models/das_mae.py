"""
DAS Foundation Model - Stage A: Temporal-only MAE
Physics rationale:
  - Each channel is an independent strain-rate time series
  - 1D patches along time axis only (not 2D image patches)
  - Cross-channel mixing added in Stage B once temporal learning is verified
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPatchEmbed(nn.Module):
    """
    Splits (B, C, T) into non-overlapping 1D patches of size patch_t.
    All channels processed in parallel via grouped conv.
    Output: (B, C, N, D) where N = T // patch_t
    """
    def __init__(self, patch_t=16, embed_dim=128):
        super().__init__()
        self.patch_t = patch_t
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_t, stride=patch_t)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.proj(x)                          # (B*C, D, N)
        N = x.shape[-1]
        x = x.transpose(1, 2)                     # (B*C, N, D)
        return x.reshape(B, C, N, self.embed_dim) # (B, C, N, D)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    """Runs transformer blocks along time axis for each channel independently."""
    def __init__(self, dim, depth, heads, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, N, D = x.shape
        x = x.reshape(B * C, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.reshape(B, C, N, D)


class DASTemporalMAE(nn.Module):
    """
    Stage A: temporal-only MAE.
    Input: (B, 1, T, C)
    Masks temporal patches per channel independently.
    Reconstructs masked patches from visible ones.
    """
    def __init__(
        self,
        win_t=512, win_c=256,
        patch_t=16,
        enc_dim=128, enc_depth=4, enc_heads=4,
        dec_dim=64,  dec_depth=2, dec_heads=4,
        mask_ratio=0.75, dropout=0.0,
        var_floor=1e-3,
    ):
        super().__init__()
        self.patch_t = patch_t
        self.mask_ratio = mask_ratio
        self.n_patches = win_t // patch_t
        self.var_floor = var_floor

        # Encoder
        self.patch_embed = TemporalPatchEmbed(patch_t, enc_dim)
        self.pos_enc = nn.Parameter(
            torch.zeros(1, 1, self.n_patches, enc_dim))
        self.encoder = TemporalEncoder(enc_dim, enc_depth, enc_heads, dropout)
        self.enc_norm = nn.LayerNorm(enc_dim)

        # Decoder
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dec_dim))
        self.pos_dec = nn.Parameter(
            torch.zeros(1, 1, self.n_patches, dec_dim))
        self.decoder = TemporalEncoder(dec_dim, dec_depth, dec_heads, dropout)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred_head = nn.Linear(dec_dim, patch_t)

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
        """Independent random masking per channel."""
        B, C, N, D = x.shape
        n_keep = max(1, int(N * (1 - self.mask_ratio)))
        noise = torch.rand(B, C, N, device=x.device)
        ids_shuf = torch.argsort(noise, dim=-1)
        ids_rest = torch.argsort(ids_shuf, dim=-1)
        ids_keep = ids_shuf[:, :, :n_keep]
        x_keep = torch.gather(
            x, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, D))
        mask = torch.ones(B, C, N, device=x.device)
        mask[:, :, :n_keep] = 0
        mask = torch.gather(mask, 2, ids_rest)
        return x_keep, mask, ids_rest

    def encode(self, x):
        """
        Global embedding for downstream tasks.
        x: (B, 1, T, C)
        Returns: (B, enc_dim)
        """
        B, _, T, C = x.shape
        x = x.squeeze(1).permute(0, 2, 1)        # (B, C, T)
        tokens = self.patch_embed(x) + self.pos_enc
        tokens = self.encoder(tokens)
        tokens = self.enc_norm(tokens)
        return tokens.mean(dim=(1, 2))            # (B, enc_dim)

    def forward(self, x):
        B, _, T, C = x.shape
        x_in = x.squeeze(1).permute(0, 2, 1)     # (B, C, T)

        tokens = self.patch_embed(x_in) + self.pos_enc
        tokens, mask, ids_rest = self._mask(tokens)
        tokens = self.encoder(tokens)
        tokens = self.enc_norm(tokens)

        # Decode
        tokens = self.enc_to_dec(tokens)
        B2, C2, n_keep, D = tokens.shape
        N = self.n_patches
        mt = self.mask_token.expand(B2, C2, N - n_keep, D)
        full = torch.cat([tokens, mt], dim=2)
        full = torch.gather(
            full, 2, ids_rest.unsqueeze(-1).expand(-1, -1, -1, D))
        full = full + self.pos_dec
        full = self.decoder(full)
        full = self.dec_norm(full)
        pred = self.pred_head(full)               # (B, C, N, patch_t)
        return pred, mask

    def loss(self, x, pred, mask):
        B, _, T, C = x.shape
        x_in = x.squeeze(1).permute(0, 2, 1)     # (B, C, T)
        target = x_in.reshape(B, C, self.n_patches, self.patch_t)

        # Per-patch normalisation with variance floor
        # so silent patches don't explode the loss
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True).clamp(min=self.var_floor)
        target = (target - mean) / var.sqrt()

        m = mask.unsqueeze(-1)
        loss = ((pred - target) ** 2 * m).sum() / (
            m.sum() * self.patch_t + 1e-6)
        return loss


def build_das_mae(**kwargs):
    return DASTemporalMAE(**kwargs)
