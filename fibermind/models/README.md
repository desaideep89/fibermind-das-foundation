# fibermind.models

Encoder blocks and the MAE model. Token tensors are (B, C, N, D).

Interface:
- `TemporalPatchEmbed(patch_t, dim)`: (B, C, T) -> (B, C, N, D)
- `TemporalEncoder(dim, depth, heads)`: per-channel transformer over N (Stage A)
- `CrossChannelBlock(dim, heads, k, downsample_to)`: mixes across C at each N (Stage B); requires a complete channel grid per time patch
- `DASTemporalMAE(...)` with `embed`, `encode`, `forward` -> `(pred, mask, amp_pred)`, `loss(...)` -> `(total, struct, amp)`
- `build_das_mae(**kwargs)`

State-dict keys are unchanged from June 2026, so old checkpoints load. Tested in `tests/test_model.py`.
