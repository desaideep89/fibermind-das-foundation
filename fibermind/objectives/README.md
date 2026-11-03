# fibermind.objectives

Masking strategies and losses, independent of any model.

Interface:
- `random_patch_mask(x, mask_ratio)` -> `MaskResult(x_keep, mask, ids_restore)`
- `restore_with_mask_tokens(x_keep, mask_token, ids_restore, n_patches)` -> (B, C, N, D)
- `patch_targets(w, n_patches, patch_t, var_floor)`, `masked_reconstruction_loss(pred, target, mask, patch_t)`, `amplitude_loss(amp_pred, log_rms)`

Decision 0004: tube masking is to be added here and made the only strategy allowed with a cross-channel block. Until then `tests/test_masking.py::test_stage_b_sees_simultaneous_tokens` is an expected failure.
