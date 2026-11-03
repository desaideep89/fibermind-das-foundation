# fibermind.training

Lightning module (`MAELightningModule`), data module (`MAEDataModule`) and callbacks. Logged keys: `train/loss`, `train/struct`, `train/amp` and `val/` counterparts. Driven by `scripts/pretrain.py` with Hydra config from `configs/`.

`RcloneCheckpointBackup` still targets Google Drive; decision 0005 moves it to R2.
