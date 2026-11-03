"""Entry points import and the Hydra config composes. Full training runs are
GPU work and are not exercised here; the Lightning module is checked in-process
on CPU instead."""

import importlib.util
import pathlib
import sys

import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scripts_import():
    assert hasattr(_load(ROOT / "scripts" / "prepare_data.py"), "main")
    assert hasattr(_load(ROOT / "scripts" / "pretrain.py"), "main")


def test_config_composes():
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    assert cfg.model.win_t == 512 and cfg.model.use_cross_channel is True
    assert cfg.wandb.project == "fibermind-das"


def test_lightning_module_fits_one_step_on_cpu(tmp_path):
    from fibermind.training import MAEDataModule, MAELightningModule

    arr = tmp_path / "arrays"
    arr.mkdir()
    for i in range(6):
        w = np.random.randn(128, 16).astype(np.float32)
        w = w / np.sqrt((w**2).mean(axis=0, keepdims=True))
        np.savez(arr / f"w_{i}.npz", w_norm=w, log_rms=np.random.randn(16).astype(np.float32))

    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cfg.data.arr_dir = str(arr)
    cfg.data.val_split = 0.34
    cfg.model.update(
        dict(
            win_t=128,
            win_c=16,
            enc_dim=32,
            enc_depth=1,
            dec_dim=16,
            dec_depth=1,
            k_neighbours=4,
            global_downsample=4,
        )
    )
    cfg.training.update(dict(epochs=1, batch_size=2, num_workers=0))

    torch.manual_seed(0)
    model = MAELightningModule(cfg)
    data = MAEDataModule(cfg)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, data)
    assert trainer.state.finished
