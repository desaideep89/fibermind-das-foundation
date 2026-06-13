import torch
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from src.fibermind.training.trainer import MAELightningModule, MAEDataModule

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(42)

    model = MAELightningModule(cfg)
    data = MAEDataModule(cfg)

    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        save_dir="/tmp/wandb",
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="/tmp/checkpoints/",
            filename="mae-{epoch:03d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        default_root_dir="/tmp/lightning",
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
