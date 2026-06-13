import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from src.fibermind.models.das_mae import build_das_mae
from src.fibermind.data.das_dataset import DASWindowDataset


class MAELightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = build_das_mae(
            win_t=cfg.model.win_t,
            win_c=cfg.model.win_c,
            patch_t=cfg.model.patch_t,
            enc_dim=cfg.model.enc_dim,
            enc_depth=cfg.model.enc_depth,
            enc_heads=cfg.model.enc_heads,
            dec_dim=cfg.model.dec_dim,
            dec_depth=cfg.model.dec_depth,
            dec_heads=cfg.model.dec_heads,
            mask_ratio=cfg.model.mask_ratio,
            dropout=cfg.model.dropout,
            var_floor=cfg.model.var_floor,
            amplitude_weight=cfg.model.amplitude_weight,
            use_cross_channel=cfg.model.use_cross_channel,
            k_neighbours=cfg.model.k_neighbours,
            global_downsample=cfg.model.global_downsample,
        )

    def _step(self, batch, stage):
        w = batch["w"]
        log_rms = batch["log_rms"]
        pred, mask, amp_pred = self.model(w, log_rms)
        loss, struct, amp = self.model.loss(w, log_rms, pred, mask, amp_pred)
        self.log(stage + "/loss", loss, on_epoch=True, prog_bar=True)
        self.log(stage + "/struct", struct, on_epoch=True)
        if self.cfg.model.amplitude_weight > 0:
            self.log(stage + "/amp", amp, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        steps = len(self.trainer.datamodule.train_ds) // self.cfg.training.batch_size + 1
        scheduler = OneCycleLR(
            opt, max_lr=self.cfg.training.lr,
            epochs=self.cfg.training.epochs,
            steps_per_epoch=steps,
            pct_start=0.05, anneal_strategy="cos",
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]


class MAEDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        full = DASWindowDataset(arr_dir=self.cfg.data.arr_dir)
        n_val = max(1, int(len(full) * self.cfg.data.val_split))
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(full, [n_train, n_val])
        print("Train: " + str(n_train) + " | Val: " + str(n_val))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.training.batch_size,
                          shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.training.batch_size,
                          shuffle=False, num_workers=self.cfg.training.num_workers, pin_memory=True)
