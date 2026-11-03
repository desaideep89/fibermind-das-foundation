"""Dataset over preprocessed windows.

Returns ``w`` of shape (1, T, C) (``w_norm`` with a leading singleton) and
``log_rms`` of shape (C,). No further normalisation: ``w_norm`` is already unit
RMS per channel. Unchanged in behaviour from the June 2026 ``das_dataset.py``.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DASWindowDataset(Dataset):
    def __init__(self, arr_dir: str = "/data/arrays", log=print):
        self.files = sorted(glob.glob(os.path.join(arr_dir, "*.npz")))
        assert len(self.files) > 0, "No npz files in " + arr_dir
        z = np.load(self.files[0])
        w = z["w_norm"]
        z.close()
        log("Dataset: " + str(len(self.files)) + " windows | shape: " + str(w.shape))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        z = np.load(self.files[idx])
        w_norm = z["w_norm"].astype(np.float32)
        log_rms = z["log_rms"].astype(np.float32)
        z.close()
        return {
            "w": torch.from_numpy(w_norm).unsqueeze(0),
            "log_rms": torch.from_numpy(log_rms),
        }


MultiOEMDASDataset = DASWindowDataset
