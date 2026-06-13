"""
DAS dataset reading RMS-factored windows.
Returns w_norm (1, T, C) and log_rms (C,).
No further normalisation: w_norm is already per-channel unit RMS.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class DASWindowDataset(Dataset):
    def __init__(self, arr_dir="/data/arrays"):
        self.files = sorted(glob.glob(os.path.join(arr_dir, "*.npz")))
        assert len(self.files) > 0, "No npz files in " + arr_dir
        z = np.load(self.files[0])
        w = z["w_norm"]
        z.close()
        print("Dataset: " + str(len(self.files)) + " windows | shape: " + str(w.shape))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        w_norm = z["w_norm"].astype(np.float32)
        log_rms = z["log_rms"].astype(np.float32)
        z.close()
        # w_norm: (T, C) -> (1, T, C); log_rms: (C,)
        return {
            "w": torch.from_numpy(w_norm).unsqueeze(0),
            "log_rms": torch.from_numpy(log_rms),
        }


MultiOEMDASDataset = DASWindowDataset
