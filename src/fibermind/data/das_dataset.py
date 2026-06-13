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
        w = z["w"]
        z.close()
        print("Dataset: " + str(len(self.files)) + " windows | shape: " + str(w.shape))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        w = z["w"].astype(np.float32)
        z.close()
        w = self._normalise_window(w)
        return torch.from_numpy(w).unsqueeze(0)  # (1, T, C)

    @staticmethod
    def _normalise_window(w, eps=1e-10):
        lo = np.percentile(w, 1)
        hi = np.percentile(w, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            m = np.max(np.abs(w)) + eps
            return np.clip(w / m, -1, 1).astype(np.float32)
        w = np.clip(w, lo, hi)
        w = 2 * (w - lo) / (hi - lo + eps) - 1
        return w.astype(np.float32)


# Keep MultiOEMDASDataset as alias for compatibility
MultiOEMDASDataset = DASWindowDataset
