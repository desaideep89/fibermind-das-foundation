"""
Multi-OEM DAS dataset.
Samples contiguous (time x channel) tiles directly from H5 files at train time.
- Contiguous channel blocks preserve real spatial structure
- Balanced sampling across OEMs so neither swamps the other
- HPF applied per-tile on the fly
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from scipy import signal


SOURCES = [
    {"path": "/workspace/data/raw/10_44_45.h5",
     "dset": "Acquisition/Raw[0]/RawData", "oem": "sintela"},
    {"path": "/workspace/data/raw/2023_02_14_00h00m28s_HDAS_1H_Strain.h5",
     "dset": "HDAS_DATA", "oem": "adif"},
    {"path": "/workspace/data/raw/2023_02_14_01h00m28s_HDAS_1H_Strain.h5",
     "dset": "HDAS_DATA", "oem": "adif"},
]


def _highpass_sos(cutoff=0.5, order=8, fs=100.0):
    return signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")


class MultiOEMDASDataset(Dataset):
    def __init__(self, win_t=512, win_c=256, n_per_oem=8000,
                 fs_hz=100.0, hp_cutoff=0.5, seed=42):
        self.win_t = win_t
        self.win_c = win_c
        self.fs_hz = fs_hz
        self.sos = _highpass_sos(hp_cutoff, 8, fs_hz)

        # Build shape index without loading data
        self.meta = []
        for s in SOURCES:
            with h5py.File(s["path"], "r") as f:
                T, C = f[s["dset"]].shape
            self.meta.append({**s, "T": T, "C": C})
            print(f"{s['oem']:8s} {s['path'].split('/')[-1]:45s} ({T}, {C})")

        # Balanced sample plan: equal windows per OEM
        oems = sorted(set(m["oem"] for m in self.meta))
        rng = np.random.default_rng(seed)
        self.samples = []  # (source_idx, t0, c0)
        for oem in oems:
            src_idxs = [i for i, m in enumerate(self.meta) if m["oem"] == oem]
            for _ in range(n_per_oem):
                si = int(rng.choice(src_idxs))
                m = self.meta[si]
                t0 = int(rng.integers(0, m["T"] - win_t))
                c0 = int(rng.integers(0, m["C"] - win_c))
                self.samples.append((si, t0, c0))
        rng.shuffle(self.samples)
        print(f"Total windows: {len(self.samples)} "
              f"({n_per_oem} per OEM x {len(oems)} OEMs)")

        # Lazy per-worker file handles
        self._handles = {}

    def _get_handle(self, si):
        if si not in self._handles:
            m = self.meta[si]
            self._handles[si] = h5py.File(m["path"], "r")[m["dset"]]
        return self._handles[si]

    def __len__(self):
        return len(self.samples)

    def _normalise(self, x, eps=1e-6):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            m = np.max(np.abs(x)) + eps
            return np.clip(x / m, -1, 1).astype(np.float32)
        y = np.clip(x, lo, hi)
        return (2 * (y - lo) / (hi - lo + eps) - 1).astype(np.float32)

    def __getitem__(self, idx):
        si, t0, c0 = self.samples[idx]
        dset = self._get_handle(si)
        raw = np.asarray(dset[t0:t0+self.win_t, c0:c0+self.win_c],
                         dtype=np.float32)
        hp = signal.sosfiltfilt(self.sos, raw, axis=0).astype(np.float32)
        hp = self._normalise(hp)
        return torch.from_numpy(hp).unsqueeze(0)  # (1, T, C)
