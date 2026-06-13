"""
DAS preprocessing with per-channel RMS factoring.
Chain: strain_rate -> cumsum/fs -> HPF 0.5Hz -> dynamic strain
Then factor each window into:
  w_norm  : per-channel RMS-normalised waveform (pure structure)
  log_rms : per-channel log RMS (spatial energy profile)
"""

import os
import gc
import json
import time
import numpy as np
import h5py
from scipy import signal

SOURCES = [
    {"path": "/workspace/data/raw/2023_02_14_00h00m28s_HDAS_1H_Strain.h5", "dset": "HDAS_DATA", "name": "adif_00h"},
    {"path": "/workspace/data/raw/2023_02_14_06h00m28s_HDAS_1H_Strain.h5", "dset": "HDAS_DATA", "name": "adif_06h"},
    {"path": "/workspace/data/raw/2023_02_14_12h00m28s_HDAS_1H_Strain.h5", "dset": "HDAS_DATA", "name": "adif_12h"},
    {"path": "/workspace/data/raw/2023_02_14_18h00m28s_HDAS_1H_Strain.h5", "dset": "HDAS_DATA", "name": "adif_18h"},
]

WIN_T = 512
WIN_C = 256
N_PER_SOURCE = 2000
OUT_DIR = "/data/arrays"
SEED = 42
FS = 100.0
EPS = 1e-20


def factor_window(dynamic):
    """dynamic: (T, C) -> w_norm (T, C) per-channel unit RMS, log_rms (C,)"""
    rms = np.sqrt(np.mean(dynamic ** 2, axis=0))
    rms_safe = np.maximum(rms, EPS)
    w_norm = dynamic / rms_safe[None, :]
    log_rms = np.log(rms_safe).astype(np.float32)
    return w_norm.astype(np.float32), log_rms


def process_source(spec, out_dir, n_windows, rng):
    os.makedirs(out_dir, exist_ok=True)
    name = spec["name"]
    print("=== " + name + " ===")
    sos = signal.butter(8, 0.5, btype="highpass", fs=FS, output="sos")
    with h5py.File(spec["path"], "r") as f:
        dset = f[spec["dset"]]
        T, C = dset.shape
        print("Shape: (" + str(T) + ", " + str(C) + ")")
        t0_run = time.time()
        for i in range(n_windows):
            ts = int(rng.integers(0, T - WIN_T))
            cs = int(rng.integers(0, C - WIN_C))
            raw = np.asarray(dset[ts:ts + WIN_T, cs:cs + WIN_C], dtype=np.float64)
            strain = np.cumsum(raw, axis=0) / FS
            dynamic = signal.sosfiltfilt(sos, strain, axis=0)
            w_norm, log_rms = factor_window(dynamic)
            out_path = os.path.join(out_dir, name + "_" + str(i).zfill(5) + ".npz")
            np.savez_compressed(out_path, w_norm=w_norm, log_rms=log_rms, t0=ts, c0=cs)
            if i % 200 == 0:
                elapsed = round(time.time() - t0_run, 1)
                print("  " + str(i) + "/" + str(n_windows) + " | " + str(elapsed) + "s")
            del raw, strain, dynamic, w_norm, log_rms
            gc.collect()
    elapsed = round((time.time() - t0_run) / 60, 1)
    print("  Done: " + str(n_windows) + " windows in " + str(elapsed) + " min")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for spec in SOURCES:
        process_source(spec, OUT_DIR, N_PER_SOURCE, rng)
    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
    print("Total: " + str(len(files)) + " windows in " + OUT_DIR)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "total": len(files),
            "sources": [s["name"] for s in SOURCES],
            "win_t": WIN_T, "win_c": WIN_C,
            "chain": "strain_rate->cumsum/fs->HPF->dynamic->per_channel_RMS_factor",
        }, f)


if __name__ == "__main__":
    main()
