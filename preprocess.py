import os, gc, json, time
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
            raw = np.asarray(dset[ts:ts+WIN_T, cs:cs+WIN_C], dtype=np.float64)
            strain = np.cumsum(raw, axis=0) / FS
            dynamic = signal.sosfiltfilt(sos, strain, axis=0).astype(np.float32)
            out_path = os.path.join(out_dir, name + "_" + str(i).zfill(5) + ".npz")
            np.savez_compressed(out_path, w=dynamic, t0=ts, c0=cs)
            if i % 200 == 0:
                elapsed = time.time() - t0_run
                print("  " + str(i) + "/" + str(n_windows) + " | " + str(round(elapsed, 1)) + "s")
            del raw, strain, dynamic
            gc.collect()
    elapsed = (time.time() - t0_run) / 60
    print("  Done: " + str(n_windows) + " windows in " + str(round(elapsed, 1)) + " min")


os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

for spec in SOURCES:
    process_source(spec, OUT_DIR, N_PER_SOURCE, rng)

files = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
print("Total: " + str(len(files)) + " windows in " + OUT_DIR)

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump({"total": len(files), "sources": [s["name"] for s in SOURCES],
               "chain": "strain_rate->cumsum/fs->HPF_0.5Hz->dynamic_strain"}, f)
