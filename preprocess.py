"""
DAS preprocessing with per-channel RMS factoring + event-aware sampling.

Chain: strain_rate -> cumsum/fs -> HPF 0.5Hz -> dynamic strain
Factor: w_norm (per-channel unit RMS structure) + log_rms (per-channel energy)

Event-aware sampling:
  Score each candidate window by energy spread = p90(log_rms) - median(log_rms).
  A train lights up a band of channels (high spread); ambient noise is flat (low spread).
  Keep the top EVENT_FRAC by spread, plus AMBIENT_FRAC random windows so the
  model still sees noise. Oversample candidates, then select.
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
N_KEEP_PER_SOURCE = 2000      # final windows kept per source
CANDIDATE_MULT = 4            # scan 4x candidates, then select
EVENT_FRAC = 0.7              # fraction of kept windows that are high-energy events
OUT_DIR = "/data/arrays"
SEED = 42
FS = 100.0
EPS = 1e-20


def factor_window(dynamic):
    rms = np.sqrt(np.mean(dynamic ** 2, axis=0))
    rms_safe = np.maximum(rms, EPS)
    w_norm = dynamic / rms_safe[None, :]
    log_rms = np.log(rms_safe).astype(np.float32)
    return w_norm.astype(np.float32), log_rms


def energy_spread(log_rms):
    """High when a band of channels is much hotter than the median (a real event)."""
    return float(np.percentile(log_rms, 90) - np.median(log_rms))


def process_source(spec, out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)
    name = spec["name"]
    print("=== " + name + " ===")
    sos = signal.butter(8, 0.5, btype="highpass", fs=FS, output="sos")

    n_cand = N_KEEP_PER_SOURCE * CANDIDATE_MULT
    with h5py.File(spec["path"], "r") as f:
        dset = f[spec["dset"]]
        T, C = dset.shape
        print("Shape: (" + str(T) + ", " + str(C) + ") | scanning " + str(n_cand) + " candidates")

        cands = []  # (spread, ts, cs)
        t0_run = time.time()
        for i in range(n_cand):
            ts = int(rng.integers(0, T - WIN_T))
            cs = int(rng.integers(0, C - WIN_C))
            raw = np.asarray(dset[ts:ts + WIN_T, cs:cs + WIN_C], dtype=np.float64)
            strain = np.cumsum(raw, axis=0) / FS
            dynamic = signal.sosfiltfilt(sos, strain, axis=0)
            rms = np.sqrt(np.mean(dynamic ** 2, axis=0))
            lr = np.log(np.maximum(rms, EPS))
            cands.append((energy_spread(lr), ts, cs))
            if i % 500 == 0:
                print("  scan " + str(i) + "/" + str(n_cand) + " | " + str(round(time.time() - t0_run, 1)) + "s")
            del raw, strain, dynamic
            gc.collect()

        # Select: top EVENT_FRAC by spread + remaining random ambient
        cands.sort(key=lambda x: x[0], reverse=True)
        n_event = int(N_KEEP_PER_SOURCE * EVENT_FRAC)
        n_ambient = N_KEEP_PER_SOURCE - n_event
        events = cands[:n_event]
        rest = cands[n_event:]
        rng.shuffle(rest)
        ambient = rest[:n_ambient]
        selected = events + ambient
        rng.shuffle(selected)
        print("  selected " + str(len(selected)) + " (" + str(n_event) + " events, " + str(n_ambient) + " ambient)")
        print("  event spread range: " + str(round(events[-1][0], 2)) + " to " + str(round(events[0][0], 2)))

        # Now write the selected windows
        for j, (spread, ts, cs) in enumerate(selected):
            raw = np.asarray(dset[ts:ts + WIN_T, cs:cs + WIN_C], dtype=np.float64)
            strain = np.cumsum(raw, axis=0) / FS
            dynamic = signal.sosfiltfilt(sos, strain, axis=0)
            w_norm, log_rms = factor_window(dynamic)
            out_path = os.path.join(out_dir, name + "_" + str(j).zfill(5) + ".npz")
            np.savez_compressed(out_path, w_norm=w_norm, log_rms=log_rms,
                                t0=ts, c0=cs, spread=spread)
            del raw, strain, dynamic, w_norm, log_rms
            gc.collect()
    print("  Done in " + str(round((time.time() - t0_run) / 60, 1)) + " min")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for spec in SOURCES:
        process_source(spec, OUT_DIR, rng)
    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
    print("Total: " + str(len(files)) + " windows in " + OUT_DIR)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "total": len(files),
            "sources": [s["name"] for s in SOURCES],
            "win_t": WIN_T, "win_c": WIN_C,
            "event_frac": EVENT_FRAC,
            "chain": "strain_rate->cumsum/fs->HPF->dynamic->RMS_factor->event_aware_select",
        }, f)


if __name__ == "__main__":
    main()
