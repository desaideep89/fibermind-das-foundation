"""Event-aware window selection and window writing.

Carried from the June 2026 ``preprocess.py`` with the same numerics and the
same random draw order, so a given (array, config, seed) selects the same
windows as before. The selection logic takes any object that supports
``obj.shape`` and 2-D slicing (an h5py dataset, a numpy array, a memmap), so
it is testable without HDF5 files.

Storage is still one compressed ``.npz`` per window in this version. Decision
0003 replaces that with a memory-mapped store in a later pull request.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from fibermind.physics import (
    DEFAULT_FS,
    energy_spread,
    factor_window,
    highpass_sos,
    log_rms_of,
    strain_rate_to_dynamic,
)


@dataclass(frozen=True)
class SelectionConfig:
    win_t: int = 512
    win_c: int = 256
    n_keep: int = 2000  # final windows kept per source
    candidate_mult: int = 4  # scan n_keep * candidate_mult candidates, then select
    event_frac: float = 0.7  # fraction of kept windows that are high-energy events
    fs: float = DEFAULT_FS


@dataclass(frozen=True)
class Candidate:
    spread: float
    t0: int
    c0: int


def score_candidates(
    dset, cfg: SelectionConfig, rng: np.random.Generator, sos: np.ndarray, log=print
) -> list[Candidate]:
    """Draw random window positions and score each by energy spread."""
    T, C = dset.shape
    n_cand = cfg.n_keep * cfg.candidate_mult
    cands: list[Candidate] = []
    t0_run = time.time()
    for i in range(n_cand):
        ts = int(rng.integers(0, T - cfg.win_t))
        cs = int(rng.integers(0, C - cfg.win_c))
        raw = np.asarray(dset[ts : ts + cfg.win_t, cs : cs + cfg.win_c], dtype=np.float64)
        dynamic = strain_rate_to_dynamic(raw, cfg.fs, sos)
        cands.append(Candidate(energy_spread(log_rms_of(dynamic)), ts, cs))
        if i % 500 == 0:
            log("  scan " + str(i) + "/" + str(n_cand) + " | " + str(round(time.time() - t0_run, 1)) + "s")
        del raw, dynamic
        gc.collect()
    return cands


def select_windows(
    cands: list[Candidate], cfg: SelectionConfig, rng: np.random.Generator, log=print
) -> list[Candidate]:
    """Keep the top event_frac by spread plus random ambient, then shuffle."""
    cands = sorted(cands, key=lambda c: c.spread, reverse=True)
    n_event = int(cfg.n_keep * cfg.event_frac)
    n_ambient = cfg.n_keep - n_event
    events = cands[:n_event]
    rest = cands[n_event:]
    rng.shuffle(rest)
    ambient = rest[:n_ambient]
    selected = events + ambient
    rng.shuffle(selected)
    log("  selected " + str(len(selected)) + " (" + str(n_event) + " events, " + str(n_ambient) + " ambient)")
    if events:
        log(
            "  event spread range: "
            + str(round(events[-1].spread, 2))
            + " to "
            + str(round(events[0].spread, 2))
        )
    return selected


def write_windows(
    dset, selected: list[Candidate], cfg: SelectionConfig, sos: np.ndarray, out_dir: str, name: str
) -> list[str]:
    """Preprocess each selected window and write it as a compressed npz."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for j, cand in enumerate(selected):
        raw = np.asarray(dset[cand.t0 : cand.t0 + cfg.win_t, cand.c0 : cand.c0 + cfg.win_c], dtype=np.float64)
        dynamic = strain_rate_to_dynamic(raw, cfg.fs, sos)
        w_norm, log_rms = factor_window(dynamic)
        out_path = os.path.join(out_dir, name + "_" + str(j).zfill(5) + ".npz")
        np.savez_compressed(
            out_path, w_norm=w_norm, log_rms=log_rms, t0=cand.t0, c0=cand.c0, spread=cand.spread
        )
        paths.append(out_path)
        del raw, dynamic, w_norm, log_rms
        gc.collect()
    return paths


def process_source(
    dset, name: str, cfg: SelectionConfig, out_dir: str, rng: np.random.Generator, log=print
) -> list[str]:
    """Score, select and write windows for one source array."""
    log("=== " + name + " ===")
    sos = highpass_sos(cfg.fs)
    T, C = dset.shape
    log(
        "Shape: ("
        + str(T)
        + ", "
        + str(C)
        + ") | scanning "
        + str(cfg.n_keep * cfg.candidate_mult)
        + " candidates"
    )
    t0_run = time.time()
    cands = score_candidates(dset, cfg, rng, sos, log=log)
    selected = select_windows(cands, cfg, rng, log=log)
    paths = write_windows(dset, selected, cfg, sos, out_dir, name)
    log("  Done in " + str(round((time.time() - t0_run) / 60, 1)) + " min")
    return paths


def process_h5_sources(sources: list[dict], cfg: SelectionConfig, out_dir: str, seed: int, log=print) -> dict:
    """Run process_source over a list of {path, dset, name} HDF5 specs and write a manifest."""
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for spec in sources:
        with h5py.File(spec["path"], "r") as f:
            process_source(f[spec["dset"]], spec["name"], cfg, out_dir, rng, log=log)
    files = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
    manifest = {
        "total": len(files),
        "sources": [s["name"] for s in sources],
        "win_t": cfg.win_t,
        "win_c": cfg.win_c,
        "event_frac": cfg.event_frac,
        "seed": seed,
        "selection": asdict(cfg),
        "chain": "strain_rate->cumsum/fs->HPF->dynamic->RMS_factor->event_aware_select",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    log("Total: " + str(len(files)) + " windows in " + out_dir)
    return manifest
