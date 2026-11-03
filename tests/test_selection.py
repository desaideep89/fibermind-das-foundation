"""Window selection is reproducible from (array, config, seed) and writes what
the dataset expects."""

import json
import os

import numpy as np

from fibermind.data import DASWindowDataset, SelectionConfig, process_source, score_candidates, select_windows
from fibermind.physics import highpass_sos

CFG = SelectionConfig(win_t=128, win_c=16, n_keep=12, candidate_mult=3, event_frac=0.5, fs=100.0)


def _quiet(*_):
    pass


def test_selection_is_reproducible(synthetic_strain_rate):
    sos = highpass_sos(CFG.fs)
    a = select_windows(
        score_candidates(synthetic_strain_rate, CFG, np.random.default_rng(7), sos, log=_quiet),
        CFG,
        np.random.default_rng(7),
        log=_quiet,
    )
    b = select_windows(
        score_candidates(synthetic_strain_rate, CFG, np.random.default_rng(7), sos, log=_quiet),
        CFG,
        np.random.default_rng(7),
        log=_quiet,
    )
    assert a == b
    assert len(a) == CFG.n_keep


def test_selection_keeps_event_fraction(synthetic_strain_rate):
    sos = highpass_sos(CFG.fs)
    rng = np.random.default_rng(3)
    cands = score_candidates(synthetic_strain_rate, CFG, rng, sos, log=_quiet)
    selected = select_windows(cands, CFG, rng, log=_quiet)
    top = sorted(cands, key=lambda c: c.spread, reverse=True)[: int(CFG.n_keep * CFG.event_frac)]
    assert set(top).issubset(set(selected))


def test_process_source_writes_dataset_compatible_files(tmp_path, synthetic_strain_rate):
    out = str(tmp_path / "arrays")
    paths = process_source(synthetic_strain_rate, "syn", CFG, out, np.random.default_rng(1), log=_quiet)
    assert len(paths) == CFG.n_keep
    z = np.load(paths[0])
    assert z["w_norm"].shape == (CFG.win_t, CFG.win_c)
    assert z["log_rms"].shape == (CFG.win_c,)
    ds = DASWindowDataset(out, log=_quiet)
    item = ds[0]
    assert item["w"].shape == (1, CFG.win_t, CFG.win_c)
    assert item["log_rms"].shape == (CFG.win_c,)
    assert item["w"].dtype.is_floating_point


def test_h5_path_writes_manifest(tmp_path, synthetic_strain_rate):
    import h5py

    from fibermind.data import process_h5_sources

    h5 = tmp_path / "raw.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("HDAS_DATA", data=synthetic_strain_rate)
    out = str(tmp_path / "arrays")
    m = process_h5_sources(
        [{"path": str(h5), "dset": "HDAS_DATA", "name": "syn"}], CFG, out, seed=5, log=_quiet
    )
    assert m["total"] == CFG.n_keep
    assert json.load(open(os.path.join(out, "manifest.json")))["seed"] == 5
