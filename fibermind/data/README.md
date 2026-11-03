# fibermind.data

Readers, window selection and storage, and the dataset.

Interface:
- `SelectionConfig(win_t, win_c, n_keep, candidate_mult, event_frac, fs)`
- `score_candidates(dset, cfg, rng, sos)` -> list of `Candidate(spread, t0, c0)`; `dset` is anything with `.shape` and 2-D slicing
- `select_windows(cands, cfg, rng)` -> the kept candidates (top `event_frac` by spread plus random ambient), shuffled
- `write_windows(dset, selected, cfg, sos, out_dir, name)` -> npz paths
- `process_source(...)`, `process_h5_sources(...)` -> the above end to end, with `manifest.json`
- `DASWindowDataset(arr_dir)` -> items `{"w": (1, T, C) float32, "log_rms": (C,) float32}`

Invariants: same `(dset, cfg, seed)` selects the same windows. Storage is still per-window npz; decision 0003 replaces it with a memmap store. Tested in `tests/test_selection.py`.
