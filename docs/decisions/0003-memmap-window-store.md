# 0003: Memory-mapped window store instead of per-window compressed files

Status: accepted, 2026-11-03

## Context

The June `preprocess.py` writes one compressed `.npz` per window, two thousand per source, eight thousand in total. Loading them is dominated by file opens and decompression, the DataLoader cannot read a contiguous batch, and the layout cannot be hashed or described as a single artefact. The v0.3 loader already read from a single memory-mapped array per source and was much faster on the pod.

## Decision

One memory-mapped array per source: `w_norm` as `float32` `(n_windows, win_t, win_c)`, `log_rms` as `float32` `(n_windows, win_c)`, and a side table `(t0, c0, spread)` per window, stored as `.npy` files written with `numpy.lib.format.open_memmap` so they can be read without loading. One manifest per source names all three files with their hashes.

## Alternatives considered

- Keep npz per window. Rejected for the throughput and bookkeeping reasons above.
- HDF5 with chunking. Rejected for now: adds a dependency on the read path, chunk tuning, and lock issues with multiple DataLoader workers. Acceptable later for very large stores.
- Zarr. Reasonable, and would be the choice for cloud-native reading straight from R2. Deferred until reading from the bucket without a local copy is actually a requirement.

## Consequences

Preprocessing writes windows in a known order with a fixed count, so the array is preallocated. Splits are index ranges or index lists over the array, stored in the split manifest. `float32` at the current window size (512 by 256) is 512 KiB per window, so 8,000 windows is about 4 GiB uncompressed; the bucket cost is acceptable and local disk on the pod is the constraint to watch (`/data/arrays`, not the workspace mount).
