# fibermind.physics

Pure functions, no I/O, no state. Arrays are (T, C), time first.

Interface:
- `highpass_sos(fs, order, cutoff_hz)` -> second-order sections for the zero-phase high-pass
- `strain_rate_to_strain(x, fs)`, `strain_to_dynamic(strain, sos)`, `strain_rate_to_dynamic(x, fs, sos)` -> (T, C) float64
- `factor_window(dynamic)` -> `w_norm` (T, C) float32 unit RMS per channel, `log_rms` (C,) float32
- `log_rms_of(dynamic)` -> (C,) float64, `energy_spread(log_rms)` -> float

Invariants: `w_norm * exp(log_rms)` reconstructs `dynamic`; no decibel conversion anywhere. Tested in `tests/test_physics.py`.
