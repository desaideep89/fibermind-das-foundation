"""The DAS preprocessing chain as pure functions.

Chain (docs/ARCHITECTURE.md section 4.2):

    strain_rate -> cumsum / fs -> high-pass (Butterworth, zero-phase) -> dynamic strain
    dynamic strain -> w_norm (per-channel unit RMS) + log_rms (per-channel energy)

Conventions: arrays are (T, C), time first. No decibel conversion anywhere.
Numerics are unchanged from the June 2026 ``preprocess.py``; the equivalence
check in the refactor pull request proves this bitwise on synthetic input.

Nothing in this module reads or writes files or holds state, so it can be
tested in isolation and handed to an outside contributor with a fixture.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-20
DEFAULT_FS = 100.0
DEFAULT_HPF_ORDER = 8
DEFAULT_HPF_CUTOFF_HZ = 0.5


def highpass_sos(
    fs: float = DEFAULT_FS, order: int = DEFAULT_HPF_ORDER, cutoff_hz: float = DEFAULT_HPF_CUTOFF_HZ
) -> np.ndarray:
    """Second-order sections for the zero-phase high-pass used on strain."""
    return signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")


def strain_rate_to_strain(strain_rate: np.ndarray, fs: float) -> np.ndarray:
    """Integrate strain rate along time. Input (T, C), float64 recommended."""
    return np.cumsum(strain_rate, axis=0) / fs


def strain_to_dynamic(strain: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase high-pass along time, removing drift from the integration."""
    return signal.sosfiltfilt(sos, strain, axis=0)


def strain_rate_to_dynamic(strain_rate: np.ndarray, fs: float, sos: np.ndarray) -> np.ndarray:
    """Full chain from strain rate to dynamic strain, (T, C) in, (T, C) out."""
    return strain_to_dynamic(strain_rate_to_strain(strain_rate, fs), sos)


def channel_rms(dynamic: np.ndarray) -> np.ndarray:
    """Per-channel RMS over the window, shape (C,)."""
    return np.sqrt(np.mean(dynamic**2, axis=0))


def factor_window(dynamic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a dynamic-strain window into structure and amplitude.

    Returns ``w_norm`` (T, C) float32 with unit RMS per channel, and
    ``log_rms`` (C,) float32, the natural log of the per-channel RMS.
    """
    rms_safe = np.maximum(channel_rms(dynamic), EPS)
    w_norm = dynamic / rms_safe[None, :]
    log_rms = np.log(rms_safe).astype(np.float32)
    return w_norm.astype(np.float32), log_rms


def log_rms_of(dynamic: np.ndarray) -> np.ndarray:
    """Per-channel log RMS without producing w_norm (used when scoring candidates)."""
    return np.log(np.maximum(channel_rms(dynamic), EPS))


def energy_spread(log_rms: np.ndarray) -> float:
    """High when a band of channels is much hotter than the median (a real event)."""
    return float(np.percentile(log_rms, 90) - np.median(log_rms))
