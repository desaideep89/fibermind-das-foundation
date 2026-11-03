"""The preprocessing chain is deterministic, invertible where it should be,
and does what the physics says on a known signal."""

import numpy as np

from fibermind.physics import (
    channel_rms,
    energy_spread,
    factor_window,
    highpass_sos,
    log_rms_of,
    strain_rate_to_dynamic,
    strain_rate_to_strain,
    strain_to_dynamic,
)


def test_integration_recovers_known_strain():
    """cumsum/fs is a rectangle rule, so it reproduces the strain sampled half a
    sample late; against that reference the residual is second order in dt."""
    fs = 100.0
    f = 2.0
    k = np.arange(1000)
    strain_rate = (2 * np.pi * f * np.cos(2 * np.pi * f * k / fs))[:, None]
    rec = strain_rate_to_strain(strain_rate, fs)
    ref = np.sin(2 * np.pi * f * (k + 0.5) / fs)[:, None]
    assert np.abs((rec - rec.mean()) - (ref - ref.mean())).max() < 2e-3


def test_highpass_removes_drift_and_keeps_signal():
    fs = 100.0
    t = np.arange(4000) / fs
    drift = 0.5 * t
    tone = np.sin(2 * np.pi * 5.0 * t)
    x = (drift + tone)[:, None]
    y = strain_to_dynamic(x, highpass_sos(fs))
    mid = slice(500, 3500)
    assert np.abs(y[mid, 0] - tone[mid]).max() < 0.05
    assert np.abs(y[mid, 0].mean()) < 0.01


def test_chain_is_deterministic(synthetic_strain_rate):
    sos = highpass_sos(100.0)
    a = strain_rate_to_dynamic(synthetic_strain_rate, 100.0, sos)
    b = strain_rate_to_dynamic(synthetic_strain_rate, 100.0, sos)
    assert np.array_equal(a, b)


def test_factor_window_is_unit_rms_and_invertible(synthetic_strain_rate):
    sos = highpass_sos(100.0)
    dyn = strain_rate_to_dynamic(synthetic_strain_rate, 100.0, sos)
    w_norm, log_rms = factor_window(dyn)
    assert w_norm.dtype == np.float32 and log_rms.dtype == np.float32
    assert w_norm.shape == dyn.shape and log_rms.shape == (dyn.shape[1],)
    assert np.allclose(channel_rms(w_norm.astype(np.float64)), 1.0, atol=1e-4)
    rebuilt = w_norm.astype(np.float64) * np.exp(log_rms.astype(np.float64))[None, :]
    assert np.allclose(rebuilt, dyn, rtol=1e-4, atol=1e-12)


def test_log_rms_of_matches_factor_window(synthetic_strain_rate):
    sos = highpass_sos(100.0)
    dyn = strain_rate_to_dynamic(synthetic_strain_rate, 100.0, sos)
    _, lr_a = factor_window(dyn)
    lr_b = log_rms_of(dyn)
    assert np.allclose(lr_a, lr_b.astype(np.float32))


def test_energy_spread_separates_event_from_ambient(synthetic_strain_rate, rng):
    sos = highpass_sos(100.0)
    event = synthetic_strain_rate[500:1012]
    ambient = rng.standard_normal((512, 64)) * 1e-9
    s_event = energy_spread(log_rms_of(strain_rate_to_dynamic(event, 100.0, sos)))
    s_amb = energy_spread(log_rms_of(strain_rate_to_dynamic(ambient, 100.0, sos)))
    assert s_event > s_amb + 0.5
