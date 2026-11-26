"""Band-limited phase synchronization measures for EEG epochs.

Input EEG shape: (epochs, channels, samples). Connectivity is computed
independently for each epoch. Use beta (13-30 Hz) with the manuscript's
4-30 Hz passband. Gamma above 30 Hz is unavailable after that filtering.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def bandpass_epochs(eeg, sampling_rate, low_hz, high_hz, order=4):
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected EEG shaped (epochs, channels, samples).")
    nyquist = float(sampling_rate) / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError("Band edges must lie strictly between 0 and Nyquist.")
    if x.shape[-1] < 16:
        raise ValueError("Epoch is too short for stable filtering.")
    sos = butter(order, [low_hz, high_hz], btype="bandpass",
                 fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _phase_difference(eeg, sampling_rate, low_hz, high_hz, order):
    filtered = bandpass_epochs(eeg, sampling_rate, low_hz, high_hz, order)
    analytic = hilbert(filtered, axis=-1)
    return np.angle(analytic)


def compute_plv(eeg, sampling_rate, low_hz=13.0, high_hz=30.0, order=4):
    """Return PLV matrices (epochs, channels, channels), in [0, 1]."""
    phase = _phase_difference(eeg, sampling_rate, low_hz, high_hz, order)
    n_channels = phase.shape[1]
    result = np.empty((phase.shape[0], n_channels, n_channels), dtype=np.float32)
    for e in range(phase.shape[0]):
        for i in range(n_channels):
            delta = phase[e, i][None, :] - phase[e]
            result[e, i] = np.abs(np.mean(np.exp(1j * delta), axis=-1))
    result = 0.5 * (result + np.swapaxes(result, 1, 2))
    for e in range(result.shape[0]):
        np.fill_diagonal(result[e], 1.0)
    return np.clip(result, 0.0, 1.0)


def compute_pli(eeg, sampling_rate, low_hz=13.0, high_hz=30.0, order=4):
    """Return PLI matrices (epochs, channels, channels), in [0, 1]."""
    phase = _phase_difference(eeg, sampling_rate, low_hz, high_hz, order)
    n_channels = phase.shape[1]
    result = np.zeros((phase.shape[0], n_channels, n_channels), dtype=np.float32)
    for e in range(phase.shape[0]):
        for i in range(n_channels):
            delta = phase[e, i][None, :] - phase[e]
            result[e, i] = np.abs(np.mean(np.sign(np.sin(delta)), axis=-1))
    result = 0.5 * (result + np.swapaxes(result, 1, 2))
    for e in range(result.shape[0]):
        np.fill_diagonal(result[e], 0.0)
    return np.clip(result, 0.0, 1.0)
