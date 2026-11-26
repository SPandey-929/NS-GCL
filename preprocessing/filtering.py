"""Zero-phase Butterworth band-pass filtering."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(eeg, sampling_rate=128.0, low_hz=4.0,
                    high_hz=30.0, order=4):
    """Filter along the final time axis of an EEG array."""
    x = np.asarray(eeg, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError("EEG must include channel and time dimensions.")
    nyquist = float(sampling_rate) / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Require 0 < low_hz < high_hz < Nyquist ({nyquist:g} Hz)."
        )
    if order < 1:
        raise ValueError("Filter order must be positive.")
    sos = butter(order, [low_hz, high_hz], btype="bandpass",
                 fs=sampling_rate, output="sos")
    try:
        y = sosfiltfilt(sos, x, axis=-1)
    except ValueError as exc:
        raise ValueError("Signal is too short for the selected zero-phase filter.") from exc
    return y.astype(np.float32, copy=False)
