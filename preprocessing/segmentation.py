"""Non-overlapping segmentation for trial-wise EEG."""
import numpy as np


def segment_trials(eeg, labels, sampling_rate=128.0,
                   window_seconds=3.0, drop_incomplete=True):
    """Return windows, repeated trial labels, and (trial, window) metadata.

    EEG input: (trials, channels, samples).
    Output windows: (windows, channels, window_samples).
    """
    x = np.asarray(eeg, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim != 3:
        raise ValueError("Expected EEG shape (trials, channels, samples).")
    if y.ndim == 0 or y.shape[0] != x.shape[0]:
        raise ValueError("Labels must have one row per trial.")
    n_float = float(sampling_rate) * float(window_seconds)
    n_window = int(round(n_float))
    if sampling_rate <= 0 or window_seconds <= 0 or not np.isclose(n_float, n_window):
        raise ValueError("Invalid sampling rate/window: product must be an integer.")
    windows, window_labels, metadata = [], [], []
    for trial in range(x.shape[0]):
        n = x.shape[-1]
        count = n // n_window
        if not drop_incomplete and n % n_window:
            count += 1
        for w in range(count):
            start = w * n_window
            part = x[trial, :, start:start + n_window]
            if part.shape[-1] != n_window:
                if drop_incomplete:
                    continue
                padded = np.zeros((x.shape[1], n_window), dtype=np.float32)
                padded[:, :part.shape[-1]] = part
                part = padded
            windows.append(part)
            window_labels.append(y[trial])
            metadata.append((trial, w))
    if not windows:
        raise ValueError("No windows produced; check signal duration and window length.")
    return (np.stack(windows).astype(np.float32),
            np.asarray(window_labels),
            np.asarray(metadata, dtype=np.int64))
