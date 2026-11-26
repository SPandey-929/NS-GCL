"""Optional baseline correction for DEAP trials.

Enable only when the source file retains the initial baseline interval.
"""
import numpy as np


def baseline_correct_trials(eeg, sampling_rate=128.0, baseline_seconds=3.0,
                            remove_baseline=True):
    """Subtract each trial/channel baseline mean from stimulus EEG.

    Input shape: (trials, channels, samples). When remove_baseline is True,
    return only the post-baseline stimulus samples.
    """
    x = np.asarray(eeg, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("Expected (trials, channels, samples) EEG.")
    n_float = sampling_rate * baseline_seconds
    n_base = int(round(n_float))
    if sampling_rate <= 0 or baseline_seconds <= 0 or not np.isclose(n_float, n_base):
        raise ValueError("Baseline duration must map to a positive integer sample count.")
    if x.shape[-1] <= n_base:
        raise ValueError("Trial contains no samples after the baseline interval.")
    mean = x[..., :n_base].mean(axis=-1, keepdims=True)
    if remove_baseline:
        return (x[..., n_base:] - mean).astype(np.float32)
    return (x - mean).astype(np.float32)
