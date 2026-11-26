"""Prepare the official preprocessed DEAP MATLAB release for NS-GCL.

This function deliberately does not downsample, baseline-correct, or run ICA.
The standard preprocessed release is already 128 Hz and has provider-side
EOG processing and 4-45 Hz filtering. The manuscript's 4-30 Hz band is
applied as a narrower analysis filter; the first 3 seconds, when present,
are discarded as the pre-stimulus baseline (not subtracted).
"""
import numpy as np
from .deap_loader import load_deap_mat, select_channels, SELECTED_CHANNELS
from .filtering import bandpass_filter
from .segmentation import segment_trials


def prepare_preprocessed_deap_file(
    file_path,
    data_key="data",
    labels_key="labels",
    sampling_rate=128.0,
    low_hz=4.0,
    high_hz=30.0,
    window_seconds=3.0,
    channel_names=SELECTED_CHANNELS,
):
    """Return windows, repeated ratings, and (trial,window) metadata.

    Accepts standard 63-s (8064 samples at 128 Hz) trials and already-trimmed
    60-s (7680 samples) trials. For 63-s trials, removes only the first 3-s
    baseline segment; it does not perform baseline subtraction. Refuses other
    lengths to avoid silently misinterpreting the input release.
    """
    if not np.isclose(float(sampling_rate), 128.0):
        raise ValueError(
            "The preprocessed DEAP MATLAB release is expected at 128 Hz; "
            "this function never downsamples."
        )
    eeg, ratings = load_deap_mat(file_path, data_key=data_key, labels_key=labels_key)
    eeg = select_channels(eeg, channel_names)
    n_samples = eeg.shape[-1]
    baseline_samples = int(round(3.0 * sampling_rate))
    stimulus_samples = int(round(60.0 * sampling_rate))
    if n_samples == baseline_samples + stimulus_samples:
        eeg = eeg[..., baseline_samples:]
    elif n_samples == stimulus_samples:
        pass
    else:
        raise ValueError(
            f"Unexpected DEAP trial length: {n_samples} samples at "
            f"{sampling_rate:g} Hz. Expected 8064 (63 s incl. baseline) "
            "or 7680 (60 s stimulus only). Check the exact dataset release."
        )
    # Apply the manuscript's narrower 4-30 Hz analysis passband.
    # No second downsampling, baseline subtraction, or ICA is performed.
    filtered = bandpass_filter(eeg, sampling_rate, low_hz, high_hz)
    windows, labels, window_meta = segment_trials(
        filtered, ratings, sampling_rate, window_seconds, drop_incomplete=True
    )
    return windows, labels, window_meta
