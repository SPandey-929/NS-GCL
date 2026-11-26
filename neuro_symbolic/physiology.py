"""EEG physiological descriptors used by the neuro-symbolic loss.

Expected input is (epochs, channels, samples). Channel indices must refer to
the channel axis supplied by the caller. The functions return one scalar per
epoch. They do not infer emotion labels or physiological sign conventions.
"""
import numpy as np
from scipy.signal import welch
from graph.connectivity import compute_plv


def compute_frontal_alpha_asymmetry(
    eeg, sampling_rate, channel_names, left_channel="F3",
    right_channel="F4", alpha_band=(8.0, 13.0), epsilon=1e-8,
):
    """Compute FAA = ln(alpha PSD right) - ln(alpha PSD left).

    Alpha power is estimated with Welch's method and integrated over the
    requested band. With the default montage, F3 is left and F4 is right.
    The returned sign convention is explicit; it must not be silently
    interpreted as positive/negative valence without study-specific support.
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected EEG shaped (epochs, channels, samples).")
    names = [str(name).upper() for name in channel_names]
    if len(names) != x.shape[1]:
        raise ValueError("channel_names length must match the EEG channel axis.")
    if left_channel.upper() not in names or right_channel.upper() not in names:
        raise ValueError("Both requested frontal channels must be present.")
    low, high = map(float, alpha_band)
    if not (0 < low < high < sampling_rate / 2):
        raise ValueError("Alpha band must lie below the Nyquist frequency.")
    left_idx, right_idx = names.index(left_channel.upper()), names.index(right_channel.upper())
    nperseg = min(x.shape[-1], max(16, int(round(sampling_rate * 2))))
    frequencies, psd = welch(x[:, [left_idx, right_idx], :],
                             fs=sampling_rate, nperseg=nperseg, axis=-1)
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        raise ValueError("No Welch frequency bins fall inside the alpha band.")
    # NumPy 2.0 renamed trapz to trapezoid; retain compatibility with
    # the NumPy >=1.24 versions allowed by requirements.txt.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    power = integrate(psd[..., band], frequencies[band], axis=-1)
    left_power, right_power = power[:, 0], power[:, 1]
    return (np.log(right_power + epsilon) -
            np.log(left_power + epsilon)).astype(np.float32)


def compute_beta_plv_strength(
    eeg, sampling_rate, low_hz=13.0, high_hz=30.0,
    filter_order=4, edge_mask=None,
):
    """Return mean off-diagonal beta PLV per epoch, optionally over edges.

    edge_mask may be a symmetric (channels, channels) boolean array. If it is
    omitted, all unique channel pairs are averaged. Self-connections are
    excluded. The beta band must be supported by the input sampling rate and
    by the preprocessing applied to the EEG.
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected EEG shaped (epochs, channels, samples).")
    plv = compute_plv(x, sampling_rate, low_hz, high_hz, filter_order)
    n_channels = x.shape[1]
    if edge_mask is None:
        mask = np.triu(np.ones((n_channels, n_channels), dtype=bool), k=1)
    else:
        mask = np.asarray(edge_mask, dtype=bool)
        if mask.shape != (n_channels, n_channels):
            raise ValueError("edge_mask must have shape (channels, channels).")
        if not np.array_equal(mask, mask.T):
            raise ValueError("edge_mask must be symmetric.")
        mask = np.triu(mask, k=1)
    if not np.any(mask):
        raise ValueError("At least one off-diagonal edge must be selected.")
    return plv[:, mask].mean(axis=1).astype(np.float32)
