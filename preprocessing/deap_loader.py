"""Loader for common trial-wise DEAP MATLAB files.

Common layout: data=(trials, 40, samples), labels=(trials, 4).
The first 32 channels are EEG and the remaining eight are peripheral signals.
Channel order must be checked against the actual data release before selection.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# Standard 32-channel montage used by the commonly distributed DEAP release.
# Note: this list does not contain Fz or Cz.
DEAP_CHANNELS = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "CPz", "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4",
    "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)
SELECTED_CHANNELS = (
    "Fp1", "Fp2", "F3", "Fz", "F4",
    "C3", "Cz", "C4", "P3", "Pz", "P4",
)


def load_deap_mat(file_path, data_key="data", labels_key="labels",
                  eeg_channels=32):
    """Load a participant .mat file and return EEG and trial ratings.

    Returns EEG as (trials, eeg_channels, samples), labels as (trials, ratings).
    Supports the common DEAP MATLAB array layout. Other layouts raise an
    informative error rather than being reshaped by guesswork.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB file not found: {path}")
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if data_key not in mat or labels_key not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Expected keys {data_key!r} and {labels_key!r}; found {keys}")
    data = np.asarray(mat[data_key], dtype=np.float32)
    labels = np.asarray(mat[labels_key], dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(
            f"Expected data with 3 dimensions (trials, channels, samples); got {data.shape}. "
            "Inspect the .mat file before changing axis order."
        )
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D trial labels; got {labels.shape}.")
    # Standard files have 40 channels: 32 EEG followed by 8 peripheral.
    if data.shape[1] == 40:
        data = data[:, :eeg_channels, :]
    elif data.shape[1] != eeg_channels:
        raise ValueError(
            f"Expected {eeg_channels} EEG channels or 40 total channels; got {data.shape[1]}."
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Trial count in data and labels does not match.")
    if not np.isfinite(data).all() or not np.isfinite(labels).all():
        raise ValueError("MATLAB data contain NaN or infinite values.")
    return data, labels


def select_channels(eeg, requested_channels=SELECTED_CHANNELS,
                    source_channel_names=DEAP_CHANNELS):
    """Select channels by name; never silently guess electrode indices."""
    x = np.asarray(eeg)
    if x.ndim != 3:
        raise ValueError("Expected EEG shape (trials, channels, samples).")
    if x.shape[1] != len(source_channel_names):
        raise ValueError(
            f"Data contain {x.shape[1]} channels but the supplied channel map "
            f"contains {len(source_channel_names)} names."
        )
    missing = [name for name in requested_channels if name not in source_channel_names]
    if missing:
        raise ValueError(
            f"Requested electrodes {missing} are absent from the supplied montage. "
            "Confirm the actual DEAP file montage/channel order. The standard "
            "32-channel DEAP montage does not include Fz or Cz."
        )
    indices = [source_channel_names.index(name) for name in requested_channels]
    return np.asarray(x[:, indices, :], dtype=np.float32)
