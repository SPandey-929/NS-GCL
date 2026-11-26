"""Loader for common trial-wise DEAP MATLAB files.

Common layout: data=(trials, 40, samples), labels=(trials, 4).
The first 32 channels are EEG and the remaining eight are peripheral signals.
Verify channel order against the precise DEAP release before use.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# Common DEAP 32-channel EEG order (EEG channels only).
DEAP_CHANNELS = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)
SELECTED_CHANNELS = (
    "Fp1", "Fp2", "F3", "Fz", "F4",
    "C3", "Cz", "C4", "P3", "Pz", "P4",
)


def load_deap_mat(file_path, data_key="data", labels_key="labels",
                  eeg_channels=32):
    """Load a participant .mat file; return EEG and ratings.

    Expected common data shape: (trials, 40, samples) or (trials, 32, samples).
    For 40-channel files, the first 32 channels are EEG and the final eight
    are peripheral signals. Returned labels retain their original ratings.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB file not found: {path}")
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if data_key not in mat or labels_key not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Expected {data_key!r} and {labels_key!r}; found {keys}")
    data = np.asarray(mat[data_key], dtype=np.float32)
    labels = np.asarray(mat[labels_key], dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D trial/channel/time data; got {data.shape}")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D trial/rating labels; got {labels.shape}")
    if data.shape[1] == 40:
        data = data[:, :eeg_channels, :]
    elif data.shape[1] != eeg_channels:
        raise ValueError(f"Expected {eeg_channels} EEG or 40 total channels; got {data.shape[1]}")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("Trial count in data and labels differs.")
    if not np.isfinite(data).all() or not np.isfinite(labels).all():
        raise ValueError("MATLAB data contain NaN or infinite values.")
    return data, labels


def select_channels(eeg, requested_channels=SELECTED_CHANNELS,
                    source_channel_names=DEAP_CHANNELS):
    """Select electrodes by explicit channel names, never guessed indices."""
    x = np.asarray(eeg)
    if x.ndim != 3 or x.shape[1] != len(source_channel_names):
        raise ValueError("EEG channel count does not match the supplied channel map.")
    missing = [name for name in requested_channels if name not in source_channel_names]
    if missing:
        raise ValueError(f"Requested electrodes absent from montage: {missing}")
    indices = [source_channel_names.index(name) for name in requested_channels]
    return np.asarray(x[:, indices, :], dtype=np.float32)
