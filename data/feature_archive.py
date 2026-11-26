"""Load prepared time-frequency feature archives for LOSO experiments."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_KEYS = ("features", "labels", "subject_ids", "trial_ids")

def load_feature_archive(path):
    """Load NPZ arrays: features (N,S,E,F,T), labels, subject/trial IDs, adjacency."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise KeyError(f"Feature archive is missing required keys: {missing}")
        data = {key: archive[key] for key in archive.files}
    features = np.asarray(data["features"], dtype=np.float32)
    labels, subjects, trials = np.asarray(data["labels"]), np.asarray(data["subject_ids"]), np.asarray(data["trial_ids"])
    if features.ndim != 5:
        raise ValueError("features must have shape (N, sequence, electrodes, freq, time).")
    n, seq, electrodes = features.shape[:3]
    if labels.shape != (n,) or subjects.shape != (n,) or trials.shape != (n,):
        raise ValueError("features, labels, subject_ids and trial_ids must share N samples.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs.")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values.")
    if "adjacency" not in data:
        raise KeyError("An explicit PLV/PLI adjacency matrix is required in the archive.")
    adjacency = np.asarray(data["adjacency"], dtype=np.float32)
    if adjacency.shape not in ((electrodes, electrodes), (n, electrodes, electrodes), (n, seq, electrodes, electrodes)):
        raise ValueError("adjacency must be (E,E), (N,E,E), or (N,sequence,E,E).")
    if not np.isfinite(adjacency).all():
        raise ValueError("adjacency contains NaN or infinite values.")
    data.update(features=features, labels=labels.astype(np.int64), subject_ids=subjects,
                trial_ids=trials, adjacency=adjacency)
    return data

class EEGFeatureDataset(Dataset):
    def __init__(self, archive, indices, features=None):
        self.archive = archive
        self.indices = np.asarray(indices, dtype=np.int64)
        self.features = archive["features"] if features is None else features
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        idx = int(self.indices[item])
        result = {
            "time_frequency": torch.as_tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(int(self.archive["labels"][idx]), dtype=torch.long),
            "subject_id": str(self.archive["subject_ids"][idx]),
            "trial_id": int(self.archive["trial_ids"][idx]),
        }
        adjacency = self.archive["adjacency"]
        result["adjacency"] = torch.as_tensor(adjacency if adjacency.ndim == 2 else adjacency[idx], dtype=torch.float32)
        for key in ("faa", "beta_plv", "arousal_labels"):
            if key in self.archive:
                result[key] = torch.as_tensor(self.archive[key][idx])
        result["archive_index"] = idx
        return result
