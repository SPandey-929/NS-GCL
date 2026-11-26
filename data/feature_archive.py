"""Validated loader for NS-GCL NPZ feature archives."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_KEYS = ("features", "labels", "subject_ids", "trial_ids", "adjacency", "metadata_json")


def load_feature_archive(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z]
        if missing:
            raise KeyError(f"Feature archive is missing required keys: {missing}")
        data = {k: z[k] for k in z.files}
    x = np.asarray(data["features"], dtype=np.float32)
    y, subjects, trials = data["labels"], data["subject_ids"], data["trial_ids"]
    if x.ndim != 5 or any(d < 1 for d in x.shape):
        raise ValueError("features must be non-empty (N,sequence,electrodes,freq,time).")
    n, seq, electrodes = x.shape[:3]
    if y.shape != (n,) or subjects.shape != (n,) or trials.shape != (n,):
        raise ValueError("features, labels, subject_ids and trial_ids must share N.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs.")
    if not np.issubdtype(trials.dtype, np.integer):
        raise ValueError("trial_ids must contain integer IDs.")
    if subjects.dtype.kind not in ("U", "S", "i", "u"):
        raise ValueError("subject_ids must be integer or string identifiers.")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinite values.")
    adj = np.asarray(data["adjacency"], dtype=np.float32)
    if adj.shape not in ((electrodes,electrodes),(n,electrodes,electrodes),(n,seq,electrodes,electrodes)):
        raise ValueError("adjacency must be (E,E), (N,E,E), or (N,sequence,E,E).")
    if not np.isfinite(adj).all():
        raise ValueError("adjacency contains NaN or infinite values.")
    raw_meta = np.asarray(data["metadata_json"])
    if raw_meta.ndim != 0 or raw_meta.dtype.kind not in ("U","S"):
        raise ValueError("metadata_json must be a scalar UTF-8 JSON string.")
    try:
        meta = json.loads(str(raw_meta.item()))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json must contain valid JSON.") from exc
    if not isinstance(meta, dict):
        raise ValueError("metadata_json must decode to a JSON object.")
    required_meta = ("schema_version","provenance","dataset_name","feature_method",
                     "channel_names","sampling_rate_hz","preprocessing","random_seed")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        raise ValueError(f"metadata_json is missing required fields: {missing}")
    if meta["schema_version"] != "1.0":
        raise ValueError("Unsupported archive schema_version; expected '1.0'.")
    if meta["provenance"] not in ("synthetic","real"):
        raise ValueError("provenance must be 'synthetic' or 'real'.")
    if not isinstance(meta["dataset_name"],str) or not meta["dataset_name"].strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(meta["feature_method"],str) or not meta["feature_method"].strip():
        raise ValueError("feature_method must be a non-empty string.")
    if not isinstance(meta["channel_names"],list) or len(meta["channel_names"]) != electrodes:
        raise ValueError("channel_names must be a list matching the electrode dimension.")
    if not all(isinstance(v,str) and v.strip() for v in meta["channel_names"]):
        raise ValueError("channel_names must contain non-empty strings.")
    if not isinstance(meta["preprocessing"],(dict,list)):
        raise ValueError("preprocessing must be a JSON object or list.")
    rate = meta["sampling_rate_hz"]
    if rate is not None and (not isinstance(rate,(int,float)) or rate <= 0):
        raise ValueError("sampling_rate_hz must be positive or null.")
    seed = meta["random_seed"]
    if seed is not None and (not isinstance(seed,int) or seed < 0):
        raise ValueError("random_seed must be a non-negative integer or null.")
    if meta["provenance"] == "synthetic" and not meta.get("synthetic_data_note"):
        raise ValueError("Synthetic archives require synthetic_data_note.")
    if meta["provenance"] == "real" and meta["dataset_name"].lower() == "synthetic":
        raise ValueError("A real archive cannot be named synthetic.")
    data.update(features=x, labels=y.astype(np.int64), subject_ids=subjects,
                trial_ids=trials.astype(np.int64), adjacency=adj, metadata=meta)
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
        result["adjacency"] = torch.as_tensor(
            adjacency if adjacency.ndim == 2 else adjacency[idx], dtype=torch.float32)
        for key in ("faa","beta_plv","arousal_labels"):
            if key in self.archive:
                result[key] = torch.as_tensor(self.archive[key][idx])
        result["archive_index"] = idx
        return result
