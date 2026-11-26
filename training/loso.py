"""Leakage-aware subject-wise splits and training-fold feature scaling."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LOSOFold:
    held_out_subject: object
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    scaler_fit_indices: np.ndarray

def make_loso_folds(subject_ids, validation_subject_count=1, random_state=42):
    """Hold out one test subject and one validation subject per fold."""
    subjects = np.asarray(subject_ids)
    if subjects.ndim != 1 or subjects.size == 0:
        raise ValueError("subject_ids must be a non-empty 1-D array.")
    unique = np.unique(subjects)
    if unique.size < 3:
        raise ValueError("At least three subjects are required for train/validation/test.")
    if validation_subject_count < 1 or validation_subject_count >= unique.size - 1:
        raise ValueError("validation_subject_count must leave at least one training subject.")
    rng = np.random.default_rng(random_state)
    for held_out in unique:
        test_idx = np.flatnonzero(subjects == held_out)
        remaining = unique[unique != held_out]
        val_subjects = rng.permutation(remaining)[:validation_subject_count]
        val_idx = np.flatnonzero(np.isin(subjects, val_subjects))
        train_idx = np.flatnonzero(~np.isin(subjects, np.concatenate(([held_out], val_subjects))))
        if not train_idx.size or not val_idx.size or not test_idx.size:
            raise RuntimeError("A LOSO fold produced an empty partition.")
        yield LOSOFold(held_out, train_idx, val_idx, test_idx, train_idx.copy())

class FeatureStandardizer:
    """Scalar standardization fitted only on training feature tensors."""
    def __init__(self, epsilon=1e-6):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.epsilon, self.mean_, self.scale_ = float(epsilon), None, None
    def fit(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 5 or x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("Expected finite non-empty 5-D time-frequency features.")
        # Per-electrode statistics: (1, 1, electrodes, 1, 1).
        self.mean_ = x.mean(axis=(0, 1, 3, 4), keepdims=True, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=(0, 1, 3, 4), keepdims=True, dtype=np.float64).astype(np.float32)
        self.scale_ = np.maximum(scale, self.epsilon)
        return self
    def transform(self, features):
        if self.mean_ is None:
            raise RuntimeError("Fit the standardizer on training data first.")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 5 or x.shape[2] != self.mean_.shape[2] or not np.isfinite(x).all():
            raise ValueError("Expected finite 5-D features with the fitted electrode count.")
        return ((x - self.mean_) / self.scale_).astype(np.float32)
    def fit_transform(self, features):
        return self.fit(features).transform(features)
