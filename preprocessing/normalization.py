"""Channel-wise z-score normalization; fit using training subjects only."""
import numpy as np

class ChannelZScore:
    def __init__(self, epsilon=1e-8):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.mean_ = None
        self.scale_ = None

    def fit(self, eeg):
        x = np.asarray(eeg, dtype=np.float32)
        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError("Expected non-empty (segments, channels, time) training data")
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.scale_ = np.maximum(x.std(axis=(0, 2), keepdims=True), self.epsilon)
        return self

    def transform(self, eeg):
        if self.mean_ is None:
            raise RuntimeError("Fit on training-subject data before transform")
        x = np.asarray(eeg, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] != self.mean_.shape[1]:
            raise ValueError("Input channels differ from fitted training data")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, eeg):
        return self.fit(eeg).transform(eeg)
