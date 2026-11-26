"""Optional ICA utility; artifact components must be selected by the researcher."""
import numpy as np
from sklearn.decomposition import FastICA

def apply_manual_ica(eeg, exclude_components, random_state=42, max_iter=1000):
    """Apply ICA to (trials, channels, samples) with explicit exclusions.

    This function does not automatically identify artifacts. If no components
    are supplied, it returns a copy and None (no ICA rejection is performed).
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite EEG shaped (trials, channels, samples)")
    if not exclude_components:
        return x.astype(np.float32), None
    trials, channels, samples = x.shape
    continuous = np.concatenate([x[i] for i in range(trials)], axis=1).T
    ica = FastICA(n_components=channels, whiten="unit-variance",
                  random_state=random_state, max_iter=max_iter, tol=1e-4)
    sources = ica.fit_transform(continuous)
    excluded = sorted(set(int(i) for i in exclude_components))
    if any(i < 0 or i >= sources.shape[1] for i in excluded):
        raise ValueError("ICA component index out of range")
    sources[:, excluded] = 0.0
    restored = ica.inverse_transform(sources).T
    restored = restored.reshape(channels, trials, samples).transpose(1, 0, 2)
    return restored.astype(np.float32), ica
