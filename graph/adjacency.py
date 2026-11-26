"""Convert functional-connectivity matrices into graph adjacencies."""
import numpy as np


def build_adjacency(connectivity, threshold=0.0, top_k=None,
                    keep_self_loops=False, binarize=False):
    """Build symmetric weighted adjacency matrices.

    connectivity: (channels, channels) or (epochs, channels, channels).
    threshold removes edges below the supplied connectivity value.
    top_k retains at most k strongest neighbours per node before symmetrizing.
    """
    a = np.asarray(connectivity, dtype=np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[None, ...]
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("Expected square (channels, channels) connectivity matrices.")
    if not np.isfinite(a).all():
        raise ValueError("Connectivity contains NaN or infinite values.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if top_k is not None and not (1 <= int(top_k) < a.shape[1]):
        raise ValueError("top_k must be between 1 and channels-1.")
    out = np.maximum(0.0, 0.5 * (a + np.swapaxes(a, 1, 2)))
    for e in range(out.shape[0]):
        np.fill_diagonal(out[e], 0.0)
        out[e][out[e] < threshold] = 0.0
        if top_k is not None:
            keep = np.zeros_like(out[e], dtype=bool)
            for i in range(out.shape[1]):
                candidates = np.argsort(out[e, i])[-int(top_k):]
                keep[i, candidates] = True
            keep = np.logical_or(keep, keep.T)
            out[e] *= keep
        if binarize:
            out[e] = (out[e] > 0).astype(np.float32)
        if keep_self_loops:
            np.fill_diagonal(out[e], 1.0)
    return out[0] if squeeze else out


def normalize_adjacency(adjacency, add_self_loops=True, symmetric=True,
                        epsilon=1e-8):
    """Apply D^-1/2 (A+I) D^-1/2 or row normalization."""
    a = np.asarray(adjacency, dtype=np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[None, ...]
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("Expected square adjacency matrices.")
    if np.any(a < 0) or not np.isfinite(a).all():
        raise ValueError("Adjacency must be finite and non-negative.")
    out = a.copy()
    if add_self_loops:
        idx = np.arange(out.shape[1])
        out[:, idx, idx] += 1.0
    degree = out.sum(axis=-1)
    if symmetric:
        inv = np.power(np.maximum(degree, epsilon), -0.5)
        out = inv[:, :, None] * out * inv[:, None, :]
    else:
        out = out / np.maximum(degree[:, :, None], epsilon)
    return out[0] if squeeze else out
