"""Projection heads and manuscript-aligned contrastive objectives.

The manuscript's Fig. 4 defines two augmentations of the same EEG epoch as
the positive pair. A different epoch is an eligible negative when at least
one of emotion label, subject ID, or trial ID differs. Other epochs sharing
all three attributes are excluded from the negative pool.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _as_list(values, name):
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    values = list(values)
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    return values


def build_manuscript_negative_mask(labels, subject_ids, trial_ids, device=None):
    """Return NxN eligibility mask: different label OR subject OR trial.

    The diagonal is always false because an epoch cannot be its own negative.
    """
    labels = _as_list(labels, "labels")
    subjects = _as_list(subject_ids, "subject_ids")
    trials = _as_list(trial_ids, "trial_ids")
    n = len(labels)
    if len(subjects) != n or len(trials) != n:
        raise ValueError("labels, subject_ids and trial_ids must have equal length.")
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mask[i, j] = (
                labels[i] != labels[j]
                or subjects[i] != subjects[j]
                or trials[i] != trials[j]
            )
    if not bool(mask.any(dim=1).all()):
        raise ValueError(
            "Every anchor needs at least one eligible negative. "
            "Increase batch diversity across emotion, subject, or trial."
        )
    return mask


class ContrastiveProjectionHead(nn.Module):
    """Map fused clip representations into the contrastive embedding space."""
    def __init__(self, input_dim=128, hidden_dim=128, projection_dim=64,
                 dropout=0.1):
        super().__init__()
        if min(input_dim, hidden_dim, projection_dim) < 1:
            raise ValueError("Projection dimensions must be positive.")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Expected clip features shaped (batch, features).")
        return F.normalize(self.network(features), p=2, dim=1)


class InfoNCELoss(nn.Module):
    """Standard symmetric cross-view NT-Xent for generic use."""
    def __init__(self, temperature=0.1, similarity="cosine"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'.")
        self.temperature = float(temperature)
        self.similarity = similarity

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        if view1.ndim != 2 or view2.ndim != 2 or view1.shape != view2.shape:
            raise ValueError("Both views must have identical (N, D) shapes.")
        n = view1.shape[0]
        if n < 2:
            raise ValueError("InfoNCE requires at least two examples per batch.")
        if self.similarity == "cosine":
            view1 = F.normalize(view1, p=2, dim=1)
            view2 = F.normalize(view2, p=2, dim=1)
        z = torch.cat((view1, view2), dim=0)
        logits = (z @ z.T) / self.temperature
        total = 2 * n
        logits = logits.masked_fill(
            torch.eye(total, dtype=torch.bool, device=z.device),
            torch.finfo(logits.dtype).min)
        indices = torch.arange(total, device=z.device)
        positives = (indices + n) % total
        return F.cross_entropy(logits, positives)


class ManuscriptInfoNCELoss(nn.Module):
    """Symmetric InfoNCE with manuscript-defined metadata-aware negatives.

    For each anchor, its other augmented view is the sole positive. Both
    views of every eligible different epoch are negatives. An epoch is
    eligible iff its emotion label OR subject ID OR trial ID differs.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")
        self.temperature = float(temperature)

    def forward(self, view1, view2, labels, subject_ids, trial_ids):
        if view1.ndim != 2 or view2.ndim != 2 or view1.shape != view2.shape:
            raise ValueError("Both views must have identical (N, D) shapes.")
        n = view1.shape[0]
        if n < 2:
            raise ValueError("At least two epochs are required.")
        negative_mask = build_manuscript_negative_mask(
            labels, subject_ids, trial_ids, device=view1.device)
        if negative_mask.shape != (n, n):
            raise ValueError("Metadata batch size does not match embedding batch.")
        z = torch.cat((
            F.normalize(view1, p=2, dim=1),
            F.normalize(view2, p=2, dim=1)
        ), dim=0)
        logits = (z @ z.T) / self.temperature
        allowed = torch.zeros((2 * n, 2 * n), dtype=torch.bool, device=z.device)
        # Both views of eligible different epochs are negatives.
        allowed[:n, :n] = negative_mask
        allowed[:n, n:] = negative_mask
        allowed[n:, :n] = negative_mask
        allowed[n:, n:] = negative_mask
        # The corresponding cross-view representation is the positive.
        idx = torch.arange(n, device=z.device)
        allowed[idx, idx + n] = True
        allowed[idx + n, idx] = True
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        positive_indices = torch.cat((idx + n, idx), dim=0)
        return F.cross_entropy(logits, positive_indices)


def info_nce_loss(view1: torch.Tensor, view2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Functional convenience wrapper for standard symmetric InfoNCE."""
    return InfoNCELoss(temperature=temperature)(view1, view2)


class NSGCLContrastiveModel(nn.Module):
    """Attach a projection head to an encoder returning clip_embedding."""
    def __init__(self, representation_encoder: nn.Module, input_dim=128,
                 hidden_dim=128, projection_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = representation_encoder
        self.projector = ContrastiveProjectionHead(
            input_dim=input_dim, hidden_dim=hidden_dim,
            projection_dim=projection_dim, dropout=dropout)

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        if not isinstance(outputs, dict) or "clip_embedding" not in outputs:
            raise ValueError("Encoder must return a dict with clip_embedding.")
        result = dict(outputs)
        result["projection"] = self.projector(outputs["clip_embedding"])
        return result
