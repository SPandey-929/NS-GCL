"""Projection head and symmetric NT-Xent / InfoNCE objective.

Two independently augmented views of each of N examples are supplied in
matching row order. Matching rows across views are positives; all other
views in the minibatch are negatives. Pairing is instance-level and does
not infer positives from emotion labels or subject identities.
"""
import torch
from torch import nn
import torch.nn.functional as F


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
    """Symmetric cross-view InfoNCE (NT-Xent) loss.

    Inputs view1 and view2 have shape (N, embedding_dim), in matching sample
    order. N must be at least two so each anchor has negative examples.
    """
    def __init__(self, temperature=0.1, similarity="cosine"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'.")
        self.temperature = float(temperature)
        self.similarity = similarity

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        if view1.ndim != 2 or view2.ndim != 2:
            raise ValueError("Both views must have shape (batch, embedding_dim).")
        if view1.shape != view2.shape:
            raise ValueError("The two views must have identical shapes.")
        batch_size = view1.shape[0]
        if batch_size < 2:
            raise ValueError("InfoNCE requires at least two examples per batch.")
        if self.similarity == "cosine":
            view1 = F.normalize(view1, p=2, dim=1)
            view2 = F.normalize(view2, p=2, dim=1)

        representations = torch.cat((view1, view2), dim=0)
        logits = (representations @ representations.T) / self.temperature
        total = 2 * batch_size
        diagonal = torch.eye(total, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        indices = torch.arange(total, device=logits.device)
        positive_indices = (indices + batch_size) % total
        return F.cross_entropy(logits, positive_indices)


def info_nce_loss(view1: torch.Tensor, view2: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Functional convenience wrapper for symmetric cosine InfoNCE."""
    return InfoNCELoss(temperature=temperature)(view1, view2)


class NSGCLContrastiveModel(nn.Module):
    """Attach a projection head to an encoder returning clip_embedding.

    Call this wrapper twice with independently augmented versions of the
    same minibatch. Pass the returned projection tensors to InfoNCELoss.
    """
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
