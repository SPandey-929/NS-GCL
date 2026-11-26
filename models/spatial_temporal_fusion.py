"""Spatial-temporal feature fusion for sequential EEG graph representations."""
import torch
from torch import nn


class SpatialTemporalFusion(nn.Module):
    """Concatenate spatial and temporal features, project, and pool."""
    def __init__(self, spatial_dim=64, temporal_dim=128, fusion_dim=128,
                 dropout=0.1):
        super().__init__()
        if min(spatial_dim, temporal_dim, fusion_dim) < 1:
            raise ValueError("Feature dimensions must be positive.")
        self.projection = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor,
                padding_mask: torch.Tensor = None):
        if spatial.ndim != 3 or temporal.ndim != 3:
            raise ValueError("Inputs must have shape (batch, sequence, features).")
        if spatial.shape[:2] != temporal.shape[:2]:
            raise ValueError("Spatial and temporal batch/sequence dimensions must match.")
        fused = self.projection(torch.cat((spatial, temporal), dim=-1))
        if padding_mask is None:
            pooled = fused.mean(dim=1)
        else:
            if padding_mask.shape != fused.shape[:2]:
                raise ValueError("padding_mask must have shape (batch, sequence).")
            valid = (~padding_mask).to(fused.dtype).unsqueeze(-1)
            pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return fused, pooled
