"""Composition of per-epoch CNN, GCN, temporal Transformer, and fusion."""
import torch
from torch import nn
from .multiscale_cnn import MultiScaleCNN
from .gcn import GraphEncoder
from .temporal_encoder import TemporalTransformerEncoder
from .spatial_temporal_fusion import SpatialTemporalFusion


class NSGCLRepresentationEncoder(nn.Module):
    """Encode time-frequency EEG sequences into fused clip representations.

    Input: (batch, sequence, electrodes, frequency_bins, time_bins).
    Adjacency: shared (electrodes,electrodes), per-batch (batch,electrodes,electrodes),
    or per-epoch (batch,sequence,electrodes,electrodes).
    """
    def __init__(self, cnn_channels=32, spatial_dim=64, gcn_hidden_dim=64,
                 temporal_dim=128, temporal_layers=2, temporal_heads=4,
                 temporal_ff_dim=256, fusion_dim=128, dropout=0.1,
                 gcn_dropout=0.3):
        super().__init__()
        self.cnn = MultiScaleCNN(cnn_channels, spatial_dim, dropout)
        self.gcn = GraphEncoder(spatial_dim, gcn_hidden_dim, spatial_dim, gcn_dropout)
        self.temporal = TemporalTransformerEncoder(
            spatial_dim, temporal_dim, temporal_layers, temporal_heads,
            temporal_ff_dim, dropout)
        self.fusion = SpatialTemporalFusion(
            spatial_dim, temporal_dim, fusion_dim, dropout)

    def forward(self, time_frequency, adjacency, padding_mask=None):
        if time_frequency.ndim != 5:
            raise ValueError("Expected (batch, sequence, electrodes, frequency, time).")
        batch, sequence, electrodes, frequency, steps = time_frequency.shape
        x = time_frequency.reshape(batch * sequence, electrodes, frequency, steps)
        node_features = self.cnn(x)
        if adjacency.ndim == 2:
            graph_adjacency = adjacency
        elif adjacency.ndim == 3 and adjacency.shape[0] == batch:
            graph_adjacency = adjacency[:, None].expand(
                batch, sequence, electrodes, electrodes).reshape(
                    batch * sequence, electrodes, electrodes)
        elif adjacency.ndim == 4 and adjacency.shape[:2] == (batch, sequence):
            graph_adjacency = adjacency.reshape(
                batch * sequence, electrodes, electrodes)
        else:
            raise ValueError("Adjacency must be shared, per-batch, or per-epoch.")
        node_embeddings, graph_embeddings = self.gcn(node_features, graph_adjacency)
        spatial_sequence = graph_embeddings.reshape(batch, sequence, -1)
        temporal_sequence = self.temporal(spatial_sequence, padding_mask)
        fused_sequence, clip_embedding = self.fusion(
            spatial_sequence, temporal_sequence, padding_mask)
        return {
            "node_embeddings": node_embeddings.reshape(batch, sequence, electrodes, -1),
            "spatial_sequence": spatial_sequence,
            "temporal_sequence": temporal_sequence,
            "fused_sequence": fused_sequence,
            "clip_embedding": clip_embedding,
        }
