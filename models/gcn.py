"""Weighted graph convolution layers for EEG functional-connectivity graphs."""
import torch
from torch import nn


class GraphConvolution(nn.Module):
    """One graph-convolution layer: H' = A_norm H W + b."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, node_features, normalized_adjacency):
        if node_features.ndim != 3:
            raise ValueError("Node features must have shape (batch, nodes, features).")
        if normalized_adjacency.ndim == 2:
            normalized_adjacency = normalized_adjacency.unsqueeze(0)
        if normalized_adjacency.ndim != 3:
            raise ValueError("Adjacency must have shape (batch, nodes, nodes).")
        batch, nodes, _ = node_features.shape
        if normalized_adjacency.shape[0] == 1:
            normalized_adjacency = normalized_adjacency.expand(batch, -1, -1)
        elif normalized_adjacency.shape[0] != batch:
            raise ValueError("Adjacency batch dimension is incompatible.")
        if normalized_adjacency.shape[1:] != (nodes, nodes):
            raise ValueError("Adjacency dimensions must match the node count.")
        support = self.linear(node_features)
        return torch.bmm(normalized_adjacency, support)


class GraphEncoder(nn.Module):
    """Two-layer GCN with global mean pooling over EEG electrode nodes."""
    def __init__(self, in_features=64, hidden_features=64,
                 embedding_dim=64, dropout=0.3):
        super().__init__()
        self.gcn1 = GraphConvolution(in_features, hidden_features)
        self.gcn2 = GraphConvolution(hidden_features, embedding_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, normalized_adjacency):
        h = self.activation(self.gcn1(node_features, normalized_adjacency))
        h = self.dropout(h)
        h = self.activation(self.gcn2(h, normalized_adjacency))
        return h, h.mean(dim=1)
