"""Transformer encoder for sequences of EEG graph embeddings."""
import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int = 4096):
        super().__init__()
        if embedding_dim < 1 or max_length < 1:
            raise ValueError("embedding_dim and max_length must be positive.")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32)
                          * (-math.log(10000.0) / embedding_dim))
        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        if embedding_dim > 1:
            odd_width = encoding[:, 1::2].shape[1]
            encoding[:, 1::2] = torch.cos(position * scale[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected (batch, sequence, features) input.")
        length = x.shape[1]
        if length > self.encoding.shape[1]:
            raise ValueError("Sequence length exceeds configured max_length.")
        return x + self.encoding[:, :length].to(dtype=x.dtype, device=x.device)


class TemporalTransformerEncoder(nn.Module):
    """Project graph embeddings and model temporal dependencies."""
    def __init__(self, input_dim=64, embedding_dim=128, num_layers=2,
                 num_heads=4, feedforward_dim=256, dropout=0.1,
                 max_length=4096):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if num_layers < 1 or feedforward_dim < 1:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.position = SinusoidalPositionalEncoding(embedding_dim, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        if x.ndim != 3:
            raise ValueError("Expected (batch, sequence, input_dim) input.")
        if padding_mask is not None and padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask must have shape (batch, sequence).")
        z = self.position(self.input_projection(x))
        return self.norm(self.encoder(z, src_key_padding_mask=padding_mask))
