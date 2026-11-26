"""Multi-scale CNN encoder for per-electrode time-frequency maps.

Input: (batch, channels, frequency_bins, time_steps).
Output: (batch, channels, embedding_dim).
The convolutional branches use 3x3, 5x5 and 7x7 kernels; these are
implementation defaults and should be checked against the final experiment.
"""
import torch
from torch import nn


class MultiScaleCNN(nn.Module):
    def __init__(self, out_channels=32, embedding_dim=64, dropout=0.1):
        super().__init__()
        if out_channels < 1 or embedding_dim < 1:
            raise ValueError("Channel and embedding dimensions must be positive.")
        def branch(kernel):
            pad = kernel // 2
            return nn.Sequential(
                nn.Conv2d(1, out_channels, kernel_size=kernel, padding=pad,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        self.branch3 = branch(3)
        self.branch5 = branch(5)
        self.branch7 = branch(7)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(3 * out_channels, embedding_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected (batch, channels, frequency, time) input.")
        batch, channels, freq, steps = x.shape
        z = x.reshape(batch * channels, 1, freq, steps)
        a = self.branch3(z).mean(dim=(-2, -1))
        b = self.branch5(z).mean(dim=(-2, -1))
        c = self.branch7(z).mean(dim=(-2, -1))
        fused = torch.cat((a, b, c), dim=-1)
        fused = self.projection(self.dropout(fused))
        return fused.reshape(batch, channels, -1)
