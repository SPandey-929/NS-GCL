# Temporal encoder and spatial-temporal fusion

## Data flow

The representation encoder accepts time-frequency inputs shaped
(batch, sequence, electrodes, frequency_bins, time_bins) and graph adjacency
matrices shaped (batch, sequence, electrodes, electrodes), or a shared graph.
For each epoch, the multi-scale CNN generates an embedding for each electrode.
The GCN aggregates information over the functional-connectivity graph and
global mean pooling produces one spatial vector per epoch. A batch-first
Transformer models dependencies across the sequence of epoch-level spatial
vectors. The spatial vectors and Transformer outputs are concatenated,
projected, normalized, and returned as fused sequence features. A masked mean
produces a clip-level embedding when a padding mask is supplied.

## Tensor shapes

- CNN input after flattening batch and sequence: (B*S, C, F, T).
- CNN electrode features: (B*S, C, D_s).
- GCN pooled graph representation: (B*S, D_s).
- Spatial sequence: (B, S, D_s).
- Transformer sequence: (B, S, D_t).
- Fused sequence: (B, S, D_f).
- Clip embedding: (B, D_f).

## Initial defaults (not verified experimental hyperparameters)

The module defaults to a 2-layer Transformer, 4 attention heads, model
dimension 128, feed-forward dimension 256, GELU activation and dropout 0.1.
The spatial-temporal fusion concatenates spatial and temporal vectors and
projects them to dimension 128. These are software defaults only. Confirm
against the final manuscript and actual experiment before reporting them as
experimental settings.

## Mask convention

The optional padding mask follows PyTorch Transformer convention: True means
that the corresponding sequence position is padding and should be ignored;
False means a valid epoch.

## Scope and validation

This module implements representation encoding and fusion only. It does not
implement InfoNCE, the classifier, FAA/beta-PLV regularization, the optimizer,
LOSO splitting, or the complete experimental pipeline. Synthetic-input tests
check tensor shapes and gradient flow; they do not validate performance or
reproduce DEAP results.
