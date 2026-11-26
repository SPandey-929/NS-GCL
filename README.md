# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

The repository contains initial DEAP preprocessing, functional-connectivity graph construction, multi-scale CNN, GCN, Transformer temporal encoder, and spatial-temporal fusion modules. These are software implementations under development. The complete training pipeline has not yet been run on the study's DEAP recordings, and the reported manuscript metrics have not been reproduced by this repository.

## Implemented components

- DEAP MATLAB loader, channel selection, configurable filtering, segmentation, normalization, and optional baseline/ICA utilities.
- Band-limited PLV and PLI, adjacency sparsification, and graph normalization.
- Multi-scale CNN with parallel 3x3, 5x5, and 7x7 convolution branches.
- Two-layer weighted GCN with global mean pooling.
- Batch-first Transformer temporal encoder with sinusoidal positional encoding.
- Spatial-temporal concatenation, projection, and masked/unmasked pooling.
- NSGCLRepresentationEncoder composes the CNN, GCN, temporal encoder, and fusion modules.
- Contrastive projection head and symmetric cosine InfoNCE (NT-Xent) loss for two augmented views of each sample.
- SameSampleTwoViewDataset, which independently augments the same EEG epoch twice and retains subject/trial/emotion metadata.
- Integrated NSGCLModel with classification and projection heads, plus a combined supervised + contrastive + optional neuro-symbolic loss.
- FAA and beta-band PLV feature extraction, plus a differentiable neuro-symbolic regularizer with explicit physiological sign and probability inputs.

See docs/graph_and_encoder.md, docs/temporal_fusion.md, and docs/integrated_nsgcl.md for tensor shapes and assumptions.

## Installation

Use Python 3.10 or later in a virtual environment:

    python -m venv .venv

Activate the environment, then install:

    python -m pip install -r requirements.txt
    python -m pytest

PyTorch installation may require a platform-specific command. Use the official selector at https://pytorch.org/get-started/locally/ when needed.

Tests use synthetic arrays and check basic software behavior, shapes, and gradient flow. They are not validation against DEAP recordings or reproduction of manuscript results.

## Example: representation encoder

    import torch
    from models import NSGCLRepresentationEncoder

    # Example dimensions only; replace with actual preprocessed features.
    batch, sequence, electrodes, freq_bins, time_bins = 2, 4, 11, 32, 48
    x = torch.randn(batch, sequence, electrodes, freq_bins, time_bins)
    adjacency = torch.eye(electrodes).repeat(batch, sequence, 1, 1)

    encoder = NSGCLRepresentationEncoder()
    outputs = encoder(x, adjacency)
    print(outputs["clip_embedding"].shape)

## Scientific and reproducibility cautions

- The graph configuration currently uses beta PLV (13–30 Hz) as an initial setting. With the manuscript's 4–30 Hz preprocessing, conventional gamma activity above 30 Hz is not retained.
- Graph top-k, thresholds, CNN dimensions, GCN dimensions, and Transformer settings are initial software defaults, not verified optimal values or confirmed manuscript hyperparameters.
- Confirm the exact DEAP release, sampling rate, baseline handling, artifact-removal procedure, channel montage, and time-frequency representation before running experiments. Avoid duplicate preprocessing.
- Fit normalization statistics using training subjects only within each LOSO fold. Do not tune thresholds or model settings using the held-out test subject.

## Dataset access

DEAP must be obtained from its official provider under the applicable data-use terms. See docs/deap_data_access.md. This repository does not host participant recordings. Do not commit raw or restricted EEG files.

## Contrastive learning

See docs/integrated_nsgcl.md for the selected manuscript pairing strategy: two independent augmentations of the same EEG epoch form a positive pair. In a minibatch of N epochs, the paired cross-view embedding is the positive, while the other 2N-2 embeddings are negatives for each anchor. Subject, trial and emotion metadata are retained but do not determine pair membership.

## Neuro-symbolic regularization

See docs/neuro_symbolic_regularization.md for FAA/PLV definitions, loss equations, sign-convention requirements, and validation cautions. FAA uses ln(alpha PSD at F4) minus ln(alpha PSD at F3). The regularizer requires an explicit FAA direction and caller-supplied positive-valence, negative-valence, and high-arousal probabilities; it does not assume universal class indices or a universal physiological sign. Physiological descriptors are precomputed and treated as fixed inputs.

## Remaining work

The integrated classifier and same-sample contrastive pair generation are implemented and have synthetic unit tests. The full LOSO training/evaluation pipeline, fold-wise preprocessing, experiment logging, statistical analysis, and reproduction of manuscript tables remain to be implemented and validated.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
