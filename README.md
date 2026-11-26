# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

The repository contains preprocessing, functional-connectivity graph construction, multi-scale CNN, GCN, Transformer temporal encoder, spatial-temporal fusion, contrastive learning, neuro-symbolic regularization, and an initial LOSO training scaffold. The complete pipeline has not yet been run on the study's actual DEAP/SEED/SEED-IV recordings, and the reported manuscript metrics have not been reproduced by this repository.

## Implemented components

- DEAP MATLAB loader, channel selection, configurable filtering, segmentation, normalization, and optional baseline/ICA utilities.
- Band-limited PLV and PLI, adjacency sparsification, and graph normalization.
- Multi-scale CNN, weighted GCN, Transformer temporal encoder, and spatial-temporal fusion.
- `NSGCLRepresentationEncoder` composing the spatial and temporal modules.
- Two independent augmented views of each EEG epoch using `SameSampleTwoViewDataset`.
- Integrated `NSGCLModel` with classification and projection heads.
- Manuscript-specific symmetric InfoNCE: the positive is the other view of the same epoch; a different epoch is a negative if its emotion label OR subject ID OR trial ID differs. Both views of eligible negative epochs are included. Candidates matching all three metadata fields are excluded.
- FAA and beta-band PLV feature extraction and an optional differentiable neuro-symbolic regularizer.

See `docs/integrated_nsgcl.md` for the exact pairing rule and the manuscript wording inconsistency. See `docs/validation_status.md` for the distinction between software tests, synthetic experiments, and actual dataset results.

## Installation

Use Python 3.10 or later in a virtual environment:

    python -m venv .venv
    python -m pip install -r requirements.txt
    python -m pytest

PyTorch installation may require a platform-specific command. Use the official selector at https://pytorch.org/get-started/locally/ when needed.

## Example: representation encoder

    import torch
    from models import NSGCLRepresentationEncoder

    batch, sequence, electrodes, freq_bins, time_bins = 2, 4, 11, 32, 48
    x = torch.randn(batch, sequence, electrodes, freq_bins, time_bins)
    adjacency = torch.eye(electrodes).repeat(batch, sequence, 1, 1)
    encoder = NSGCLRepresentationEncoder()
    outputs = encoder(x, adjacency)
    print(outputs["clip_embedding"].shape)

## Scientific and reproducibility cautions

- The graph configuration currently uses beta PLV (13–30 Hz) as an initial setting. With preprocessing limited to 4–30 Hz, conventional gamma activity above 30 Hz is not retained.
- Graph thresholds, CNN/GCN/Transformer dimensions, augmentation settings, and loss weights are software defaults unless explicitly confirmed against the final manuscript.
- Confirm the exact dataset release, sampling rate, baseline handling, artifact-removal procedure, channel montage, and time-frequency representation before running experiments. Avoid duplicate preprocessing.
- Fit normalization statistics using training subjects only within each LOSO fold. Do not tune thresholds or model settings using the held-out test subject.
- Do not infer high-arousal labels from categorical emotion labels where arousal annotations are unavailable.

## Dataset access

DEAP, SEED, and SEED-IV recordings are not included. Obtain them from their respective official providers under applicable data-use terms. Do not commit raw or restricted EEG files.

## LOSO training scaffold

The subject-wise training scaffold is `scripts/train_loso.py`. It expects a prepared NPZ feature archive; see `docs/loso_pipeline.md` for required arrays and the run command. The current scaffold performs training-fold-only normalization, two-view augmentation, metadata-aware batch construction, supervised cross-entropy plus manuscript-specific InfoNCE, validation-based checkpoint selection, and held-out subject evaluation. It does not yet integrate the neuro-symbolic penalty or build the feature archive directly from raw recordings.

## Evidence and validation

Training uses a metadata-aware batch sampler so each anchor has at least one eligible negative under the manuscript's emotion-OR-subject-OR-trial rule. The sampler fails early if the training data cannot provide such negatives.

The repository reports three evidence categories separately:

1. **Verified software tests:** automated checks confirmed by a completed successful CI run.
2. **Synthetic experiments:** executions on artificial signals/features for software integration only.
3. **Actual dataset results:** results obtained from authorized real EEG recordings under the documented protocol.

Synthetic tests are not dataset experiments and cannot substantiate manuscript performance claims. Current actual dataset results have not been reproduced by this repository. See `docs/validation_status.md`.

## Statistical utilities

Paired t-test and Wilcoxon utilities for matched subject-level comparisons are in `training/statistics.py`. They require genuinely matched per-subject scores from two methods; they do not establish significance for experiments that have not been run.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
