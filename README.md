# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

**DEAP preprocessing module: initial implementation committed; validation on the study's actual DEAP files is pending.** Model architecture, training, neuro-symbolic loss, LOSO experiments, and reproduction of manuscript tables are not yet complete.

The code is being developed transparently. No manuscript result should be described as reproduced by this repository until the complete pipeline has been run and the outputs independently checked.

## Implemented preprocessing components

- Common participant-level DEAP MATLAB loader (data and labels arrays).
- Explicit selection of the 11 manuscript-listed channels, using a named 32-channel montage.
- Configurable zero-phase Butterworth band-pass filter.
- Non-overlapping window segmentation.
- Channel-wise z-score scaler designed to be fitted on training subjects only.
- Optional ICA utility requiring researcher-reviewed component exclusions.
- Unit tests for core preprocessing functions.

## Planned components

- Time-frequency representation (exact method to be reconciled with the final experiment).
- Functional-connectivity graph construction.
- Multi-scale CNN, graph encoder, and temporal encoder.
- InfoNCE contrastive learning.
- FAA and beta-band PLV neuro-symbolic regularization.
- LOSO training, validation, and evaluation.

## Installation

Use Python 3.10 or later in a virtual environment:

    python -m venv .venv

Activate it using the appropriate command for your operating system, then run:

    python -m pip install -r requirements.txt
    python -m pytest

These tests are initial software checks, not validation against DEAP recordings or reproduction of reported results.

## DEAP data access

DEAP must be obtained from its official provider under the applicable data-use terms. See the data-access documentation in docs/. This repository does not host participant recordings. Never commit raw or restricted EEG files.

## Important protocol checks

The manuscript currently specifies 4–30 Hz filtering, 3-second non-overlapping windows, and z-score normalization. Before running experiments, verify the exact DEAP release, sampling rate, baseline handling, artifact-removal procedure, channel montage, and feature-generation method. The preprocessed DEAP release may already have undergone filtering and artifact processing; avoid duplicate preprocessing without checking the study protocol.

For LOSO, fit normalization statistics using training subjects only and apply the fitted transform to validation/test subjects. Hyperparameters and physiological thresholds must not be selected using the held-out test subject.

## Reproducibility status

The complete model and experiment pipeline is still under development. This repository currently does not establish that the reported manuscript metrics have been reproduced.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
