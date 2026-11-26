# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

**Initial preprocessing, functional-connectivity graph, multi-scale CNN, and GCN modules have been committed.** They are initial software implementations and have not yet been validated on the study's DEAP recordings. The temporal encoder, contrastive objective, neuro-symbolic loss, complete LOSO training/evaluation pipeline, and reproduction of manuscript tables remain to be implemented and tested.

No manuscript result should be described as reproduced by this repository until the complete pipeline has been run and the outputs independently checked.

## Implemented components

### DEAP preprocessing
- Common participant-level DEAP MATLAB loader.
- Explicit selection of the 11 manuscript-listed electrodes from a named 32-channel montage.
- Configurable zero-phase Butterworth filtering.
- Non-overlapping epoch segmentation.
- Channel-wise z-score scaler intended to be fitted on training subjects only.
- Optional baseline correction and manually reviewed ICA component rejection.

### Functional-connectivity graphs
- Band-limited PLV and PLI computation from EEG epochs.
- Symmetric adjacency construction with threshold and optional top-k sparsification.
- Optional binarization, self-loops, and symmetric or row-wise adjacency normalization.

### Neural encoders
- Multi-scale CNN with parallel 3x3, 5x5, and 7x7 convolution branches, producing per-electrode embeddings from time-frequency maps.
- Two-layer weighted graph convolution encoder with ReLU, dropout, and global mean pooling.

See `docs/graph_and_encoder.md` for tensor shapes, assumptions, and limitations.

## Installation

Use Python 3.10 or later in a virtual environment:

    python -m venv .venv

Activate it using the appropriate command for your operating system, then install:

    python -m pip install -r requirements.txt
    python -m pytest

PyTorch installation may need a platform-specific command. Follow the official selector at https://pytorch.org/get-started/locally/ if the default package installation is unsuitable for your CPU/GPU.

Tests are initial software checks on synthetic inputs. They are not validation against DEAP recordings or reproduction of reported results.

## Configuration and scientific cautions

- The graph configuration currently specifies beta PLV (13–30 Hz) as an initial setting. With the manuscript's 4–30 Hz preprocessing, conventional gamma activity above 30 Hz is not retained.
- The graph top-k setting and threshold are starting configuration values, not verified optimal values or confirmed manuscript hyperparameters.
- The CNN kernel sizes and GCN dimensions are implementation defaults. Reconcile them with the final method and actual experiment before reporting them as experimental settings.
- Confirm the exact DEAP release, sampling rate, baseline handling, artifact-removal procedure, channel montage, and time-frequency feature-generation method before running experiments. Avoid duplicate filtering or artifact removal if using a preprocessed release.
- In LOSO evaluation, fit normalization statistics using training subjects only. Do not select hyperparameters or physiological thresholds using the held-out test subject.

## DEAP data access

DEAP must be obtained from its official provider under the applicable data-use terms. See `docs/deap_data_access.md`. This repository does not host participant recordings. Never commit raw or restricted EEG files.

## Reproducibility status

The full NS-GCL experiment pipeline is still under development. The current modules do not establish that the manuscript's reported metrics have been reproduced.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
