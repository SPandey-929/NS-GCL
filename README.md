# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

This repository is intended to host the research implementation accompanying the manuscript **“Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”**

## Project status

**Implementation and verification are in progress.** This initial repository setup is not yet a complete reproduction package. The manuscript's reported results must not be interpreted as independently reproduced by this repository until the training and evaluation pipelines have been implemented, executed, and checked against the reported experimental protocol.

The implementation plan covers:

- EEG preprocessing and time-frequency representation
- Functional-connectivity graph construction
- Multi-scale feature extraction and graph-based representation learning
- Temporal representation learning
- Contrastive learning with InfoNCE
- Neuro-symbolic regularization using frontal alpha asymmetry (FAA) and beta-band phase-locking value (PLV), subject to confirmation against the final manuscript and actual experimental pipeline
- Leave-One-Subject-Out (LOSO) evaluation

## Planned repository layout

```text
NS-GCL/
├── configs/             # Dataset-specific experiment configurations
├── data/                # Dataset instructions; raw EEG data are not distributed
├── docs/                # Method and reproducibility documentation
├── losses/              # Classification, contrastive and neuro-symbolic losses
├── models/              # CNN, GCN, temporal encoder and NS-GCL model
├── preprocessing/       # EEG preprocessing and graph construction
├── scripts/             # Training and evaluation entry points
├── tests/               # Unit and integration tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

A tested environment specification will be added after implementation and smoke tests are complete. Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then install the packages listed in `requirements.txt`.

## Data availability and preparation

This repository does not distribute DEAP, SEED, SEED-IV, or HBUED recordings. Users must obtain each dataset from its authorized source and comply with its terms of use. Dataset-specific loading, channel mapping, preprocessing, and label handling will be documented alongside the implementation.

Do not upload raw EEG recordings, participant identifiers, private information, or restricted dataset files to this repository.

## Reproducibility checklist

Before claiming reproduction of manuscript results, verify:

1. Dataset versions, sampling rates, channel selections, filtering, artifact handling, segmentation, and normalization.
2. The implemented CNN, GCN, temporal encoder, graph construction, and fusion operations.
3. Positive/negative pair construction and InfoNCE temperature.
4. Exact FAA and beta-PLV definitions, thresholds, loss weights, and dataset-specific label mappings.
5. LOSO subject partitioning and validation-only hyperparameter selection.
6. Fold-wise metrics and aggregation against the manuscript tables.

The manuscript describes a 4–30 Hz preprocessing range. Conventional gamma-band PLV above 30 Hz must not be computed from signals already restricted to 4–30 Hz. The implementation and manuscript must use a consistent frequency range.

## Citation

A complete citation record will be added when the manuscript's publication details are available. Until then, cite the manuscript title and author list as stated in the submitted manuscript.

## License

No reuse license has been selected yet. Until a license is added, all rights are reserved by the repository owner.
