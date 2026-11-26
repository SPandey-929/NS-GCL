# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

The repository contains preprocessing utilities, graph construction, multi-scale CNN, GCN, Transformer temporal encoding, contrastive learning, neuro-symbolic regularization, and an initial LOSO training scaffold. The complete pipeline has not yet been run on the study's actual DEAP/SEED/SEED-IV recordings, and manuscript metrics have not been reproduced by this repository.

## Confirmed DEAP settings and remaining work

The author has confirmed 4–30 Hz band-pass filtering and non-overlapping 3-second segmentation. The current filtering and segmentation defaults match these settings.

A complete DEAP-to-feature-archive command is not yet available. Baseline handling, the exact DEAP file release, ICA artifact-component identification, wavelet family/level, graph parameters, and final feature tensor construction still need to be specified and implemented. Also, a 4–30 Hz signal cannot contain conventional delta 1–4 Hz and gamma 30–45 Hz components; the manuscript's five-band description needs reconciliation. See `docs/deap_preprocessing_audit.md`.


## DEAP preprocessed MATLAB pipeline

For the common provider-preprocessed MATLAB release, use:

```python
from preprocessing.deap_pipeline import prepare_preprocessed_deap_file
windows, labels, window_meta = prepare_preprocessed_deap_file(
    "data/DEAP/data_preprocessed_matlab/s01.mat"
)
```

The utility expects 128-Hz files with either 8064 samples per trial (63 s, including the initial 3-s baseline interval) or 7680 samples (60-s stimulus already isolated). It does not downsample, repeat ICA/EOG artifact processing, or perform baseline subtraction. When 8064 samples are present, it discards the initial 384 baseline samples, applies the manuscript's additional 4–30 Hz analysis filter, and creates non-overlapping 384-sample (3-s) windows. For the common 40-trial, 11-channel file, the expected output is 800 windows with shape `(800, 11, 384)`.

This output is filtered time-domain data, not yet the final manuscript wavelet/graph feature archive. Confirm the downloaded release's dimensions and processing notes before use. Focused software tests:

```bash
python -m pytest tests/test_deap_preprocessed_pipeline.py -q
```

## Synthetic archive: software tests only

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

The generator creates artificial random tensors and explicit synthetic provenance metadata. It does not generate recorded EEG. Its scores must never be reported as dataset results.

## NPZ archive format

Required arrays: `features` shaped `(N, sequence, electrodes, frequency_bins, time_bins)`; integer `labels`; `subject_ids`; integer `trial_ids`; `adjacency`; and scalar JSON `metadata_json`. Metadata records schema version, provenance (`synthetic` or `real`), dataset name, feature method, channel names, sampling rate, preprocessing record, and random seed. Synthetic archives require an explicit warning note. Use `data.feature_archive.load_feature_archive` to validate archives.

## Wavelet feature-generation status

The author has confirmed DWT with Daubechies-4 (`db4`) and separate frequency-band processing. The manuscript and current repository configuration files do not specify the decomposition level, coefficient representation, boundary mode, band-filter design, normalization, or final feature dimensions. No manuscript-specific wavelet extractor is implemented until the original experiment records/configuration establish these settings. See `docs/wavelet_feature_specification.md`.

## Dataset access and reviewer execution

DEAP, SEED, and SEED-IV recordings are not included. Obtain them from their official providers and follow their access terms. Do not commit raw or restricted EEG files. See:
- `docs/deap_data_access.md` — access notes.
- `docs/deap_preprocessing_audit.md` — preprocessing protocol and remaining gaps.
- `docs/synthetic_data.md` — artificial archive generator.
- `docs/real_data_reproducibility.md` — reviewer execution guide.
- `docs/loso_pipeline.md` — archive contract and training scaffold.

After a verified real-data archive is prepared, the current LOSO scaffold can be invoked:

```bash
python -m scripts.train_loso --data path/to/real_features.npz --config configs/loso_deap.yaml --output outputs/real_deap_loso
```

This scaffold performs training-fold-only normalization, two-view augmentation, metadata-aware batch construction, supervised cross-entropy plus contrastive loss, validation-based checkpoint selection, and held-out subject evaluation. It does not yet integrate the neuro-symbolic penalty or generate archives from raw recordings. Configuration values are initial software defaults unless confirmed against the manuscript.

## Evidence categories

1. **Verified software tests:** automated checks confirmed by completed CI runs.
2. **Synthetic software experiments:** artificial inputs for integration and numerical checks only.
3. **Actual dataset results:** authorized real EEG recordings processed under a documented protocol.

Synthetic tests cannot substantiate manuscript performance claims. Actual dataset results have not yet been reproduced by this repository. The latest archive/generator CI workflow completed successfully; consult GitHub Actions for the run associated with the current commit.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
