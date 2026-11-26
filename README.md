# NS-GCL: Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition

Research-code repository for the manuscript “Neuro-Symbolic Graph Contrastive Learning for Cross-Subject EEG Emotion Recognition.”

## Current status

The repository contains preprocessing utilities, functional-connectivity graph construction, multi-scale CNN, GCN, Transformer temporal encoder, spatial-temporal fusion, contrastive learning, neuro-symbolic regularization, and an initial LOSO training scaffold. The complete pipeline has not yet been run on the study's actual DEAP/SEED/SEED-IV recordings, and the reported manuscript metrics have not been reproduced by this repository.

## Reproducibility status

The latest previously confirmed CI milestone was 40 passed tests (Run #73). New archive and synthetic-generator changes require a fresh successful CI run before they can be described as verified. Software tests, synthetic runs, and actual dataset experiments are distinct evidence categories.

## Synthetic archive: software tests only

Generate a deterministic artificial archive:

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

This generator creates random feature tensors and explicit synthetic provenance metadata. It does not generate recorded EEG; its scores must never be reported as dataset results.

## NPZ archive format

Required arrays: `features` shaped `(N, sequence, electrodes, frequency_bins, time_bins)`; integer `labels`; `subject_ids`; integer `trial_ids`; explicit `adjacency`; and scalar JSON `metadata_json`. The metadata records schema version, provenance (`synthetic` or `real`), dataset name, feature method, channel names, sampling rate, preprocessing record, and random seed. Synthetic archives require an explicit warning note. Use `data.feature_archive.load_feature_archive` to validate archives.

## Dataset access and reviewer execution

DEAP, SEED, and SEED-IV recordings are not included. Obtain them from their official providers and follow their access terms. Do not commit raw or restricted EEG files.

The repository does not yet provide a complete raw-recording-to-NPZ feature-generation command. The DEAP protocol has unresolved differences in filter range, baseline handling, segmentation, resampling, ICA, and wavelet specification. See:
- `docs/deap_data_access.md` — access notes.
- `docs/deap_preprocessing_audit.md` — preprocessing conflicts.
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

Synthetic tests cannot substantiate manuscript performance claims. Actual dataset results have not yet been reproduced by this repository.

## Citation and license

A complete citation record will be added when publication details are available. No reuse license has yet been selected; until then, all rights are reserved by the repository owner.
