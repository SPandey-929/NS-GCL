# Real-data execution guide for reviewers

## Dataset access

The repository does not redistribute EEG recordings.

- DEAP: obtain the data through the official [DEAP dataset page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) and follow its access and data-use terms.
- SEED / SEED-IV: use the official [SEED project portal](https://bcmi.sjtu.edu.cn/home/seed/) and follow its access and use requirements.

Keep downloaded data outside the public repository. Do not commit raw recordings, restricted derived data, or credentials.

## Environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

For GPU-specific PyTorch builds, use the command from the official PyTorch installation selector.

## Confirmed DEAP preprocessing settings

The author has confirmed:
- 4–30 Hz band-pass filtering.
- Non-overlapping 3-second segmentation.

The repository defaults match these two settings. The remaining steps (source file release, baseline handling, ICA component identification, wavelet details, graph parameters, and feature tensor layout) are not yet sufficiently specified to claim a complete manuscript-compliant conversion. See `docs/deap_preprocessing_audit.md`.

A 4–30 Hz filter cannot retain conventional delta 1–4 Hz or gamma 30–45 Hz. The manuscript's five-band description must be reconciled before implementing the time-frequency feature generator.

## Synthetic software check

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

This is an artificial-data software test, not a real-data experiment.

## Real-data archive status

The current repository does not yet provide a complete raw-DEAP/SEED/SEED-IV-to-NPZ feature-generation command. Once the exact preprocessing and feature-generation protocol is finalized and implemented, an archive can be validated with:

```python
from data.feature_archive import load_feature_archive
archive = load_feature_archive("path/to/real_features.npz")
print(archive["features"].shape)
print(archive["metadata"])
```

## LOSO command after archive preparation

```bash
python -m scripts.train_loso --data path/to/real_features.npz --config configs/loso_deap.yaml --output outputs/real_deap_loso
```

The YAML currently contains initial software defaults, not confirmed manuscript hyperparameters. Do not interpret this command as a completed reproduction until a verified real-data archive and matching configuration are available.

## Evidence separation

Keep CI software tests, synthetic integration runs, and actual dataset experiments separate. Only the last category supports DEAP/SEED/SEED-IV performance claims. Record dataset release, preprocessing configuration, archive metadata, code commit, random seed, software environment, fold outputs, and checkpoints for each real run.
