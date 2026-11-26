# Real-data execution guide for reviewers

## Dataset access

The repository does not redistribute EEG recordings.

- DEAP: follow the access procedure and data-use terms on the official [DEAP page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html).
- SEED and SEED-IV: obtain them from the official [SEED project portal](https://bcmi.sjtu.edu.cn/home/seed/), subject to its access and use requirements.

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

## Synthetic software check

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

This is an artificial-data software test, not a real-data experiment.

## Real-data archive status

The current repository does not yet provide a complete raw-DEAP/SEED/SEED-IV-to-NPZ command. The DEAP protocol also has unresolved manuscript inconsistencies documented in `docs/deap_preprocessing_audit.md`. Before real-data execution, the authors must release the confirmed dataset-specific preprocessing configuration, exact wavelet/STFT implementation, graph settings, and archive-generation command.

When a manuscript-compliant archive is available, inspect it:

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

The YAML currently contains initial software defaults, not confirmed manuscript hyperparameters. Use dataset-specific settings only after confirming channel layout, sample rate, labels, preprocessing, and model configuration.

## Evidence separation

Keep CI software tests, synthetic integration runs, and actual dataset experiments separate. Only the last category supports DEAP/SEED/SEED-IV performance claims. Record dataset release, preprocessing configuration, archive metadata, code commit, random seed, software environment, fold outputs, and checkpoints for each real run.
