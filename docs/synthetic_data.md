# Synthetic archive generation (software validation only)

Generate a deterministic artificial feature archive:

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
```

The generator creates random Gaussian tensors, class IDs, subject/trial identifiers, a simple symmetric adjacency matrix, and JSON provenance metadata. It does not load, simulate, or approximate recorded EEG.

Validate the archive and run tests:

```bash
python -m pytest tests/test_synthetic_archive.py -q
python -m pytest -q
```

Synthetic scores must never be reported as DEAP/SEED/SEED-IV results or used in manuscript tables. Keep generated files under `synthetic_data/`, excluded from version control.
