# Validation and evidence status

This repository distinguishes three kinds of evidence.

## 1. Verified software tests

Automated tests check implementation properties such as tensor shapes,
metadata-aware pair eligibility, gradient propagation, fold partitioning,
normalization behavior, and metric calculations. A test is marked verified
only when the corresponding CI run completes successfully. The tests use
small synthetic inputs unless explicitly stated otherwise.

### Recorded software-validation baseline

- Branch: `main`
- Tested commit: `d8dd0b2bc605b390d0b4df2f11d79f638e8dfb98`
- GitHub Actions run: [45901507692](https://github.com/SPandey-929/NS-GCL/actions/runs/45901507692)
- Command: `python -m pytest -q`
- Result: **46 passed, 7 warnings**
- Interpretation: software tests passed; this is not evidence of real EEG
  performance or reproduction of manuscript results.

The three cleanup commits (configuration, preprocessing audit, and reviewer
guide) each triggered CI. All three workflows completed successfully with
46 passed and 7 warnings. The latest tested cleanup commit above is the
recorded baseline.

## 2. Synthetic experiments and archive validation

Synthetic experiments execute model or pipeline code on artificially
generated signals or feature arrays. They are intended to expose runtime,
integration, and numerical problems. Synthetic inputs do not represent
recorded participants and cannot support claims about EEG emotion-recognition
accuracy, cross-subject generalization, or reproduction of manuscript results.

The seeded archive generator and validated loader were smoke-checked with
seed 2026. The generated archive contained all six mandatory arrays
(`features`, `labels`, `subject_ids`, `trial_ids`, `adjacency`,
`metadata_json`) and loaded successfully. Its feature tensor had shape
`(12, 2, 11, 5, 16)`; labels were integer class IDs; adjacency and features
were finite; metadata was a scalar JSON string with `provenance="synthetic"`
and an explicit non-EEG warning. Re-generation with the same seed produced
identical features and labels. This is an archive/software check only.

Re-run it with:

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

## 3. Actual dataset results

This category is reserved for experiments run on the real, appropriately
authorized DEAP, SEED, or SEED-IV recordings using a documented preprocessing
and evaluation protocol. Each result should record dataset/version, subject
folds, preprocessing, configuration, random seed, software environment,
checkpoint, and metrics. Do not place synthetic scores in result tables or
describe them as dataset performance.

## Current status at repository update

- Software implementation: ongoing; the latest recorded software baseline
  above passed its full automated test suite.
- Synthetic archive: generator and mandatory metadata validated as described
  above; synthetic inputs remain software checks only.
- Actual dataset experiments: not yet run/reproduced by this repository.
- Manuscript performance tables: not generated or verified by this repository.
- Wavelet feature extractor: intentionally unimplemented pending original
  experimental settings.

Update this document whenever evidence status changes. Never infer a successful
test run from a commit or from the presence of a test file.
