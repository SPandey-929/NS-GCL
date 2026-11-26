# Validation and evidence status

This repository distinguishes three kinds of evidence.

## 1. Verified software tests

Automated tests check implementation properties such as tensor shapes,
metadata-aware pair eligibility, gradient propagation, fold partitioning,
normalization behavior, and metric calculations. A test is marked verified
only when the corresponding CI run completes successfully. The tests use
small synthetic inputs unless explicitly stated otherwise.

## 2. Synthetic experiments

Synthetic experiments execute model or pipeline code on artificially
generated signals or feature arrays. They are intended to expose runtime,
integration, and numerical problems. Synthetic inputs do not represent
recorded participants and cannot support claims about EEG emotion-recognition
accuracy, cross-subject generalization, or reproduction of manuscript results.

## 3. Actual dataset results

This category is reserved for experiments run on the real, appropriately
authorized DEAP, SEED, or SEED-IV recordings using a documented preprocessing
and evaluation protocol. Each result should record dataset/version, subject
folds, preprocessing, configuration, random seed, software environment,
checkpoint, and metrics. Do not place synthetic scores in result tables or
describe them as dataset performance.

## Current status at repository update

- Software implementation: ongoing; the relevant automated test status is
  reported by GitHub Actions.
- Synthetic pipeline checks: only those explicitly logged as synthetic.
- Actual dataset experiments: not yet run/reproduced by this repository.
- Manuscript performance tables: not generated or verified by this repository.

Update this document whenever evidence status changes. Never infer a successful
test run from a commit or from the presence of a test file.
