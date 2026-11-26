# LOSO training pipeline

## Input feature archive

The first pipeline stage consumes a prepared NumPy NPZ archive. Required arrays:

- features: float array shaped (N, sequence, electrodes, frequency_bins, time_bins).
- labels: integer class IDs shaped (N,), contiguous from zero.
- subject_ids and trial_ids: one identifier per sample.
- adjacency: explicit PLV/PLI graph shaped (E,E), (N,E,E), or (N,sequence,E,E).

The archive is expected to be created from preprocessed EEG and time-frequency maps after the exact manuscript feature-extraction procedure is fixed. This repository does not yet convert raw DEAP MATLAB recordings into the final time-frequency archive. Do not commit restricted participant recordings to public GitHub.

## Fold protocol

Each unique subject is held out once for final testing. From the remaining subjects, one complete subject is reserved for validation. Early stopping and model selection use only this validation subject. Feature standardization is fitted only on the remaining training-subject samples and applied to training, validation and test data. No held-out test samples enter fitting or hyperparameter selection.

## Training

Two independent augmentations of each same EEG epoch form the positive pair. Other embeddings in the minibatch are negatives under symmetric InfoNCE. Batches smaller than two are dropped. The model is trained with mean cross-entropy from both views plus weighted InfoNCE. Validation uses unaugmented single-view classification. The best validation macro-F1 checkpoint is restored before held-out subject evaluation.

## Outputs

Each held-out subject gets training_history.csv, best_model.pt and test_metrics.json. The root output contains loso_subject_metrics.csv and loso_summary.json. Report mean and standard deviation across held-out subjects; per-subject confusion matrices are saved separately.

## Neuro-symbolic limitation

This first executable LOSO scaffold currently optimizes supervised classification plus contrastive loss. The physiological regularizer exists as a standalone module but is not silently enabled: it requires a scientifically defined high-arousal probability and validated FAA sign/class mappings. Do not infer high-arousal probability from a valence-only class label. We must connect this term only after confirming the exact manuscript rule and available annotations.

## Run

Install requirements, then run:

    python -m scripts.train_loso --data path/to/prepared_features.npz --config configs/loso_deap.yaml --output outputs/loso

Configuration values are initial defaults, not evidence of optimality or confirmation of reported experimental settings. Synthetic tests check software behavior only; actual dataset runs are required for scientific reproducibility.
