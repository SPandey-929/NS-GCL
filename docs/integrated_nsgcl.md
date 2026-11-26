# Integrated NS-GCL model and same-sample contrastive pairs

## Positive-pair definition

The selected manuscript strategy is instance-level augmentation. For each EEG
epoch, the dataset wrapper applies the augmentation transform independently
twice. The resulting views are a positive pair because they originate from
the same epoch. The emotion, subject and trial fields are retained as metadata;
they do not determine positive-pair membership.

For a minibatch of N epochs, symmetric NT-Xent/InfoNCE receives two tensors
with N rows in matching sample order. The paired row across the two views is
the positive. All other 2N-2 representations are negatives for that anchor.
The batch must contain at least two distinct epochs.

## Dataset wrapper

The wrapped dataset must return a dictionary with an EEG tensor under the
'eeg' or 'time_frequency' key. It should also return the class label and may
include adjacency, subject_id and trial_id fields. A transform is called
separately on cloned tensors for each view.

## Integrated model

NSGCLModel combines the existing multi-scale CNN, graph encoder, temporal
Transformer and spatial-temporal fusion with:
- an emotion classification head;
- a contrastive projection head;
- symmetric InfoNCE loss.

Its loss method combines the mean supervised cross-entropy from both views,
weighted InfoNCE and an optional scalar neuro-symbolic penalty.

The dataset-specific mapping from class probabilities to positive valence,
negative valence and high arousal must be supplied explicitly. Do not infer
high-arousal labels for datasets that do not provide a defensible arousal
annotation. The physiological sign convention and regularization thresholds
also require study-specific justification.

## Current limitations

This is the model and pair-generation stage, not yet the complete experiment
pipeline. Dataset loaders, signal augmentation policies, leakage-safe
preprocessing, LOSO orchestration, checkpointing and evaluation scripts are
still to be integrated and tested on the actual datasets. Synthetic unit tests
check software behavior; they do not reproduce manuscript results.
