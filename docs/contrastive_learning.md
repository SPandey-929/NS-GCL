# Contrastive projection and InfoNCE objective

## Intended data flow

The NS-GCL representation encoder produces one clip-level fused feature vector
for each EEG sample. A projection head maps this vector to a normalized
contrastive embedding. During training, two independently augmented views
of each sample are passed through the same encoder and projection head.

For a minibatch of N original samples, let z_i^(1) and z_i^(2) denote the
normalized embeddings from the two views of sample i. The positive pair is
the two views of the same sample. The other 2N-2 embeddings in the minibatch
are negatives for that anchor.

## Symmetric NT-Xent / InfoNCE

For anchor a and its matching positive p(a), the loss is

L_a = -log [ exp(sim(z_a,z_p(a))/tau) /
             sum_{k != a} exp(sim(z_a,z_k)/tau) ].

The implementation concatenates both views, masks each embedding's
self-similarity, and uses the corresponding other-view embedding as its
positive target. Cross-entropy is averaged over all 2N anchors. Cosine
similarity is used by default, with L2-normalized projection vectors.

## Pair-construction assumption

This implementation uses instance-level self-supervised pairing:
- Positive: two independently augmented views of the same original EEG sample.
- Negative: every other view in the current minibatch.
- Samples are paired by their position in the two input tensors.

It does not use emotion labels to define positives or negatives, and it does
not impose cross-subject positive-pair restrictions. If the manuscript's
implemented pair rule differs (for example, same-emotion samples from
different subjects), this module must be adapted before being described as
the exact manuscript implementation. Do not claim a pair policy that was
not used in the actual experiment.

## Example

    import torch
    from models.contrastive import InfoNCELoss

    # Each row in view1 and view2 represents the same original EEG sample.
    view1 = torch.randn(16, 64)
    view2 = torch.randn(16, 64)
    criterion = InfoNCELoss(temperature=0.1)
    loss = criterion(view1, view2)
    loss.backward()

A batch must contain at least two original samples so that each anchor has
at least one negative. The temperature must be positive.

## Scope and validation

The module includes a projection head, symmetric cosine InfoNCE loss, and a
wrapper that attaches the projection head to an encoder returning the
clip_embedding field. Unit tests use synthetic tensors to check finite loss,
gradient flow, normalized projections, and basic pair-alignment behavior.
They do not validate EEG augmentation choices, subject-wise leakage,
experimental performance, or reproduce manuscript metrics.

The temperature default (0.1) is a software starting value, not a verified
optimal or manuscript-confirmed hyperparameter. Select it using training/
validation subjects only within each LOSO fold.
