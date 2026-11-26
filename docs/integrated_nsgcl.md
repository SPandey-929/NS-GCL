# Integrated NS-GCL model and manuscript-specific contrastive pairing

## Pair definitions taken from Fig. 4 and its caption

For each EEG epoch, two independent augmentations are generated. The two
resulting embeddings of that **same epoch** form the positive pair.

For anchor epoch i and candidate epoch j, j is an eligible negative when:

- its emotion label differs from i, **or**
- its subject identity differs from i, **or**
- its trial identity differs from i.

Thus, a candidate sharing the same emotion label, subject identity, and trial
identity is not treated as a negative. The diagonal (the anchor itself) is
never a negative. Both augmented views of each eligible candidate are included
in the denominator. The positive is the matching cross-view embedding of the
anchor epoch.

This is a literal operationalization of the “different emotion classes,
trials, or subjects” wording in Fig. 4 and its caption. It is not equivalent
to ordinary NT-Xent, which treats every other instance in the batch as a
negative. The training code therefore requires labels, subject IDs, and trial
IDs in each minibatch.

## Important manuscript wording inconsistency

The opening sentence of Section 3.5 describes positive pairs as “different
segments or augmentations of the same emotional state across subjects,” while
Fig. 4 and its caption explicitly show two augmented views of the same sample
as the positive pair. The released implementation follows the more explicit
Fig. 4/caption rule above. Please revise Section 3.5 so the manuscript and
repository describe one consistent pairing protocol before final submission.

## Dataset wrapper and model

`SameSampleTwoViewDataset` independently calls the augmentation transform
twice on cloned input tensors and retains `label`, `subject_id`, and
`trial_id`. `NSGCLModel` encodes both views, produces class logits and
projection embeddings, and combines mean two-view cross-entropy with the
metadata-aware symmetric InfoNCE objective. An optional scalar
neuro-symbolic penalty can also be added.

Every anchor must have at least one eligible negative in its minibatch. The
loss raises an informative error otherwise; the batch sampler or batch size
must then be adjusted to provide adequate diversity. The standard shuffled
minibatch loader does not guarantee that condition for every possible dataset
or batch, so this is checked at runtime.

## Validation scope

Unit tests exercise the negative eligibility rule, positive-pair alignment,
loss finiteness, gradient propagation, and invalid-batch handling using
synthetic tensors. These are software tests, not EEG experiments. No result
from these tests is a DEAP, SEED, or SEED-IV emotion-recognition result.
