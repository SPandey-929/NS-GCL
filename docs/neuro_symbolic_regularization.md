# Neuro-symbolic physiological regularization

## 1. Physiological descriptors

Inputs to both extractors have shape (epochs, channels, samples).

### Frontal alpha asymmetry

The implementation uses F3 (left frontal) and F4 (right frontal), estimates
alpha-band power with Welch's method, and defines

FAA_i = ln(PSD_{F4,i}^alpha + epsilon)
        - ln(PSD_{F3,i}^alpha + epsilon).

The sign convention is explicit. FAA direction is not a universal positive-
versus-negative emotion rule. The sign mapping used by the loss must be
supported by the study's stated hypothesis and validated without looking at
the held-out LOSO subject.

### Beta-band phase locking value

For channel pair (p,q),

PLV_{pq} = | mean_t exp(j (phi_p(t)-phi_q(t))) |.

The utility filters each epoch in the requested beta range (default 13-30 Hz),
computes the Hilbert phase, and returns mean PLV across unique off-diagonal
channel pairs. An optional symmetric edge mask restricts the average to a
predefined set of graph edges. The beta band must be consistent with the
signal preprocessing. If preprocessing is 4-30 Hz, gamma above 30 Hz cannot
be recovered from those filtered signals.

## 2. Differentiable physiological penalties

The regularizer accepts precomputed FAA and beta-PLV values plus predicted
probabilities supplied by the caller. It does not hard-code class indices.

Let p_i^+ and p_i^- be predicted positive- and negative-valence probabilities,
and p_i^H the predicted high-arousal probability. Let s_F in {-1,+1} be the
study-justified FAA sign convention, m_F a non-negative FAA margin, and
tau_beta a beta-PLV threshold.

L_FAA = (1/N) sum_i [
 p_i^+ max(0, m_F - s_F FAA_i)^2
 + p_i^- max(0, m_F + s_F FAA_i)^2
].

L_beta = (1/N) sum_i [
 p_i^H max(0, tau_beta - S_i^beta)^2
].

L_NS = lambda_FAA L_FAA + lambda_beta L_beta.

Gradients flow through predicted probabilities. FAA and beta-PLV descriptors
are fixed signal-derived inputs and are detached inside the loss. The beta
penalty encodes the specific hypothesis that a high-arousal prediction should
coincide with beta synchrony above a threshold. This is a modelling
assumption requiring empirical validation, not a universal physiological law.

## 3. Use in the complete objective

Once the classifier and contrastive objective are connected, a candidate
training objective is

L_total = L_classification + lambda_con L_InfoNCE + zeta_NS L_NS.

The loss weights, FAA margin, FAA sign, beta threshold, and class-probability
mapping must be selected from the actual study protocol and training/validation
subjects only. Do not report the software defaults as tuned experimental
settings. Do not apply the same binary valence/arousal mapping to SEED or
SEED-IV without defining and justifying the mapping for those datasets.

## 4. Pair-construction reconciliation

The manuscript figure describing contrastive learning shows two augmented
views of the same EEG sample as a positive pair, labels the positive as
"same sample/same emotion", and describes negatives as samples from different
emotions, other subjects, or trials. The current InfoNCE code uses the two
views of the same sample as positives but treats every other minibatch view
as a negative. It therefore does not yet enforce the figure's different-
emotion / subject / trial negative-selection conditions. The detailed
candidate-pool and exclusion rules need confirmation from the experiment
protocol or training code before claiming exact reproduction.

## 5. Validation scope

Unit tests use synthetic signals and probabilities to check output ranges,
shape handling, finite losses, and gradient propagation. They do not
establish that the physiological assumptions improve emotion recognition,
nor do they reproduce results on DEAP, SEED, or SEED-IV.
