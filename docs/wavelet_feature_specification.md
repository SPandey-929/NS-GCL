# Wavelet feature-generation specification status

## 1. Author-confirmed wavelet choices

| Item | Confirmed value | Evidence |
|---|---|---|
| Transform | Discrete Wavelet Transform (DWT) | Author confirmation |
| Wavelet family | Daubechies-4 (`db4`) | Author confirmation |
| Frequency-band procedure | Process each frequency band separately | Author confirmation |

These values are not inferred from generic EEG practice.

## 2. Preprocessing details found in the manuscript

The manuscript provides the following information:

| Processing item | Manuscript statement | Exact implementation detail available? |
|---|---|---|
| DEAP sampling rate | 512 Hz, downsampled to 128 Hz (Section 4.1 / Table 3) | Yes as described experimental history; do not downsample again when using the preprocessed 128-Hz release |
| Band-pass filtering | 4–30 Hz (Table 3; Section 4.2 says band-pass filtering) | Cutoff range yes; filter type/order/ripple/phase implementation no |
| Artifact processing | ICA artifact removal for DEAP (Table 3); generic artifact rejection elsewhere | Method named, but ICA algorithm, component selection and rejection criteria not given |
| Normalization | Z-score normalization for DEAP (Table 3) | Z-score named; axis/scope and fit/apply protocol not stated |
| Segmentation | 3-second non-overlapping windows (Table 3) | Yes |
| Time-frequency transform | Wavelet decomposition (Section 4.2) | General method only; DWT/db4 supplied separately by author |
| Frequency bands | Delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz (Section 4.2) | Band edges stated |
| Bandwise extraction | Each band processed separately | Author confirmation |

Important distinction: the manuscript's historical description says the original DEAP recordings were downsampled from 512 to 128 Hz. The repository is intended to start from the provider-preprocessed 128-Hz MATLAB release, so the code must not repeat that downsampling. Likewise, do not repeat provider-side artifact processing. The manuscript's additional 4–30 Hz filter can only be applied after confirming that the chosen input release's prior filtering and the intended branch order make it scientifically appropriate.

## 3. DWT parameters not found in manuscript/configuration

Inspection of manuscript Sections 3.2 and 4.2, Algorithm 1, and the current repository configurations (`configs/deap.yaml`, `configs/loso_deap.yaml`, and `configs/graph.yaml`) did not identify:
- DWT decomposition level;
- coefficient representation (raw signed coefficients, absolute values, squared magnitude/energy, or other);
- boundary-extension mode;
- coefficient normalization;
- coefficient aggregation, padding, truncation or resizing;
- final time-frequency feature-map dimensions.

The manuscript itself does not mention `db4`; that wavelet choice is confirmed by the author in the project conversation. None of the inspected config files records the missing DWT parameters.

## 4. Frequency-band / filtering conflict

Section 4.2 lists the five conventional bands:
- delta: 1–4 Hz;
- theta: 4–8 Hz;
- alpha: 8–13 Hz;
- beta: 13–30 Hz;
- gamma: 30–45 Hz.

However, Table 3 states that DEAP is band-pass filtered at 4–30 Hz. If that 4–30 Hz filter is applied to the same signal before bandwise DWT extraction, conventional delta and gamma components cannot be fully represented. In addition, the common provider-preprocessed DEAP release has already undergone provider-side filtering (commonly documented as approximately 4–45 Hz), so a delta band below 4 Hz cannot be reconstructed from that release.

The manuscript does not explain whether:
1. the 4–30 Hz filter is used for a separate preprocessing/graph branch while bandwise features are derived from another signal; or
2. the five-band description is generic and the actual DEAP features use only bands supported by the 4–30 Hz preprocessing; or
3. the authors used a different DEAP release or processing sequence.

Do not silently choose one of these interpretations. In particular, do not apply 4–30 Hz globally and then label resulting features as complete delta and gamma bands.

## 5. Original records still required before exact extractor implementation

The manuscript supplies the filter cutoff range, named artifact method, normalization type, epoch duration, and band edges. It does not supply enough detail to reproduce the exact signal-processing implementation.

Please locate the original experimental script/configuration or logs that establish:
1. exact band-pass filter design (Butterworth/FIR/etc.), order, phase handling, and per-band filter settings;
2. how the 4–30 Hz preprocessing relates to separate delta/theta/alpha/beta/gamma extraction;
3. whether DEAP provider-preprocessed files were used, and exactly which processing operations were retained or skipped;
4. DWT level;
5. coefficient representation;
6. boundary-extension mode;
7. coefficient normalization and final feature dimensions;
8. ICA/component-rejection details only if an additional ICA stage was truly run beyond provider preprocessing.

Until these are found, repository code may document confirmed manuscript and author statements, but must not claim an exact manuscript-specific DWT feature extractor. No arbitrary wavelet defaults should be selected merely to make the code run.
