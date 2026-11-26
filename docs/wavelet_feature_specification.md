# Wavelet feature-generation specification status

## Confirmed by the author

| Item | Confirmed value | Evidence |
|---|---|---|
| Transform | Discrete Wavelet Transform (DWT) | Author confirmation in the project conversation |
| Wavelet family | Daubechies-4 (`db4`) | Author confirmation in the project conversation |
| Frequency-band procedure | Each frequency band is processed separately | Author confirmation in the project conversation |

These three values are treated as author-confirmed protocol details. They are not inferred from generic EEG practice.

## Inspection of the manuscript and repository configuration

The current manuscript (Sections 3.2 and 4.2, and Algorithm 1) says that wavelet decomposition is used to obtain time-frequency and frequency-band information. It lists delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz). It does not state the DWT family, decomposition level, coefficient representation, boundary-extension mode, normalization, coefficient-to-frequency mapping, or feature-map dimensions.

The repository configurations inspected were:
- `configs/deap.yaml`
- `configs/loso_deap.yaml`
- `configs/graph.yaml`

None contains a wavelet configuration or records the missing DWT parameters. The current repository has no original experiment script/configuration that establishes these values. The manuscript itself contains no occurrence of `db4`.

## Settings deliberately left unresolved

| Parameter | Status | Why it must not be guessed |
|---|---|---|
| DWT level | Not found | Determines approximation/detail scales and coefficient lengths |
| Coefficient representation | Not found | Raw signed coefficients, absolute values, and squared energy are different features |
| Extension/boundary mode | Not found | Changes coefficients near each epoch boundary |
| Per-band filtering implementation | Partially specified | “Separate bands” is confirmed, but exact filter design/order and edge handling are not recorded |
| Band definitions | Listed in manuscript | Must be reconciled with the input signal's existing 4–45 Hz provider filtering and the manuscript's 4–30 Hz analysis filter |
| Coefficient normalization | Not found | Changes feature scale and can affect downstream learning |
| Time-frequency image/feature shape | Not found | Needed to match the model input contract |
| Resizing, padding, or truncation | Not found | Must not be introduced merely to force a tensor shape |

## Important signal-band compatibility issue

The common preprocessed DEAP release is supplied after provider-side filtering in approximately the 4–45 Hz range. If this exact release is used, a separate 1–4 Hz delta band cannot be recovered from it because frequencies below the provider's lower cutoff have already been attenuated. Its 30–45 Hz gamma range may remain available, subject to the exact file version and filter response.

The manuscript separately specifies a 4–30 Hz filtering step. That step is not compatible with extracting a conventional 30–45 Hz gamma band from the same filtered signal. A defensible implementation therefore needs an explicitly documented branch design: which signal version is input to each band filter, and whether the 4–30 Hz step applies to the time-domain/graph branch or to all feature branches. Do not silently apply 4–30 Hz before generating gamma.

No repeated downsampling, baseline subtraction, or provider-side artifact processing should be added to the preprocessed DEAP input.

## Required original experimental record

Before coding the final feature extractor, locate or provide the original experiment source/configuration (for example, Python/MATLAB script, notebook, YAML/JSON config, saved feature-generation command, or a methods log) that records:
1. DWT level;
2. coefficient representation (raw, absolute, power/energy, or another explicit calculation);
3. extension mode;
4. exact band-pass filter definitions for each separate band (filter type/order and cutoffs);
5. signal branch used for each band, including whether the 4–30 Hz filter is applied before or separately from the five-band branch;
6. coefficient normalization and final feature-map dimensions;
7. any padding, truncation, resizing, or coefficient aggregation.

Until those records are available, the repository should document DWT + db4 + separate band processing as author-confirmed, but should not implement a purported manuscript-exact extractor. A generic parameterized DWT utility may be added later as a separate optional tool, clearly labeled as such, but it must not be presented as reproducing the reported experiments.

## Current status

**No manuscript-specific wavelet extractor has been implemented or validated.** This is intentional. Software tests on an arbitrarily configured DWT would only validate that arbitrary implementation, not the manuscript's actual feature-generation procedure.
