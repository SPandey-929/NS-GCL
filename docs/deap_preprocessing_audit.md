# DEAP preprocessing protocol audit

## Author-confirmed manuscript settings

- Analysis band-pass: 4–30 Hz.
- Segmentation: non-overlapping 3-second windows.
- Selected electrodes: Fp1, Fp2, F3, Fz, F4, C3, Cz, C4, P3, Pz, P4.

## Input release and operations that must not be repeated

The intended input is the provider's preprocessed DEAP MATLAB release (`data_preprocessed_matlab`). The commonly distributed preprocessed files are sampled at 128 Hz and are supplied with provider-side signal processing, including EOG-related artifact processing and 4–45 Hz filtering. A common trial array has shape `(40 trials, 40 channels, 8064 samples)`: 32 EEG channels plus 8 peripheral channels, with 63 seconds per trial.

Accordingly, for this input variant, NS-GCL must:
- NOT downsample these files again.
- NOT run ICA or repeat provider-side EOG/artifact processing.
- NOT apply a second baseline subtraction.
- Remove the first 3 seconds (384 samples at 128 Hz) from 63-second trials as the pre-stimulus baseline interval; this is trimming, not baseline correction.
- Accept a 60-second / 7680-sample trial if the baseline interval has already been removed.
- Apply the manuscript's narrower 4–30 Hz analysis filter to the remaining stimulus signal.
- Segment the 60-second stimulus into 20 non-overlapping windows of 3 seconds (384 samples each).

The precise release and its accompanying documentation must be checked locally before processing. Do not infer its history from the filename alone.

## Repository implementation

`preprocessing/deap_pipeline.py` provides `prepare_preprocessed_deap_file`, which loads a participant file, selects the named 11 electrodes, checks for 8064 or 7680 samples per trial at 128 Hz, trims the baseline interval only when present, applies 4–30 Hz filtering, and segments into 3-second windows. It deliberately has no downsampling, baseline-subtraction, or ICA step.

Run focused tests:

```bash
python -m pytest tests/test_deap_preprocessed_pipeline.py -q
```

## Frequency-band consistency

A 4–30 Hz analysis filter cannot retain conventional delta 1–4 Hz or gamma 30–45 Hz. Do not report five-band delta/theta/alpha/beta/gamma features as being extracted from this filtered signal. The manuscript must either restrict this branch to compatible bands or describe and justify a separate signal branch for other frequency ranges.

## Remaining items before claiming complete manuscript reproduction

1. Verify the exact DEAP release and inspect participant file dimensions/keys and accompanying processing notes.
2. Confirm the manuscript's time-frequency transform details (wavelet family, decomposition level, coefficient/scalogram representation, frequency-bin mapping).
3. Confirm graph estimator, band, sparsification, and normalization settings.
4. Confirm final feature tensor shape and label threshold/equality rule.
5. Verify end-to-end conversion on authorized participant data and record archive metadata.

Until these are complete, the function is a preprocessing utility for the specified common file layouts, not a claim that the complete manuscript feature-generation and model pipeline has been reproduced.


## Manuscript-specific time-frequency extraction: specification gap

The manuscript's Section 3.2 says that time-frequency representations are obtained using "wavelet decomposition or STFT"; Section 4.2 more specifically says "wavelet decomposition". Neither section specifies the exact transform implementation or its parameters. The manuscript does not identify:
- Discrete wavelet transform (DWT), wavelet packet transform (WPT), or continuous wavelet transform (CWT).
- Wavelet family/mother wavelet and its parameters.
- Decomposition level or scale/frequency grid.
- Whether the representation uses coefficients, absolute coefficients, squared magnitude/power, or a scalogram.
- Any coefficient normalization, boundary/extension mode, or output resizing/binning rule.

The code must not select these settings silently. A reproducible extractor can be implemented as a parameterized utility only after the actual transform and parameter values are confirmed from the experiment code/configuration or author records. Until then, no transform-specific defaults are asserted as manuscript settings.

### Frequency-band conflict requiring correction

Section 4.2 lists delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz), while the DEAP protocol specifies 4–30 Hz filtering. A signal filtered to 4–30 Hz cannot support the full conventional delta and gamma bands. The author must confirm whether:
1. the time-frequency branch used a separate signal before the 4–30 Hz filter, or
2. the actual wavelet feature branch was restricted to frequencies within 4–30 Hz and the five-band description needs correction.

Do not generate, label, or publish five-band features from the 4–30 Hz signal. Do not claim the final feature-generation pipeline is manuscript-reproduced until this is resolved.
