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
