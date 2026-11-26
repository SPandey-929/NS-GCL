# DEAP preprocessing protocol audit

## Manuscript statements located

The current manuscript contains conflicting preprocessing descriptions:

- Section 4.1 / Table 3: DEAP sampled at 512 Hz and downsampled to 128 Hz; 4–30 Hz filtering, ICA artifact removal, z-score normalization, and 3-second non-overlapping windows.
- Section 3.2: filtering, artifact removal, and baseline correction; wavelet decomposition or STFT; delta, theta, alpha, beta, and gamma bands.
- Section 4.2: wavelet decomposition; delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz.
- A separate manuscript excerpt/version describes 4–45 Hz filtering and one-second windows. It conflicts with the main table and Section 4.2 and must not silently override them.

## Repository implementation

Available utilities include a common-layout MATLAB loader, named 11-channel selection, configurable zero-phase Butterworth filtering (default 4–30 Hz), optional manual baseline correction, optional ICA with explicitly supplied component exclusions, three-second non-overlapping segmentation, and channel-wise z-score normalization.

The repository does not yet implement a complete, verified raw-DEAP-to-feature-archive pipeline. In particular, there is no documented 512-to-128 Hz conversion command, reproducible ICA component identification procedure, wavelet feature extraction, or archive-generation pipeline.

## Decisions required before claiming manuscript compliance

1. Confirm 4–30 Hz versus 4–45 Hz.
2. Confirm whether DEAP baseline correction is actually applied and whether the chosen files are raw or already preprocessed.
3. Confirm 3-second versus 1-second segmentation.
4. Specify the resampling method and whether resampling is required for the exact DEAP release.
5. Specify ICA implementation, fitting scope, artifact-component criteria, and exclusions.
6. Specify wavelet family, decomposition level, coefficient/scalogram construction, frequency-bin mapping, and gamma handling.
7. Confirm the exact feature tensor dimensions and graph construction settings.

A 4–30 Hz band-pass cannot retain conventional gamma activity at 30–45 Hz. Until these questions are resolved and implemented, preprocessing modules are reusable utilities—not a manuscript-verified feature-generation pipeline.
