# DEAP preprocessing protocol audit

## Author-confirmed settings

The author has confirmed that the controlling DEAP settings are:
- Band-pass filtering: 4–30 Hz.
- Segmentation: non-overlapping 3-second windows.

These settings supersede the conflicting 4–45 Hz / one-second description in an earlier manuscript excerpt. The code defaults in `preprocessing/filtering.py` and `preprocessing/segmentation.py` match the confirmed filter range and window duration.

## Manuscript-to-code comparison

| Processing step | Confirmed manuscript protocol / status | Repository status |
|---|---|---|
| MATLAB loading | DEAP participant files | `load_deap_mat` supports common trial-wise `data` / `labels` arrays; exact source release and variable shapes must still be checked |
| EEG channels | 11 selected channels: Fp1, Fp2, F3, Fz, F4, C3, Cz, C4, P3, Pz, P4 | Name-based channel selection is implemented |
| Filtering | 4–30 Hz | Zero-phase Butterworth implementation; default matches |
| Segmentation | 3 seconds, non-overlapping | Implemented; at 128 Hz this is 384 samples per window |
| Baseline correction | Not yet confirmed for the actual DEAP file variant | Optional utility only; must not be enabled by assumption |
| ICA | Manuscript states artifact removal, but component-selection method is not specified in enough detail | Optional manual FastICA utility; no automatic artifact detection |
| Normalization | Z-score | Channel-wise scaler exists; fit it on training-subject data only in each LOSO fold |
| Time-frequency representation | Wavelet decomposition is described | Exact wavelet family, level, coefficient/scalogram construction, and binning are not yet implemented as a manuscript-verified step |
| Graph construction | Connectivity graph | Verify frequency band, edge estimator, threshold/top-k and normalization settings against the final method description |

## Important frequency-band inconsistency

The manuscript lists delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz). However, the author-confirmed 4–30 Hz filter removes frequencies below 4 Hz and above 30 Hz. Therefore, the filtered signal cannot contain the conventional delta 1–4 Hz or gamma 30–45 Hz components. The current 4–30 Hz signal can support theta, alpha, and beta analysis (subject to filter transition characteristics).

Do not claim five-band delta/theta/alpha/beta/gamma features from a signal filtered at 4–30 Hz. The authors must either:
1. revise the manuscript's frequency-band list to bands supported by the confirmed filter, or
2. confirm that a separate unfiltered / differently filtered signal is used for the five-band decomposition and document that separate branch.

## Remaining information needed to complete feature-generation verification

1. Exact DEAP file release: raw 512-Hz data or preprocessed 128-Hz data. Do not downsample a 128-Hz release again.
2. Whether baseline correction is applied to this release, and how the baseline interval is identified.
3. ICA implementation and artifact-component identification/rejection rule. Manual component indices cannot be generalized across participants.
4. Exact wavelet family, decomposition level, coefficient/scalogram representation, frequency-bin mapping, and treatment of edge coefficients.
5. Whether the graph is built from raw filtered epochs or from band-specific signals, and the exact connectivity/sparsification parameters.
6. Exact feature tensor shape supplied to the CNN/GCN/Transformer.

Until these are specified and the complete conversion is implemented and tested against an authorized DEAP file, the repository preprocessing utilities should not be described as a fully reproduced manuscript feature-generation pipeline.
