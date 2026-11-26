# Real-data execution guide for reviewers

## Dataset access

- DEAP: request data from the official [DEAP dataset page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) and comply with its access and data-use terms.
- SEED / SEED-IV: use the official [SEED project portal](https://bcmi.sjtu.edu.cn/home/seed/) and comply with its terms.

The repository does not redistribute recordings. Keep downloaded data outside the public repository.

## Environment and software tests

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## DEAP preprocessed MATLAB input

The intended input is the provider's preprocessed MATLAB release. A common file layout has `data` with shape `(40, 40, 8064)` and `labels` with shape `(40, 4)` per participant. The first 32 channels are EEG; the remaining eight are peripheral signals. The common 8064-sample trial contains 3 seconds of baseline plus 60 seconds of stimulus at 128 Hz. Confirm the exact release and metadata accompanying the downloaded files.

The preprocessing function does not downsample, run ICA/EOG removal, or subtract a baseline again. It trims the initial 3 seconds if present, applies the manuscript's 4–30 Hz analysis filter, selects the manuscript's 11 named EEG electrodes, and segments the 60-second stimulus into 20 non-overlapping 3-second windows.

**Filter implementation caveat:** the manuscript specifies the 4–30 Hz cutoffs but does not state the filter family, order, or phase handling. The current code uses a fourth-order Butterworth filter with zero-phase forward/backward application as a provisional repository choice. This is not verified as the original experimental implementation. See `configs/deap.yaml` and `docs/deap_preprocessing_audit.md`.

Check one participant file locally:

```python
from scipy.io import loadmat
m = loadmat("data/DEAP/data_preprocessed_matlab/s01.mat")
print(m["data"].shape, m["labels"].shape)
```

Run the preprocessing function:

```python
from preprocessing.deap_pipeline import prepare_preprocessed_deap_file
windows, labels, window_meta = prepare_preprocessed_deap_file(
    "data/DEAP/data_preprocessed_matlab/s01.mat"
)
print(windows.shape, labels.shape, window_meta.shape)
```

Expected shapes for the common 40-trial release: `(800, 11, 384)`, `(800, 4)`, and `(800, 2)`. This is filtered-window output, not yet the manuscript's final time-frequency feature archive.

## Real feature-archive schema 1.1

The NPZ loader requires six arrays: `features`, `labels`, `subject_ids`,
`trial_ids`, `adjacency`, and scalar JSON string `metadata_json`.
Features have shape `(N, sequence, electrodes, frequency_bins, time_bins)`.
Labels and trial IDs are integer arrays of length N; subject IDs are integer
or string arrays of length N. Adjacency may be shared `(E,E)`,
sample-specific `(N,E,E)`, or sequence-specific `(N,sequence,E,E)`.

Synthetic archives remain compatible with schema 1.0 and must explicitly
identify synthetic provenance. Real archives require metadata schema 1.1 and
the following additional traceability fields:

| Field | Required information |
|---|---|
| `dataset_version`, `source_release` | Exact dataset version and source/release description |
| `feature_generation` | Method, parameter object, and parameter status |
| `frequency_bands_hz` | Named band edges, or null if not applicable |
| `label_mapping` | Class names and explicit threshold/equality/mapping rule |
| `graph_construction` | Connectivity, sparsification, and normalization |
| `evaluation_protocol` | Strategy, split definition, validation policy |
| `software_environment` | Python version and package-version map |
| `data_access` | Access status and usage/permission note |

Base metadata remains mandatory: dataset name, feature method, channel names,
sampling rate, preprocessing description, and random seed. Unresolved values
must be represented explicitly rather than guessed. For example, a DWT level
may remain null while `parameter_status` is `partially_confirmed`.
The validator checks presence and basic structure; it does not certify
scientific correctness or prove that the stated operations were performed.

A minimal schema illustration (not a completed experimental record):

```json
{
  "schema_version": "1.1",
  "provenance": "real",
  "dataset_name": "DEAP",
  "dataset_version": "record exact downloaded release",
  "source_release": "record provider release and file variant",
  "feature_method": "time_frequency_features",
  "channel_names": ["F3", "F4"],
  "sampling_rate_hz": 128,
  "preprocessing": {"record exact steps and parameters": true},
  "random_seed": 42,
  "feature_generation": {
    "method": "DWT",
    "parameters": {"wavelet": "db4", "level": null},
    "parameter_status": "partially_confirmed"
  },
  "frequency_bands_hz": null,
  "label_mapping": {
    "class_names": ["negative", "positive"],
    "rule": "record exact threshold and equality handling"
  },
  "graph_construction": {
    "connectivity": "record method",
    "sparsification": "record method and parameters",
    "normalization": "record method"
  },
  "evaluation_protocol": {
    "strategy": "LOSO",
    "split_definition": "record held-out subject/fold assignment",
    "validation_policy": "record validation subject selection"
  },
  "software_environment": {
    "python": "record exact version",
    "packages": {"numpy": "record exact version", "torch": "record exact version"}
  },
  "data_access": {
    "access_status": "authorized",
    "usage_note": "record applicable access/use terms; do not include credentials"
  }
}
```

This example contains placeholders and must not be copied as if it described
an actual experiment.

Run the archive tests:

```bash
python -m pytest tests/test_synthetic_archive.py -q
```

## Synthetic software check

```bash
python -m scripts.generate_synthetic_archive --output synthetic_data/synthetic_features.npz --seed 2026
python -m pytest tests/test_synthetic_archive.py -q
```

Synthetic inputs are not EEG and must never be reported as experimental results.

## Remaining feature-generation stage

The repository still needs the exact manuscript wavelet implementation, graph construction settings, final tensor assembly, and archive-writing command. The current LOSO trainer must not be treated as a complete reproduction until a real archive has been generated and validated using those specifications.

Once a verified real archive exists, the current training scaffold command is:

```bash
python -m scripts.train_loso --data path/to/real_features.npz --config configs/loso_deap.yaml --output outputs/real_deap_loso
```

Keep software tests, synthetic runs, and actual dataset results separate. Only actual experiments on authorized recordings can support manuscript performance claims.
