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

Run focused tests:

```bash
python -m pytest tests/test_deap_preprocessed_pipeline.py -q
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
