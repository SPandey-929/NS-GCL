import numpy as np
import pytest
from scipy.io import savemat
from preprocessing.deap_pipeline import prepare_preprocessed_deap_file


def _write_deap(path, samples):
    rng = np.random.default_rng(13)
    # Common DEAP preprocessed MATLAB structure: trials x 40 channels x time.
    data = rng.normal(size=(2, 40, samples)).astype(np.float32)
    labels = np.array([[6, 7, 5, 5], [3, 4, 5, 5]], dtype=np.float32)
    savemat(path, {"data": data, "labels": labels})


def test_preprocessed_63s_release_discards_baseline_without_subtraction(tmp_path):
    path = tmp_path / "s01.mat"
    _write_deap(path, 63 * 128)
    windows, labels, meta = prepare_preprocessed_deap_file(path)
    # 60 seconds / 3-second windows = 20 windows per trial.
    assert windows.shape == (40, 11, 384)
    assert labels.shape == (40, 4)
    assert meta.shape == (40, 2)
    assert meta[0].tolist() == [0, 0]
    assert meta[19].tolist() == [0, 19]
    assert meta[20].tolist() == [1, 0]


def test_already_trimmed_60s_release_is_accepted(tmp_path):
    path = tmp_path / "s01.mat"
    _write_deap(path, 60 * 128)
    windows, _, _ = prepare_preprocessed_deap_file(path)
    assert windows.shape == (40, 11, 384)


def test_pipeline_refuses_unknown_sampling_rate_and_length(tmp_path):
    path = tmp_path / "s01.mat"
    _write_deap(path, 60 * 128)
    with pytest.raises(ValueError, match="never downsamples"):
        prepare_preprocessed_deap_file(path, sampling_rate=512)
    bad = tmp_path / "bad.mat"
    _write_deap(bad, 62 * 128)
    with pytest.raises(ValueError, match="Unexpected DEAP trial length"):
        prepare_preprocessed_deap_file(bad)
