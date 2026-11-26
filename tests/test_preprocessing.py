import numpy as np
from preprocessing.filtering import bandpass_filter
from preprocessing.segmentation import segment_trials
from preprocessing.normalization import ChannelZScore
from preprocessing.deap_loader import DEAP_CHANNELS, SELECTED_CHANNELS, select_channels
from preprocessing.baseline import baseline_correct_trials


def test_three_second_non_overlapping_segmentation():
    x = np.zeros((2, 3, 128 * 7), dtype=np.float32)
    labels = np.array([[1, 2], [3, 4]])
    windows, y, meta = segment_trials(x, labels, 128, 3)
    assert windows.shape == (4, 3, 384)
    assert y.shape == (4, 2)
    assert meta.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_zscore_scaler_uses_training_statistics():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(5, 2, 100)).astype(np.float32)
    test = np.ones((2, 2, 100), dtype=np.float32) * 50
    scaler = ChannelZScore().fit(train)
    transformed = scaler.transform(test)
    assert transformed.shape == test.shape
    assert np.isfinite(transformed).all()
    assert not np.allclose(transformed.mean(axis=(0, 2)), 0)


def test_filter_preserves_shape():
    x = np.random.default_rng(9).normal(size=(2, 3, 1024)).astype(np.float32)
    y = bandpass_filter(x, 128, 4, 30)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_manuscript_electrodes_in_declared_deap_montage():
    x = np.zeros((1, len(DEAP_CHANNELS), 100), dtype=np.float32)
    selected = select_channels(x, SELECTED_CHANNELS, DEAP_CHANNELS)
    assert selected.shape == (1, 11, 100)


def test_baseline_correction_returns_stimulus_interval():
    x = np.ones((2, 3, 128 * 5), dtype=np.float32) * 4
    corrected = baseline_correct_trials(x, sampling_rate=128, baseline_seconds=3)
    assert corrected.shape == (2, 3, 128 * 2)
    assert np.allclose(corrected, 0)
