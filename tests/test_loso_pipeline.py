import numpy as np
import pytest
from training.loso import make_loso_folds, FeatureStandardizer
from training.metrics import classification_metrics
from data.feature_archive import load_feature_archive

def test_loso_subject_partitions_are_disjoint_and_scaler_uses_train_only():
    subjects = np.repeat(np.arange(5), 3)
    folds = list(make_loso_folds(subjects, random_state=13))
    assert len(folds) == 5
    for fold in folds:
        train_s, val_s, test_s = set(subjects[fold.train_indices]), set(subjects[fold.validation_indices]), set(subjects[fold.test_indices])
        assert len(test_s) == 1
        assert train_s.isdisjoint(val_s) and train_s.isdisjoint(test_s) and val_s.isdisjoint(test_s)
        assert np.array_equal(fold.scaler_fit_indices, fold.train_indices)

def test_feature_standardizer_uses_only_fit_array():
    train = np.ones((3, 2, 4, 5, 6), dtype=np.float32) * 2
    heldout = np.ones((2, 2, 4, 5, 6), dtype=np.float32) * 100
    scaler = FeatureStandardizer().fit(train)
    assert np.allclose(scaler.mean_, 2.0) and np.all(scaler.transform(heldout) > 0)

def test_metrics_report_macro_and_confusion_matrix():
    result = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], labels=[0, 1])
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]

def test_archive_rejects_missing_adjacency(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, features=np.zeros((4, 2, 3, 4, 5), np.float32), labels=np.array([0, 1, 0, 1]), subject_ids=np.array([0, 0, 1, 1]), trial_ids=np.arange(4))
    with pytest.raises(KeyError):
        load_feature_archive(path)
