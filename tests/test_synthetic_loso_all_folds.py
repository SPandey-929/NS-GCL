"""End-to-end synthetic LOSO validation across every held-out subject.

This test validates software execution only. Its synthetic scores are not
DEAP/SEED/SEED-IV experimental results and must not be cited as such.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.loso import make_loso_folds
from scripts.train_loso import train_fold


def test_synthetic_loso_completes_all_subject_folds(tmp_path):
    rng = np.random.default_rng(2026)
    n_subjects, samples_per_subject = 3, 4
    n = n_subjects * samples_per_subject
    archive = {
        "features": rng.normal(
            size=(n, 2, 5, 8, 12)
        ).astype(np.float32),
        "labels": np.tile(np.array([0, 1, 0, 1], dtype=np.int64), n_subjects),
        "subject_ids": np.repeat(np.arange(n_subjects), samples_per_subject),
        "trial_ids": np.tile(np.arange(samples_per_subject), n_subjects),
        "adjacency": np.eye(5, dtype=np.float32),
    }
    folds = list(make_loso_folds(
        archive["subject_ids"], validation_subject_count=1, random_state=17
    ))
    assert len(folds) == n_subjects

    config = {
        "seed": 17,
        "augmentation": {
            "noise_std": 0.01,
            "max_time_shift_fraction": 0.05,
            "frequency_mask_probability": 0.1,
            "channel_dropout_probability": 0.0,
        },
        "model": {
            "cnn_channels": 2, "spatial_dim": 4, "gcn_hidden_dim": 4,
            "temporal_dim": 8, "temporal_layers": 1, "temporal_heads": 2,
            "temporal_ff_dim": 16, "fusion_dim": 8,
            "projection_hidden_dim": 8, "projection_dim": 4,
            "classifier_hidden_dim": 4, "dropout": 0.0,
            "gcn_dropout": 0.0, "contrastive_temperature": 0.1,
        },
        "training": {
            "batch_size": 2, "eval_batch_size": 4,
            "learning_rate": 0.001, "weight_decay": 0.0,
            "lambda_contrastive": 0.1, "max_epochs": 1,
            "patience": 1, "gradient_clip_norm": 1.0,
            "num_workers": 0,
        },
    }

    results = []
    for fold in folds:
        assert set(archive["subject_ids"][fold.train_indices]).isdisjoint(
            set(archive["subject_ids"][fold.validation_indices])
        )
        assert set(archive["subject_ids"][fold.train_indices]).isdisjoint(
            set(archive["subject_ids"][fold.test_indices])
        )
        assert set(archive["subject_ids"][fold.validation_indices]).isdisjoint(
            set(archive["subject_ids"][fold.test_indices])
        )
        result = train_fold(
            archive, fold, config, tmp_path, torch.device("cpu"), num_classes=2
        )
        fold_dir = tmp_path / f"heldout_{fold.held_out_subject}"
        assert (fold_dir / "best_model.pt").is_file()
        assert (fold_dir / "training_history.csv").is_file()
        assert (fold_dir / "test_metrics.json").is_file()
        assert result["n_test"] == samples_per_subject
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["f1_macro"] <= 1.0
        results.append(result)

    assert len({r["held_out_subject"] for r in results}) == n_subjects
    assert all(r["n_train"] == samples_per_subject for r in results)
    assert all(r["n_validation"] == samples_per_subject for r in results)
