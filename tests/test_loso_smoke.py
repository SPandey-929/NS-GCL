import numpy as np
import pytest
torch = pytest.importorskip("torch")
from training.loso import make_loso_folds
from scripts.train_loso import train_fold


def test_one_tiny_loso_fold_trains_saves_and_evaluates(tmp_path):
    rng = np.random.default_rng(4)
    n_subjects, samples_per_subject = 3, 4
    n = n_subjects * samples_per_subject
    features = rng.normal(size=(n, 2, 5, 12, 16)).astype(np.float32)
    labels = np.tile(np.array([0, 1, 0, 1], dtype=np.int64), n_subjects)
    subjects = np.repeat(np.arange(n_subjects), samples_per_subject)
    archive = {
        "features": features,
        "labels": labels,
        "subject_ids": subjects,
        "trial_ids": np.tile(np.arange(samples_per_subject), n_subjects),
        "adjacency": np.eye(5, dtype=np.float32),
    }
    fold = next(make_loso_folds(subjects, random_state=3))
    config = {
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
    result = train_fold(archive, fold, config, tmp_path, torch.device("cpu"), 2)
    fold_dir = tmp_path / f"heldout_{fold.held_out_subject}"
    assert (fold_dir / "best_model.pt").is_file()
    assert (fold_dir / "training_history.csv").is_file()
    assert (fold_dir / "test_metrics.json").is_file()
    assert 0.0 <= result["accuracy"] <= 1.0
