import numpy as np
import pytest

torch = pytest.importorskip("torch")
from neuro_symbolic.physiology import (
    compute_frontal_alpha_asymmetry, compute_beta_plv_strength
)
from models.neuro_symbolic import NeuroSymbolicRegularizer


def test_faa_and_beta_plv_feature_extractors():
    fs = 128
    time = np.arange(512) / fs
    left = np.sin(2 * np.pi * 10 * time)
    right = 1.5 * np.sin(2 * np.pi * 10 * time + 0.2)
    eeg = np.stack([left, right, np.sin(2*np.pi*20*time)], axis=0)[None, ...]
    faa = compute_frontal_alpha_asymmetry(
        eeg, fs, ["F3", "F4", "Cz"])
    beta = compute_beta_plv_strength(eeg, fs)
    assert faa.shape == (1,)
    assert beta.shape == (1,)
    assert np.isfinite(faa).all()
    assert 0 <= beta[0] <= 1


def test_neuro_symbolic_loss_and_probability_gradients():
    criterion = NeuroSymbolicRegularizer(
        faa_margin=0.1, beta_plv_threshold=0.3,
        positive_faa_sign=1, lambda_faa=1.0, lambda_beta=1.0)
    p_pos = torch.tensor([0.7, 0.2], requires_grad=True)
    p_neg = torch.tensor([0.3, 0.8], requires_grad=True)
    p_high = torch.tensor([0.8, 0.6], requires_grad=True)
    result = criterion(
        faa=torch.tensor([0.2, -0.2]),
        beta_plv=torch.tensor([0.2, 0.4]),
        positive_probability=p_pos,
        negative_probability=p_neg,
        high_arousal_probability=p_high)
    assert set(result) == {"loss", "faa_loss", "beta_plv_loss"}
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert p_pos.grad is not None
    assert p_neg.grad is not None
    assert p_high.grad is not None


def test_regularizer_requires_explicit_faa_direction():
    with pytest.raises(ValueError):
        NeuroSymbolicRegularizer()
    with pytest.raises(ValueError):
        NeuroSymbolicRegularizer(positive_faa_sign=0)


def test_invalid_beta_plv_is_rejected():
    criterion = NeuroSymbolicRegularizer(positive_faa_sign=1)
    with pytest.raises(ValueError):
        criterion(torch.tensor([0.1]), torch.tensor([1.2]),
                  torch.tensor([0.5]), torch.tensor([0.5]),
                  torch.tensor([0.5]))
