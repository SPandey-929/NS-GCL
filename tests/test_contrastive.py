import pytest

torch = pytest.importorskip("torch")
from models.contrastive import (
    ContrastiveProjectionHead, InfoNCELoss, NSGCLContrastiveModel
)


def test_info_nce_returns_finite_scalar_and_gradients():
    z1 = torch.randn(6, 16, requires_grad=True)
    z2 = (z1.detach() + 0.05 * torch.randn(6, 16)).requires_grad_()
    loss = InfoNCELoss(temperature=0.1)(z1, z2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None
    assert z2.grad is not None


def test_matching_views_have_lower_loss_than_shuffled_positives():
    torch.manual_seed(7)
    z1 = torch.randn(12, 32)
    z2 = z1 + 0.02 * torch.randn_like(z1)
    criterion = InfoNCELoss(temperature=0.1)
    matched = criterion(z1, z2)
    shuffled = criterion(z1, z2[torch.randperm(z2.shape[0])])
    assert matched < shuffled


def test_projection_head_normalizes_embeddings():
    head = ContrastiveProjectionHead(20, 16, 8, dropout=0.0)
    output = head(torch.randn(5, 20))
    assert output.shape == (5, 8)
    assert torch.allclose(output.norm(dim=1), torch.ones(5), atol=1e-5)


def test_info_nce_rejects_invalid_batch_and_temperature():
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)
    with pytest.raises(ValueError):
        InfoNCELoss()(torch.randn(1, 8), torch.randn(1, 8))


def test_contrastive_wrapper_attaches_projection():
    class DummyEncoder(torch.nn.Module):
        def forward(self, x):
            return {"clip_embedding": x, "other": "kept"}

    model = NSGCLContrastiveModel(DummyEncoder(), input_dim=10,
                                  hidden_dim=12, projection_dim=6, dropout=0.0)
    result = model(torch.randn(4, 10))
    assert result["projection"].shape == (4, 6)
    assert result["other"] == "kept"
