import pytest
torch = pytest.importorskip("torch")
from models.nsgcl import NSGCLModel


def test_integrated_model_forward_loss_and_backward():
    model = NSGCLModel(
        num_classes=2, cnn_channels=4, spatial_dim=8, gcn_hidden_dim=8,
        temporal_dim=16, temporal_layers=1, temporal_heads=4,
        temporal_ff_dim=32, fusion_dim=12, projection_hidden_dim=10,
        projection_dim=6, classifier_hidden_dim=8, dropout=0.0, gcn_dropout=0.0)
    x1 = torch.randn(3, 2, 5, 16, 20)
    x2 = x1 + 0.02 * torch.randn_like(x1)
    adjacency = torch.eye(5)
    out1, out2 = model(x1, adjacency), model(x2, adjacency)
    losses = model.loss(out1, out2, torch.tensor([0, 1, 0]))
    assert out1["logits"].shape == (3, 2)
    assert out1["projection"].shape == (3, 6)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.classifier[0].weight.grad is not None


def test_integrated_model_rejects_invalid_class_count():
    with pytest.raises(ValueError):
        NSGCLModel(num_classes=1)
