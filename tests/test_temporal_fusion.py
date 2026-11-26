import pytest

torch = pytest.importorskip("torch")
from models.temporal_encoder import TemporalTransformerEncoder
from models.spatial_temporal_fusion import SpatialTemporalFusion
from models.spatiotemporal_encoder import NSGCLRepresentationEncoder


def test_temporal_encoder_shape_and_gradient():
    model = TemporalTransformerEncoder(input_dim=16, embedding_dim=32,
        num_layers=2, num_heads=4, feedforward_dim=64, dropout=0.0)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 5, 32)
    y.mean().backward()
    assert x.grad is not None


def test_fusion_shape_and_padding_mask():
    fusion = SpatialTemporalFusion(16, 32, 24, dropout=0.0)
    spatial, temporal = torch.randn(2, 4, 16), torch.randn(2, 4, 32)
    mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    sequence, pooled = fusion(spatial, temporal, mask)
    assert sequence.shape == (2, 4, 24)
    assert pooled.shape == (2, 24)
    assert torch.isfinite(pooled).all()


def test_composed_representation_encoder_shapes():
    model = NSGCLRepresentationEncoder(cnn_channels=4, spatial_dim=8,
        gcn_hidden_dim=8, temporal_dim=16, temporal_layers=1,
        temporal_heads=4, temporal_ff_dim=32, fusion_dim=12,
        dropout=0.0, gcn_dropout=0.0)
    x = torch.randn(2, 3, 5, 16, 20)
    adjacency = torch.eye(5).repeat(2, 3, 1, 1)
    out = model(x, adjacency)
    assert out["node_embeddings"].shape == (2, 3, 5, 8)
    assert out["spatial_sequence"].shape == (2, 3, 8)
    assert out["temporal_sequence"].shape == (2, 3, 16)
    assert out["fused_sequence"].shape == (2, 3, 12)
    assert out["clip_embedding"].shape == (2, 12)
