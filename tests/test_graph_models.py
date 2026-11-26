import numpy as np
import pytest

torch = pytest.importorskip("torch")
from graph.connectivity import compute_plv, compute_pli
from graph.adjacency import build_adjacency, normalize_adjacency
from models.multiscale_cnn import MultiScaleCNN
from models.gcn import GraphEncoder


def test_connectivity_outputs_symmetric_bounded_matrices():
    rng = np.random.default_rng(4)
    eeg = rng.normal(size=(2, 4, 768))
    plv = compute_plv(eeg, 128, 13, 30)
    pli = compute_pli(eeg, 128, 13, 30)
    assert plv.shape == (2, 4, 4)
    assert pli.shape == (2, 4, 4)
    assert np.allclose(plv, plv.transpose(0, 2, 1), atol=1e-6)
    assert np.allclose(pli, pli.transpose(0, 2, 1), atol=1e-6)
    assert np.all((plv >= 0) & (plv <= 1))
    assert np.all((pli >= 0) & (pli <= 1))
    assert np.allclose(np.diagonal(plv, axis1=1, axis2=2), 1)
    assert np.allclose(np.diagonal(pli, axis1=1, axis2=2), 0)


def test_adjacency_is_symmetric_and_normalized():
    c = np.array([[1, .8, .2], [.8, 1, .6], [.2, .6, 1]], dtype=np.float32)
    a = build_adjacency(c, threshold=.1, top_k=1)
    assert np.allclose(a, a.T)
    n = normalize_adjacency(a)
    assert n.shape == (3, 3)
    assert np.isfinite(n).all()


def test_multiscale_cnn_and_gcn_shapes():
    x = torch.randn(2, 11, 32, 48)
    cnn = MultiScaleCNN(out_channels=8, embedding_dim=16)
    node_features = cnn(x)
    assert node_features.shape == (2, 11, 16)
    adjacency = torch.eye(11).repeat(2, 1, 1)
    gcn = GraphEncoder(in_features=16, hidden_features=12, embedding_dim=10)
    nodes, graph = gcn(node_features, adjacency)
    assert nodes.shape == (2, 11, 10)
    assert graph.shape == (2, 10)
