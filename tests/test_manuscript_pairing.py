import pytest

torch = pytest.importorskip("torch")
from models.contrastive import (
    ManuscriptInfoNCELoss, build_manuscript_negative_mask
)


def test_negative_eligibility_matches_manuscript_or_rule():
    labels = torch.tensor([0, 0, 1, 0])
    subjects = ["S1", "S1", "S1", "S2"]
    trials = torch.tensor([1, 2, 1, 1])
    mask = build_manuscript_negative_mask(labels, subjects, trials)
    # 0 vs 1: same class, subject and trial? Different trial -> negative.
    assert mask[0, 1]
    # 0 vs 2: different emotion -> negative.
    assert mask[0, 2]
    # 0 vs 3: different subject -> negative.
    assert mask[0, 3]
    assert not mask.diagonal().any()


def test_same_label_subject_trial_is_not_a_negative():
    labels = [1, 1, 0]
    subjects = ["S1", "S1", "S1"]
    trials = [4, 4, 5]
    mask = build_manuscript_negative_mask(labels, subjects, trials)
    assert not mask[0, 1]
    assert mask[0, 2]


def test_manuscript_info_nce_is_finite_and_differentiable():
    torch.manual_seed(13)
    z1 = torch.randn(4, 12, requires_grad=True)
    z2 = (z1.detach() + 0.1 * torch.randn(4, 12)).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    subjects = ["A", "A", "A", "B"]
    trials = torch.tensor([1, 2, 1, 1])
    loss = ManuscriptInfoNCELoss(temperature=0.1)(
        z1, z2, labels, subjects, trials)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    assert z2.grad is not None and torch.isfinite(z2.grad).all()


def test_manuscript_info_nce_requires_eligible_negatives():
    z1 = torch.randn(2, 5)
    z2 = torch.randn(2, 5)
    with pytest.raises(ValueError, match="eligible negative"):
        ManuscriptInfoNCELoss()(
            z1, z2, [0, 0], ["S1", "S1"], [1, 1])


def test_metadata_lengths_must_match_embeddings():
    z1 = torch.randn(3, 5)
    z2 = torch.randn(3, 5)
    with pytest.raises(ValueError, match="equal length"):
        ManuscriptInfoNCELoss()(
            z1, z2, [0, 1, 0], ["S1", "S2"], [1, 2, 3])
