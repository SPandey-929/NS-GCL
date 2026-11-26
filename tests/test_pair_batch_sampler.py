import pytest
torch = pytest.importorskip("torch")
from data.contrastive_pairs import ManuscriptPairBatchSampler


def test_sampler_builds_batches_with_negative_for_every_anchor():
    labels = [0, 0, 1, 1, 0, 1]
    subjects = ["S1"] * 6
    trials = [1, 1, 2, 2, 3, 3]
    sampler = ManuscriptPairBatchSampler(labels, subjects, trials, batch_size=3, seed=5)
    batches = list(sampler)
    assert batches
    for batch in batches:
        keys = {(labels[i], subjects[i], trials[i]) for i in batch}
        assert len(keys) >= 2


def test_sampler_rejects_data_without_any_eligible_negatives():
    with pytest.raises(ValueError, match="fewer than two distinct"):
        ManuscriptPairBatchSampler([0, 0], ["S1", "S1"], [1, 1], batch_size=2)
