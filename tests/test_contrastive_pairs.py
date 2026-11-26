import pytest
torch = pytest.importorskip("torch")
from torch.utils.data import Dataset
from data.contrastive_pairs import SameSampleTwoViewDataset, validate_same_sample_metadata


class ToyEEGDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, index):
        return {"eeg": torch.full((2, 3), float(index)),
                "label": index % 2, "subject_id": index // 2, "trial_id": index}


def test_two_views_come_from_same_epoch_and_transform_runs_twice():
    calls = []
    def augment(x):
        calls.append(float(x[0, 0]))
        return x + torch.randn_like(x)
    item = SameSampleTwoViewDataset(ToyEEGDataset(), transform=augment)[2]
    assert item["view1"].shape == item["view2"].shape == (2, 3)
    assert calls == [2.0, 2.0]
    assert item["subject_id"] == 1 and item["trial_id"] == 2
    assert item["sample_index"] == 2
    assert not torch.equal(item["view1"], item["view2"])


def test_identity_views_are_copies():
    item = SameSampleTwoViewDataset(ToyEEGDataset())[1]
    assert torch.equal(item["view1"], item["view2"])
    assert item["label"] == 1


def test_metadata_validation():
    assert validate_same_sample_metadata(
        subject_ids=[1, 2], trial_ids=[3, 4],
        emotion_labels=[0, 1], num_samples=2)
    with pytest.raises(ValueError):
        validate_same_sample_metadata(subject_ids=[1], num_samples=2)
