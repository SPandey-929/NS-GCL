"""Same-instance two-view dataset wrapper for NS-GCL."""
from collections.abc import Mapping
import torch
from torch.utils.data import Dataset


class SameSampleTwoViewDataset(Dataset):
    """Create two independent augmentations of each single EEG epoch.

    The base dataset must return a mapping containing 'eeg' or
    'time_frequency'. Subject, trial and emotion fields are retained as
    metadata only; they do not alter positive-pair assignment.
    """
    def __init__(self, dataset, transform=None, eeg_key=None):
        self.dataset = dataset
        self.transform = transform
        self.eeg_key = eeg_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("Base dataset items must be mappings.")
        key = self.eeg_key
        if key is None:
            key = "eeg" if "eeg" in sample else "time_frequency"
        if key not in sample:
            raise KeyError("Sample must contain 'eeg' or 'time_frequency'.")
        original = sample[key]
        if not torch.is_tensor(original):
            original = torch.as_tensor(original)
        if self.transform is None:
            view1, view2 = original.clone(), original.clone()
        else:
            view1 = self.transform(original.clone())
            view2 = self.transform(original.clone())
            if not torch.is_tensor(view1) or not torch.is_tensor(view2):
                raise TypeError("Augmentation transform must return tensors.")
        if view1.shape != view2.shape:
            raise ValueError("Both augmented views must have identical shapes.")
        result = dict(sample)
        result["view1"] = view1
        result["view2"] = view2
        result["sample_index"] = int(index)
        return result


def validate_same_sample_metadata(subject_ids=None, trial_ids=None,
                                  emotion_labels=None, num_samples=None):
    """Validate metadata lengths without using metadata to choose positives."""
    for name, values in (
        ("subject_ids", subject_ids), ("trial_ids", trial_ids),
        ("emotion_labels", emotion_labels)
    ):
        if values is not None and num_samples is not None:
            if len(values) != num_samples:
                raise ValueError(f"{name} must have one entry per EEG sample.")
    return True
