"""Dataset wrappers and batch construction for manuscript-specific pairs."""
from collections.abc import Mapping
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SameSampleTwoViewDataset(Dataset):
    """Create two independent augmentations of each EEG epoch."""
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
        result["view1"], result["view2"] = view1, view2
        result["sample_index"] = int(index)
        return result


class ManuscriptPairBatchSampler(Sampler):
    """Yield batches containing at least two distinct (emotion, subject, trial) keys.

    This ensures every anchor has at least one eligible negative under the
    manuscript rule: another epoch differing in emotion OR subject OR trial.
    Each dataset index appears at most once per epoch.
    """
    def __init__(self, labels, subject_ids, trial_ids, batch_size,
                 drop_last=True, shuffle=True, seed=42):
        self.labels = np.asarray(labels).tolist()
        self.subject_ids = np.asarray(subject_ids).tolist()
        self.trial_ids = np.asarray(trial_ids).tolist()
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if not (len(self.labels) == len(self.subject_ids) == len(self.trial_ids)):
            raise ValueError("Pair-sampler metadata arrays must have equal length.")
        if self.batch_size < 2:
            raise ValueError("Contrastive batch_size must be at least two.")
        keys = {(self.labels[i], self.subject_ids[i], self.trial_ids[i])
                for i in range(len(self.labels))}
        if len(keys) < 2:
            raise ValueError(
                "Training data contain fewer than two distinct "
                "(emotion, subject, trial) combinations; no eligible negatives exist.")

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.labels)).tolist() if self.shuffle else list(range(len(self.labels)))
        batches = []
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if len(batch) < 2:
                continue
            keys = [(self.labels[i], self.subject_ids[i], self.trial_ids[i]) for i in batch]
            if len(set(keys)) < 2:
                remaining = order[start + len(batch):]
                replacement = next((j for j in remaining
                    if (self.labels[j], self.subject_ids[j], self.trial_ids[j]) != keys[0]), None)
                if replacement is None:
                    # Search all indices not already used in this epoch's earlier batches.
                    used = {i for b in batches for i in b} | set(batch)
                    replacement = next((j for j in order if j not in used and
                        (self.labels[j], self.subject_ids[j], self.trial_ids[j]) != keys[0]), None)
                if replacement is None:
                    raise RuntimeError(
                        "Could not construct a minibatch with an eligible negative. "
                        "Reduce batch_size or check metadata diversity.")
                batch[-1] = replacement
                # Keep each index unique by replacing the displaced item into a later slot.
                displaced = order[start + len(batch) - 1]
                try:
                    displaced_pos = order.index(replacement, start + len(batch))
                    order[displaced_pos] = displaced
                except ValueError:
                    pass
            batches.append(batch)
        for batch in batches:
            yield batch


def validate_same_sample_metadata(subject_ids=None, trial_ids=None,
                                  emotion_labels=None, num_samples=None):
    """Validate metadata lengths."""
    for name, values in (
        ("subject_ids", subject_ids), ("trial_ids", trial_ids),
        ("emotion_labels", emotion_labels)
    ):
        if values is not None and num_samples is not None and len(values) != num_samples:
            raise ValueError(f"{name} must have one entry per EEG sample.")
    return True
