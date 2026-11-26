"""Stochastic time-frequency augmentations for contrastive views."""
import torch

class EEGTimeFrequencyAugment:
    """Input shape: (sequence, electrodes, frequency_bins, time_bins)."""
    def __init__(self, noise_std=0.05, max_time_shift_fraction=0.05,
                 frequency_mask_probability=0.20, channel_dropout_probability=0.0):
        if noise_std < 0 or not 0 <= max_time_shift_fraction < 0.5:
            raise ValueError("Invalid noise or time-shift setting.")
        for p in (frequency_mask_probability, channel_dropout_probability):
            if not 0 <= p <= 1:
                raise ValueError("Augmentation probabilities must lie in [0, 1].")
        self.noise_std = float(noise_std)
        self.max_time_shift_fraction = float(max_time_shift_fraction)
        self.frequency_mask_probability = float(frequency_mask_probability)
        self.channel_dropout_probability = float(channel_dropout_probability)
    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("Expected tensor shaped (sequence, electrodes, frequency, time).")
        out = x.clone()
        max_shift = int(out.shape[-1] * self.max_time_shift_fraction)
        if max_shift:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            out = torch.roll(out, shifts=shift, dims=-1)
        if self.noise_std:
            out = out + torch.randn_like(out) * self.noise_std
        if self.frequency_mask_probability and torch.rand(()) < self.frequency_mask_probability:
            index = int(torch.randint(out.shape[-2], (1,)).item())
            out[..., index, :] = 0
        if self.channel_dropout_probability:
            keep = torch.rand(out.shape[-3], device=out.device) >= self.channel_dropout_probability
            out = out * keep.view(1, -1, 1, 1)
        return out
