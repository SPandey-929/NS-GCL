"""Differentiable physiological regularization for NS-GCL.

Physiological descriptors (FAA and beta PLV) are precomputed from each EEG
epoch and supplied as fixed tensors. Gradients flow through the predicted
class probabilities, not through the signal-processing descriptors.
"""
import torch
from torch import nn
import torch.nn.functional as F


class NeuroSymbolicRegularizer(nn.Module):
    """FAA and beta-PLV penalty with explicit, configurable semantics.

    FAA convention expected here is ln(alpha_PSD_right) - ln(alpha_PSD_left).
    positive_faa_sign must be +1 or -1 and must be selected from the
    physiological hypothesis supported by the study; no universal sign is
    assumed. positive_probability, negative_probability, and
    high_arousal_probability are supplied separately to avoid hard-coded
    dataset-specific class indices.

    FAA penalty:
      p_pos * relu(margin - sign * FAA)^2
      + p_neg * relu(margin + sign * FAA)^2

    Beta penalty:
      p_high * relu(beta_threshold - beta_PLV)^2

    The beta rule encodes the hypothesis that high-arousal predictions should
    be accompanied by at least the configured beta synchrony threshold. This
    is a testable modelling assumption, not a universal neurophysiological law.
    """
    def __init__(self, faa_margin=0.1, beta_plv_threshold=0.3,
                 positive_faa_sign=None, lambda_faa=1.0, lambda_beta=1.0):
        super().__init__()
        if faa_margin < 0:
            raise ValueError("faa_margin must be non-negative.")
        if not 0.0 <= beta_plv_threshold <= 1.0:
            raise ValueError("beta_plv_threshold must lie in [0, 1].")
        if positive_faa_sign not in (-1, 1):
            raise ValueError("Set positive_faa_sign to +1 or -1 based on the study hypothesis.")
        if lambda_faa < 0 or lambda_beta < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.faa_margin = float(faa_margin)
        self.beta_plv_threshold = float(beta_plv_threshold)
        self.positive_faa_sign = float(positive_faa_sign)
        self.lambda_faa = float(lambda_faa)
        self.lambda_beta = float(lambda_beta)

    @staticmethod
    def _vector(name, value, batch_size, device, dtype):
        value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if value.numel() != batch_size:
            raise ValueError(f"{name} must contain one value per sample.")
        return value

    @staticmethod
    def _probability(name, value, batch_size, device, dtype):
        p = NeuroSymbolicRegularizer._vector(name, value, batch_size, device, dtype)
        if torch.any((p < 0) | (p > 1)):
            raise ValueError(f"{name} must lie in [0, 1].")
        return p

    def forward(self, faa, beta_plv, positive_probability,
                negative_probability, high_arousal_probability):
        # Use probability tensors as the reference device/dtype so gradients
        # remain connected to the classification probabilities.
        reference = positive_probability
        if not torch.is_tensor(reference):
            raise TypeError("Predicted probabilities must be torch tensors.")
        device, dtype = reference.device, reference.dtype
        p_pos = reference.reshape(-1)
        batch_size = p_pos.numel()
        p_pos = self._probability("positive_probability", p_pos, batch_size, device, dtype)
        p_neg = self._probability("negative_probability", negative_probability, batch_size, device, dtype)
        p_high = self._probability("high_arousal_probability", high_arousal_probability, batch_size, device, dtype)
        faa = self._vector("faa", faa, batch_size, device, dtype).detach()
        beta = self._vector("beta_plv", beta_plv, batch_size, device, dtype).detach()
        if torch.any((beta < 0) | (beta > 1)):
            raise ValueError("beta_plv values must lie in [0, 1].")

        signed_faa = self.positive_faa_sign * faa
        faa_positive_penalty = F.relu(self.faa_margin - signed_faa).square()
        faa_negative_penalty = F.relu(self.faa_margin + signed_faa).square()
        faa_loss = (p_pos * faa_positive_penalty +
                    p_neg * faa_negative_penalty).mean()

        beta_penalty = F.relu(self.beta_plv_threshold - beta).square()
        beta_loss = (p_high * beta_penalty).mean()
        total = self.lambda_faa * faa_loss + self.lambda_beta * beta_loss
        return {
            "loss": total,
            "faa_loss": faa_loss,
            "beta_plv_loss": beta_loss,
        }
