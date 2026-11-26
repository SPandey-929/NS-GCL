"""Preprocessing package for NS-GCL."""
from .deap_loader import load_deap_mat, select_channels
from .filtering import bandpass_filter
from .segmentation import segment_trials
from .normalization import ChannelZScore
from .ica import apply_manual_ica
from .baseline import baseline_correct_trials

__all__ = ["load_deap_mat", "select_channels", "bandpass_filter",
           "segment_trials", "ChannelZScore", "apply_manual_ica",
           "baseline_correct_trials"]
