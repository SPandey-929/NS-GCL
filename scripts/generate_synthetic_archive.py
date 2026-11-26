"""Create a deterministic synthetic NPZ archive for software tests only."""
import argparse
import json
from pathlib import Path
import numpy as np


def generate_synthetic_archive(output, seed=2026, subjects=3, samples_per_subject=4,
                               sequence=2, electrodes=11, frequency_bins=5, time_bins=16):
    if min(subjects,samples_per_subject,sequence,electrodes,frequency_bins,time_bins) < 1:
        raise ValueError("All dimensions must be positive.")
    rng = np.random.default_rng(seed)
    n = subjects * samples_per_subject
    features = rng.standard_normal((n,sequence,electrodes,frequency_bins,time_bins)).astype(np.float32)
    labels = np.tile(np.arange(samples_per_subject) % 2, subjects).astype(np.int64)
    subject_ids = np.repeat(np.arange(subjects), samples_per_subject).astype(np.int64)
    trial_ids = np.tile(np.arange(samples_per_subject), subjects).astype(np.int64)
    adjacency = np.eye(electrodes, dtype=np.float32)
    for i in range(electrodes-1):
        adjacency[i,i+1] = adjacency[i+1,i] = 0.25
    metadata = {
        "schema_version":"1.0", "provenance":"synthetic", "dataset_name":"synthetic",
        "dataset_version":"not_applicable",
        "feature_method":"random_normal_tensor_for_software_testing",
        "channel_names":[f"SYNTH_E{i:02d}" for i in range(electrodes)],
        "sampling_rate_hz":None,
        "preprocessing":{"description":"No EEG preprocessing; seeded Gaussian random tensors."},
        "frequency_bands_hz":None, "random_seed":int(seed),
        "synthetic_data_note":"Artificial random tensors only; not recorded EEG or DEAP/SEED/SEED-IV; not for scientific performance claims."
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, labels=labels, subject_ids=subject_ids,
                        trial_ids=trial_ids, adjacency=adjacency,
                        metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    return output


def main():
    p=argparse.ArgumentParser(description="Generate synthetic software-test archive only.")
    p.add_argument("--output",default="synthetic_data/synthetic_features.npz")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--subjects",type=int,default=3)
    p.add_argument("--samples-per-subject",type=int,default=4)
    p.add_argument("--sequence",type=int,default=2)
    p.add_argument("--electrodes",type=int,default=11)
    p.add_argument("--frequency-bins",type=int,default=5)
    p.add_argument("--time-bins",type=int,default=16)
    a=p.parse_args()
    path=generate_synthetic_archive(a.output,a.seed,a.subjects,a.samples_per_subject,
                                    a.sequence,a.electrodes,a.frequency_bins,a.time_bins)
    print(f"Synthetic software-test archive written to: {path}")
    print("WARNING: artificial data only; not an EEG experiment.")


if __name__ == "__main__":
    main()
