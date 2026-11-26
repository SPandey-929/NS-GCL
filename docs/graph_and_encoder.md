# Functional-connectivity graph and CNN-GCN modules

## Input conventions

- Raw EEG epochs: NumPy array shaped (epochs, electrodes, samples).
- Time-frequency maps for the CNN: PyTorch tensor shaped (batch, electrodes, frequency bins, time steps).
- CNN node embeddings: (batch, electrodes, embedding dimension).
- Connectivity and adjacency: (epochs, electrodes, electrodes), or a single square matrix.
- GCN output: node embeddings and global-mean-pooled graph embeddings.

## Connectivity

The graph module provides band-limited phase-locking value (PLV) and phase-lag index (PLI). It filters each epoch in the requested frequency band, obtains analytic-signal phase using the Hilbert transform, and calculates pairwise phase synchronization.

For the manuscript's stated 4-30 Hz preprocessing, beta synchronization (13-30 Hz) is available. Conventional gamma synchronization above 30 Hz is not available from signals that have already been low-pass filtered at 30 Hz. Do not set a gamma band unless the preprocessing pipeline is changed and the data are reprocessed appropriately.

## Adjacency construction

The adjacency builder supports an absolute threshold, optional top-k neighbours per node, symmetric edge retention, optional binarization, and self-loops. The normalization helper supports symmetric D^-1/2 A D^-1/2 or row normalization.

Top-k=4 and the zero threshold in configs/graph.yaml are starting configuration values only; they are not asserted to be the manuscript's experimentally selected settings. Select graph construction settings using training/validation data only.

## Multi-scale CNN

MultiScaleCNN applies parallel 2-D convolution branches with 3x3, 5x5, and 7x7 kernels to each electrode's time-frequency map. Each branch is globally averaged, concatenated, and projected to a node embedding. This is an initial implementation choice and must be reconciled with the final manuscript architecture and actual experiment.

## GCN

GraphEncoder applies two weighted graph-convolution layers using the supplied normalized adjacency, ReLU activations, dropout, and global mean pooling. It returns both electrode-level and graph-level representations.

## Important limitations

These modules are initial software implementations. They have not been validated on the study's DEAP recordings, and they do not by themselves reproduce any manuscript results. The exact connectivity measure, frequency bands, graph sparsification, CNN kernels, GCN depth, dimensions, and dropout must match the implementation used for reported experiments.
