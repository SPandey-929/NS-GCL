# Obtaining DEAP data

DEAP is distributed by its custodians under an access and data-use agreement. Obtain it from the official Queen Mary University of London DEAP page:

https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

Read and accept the provider's terms and complete its access procedure. This repository does not download, mirror, or redistribute DEAP recordings. Keep participant files locally and do not commit them to this public repository.

The commonly distributed preprocessed MATLAB release contains participant files such as s01.mat through s32.mat. Typical variables are:
- data: trials × 40 channels × samples (32 EEG followed by 8 peripheral channels)
- labels: trials × 4 ratings (valence, arousal, dominance, liking)

Confirm the actual file keys, dimensions, sample rate, and preprocessing history after obtaining the data. File extensions alone do not establish the data format.

Example local layout (not committed):
data/DEAP/data_preprocessed_matlab/s01.mat

The repository .gitignore excludes common data files and local data directories.
