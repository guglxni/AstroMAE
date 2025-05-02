#!/bin/bash

# Source conda initialization (adjust path if necessary)
# Common paths: ~/miniconda3/etc/profile.d/conda.sh or ~/anaconda3/etc/profile.d/conda.sh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate astromae

# Run the training script (uses default config.yaml in the root)
# Ensure you run this script from the AstroMAE project root directory
python src/train.py