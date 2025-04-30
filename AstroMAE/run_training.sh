#!/bin/bash

# Activate the conda environment
conda activate astromae

# Run the training script (uses default config.yaml in the root)
# Ensure you run this script from the AstroMAE project root directory
python src/train.py