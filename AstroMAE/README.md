# AstroMAE

AstroMAE is a Masked Autoencoder designed for processing astrophysical data. This project aims to provide tools for downloading, preprocessing, and analyzing data from sources like SDSS or Kepler/TESS.

## Setup Instructions

To set up the environment, use the following commands:

```bash
conda env create -f environment.yml
conda activate astromae
```

## Usage Example

To start training, run:

```bash
python src/train.py --data-dir ./data --epochs 10 --batch-size 8
```

## Project Structure

- `data/`: Contains scripts for downloading and preprocessing data.
- `notebooks/`: Jupyter notebooks for data exploration.
- `src/`: Source code for dataset handling, model definition, training, and evaluation.
- `requirements.txt`: Python package dependencies.
- `environment.yml`: Conda environment specification.