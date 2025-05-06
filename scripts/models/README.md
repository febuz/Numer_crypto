# Numerai Crypto Model Runners

This directory contains Python scripts for executing Numerai Crypto models.

## Available Runners

- `run_advanced_model.py`: Advanced model with full feature set and comprehensive training pipeline
- `run_quick_model.py`: Quick model for rapid experimentation and testing

## Usage

These scripts can be run from the project root directory:

```bash
# Run the advanced model
python scripts/models/run_advanced_model.py --gpu

# Run the quick model
python scripts/models/run_quick_model.py --test
```

Run any script with `--help` to see available options.

## Command-line Tools

The package also provides command-line tools when installed:

```bash
# Install the package
pip install -e .

# Run the models using entry points
numer-crypto --model advanced
numer-crypto-eda --input data/yiedl/yiedl_latest.parquet
```