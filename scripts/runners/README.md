# Numerai Crypto Runners

This directory contains shell scripts for executing various Numerai Crypto models and pipelines.

## Available Runners

### Ensemble Models

- `run_crypto_ensemble_v2.sh`: Runs the enhanced crypto ensemble pipeline with GPU acceleration
- `run_high_mem_crypto_model.sh`: Runs the high memory crypto model with GPU acceleration

### H2O Models

- `run_h2o_simple.sh`: Runs a simple H2O AutoML model
- `run_h2o_submission.sh`: Generates a submission using H2O

### Yiedl Models

- `run_yiedl_quick.sh`: Quick model for Yiedl data
- `run_yiedl_submission.sh`: Generates a submission using Yiedl data

### Utilities

- `run_model_comparison.sh`: Compares performance of different models

## Usage

All scripts can be run from the project root directory:

```bash
# Example: Run the crypto ensemble model with GPU acceleration
./scripts/runners/run_crypto_ensemble_v2.sh --gpu
```

Run any script with `--help` to see available options.

## Output Locations

- Logs: Stored in `/media/knight2/EDB/cryptos/logs/`
- Models: Saved in `/media/knight2/EDB/cryptos/models/`
- Submissions: Output to `data/submissions/`