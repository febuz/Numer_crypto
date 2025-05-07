# Numerai Crypto Prediction Pipeline

A streamlined machine learning project for the Numerai Crypto competition, focused on high-performance prediction with a target RMSE < 0.018 for a top 100 ranking.

## Overview

This repository contains a complete machine learning pipeline for cryptocurrency predictions in the Numerai tournament. It features:

- High-memory feature engineering (up to 600GB RAM)
- GPU-accelerated model training (3 GPUs)
- Ensemble methods for combining model predictions
- Time-budgeted execution for both optimal and quick submissions

## Key Components

- **Data Retrieval**: Fetch and preprocess data from Numerai API
- **Feature Engineering**: Generate predictive features with high-memory transformations
- **Feature Selection**: Identify the most important features for prediction
- **Model Training**: Train GPU-accelerated LightGBM and XGBoost models
- **Ensemble Predictions**: Combine multiple models for improved performance
- **Submission**: Format and submit predictions to Numerai

## Pipelines

### Quick Submission (15-30 minutes)

A streamlined pipeline for rapid submissions with reasonable performance:

```bash
./run_quick_submission.sh
```

**Target Performance**: RMSE < 0.020  
**Resource Usage**: Single GPU, minimal memory  
**Time Target**: 15-30 minutes

### Optimal Pipeline (4-8 hours)

The high-performance pipeline for achieving lowest RMSE:

```bash
./run_optimal_pipeline.sh
```

**Target Performance**: RMSE < 0.018  
**Resource Usage**: 3 GPUs, up to 600GB RAM  
**Time Target**: 4-8 hours

## Requirements

- Python 3.8+
- NVIDIA GPU(s) with CUDA support
- Large memory machine (recommended: 600GB+ RAM)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Numer_crypto.git
   cd Numer_crypto
   ```

2. Create external data directory:
   ```bash
   mkdir -p /numer_crypto_temp/data/{raw,processed,features,submissions}
   mkdir -p /numer_crypto_temp/models/checkpoints
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Numerai API credentials in environment variables:
   ```bash
   export NUMERAI_PUBLIC_ID="your-public-id"
   export NUMERAI_SECRET_KEY="your-secret-key"
   ```

## Usage

### Running Pipelines

For optimal results (4-8 hours):
```bash
./run_optimal_pipeline.sh --submit
```

For quick submissions (15-30 minutes):
```bash
./run_quick_submission.sh --submit
```

### Command Line Options

Both scripts support the following options:
- `--tournament TOURNAMENT`: Tournament name (default: crypto)
- `--submission-id ID`: Custom submission ID 
- `--submit`: Submit results to Numerai API
- `--time-budget VALUE`: Time budget in hours/minutes

### Submitting Predictions

To submit an existing prediction file:
```bash
python scripts/submit.py /path/to/prediction.csv --submit --track-performance
```

## Performance

The optimal pipeline targets RMSE < 0.018 on validation data through:
- High-memory feature engineering (600GB RAM)
- GPU-accelerated modeling with 3 GPUs
- Model ensembling and feature selection

## Directory Structure

- `config/`: Configuration settings
- `data/`: Data management code
- `docs/`: Documentation files
- `features/`: Feature engineering and selection
- `models/`: Model implementations
- `pipelines/`: Pipeline implementations
- `scripts/`: Executable scripts
- `tests/`: Test scripts and validation tools
- `utils/`: Utility functions

Data and model artifacts are stored outside the repository in `/numer_crypto_temp/`.

For more detailed documentation, see the [docs directory](docs/index.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.