# Numerai Crypto Prediction Pipeline

A comprehensive pipeline for generating cryptocurrency predictions for the Numerai Crypto competition, focused on high-performance prediction with optimized strategies.

## Overview

This repository implements a high-performance pipeline for the Numerai Crypto tournament, including:

- Data retrieval from Numerai and Yiedl
- Extensive feature engineering using high-memory computation (600GB RAM)
- Multiple prediction strategies (Mean Reversion, Momentum, Trend Following, and Ensemble)
- GPU-accelerated model training using LightGBM and XGBoost
- Submission format validation and correction
- Performance comparison analytics

## Key Components

- **Data Retrieval**: Fetch and preprocess data from Numerai API
- **Feature Engineering**: Generate predictive features with high-memory transformations
- **Feature Selection**: Identify the most important features for prediction
- **Model Training**: Train GPU-accelerated LightGBM and XGBoost models
- **Ensemble Predictions**: Combine multiple models for improved performance
- **Submission**: Format and submit predictions to Numerai

## Prediction Strategies

1. **Mean Reversion Strategy**: Assumes prices will revert to historical average (RMSE: 0.0893)
2. **Momentum Strategy**: Assumes price trends will continue (RMSE: 0.1117)
3. **Trend Following Strategy**: Follows established price trends (RMSE: 0.1079)
4. **Ensemble Strategy**: Combines all strategies with intelligent weighting (RMSE: 0.1050)

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

### Advanced Pipeline

Complete pipeline with all components:

```bash
./go_pipeline.sh
```

## Hardware Requirements

- RAM: 600GB+ (minimum 16GB for simple pipeline)
- GPU: 3x GPUs with CUDA support (minimum 1 for simple pipeline)
- CPU: 96 threads (minimum 8 for simple pipeline)
- Storage: 500GB+ free space

## Submission Format

All submissions are automatically formatted to meet Numerai Crypto competition requirements:
- Lowercase headers (`symbol,prediction`)
- Only valid cryptocurrency symbols
- Predictions in valid range (0-1)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/febuz/Numer_crypto.git
   cd Numer_crypto
   ```

2. Create external data directory:
   ```bash
   mkdir -p /media/knight2/EDB/numer_crypto_temp/{data,models,submission,log}
   mkdir -p /media/knight2/EDB/numer_crypto_temp/data/{raw,processed,features}
   mkdir -p /media/knight2/EDB/numer_crypto_temp/models/checkpoints
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment:
   ```bash
   ./setup_environment.sh
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

### Multiple Model Generation

To generate predictions with all strategies:
```bash
python scripts/generate_multiple_models.py
```

### Fix Submission Formats

To ensure all submission files meet Numerai requirements:
```bash
python scripts/fix_submission_format.py --all
```

## Directory Structure

- `config/`: Configuration settings
- `data/`: Data management code
- `features/`: Feature engineering and selection
- `models/`: Model implementations
- `pipelines/`: Pipeline implementations
- `scripts/`: Executable scripts
- `tests/`: Test scripts and validation tools
- `utils/`: Utility functions (GPU, Memory, Threading, etc.)

Data and model artifacts are stored outside the repository in `set to your folder in the config files`.
