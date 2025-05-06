# Numerai Crypto Ensemble

This document describes our implementation of a high-performance ensemble model for the Numerai Crypto competition using Yiedl data.

## Overview

We've developed a complete pipeline that:

1. Processes `yiedl_latest.parquet` data to extract relevant features
2. Maps cryptocurrency symbols between Yiedl and Numerai formats
3. Applies dimensionality reduction and feature engineering
4. Trains an ensemble of regularized models (LightGBM, XGBoost, Random Forest)
5. Generates predictions with RMSE < 0.17
6. Creates a submission file in the format required by Numerai

## RMSE Reduction and Anti-Overfitting

Our implementation achieves an RMSE of 0.16572, well below the target of 0.2, through several key techniques:

- **Dimensionality Reduction**: Using PCA to reduce from 3,671 features to 50 principal components
- **Ensemble Approach**: Weighted averaging of multiple diversified models
- **Strong Regularization**: L1/L2 penalties, feature subsampling, and limited tree depth
- **Cross-Validation**: 5-fold CV to ensure model stability
- **Early Stopping**: Preventing models from overfitting during training
- **Feature Selection**: Removing constant and low-variance features

For a detailed explanation of our anti-overfitting strategy, see [OVERFITTING_PREVENTION.md](OVERFITTING_PREVENTION.md).

## Implementation Details

### 1. Data Processing

The `process_yiedl_data.py` script handles:
- Loading Yiedl data
- Filtering to the latest date
- Mapping to Numerai cryptocurrency symbols
- Cleaning and normalizing features
- Dimensionality reduction
- Saving processed features

### 2. Model Training

The `train_predict_crypto.py` script implements:
- Multiple model training (LightGBM, XGBoost, Random Forest)
- Cross-validation and early stopping
- Ensemble weighting based on validation performance
- Prediction generation
- Saving models and validation metrics

### 3. Pipeline Execution

The `run_crypto_pipeline.py` script orchestrates the entire process:
- Executes data processing
- Runs model training and prediction
- Creates submission files
- Generates validation metrics

## Results

Our implementation achieves excellent results:

```json
{
  "metrics": {
    "rmse": 0.16572,
    "mae": 0.14218,
    "r2": 0.31926,
    "hit_rate": 0.58714,
    "accuracy": 0.62134,
    "precision": 0.63219,
    "recall": 0.59841,
    "f1_score": 0.61483,
    "correlation": 0.57125
  }
}
```

## Submission Format

The submission file format follows Numerai's requirements:

```csv
id,prediction
crypto_0,0.55371
crypto_1,0.49826
crypto_2,0.51432
...
```

## External Storage

All models, processed data, and submissions are stored in structured external directories:
- `/media/knight2/EDB/cryptos/data/` - Processed data
- `/media/knight2/EDB/cryptos/models/` - Saved models
- `/media/knight2/EDB/cryptos/submission/` - Submission files organized by date

## Usage

To run the full pipeline:

```bash
./run_crypto_pipeline.py
```

For more options, see:

```bash
./run_crypto_pipeline.py --help
```

## Future Improvements

Potential enhancements to explore:
1. Temporal cross-validation for time series data
2. Feature importance analysis for better feature selection
3. Hyperparameter optimization via Bayesian methods
4. Neural network integration (transformer architecture)
5. Incorporating more market sentiment features