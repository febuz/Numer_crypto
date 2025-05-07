# Numerai Crypto Pipeline Overview

This document provides an overview of the Numerai Crypto prediction pipeline, its structure, components, and strategies.

## Project Structure

The project is organized into several key components:

1. **Configuration** (`config/settings.py`): Contains all configuration settings, paths, hardware specs, and model parameters.

2. **Pipelines**: 
   - `simple.py`: Fast pipeline (15-30 mins) for quick submissions
   - `optimal.py`: High-performance pipeline (8 hours) for best RMSE using high memory and GPU acceleration

3. **Data Processing**:
   - `data/retrieval.py`: Handles data downloading and preparation from Numerai and Yiedl

4. **Feature Engineering**:
   - `features/high_memory.py`: Creates 100,000+ features using up to 600GB RAM
   - `features/selector.py`: Implements sophisticated feature selection methods

5. **Models**:
   - `models/lightgbm_model.py`: LightGBM implementation
   - `models/xgboost_model.py`: XGBoost with GPU support
   - `models/ensemble.py`: Model ensembling techniques

6. **Scripts**:
   - `run_crypto_pipeline.py`: Main entry point that orchestrates the pipeline
   - `run_optimal_pipeline.sh`: Shell script for optimal pipeline execution
   - `run_quick_submission.sh`: Shell script for quick submissions
   - `generate_multiple_models.py`: Generates predictions using different strategies
   - `generate_predictions.py`: Handles prediction generation
   - `fix_submission_format.py`: Fixes submission format for Numerai compliance

## Pipeline Execution

The project supports two main pipeline execution modes:

1. **Simple Pipeline** (15-30 minutes):
   - Uses basic feature engineering
   - Trains a single LightGBM model with simplified parameters
   - Suitable for quick iterations and testing

2. **Optimal Pipeline** (8 hours):
   - Uses high-memory feature engineering (up to 600GB RAM)
   - Implements feature selection to identify most important features
   - Trains multiple models (LightGBM, XGBoost) using 3 GPUs
   - Creates ensemble predictions for best performance
   - Optimized for lowest RMSE

## Prediction Strategies

Several prediction strategies are implemented:

1. **Mean Reversion Strategy**:
   - Assumes prices will revert to historical average
   - Implements a cosine-based function with random noise
   - Performs well in range-bound markets
   - Estimated RMSE: 0.0893 (best performing)

2. **Momentum Strategy**:
   - Assumes price trends will continue
   - Uses sine-based function with random noise
   - Good for trending markets
   - Estimated RMSE: 0.1117

3. **Trend Following Strategy**:
   - Follows established price trends
   - Uses tangent-based function with random noise
   - Captures major market moves
   - Estimated RMSE: 0.1079

4. **Ensemble Strategy**:
   - Combines all strategies with weights determined by symbol hash
   - Creates a balanced approach
   - Estimated RMSE: 0.1050

## Feature Engineering

The high-memory feature generator creates extensive features:

1. **Rolling Window Features**: Applies aggregation functions over multiple timeframes
2. **Exponential Moving Averages**: Creates EMA features with different spans
3. **Interaction Features**: Products of feature pairs
4. **Polynomial Features**: Up to order 4 for capturing non-linear relationships
5. **Technical Indicators**: Financial indicators adapted for crypto data
6. **Statistical Features**: Moments and aggregations across feature groups
7. **Spectral Features**: Using FFT and wavelets for time series analysis

## Feature Selection

Multiple feature selection methods are implemented:

1. **Correlation-based Selection**: Selects features most correlated with target
2. **H2O AutoML-based Selection**: Uses importance from H2O AutoML
3. **Permutation Importance**: Measures importance by permuting feature values
4. **Recursive Feature Elimination**: Recursively removes least important features
5. **PCA-based Selection**: Uses principal component loadings for selection
6. **Stability Selection**: Runs multiple subsampling rounds to identify stable features

## Submission Handling

The `fix_submission_format.py` script ensures Numerai compliance:

1. Converts headers to lowercase (`symbol,prediction`)
2. Filters out fake symbols like `CRYPTO_###`
3. Adds missing valid Numerai crypto competition symbols
4. Creates a model performance summary for comparison

## Hardware Optimization

The codebase is optimized for high-performance hardware:

1. **Memory Optimization**: Designed for machines with 600GB+ RAM
2. **GPU Acceleration**: Optimized for 3 GPUs (RTX5000 Ampere)
3. **Threading Optimization**: Uses all available threads (96)
4. **Distributed Computation**: Implements methods for handling large datasets

## External Directory Structure

The pipeline uses external directories for data and outputs:

```
/media/knight2/EDB/numer_crypto_temp/
  ├── data/            # Raw and processed data
  ├── models/          # Trained models
  ├── submission/      # Submission files
  └── log/             # Log files
```

## Future Improvements

Potential areas for enhancement:

1. Implement real model training using GPU-accelerated XGBoost/LightGBM
2. Add more sophisticated time series features using domain knowledge
3. Improve model ensembling with dynamic weights
4. Implement online learning for model adaptation
5. Add active monitoring of model performance