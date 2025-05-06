# Numerai Crypto Ensemble Solution

This document explains how to use the advanced ensemble solution for the Numerai Crypto competition. The solution combines multiple machine learning models, GPU acceleration, and sophisticated feature engineering to create high-quality predictions.

## Overview

This solution includes:

1. **Multi-GPU Training** - Utilize all available GPUs for parallel model training
2. **Feature Engineering** - Generate polynomial features up to 5000 columns using GPU acceleration
3. **Model Ensemble** - Combine predictions from LightGBM, H2O XGBoost, and H2O AutoML models
4. **Iterative Feature Selection** - Improve model quality by keeping only the most important features
5. **Multiple Submission Files** - Generate two slightly different submission files for diversification

## Prerequisites

- Python 3.8+
- CUDA-compatible GPUs (recommended but not required)
- H2O and Sparkling Water (optional, for advanced models)
- RAPIDS libraries (optional, for GPU-accelerated data processing)

## Quick Start

1. **Test GPU Capabilities**

   Before running the full pipeline, check your GPU capabilities:

   ```bash
   python scripts/test_gpu_capabilities.py
   ```

2. **Run the Ensemble Pipeline**

   Run the full pipeline with a single command:

   ```bash
   ./run_crypto_ensemble.sh
   ```

   To download the latest data:

   ```bash
   ./run_crypto_ensemble.sh --download
   ```

3. **Analyze Predictions**

   After generating predictions, analyze them:

   ```bash
   python scripts/analyze_crypto_predictions.py \
     --predictions data/submissions/crypto_ensemble_TIMESTAMP.csv
   ```

## Script Descriptions

### 1. `crypto_ensemble_submission.py`

This is the main script that:
- Loads and preprocesses data
- Generates polynomial features
- Trains multiple models using GPUs 
- Creates an ensemble prediction
- Outputs two submission files

```bash
python scripts/crypto_ensemble_submission.py --help
```

Parameters:
- `--download`: Download latest data
- `--feature-count`: Number of polynomial features (default: 5000)
- `--poly-degree`: Degree of polynomial features (default: 2)
- `--gpu`: Use GPU acceleration
- `--ensemble-size`: Number of models in ensemble (default: 5)
- `--output`: Output file name
- `--random-seed`: Random seed for reproducibility
- `--iterative-pruning`: Enable iterative feature pruning
- `--prune-pct`: Percentage of features to keep in each iteration

### 2. `test_gpu_capabilities.py`

Test your system's GPU capabilities for:
- RAPIDS (cuDF, cuML)
- LightGBM GPU acceleration
- H2O XGBoost GPU acceleration
- PySpark with RAPIDS acceleration

```bash
python scripts/test_gpu_capabilities.py [--full]
```

### 3. `analyze_crypto_predictions.py`

Analyze prediction quality and distributions:
- Visualize prediction distributions
- Compare multiple prediction files
- Evaluate against validation data
- Generate detailed performance metrics

```bash
python scripts/analyze_crypto_predictions.py \
  --predictions PATH_TO_PREDICTIONS \
  [--validation PATH_TO_VALIDATION] \
  [--compare PATH_TO_SECOND_PREDICTIONS]
```

## Model Types Used

1. **LightGBM**
   - GPU-accelerated gradient boosting
   - Fast training and high performance
   - Works well with high-dimensional data

2. **H2O XGBoost**
   - Distributed XGBoost via H2O
   - Robust to overfitting
   - Excellent performance on structured data

3. **H2O AutoML** (when available)
   - Automatically finds best models
   - Tests multiple algorithms
   - Provides model stacking

## Feature Engineering

The solution generates up to 5000 features using polynomial combinations:
- Original features
- Squares of features (degree=2)
- Interactions between features
- Higher-order terms (when degree>2)
- Random features to detect spurious correlations

## GPU Acceleration

When GPUs are available, the solution uses:
- RAPIDS (cuDF) for GPU-accelerated data processing
- LightGBM with GPU tree construction
- H2O XGBoost with GPU histogram algorithm
- Multi-GPU training for parallel model creation

## Workflow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Download Data  │────▶│ Feature         │────▶│ Iterative       │
└─────────────────┘     │ Engineering     │     │ Feature         │
                        └─────────────────┘     │ Selection       │
                                                └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│  Create         │◀────│ Ensemble        │◀────│ Multi-GPU       │
│  Submissions    │     │ Predictions     │     │ Model Training  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Troubleshooting

1. **CUDA/GPU Issues**
   - Ensure NVIDIA drivers are installed
   - Check GPU status with `nvidia-smi`
   - Run the GPU capability test script

2. **H2O Issues**
   - Ensure Java is installed
   - Check H2O cluster initialization
   - Allocate sufficient memory with `-Xmx` options

3. **Out of Memory**
   - Reduce `--feature-count` parameter
   - Lower the `--ensemble-size` value
   - Process data in batches

## References

- [Numerai API Documentation](https://docs.numer.ai/numerai-tournament/api)
- [H2O Documentation](https://docs.h2o.ai/)
- [LightGBM GPU Documentation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [RAPIDS Documentation](https://docs.rapids.ai/)