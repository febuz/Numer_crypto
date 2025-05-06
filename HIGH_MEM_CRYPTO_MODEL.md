# High Memory GPU-Accelerated Crypto Model

This document provides information on using the high memory GPU-accelerated model for the Numerai Crypto competition.

## Overview

The `high_mem_crypto_model.py` script is designed to create high-quality submissions for the Numerai Crypto competition with:

- Full memory utilization (up to 600GB RAM)
- GPU-accelerated gradient boosting (LightGBM, XGBoost)
- PyTorch neural network models
- 20-day ahead forecasting
- RMSE and hit rate evaluation metrics
- Multi-GPU utilization

## Quick Start

To run the model with default settings:

```bash
./run_high_mem_crypto_model.sh
```

This will:
1. Create a virtual environment with all required dependencies (if not already created)
2. Detect available GPUs and use them for acceleration
3. Train ensemble models using LightGBM and XGBoost (with GPU acceleration when available)
4. Create a submission file in the `data/submissions` directory

## Requirements

- Python 3.8+
- CUDA-compatible GPUs (for GPU acceleration)
- 600GB+ RAM (configurable)
- Required Python packages (installed automatically by the script):
  - PyTorch (with CUDA support)
  - LightGBM
  - XGBoost
  - H2O (version 3.46.0.6 specifically)
  - scikit-learn
  - pandas, numpy, matplotlib, seaborn

## Command Line Options

The `run_high_mem_crypto_model.sh` script accepts the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--ram GB` | RAM to use in GB | 500 |
| `--forecast-days DAYS` | Number of days to forecast ahead | 20 |
| `--time-limit SECONDS` | Time limit in seconds | 900 (15 minutes) |
| `--output FILE` | Output path for submission file | Auto-generated timestamp-based filename |
| `--no-gpu` | Disable GPU usage | GPU is used if available |
| `--gpu-ids IDS` | Comma-separated list of GPU IDs to use | All available GPUs |
| `--nn-model` | Enable PyTorch neural network model | Disabled by default |
| `--evaluate` | Evaluate models on validation data | Disabled by default |
| `--test` | Run environment tests only | Disabled by default |
| `--skip-tests` | Skip environment tests | Tests run by default |
| `--fstore-dir DIR` | Feature store directory | `/media/knight2/EDB/fstore` |

## Examples

1. Run with all available GPUs, 500GB RAM:
   ```bash
   ./run_high_mem_crypto_model.sh
   ```

2. Run with 400GB RAM and neural network models:
   ```bash
   ./run_high_mem_crypto_model.sh --ram 400 --nn-model
   ```

3. Run using only GPUs 0 and 2, with evaluation:
   ```bash
   ./run_high_mem_crypto_model.sh --gpu-ids 0,2 --evaluate
   ```

4. Run CPU-only mode with 10-day forecast:
   ```bash
   ./run_high_mem_crypto_model.sh --no-gpu --forecast-days 10
   ```

5. Test the environment without running the full model:
   ```bash
   ./run_high_mem_crypto_model.sh --test
   ```

6. Run without environment tests with a 15-minute time limit:
   ```bash
   ./run_high_mem_crypto_model.sh --skip-tests --time-limit 900
   ```

7. Use an external feature store location:
   ```bash
   ./run_high_mem_crypto_model.sh --fstore-dir /path/to/external/fstore
   ```

## Features

### Multi-GPU Support

The model automatically detects available GPUs and distributes workloads across them. You can specify which GPUs to use with the `--gpu-ids` option.

### Memory Optimization

Memory is carefully allocated between different components:
- H2O: 70% of available RAM (capped at 400GB)
- Pandas operations: 20% of available RAM (capped at 100GB)
- Other operations: Remaining RAM

### Models

The script trains multiple models in parallel:

1. **LightGBM** with GPU acceleration
   - Binary classification
   - Feature importance analysis
   - Hyperparameter optimization

2. **XGBoost** with GPU acceleration
   - Binary classification
   - Tree-based feature importance
   - GPU-optimized training

3. **PyTorch Neural Network** (optional)
   - LSTM-based architecture
   - GPU-accelerated training
   - Dropout for regularization
   - Early stopping to prevent overfitting

### Ensemble Predictions

Predictions from all models are combined using simple averaging to create the final submission.

### Feature Store

The model now includes a feature store system that caches computationally expensive features outside the repository:

- Features are stored in parquet format in the external feature store directory
- Each feature computation is cached to avoid redundant calculations
- Metadata is maintained to track datasets, features, and statistics
- Features can be reused across multiple runs for faster execution
- Automatically handles caching and retrieval of:
  - Preprocessed datasets
  - Feature interactions (multiply, divide, sum, difference)
  - Polynomial features (squared, cubed)
  - Custom engineered features

The feature store is particularly useful for:
- Large datasets where feature engineering is computationally expensive
- Iterative model development where the same features are used repeatedly
- Time-constrained runs (15 minutes) where cached features enable faster processing

### Evaluation Metrics

When using the `--evaluate` option, the following metrics are calculated:
- RMSE (Root Mean Squared Error)
- AUC (Area Under the ROC Curve)
- Accuracy
- Precision
- Recall
- F1 Score
- Hit Rate (percentage of correct direction predictions)

## Environment Testing

You can verify that your environment is properly set up by running the test script:

```bash
./run_high_mem_crypto_model.sh --test
```

This will:
1. Check system information (CPU, RAM, GPUs)
2. Verify that all required Python packages are installed
3. Test GPU functionality with LightGBM and PyTorch (if applicable)
4. Test synthetic data creation and feature engineering
5. Provide a summary of test results

This is particularly useful when setting up a new environment or troubleshooting issues.

## File Structure

- `scripts/high_mem_crypto_model.py`: Main model implementation
- `scripts/test_high_mem_crypto_model.py`: Environment test script
- `scripts/install_requirements.sh`: Installs all required dependencies in a virtual environment
- `run_high_mem_crypto_model.sh`: Shell script to set up and run the model
- `data/submissions/`: Directory for generated submission files
- `data/yiedl/`: Directory for Yiedl data (both latest and historical)
- `models/yiedl/`: Directory for saved model files
- `reports/plots/`: Directory for evaluation plots and metrics

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues, try:
- Running with `--no-gpu` to use CPU only
- Specifying a specific GPU with `--gpu-ids`
- Checking GPU memory with `nvidia-smi` before running

### Memory Issues

If you experience memory-related crashes:
- Reduce the `--ram` parameter
- Ensure your system has enough physical RAM
- Check for other memory-intensive processes running

### Python Environment

If you encounter Python package compatibility issues:
- Remove the `venv` directory and let the script create a new one
- Manually install specific versions of packages if needed
- Check the installation logs for error messages

## Advanced Usage

### Custom Feature Engineering

The model performs automatic feature engineering, including:
- Feature interactions (multiplication, division, sum, difference)
- Polynomial features (squares, cubes)
- Moving averages and volatility measures (for time series data)

### Model Tuning

For model tuning and optimization:
1. Run with `--evaluate` to get performance metrics
2. Modify model parameters in the script as needed
3. Experiment with different ensemble weights
4. Add custom feature engineering steps

## Submitting to Numerai

After running the model, you can submit the predictions to Numerai with:

```python
from numer_crypto.data.retrieval import NumeraiDataRetriever
NumeraiDataRetriever().submit_predictions('path/to/submission.csv', 'crypto')
```

The exact command is provided in the output after a successful run.