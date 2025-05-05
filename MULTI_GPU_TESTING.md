# Multi-GPU Machine Learning Testing

This document provides instructions for running and comparing multi-GPU machine learning tests using H2O Sparkling Water (XGBoost) and PyTorch.

## Overview

Both tests will:
1. Use a synthetic dataset for training
2. Distribute training across all available GPUs
3. Monitor GPU utilization and memory usage
4. Compare performance metrics across GPUs
5. Save trained models and performance reports

## Prerequisites

- Multiple NVIDIA GPUs with CUDA support
- NVIDIA drivers and CUDA toolkit installed
- The EDB drive mounted at `/media/knight2/EDB`

## Running the Tests

We provide a convenient script that handles environment setup and test execution:

```bash
cd /media/knight2/EDB/repos/Numer_crypto
./setup_env_and_run_multiGPU.sh
```

This script will:
1. Create necessary directories for models and reports
2. Set up PyTorch and H2O Sparkling environments
3. Prompt you to choose which test(s) to run
4. Execute the selected test(s)
5. Save results to the reports directory

## Test Details

### H2O Sparkling Water (XGBoost)

This test:
- Creates a separate Spark session for each GPU
- Initializes H2O in each session with a unique port
- Trains an XGBoost model on each GPU using H2O
- Uses CUDA_VISIBLE_DEVICES to isolate GPUs

**Script location:** `/media/knight2/EDB/repos/Numer_crypto/tests/performance/test_multi_gpu_h2o.py`

### PyTorch (Neural Network)

This test:
- Creates a neural network for binary classification
- Distributes training to multiple GPUs in parallel
- Uses PyTorch's native CUDA device management
- Monitors GPU utilization and memory usage

**Script location:** `/media/knight2/EDB/repos/Numer_crypto/multi_gpu_pytorch.py`

## Viewing Results

After running the tests, results will be available in:
- `/media/knight2/EDB/repos/Numer_crypto/reports/`

The results include:
- Training time comparison charts
- GPU utilization over time graphs
- Memory usage charts
- Detailed JSON reports with metrics
- Trained models (in the models directory)

## Cleaning Up

To remove the repository from the home directory (saving disk space) after everything has been copied to the EDB drive:

```bash
cd /media/knight2/EDB/repos/Numer_crypto
./cleanup_home_repo.sh
```

This script will:
1. Verify that the repository exists on the EDB drive
2. Ensure a symbolic link is created for convenient access
3. Remove the repository from /home/knight2/repos/Numer_crypto

After cleanup, you'll still be able to access the repository through:
- `/home/knight2/repos/Numer_crypto_EDB` (symbolic link)
- `/media/knight2/EDB/repos/Numer_crypto` (direct path)

## Important Locations

- **Repository**: `/media/knight2/EDB/repos/Numer_crypto`
- **Symbolic Link**: `/home/knight2/repos/Numer_crypto_EDB`
- **Models**: `/media/knight2/EDB/repos/Numer_crypto/models/`
- **Reports**: `/media/knight2/EDB/repos/Numer_crypto/reports/`
- **PyTorch Environment**: `/media/knight2/EDB/repos/Numer_crypto/pytorch_env/`
- **H2O Environment**: `/media/knight2/EDB/repos/Numer_crypto/h2o_sparkling_java17_env/`