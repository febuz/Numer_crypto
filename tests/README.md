# GPU Performance Tests for Machine Learning Libraries

This directory contains tests for evaluating GPU performance with XGBoost, LightGBM, and H2O Sparkling Water. The tests measure GPU utilization, memory consumption, and training time across different configurations.

## Directory Structure

- `functional/`: Basic functionality and compatibility tests
- `performance/`: Performance benchmarks and GPU utilization tests

## Key Tests

### Functional Tests

- `test_h2o_sparkling_minimal.py`: Minimal test to verify H2O Sparkling Water functionality
- `test_h2o_java17_simple.py`: Verify Java 17 compatibility with H2O
- `test_rapids.py`: Test RAPIDS functionality
- `test_hardware.py`: Query and verify hardware capabilities

### Performance Tests

- `test_java_gpu_comparison.py`: Compare GPU performance between Java 11 and Java 17
- `test_multi_gpu_h2o.py`: Test H2O Sparkling Water with multiple GPUs
- `test_peak_gpu.py`: Measure peak GPU utilization across libraries
- `test_gpu_integration.py`: Comprehensive test of all libraries with GPU support
- `test_lightgbm_gpu.py`: Test LightGBM GPU performance
- `test_h2o_sparkling_java11.py`: Test H2O Sparkling with Java 11
- `test_h2o_sparkling_java17.py`: Test H2O Sparkling with Java 17

## Running Tests

### Setup

Before running tests, ensure your environment is properly configured:

```bash
# Configure environment for Ubuntu 24.04
bash setup_ubuntu_24.04_gpu.sh

# For Java 11 (default for H2O)
source setup_java11_env.sh

# For Java 17 (requires module options)
source setup_java17_env.sh
```

### Running Individual Tests

```bash
# Run a specific test
python tests/performance/test_peak_gpu.py

# Run with custom parameters
python tests/performance/test_multi_gpu_h2o.py --rows 200000 --cols 50
```

### Wrapper Scripts

We provide convenience wrapper scripts for common test scenarios:

```bash
# Run multi-GPU test
./run_multi_gpu_test.sh

# Run Java comparison test
./run_java_comparison_test.sh
```

## GPU Requirements

- NVIDIA GPU with CUDA support
- For multi-GPU tests: Multiple NVIDIA GPUs
- NVIDIA drivers and CUDA toolkit installed

## Java Requirements

- Java 11 (recommended for best compatibility with H2O)
- Java 17 (supported with additional module options)

## Results

Test results are saved to the `reports/` directory, including:
- JSON files with detailed metrics
- Plots comparing performance across configurations
- Summary statistics for easy analysis

## Troubleshooting

If you encounter issues:

1. Check Java version compatibility (`java -version`)
2. Verify NVIDIA drivers are installed and working (`nvidia-smi`)
3. Ensure CUDA toolkit is properly configured
4. Check environment variables in your setup