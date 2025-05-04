# Add GPU Performance Tests and H2O Sparkling Water Configuration

## Summary
- Added organized test structure with separate functional and performance tests
- Implemented Java 11 vs Java 17 GPU performance comparison test
- Created multi-GPU test for H2O Sparkling Water
- Added Ubuntu 24.04 configuration script for RAPIDS, Spark, and H2O

## Changes Made
1. **Reorganized test structure**:
   - Created `tests/functional/` for basic compatibility tests
   - Created `tests/performance/` for benchmarking and performance tests
   - Added comprehensive `tests/README.md` documentation

2. **Java GPU comparison testing**:
   - Added `test_java_gpu_comparison.py` to compare Java 11 vs Java 17 performance
   - Measures training time, GPU utilization, and model accuracy
   - Generates visualizations for easy comparison

3. **Multi-GPU utilization testing**:
   - Created `test_multi_gpu_h2o.py` to leverage multiple GPUs simultaneously
   - Tests H2O Sparkling Water with XGBoost across all available GPUs
   - Monitors and reports detailed utilization metrics

4. **Ubuntu 24.04 configuration**:
   - Added `setup_ubuntu_24.04_gpu.sh` for comprehensive environment setup
   - Configures RAPIDS, Spark, H2O Sparkling Water, and GPU dependencies
   - Creates convenience scripts for Java version switching

## Test plan
- Run Java comparison test: `./run_java_comparison_test.sh`
- Run multi-GPU test: `./run_multi_gpu_test.sh`
- Set up environment for Ubuntu 24.04: `./setup_ubuntu_24.04_gpu.sh`

## Technical Details
- The tests use threading to monitor GPU utilization in real-time
- Java 17 requires special module options for H2O Sparkling Water
- Multi-GPU tests use CUDA_VISIBLE_DEVICES to isolate specific GPUs
- Results are automatically saved as JSON and visualization plots