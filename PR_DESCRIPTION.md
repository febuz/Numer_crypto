# Add GPU Performance Tests, Multi-GPU Support, and Repository Organization

## Summary
- Added organized test structure with separate functional and performance tests
- Implemented Java 11 vs Java 17 GPU performance comparison test
- Created multi-GPU tests for H2O Sparkling Water and PyTorch neural networks
- Added Ubuntu 24.04 configuration script for RAPIDS, Spark, and H2O
- Reorganized notebooks and migrated repository to EDB drive for better storage management

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
   - Created `test_multi_gpu_h2o.py` to leverage multiple GPUs simultaneously with H2O Sparkling Water
   - Implemented `multi_gpu_pytorch.py` for neural network training across multiple GPUs
   - Created `setup_env_and_run_multiGPU.sh` for automated environment setup and test execution
   - Added comprehensive documentation with `MULTI_GPU_TESTING.md` and `multi_gpu_test_comparison.md`
   - Tests both H2O Sparkling Water with XGBoost and PyTorch neural networks across all available GPUs
   - Monitors and reports detailed utilization metrics with visualization capabilities

4. **Ubuntu 24.04 configuration**:
   - Added `setup_ubuntu_24.04_gpu.sh` for comprehensive environment setup
   - Configures RAPIDS, Spark, H2O Sparkling Water, and GPU dependencies
   - Creates convenience scripts for Java version switching

5. **Notebook reorganization and repository migration**:
   - Created dedicated `notebook/` structure with subdirectories for specific purposes
   - Moved Colab-specific notebooks to `notebook/colab/` directory
   - Added helper scripts for notebook execution and fixing
   - Migrated repository to EDB drive for better storage of large data files
   - Created symbolic links and helper scripts for easy access to EDB repository

## Test plan
- Run Java comparison test: `./run_java_comparison_test.sh`
- Run multi-GPU tests: `./setup_env_and_run_multiGPU.sh`
- Set up environment for Ubuntu 24.04: `./setup_ubuntu_24.04_gpu.sh`
- Use EDB repository: `./setup_edb_symlink.sh`

## Technical Details
- The tests use threading to monitor GPU utilization in real-time
- Java 17 requires special module options for H2O Sparkling Water
- Multi-GPU tests use CUDA_VISIBLE_DEVICES and torch.cuda.set_device() to isolate specific GPUs
- PyTorch implementation uses custom neural network architecture with CUDA acceleration
- H2O Sparkling tests use separate Spark contexts on different ports
- Results are automatically saved as JSON and visualization plots
- Repository migration uses symbolic links for transparent access between storage locations