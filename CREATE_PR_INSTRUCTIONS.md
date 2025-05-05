# Manual Pull Request Creation Instructions

To create a pull request for merging the `feature/crypto-analysis` branch into `main`, follow these steps:

1. Go to: https://github.com/febuz/Numer_crypto/pull/new/feature/crypto-analysis

2. Set the base branch to `main` and the compare branch to `feature/crypto-analysis`

3. Use the following title:
   ```
   Clean up redundant files and add multi-GPU support
   ```

4. Copy the contents of the `PR_CLEANUP.md` file as the description

5. Click "Create pull request"

After creating the pull request, you can review the changes before merging.

## Key files to review:

1. **Multi-GPU implementation**:
   - `multi_gpu_pytorch.py`
   - `tests/performance/test_multi_gpu_h2o.py`
   - `setup_env_and_run_multiGPU.sh`
   - `MULTI_GPU_TESTING.md`

2. **Notebook reorganization**:
   - `notebook/` directory structure
   - `notebook/colab/` for Colab-specific notebooks

3. **Test organization**:
   - `tests/functional/` for basic compatibility tests
   - `tests/performance/` for benchmarking tests

The PR focuses on cleaning up redundant files that were previously in the main directory while adding new multi-GPU capabilities.