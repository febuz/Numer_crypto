# Remove Redundant Files from Main Directory

## Problem

The repository structure was partially cleaned up, but redundant files still remained in the main directory, causing confusion and inconsistency.

Specifically:
- Setup scripts existed both in the main directory and in `scripts/setup/`
- Test files existed both in the main directory and in appropriate test directories

## Solution

This PR removes all redundant files from the main directory:

1. **Removed redundant setup scripts**:
   - `setup_env.sh` → already in `scripts/setup/`
   - `setup_env_venv.sh` → already in `scripts/setup/`
   - `setup_gpu_test_env.sh` → already in `scripts/setup/`
   - `setup_h2o_sparkling_java17.sh` → already in `scripts/setup/`
   - `setup_test_env_java11.sh` → already in `scripts/setup/`
   - `setup_test_env_java17.sh` → already in `scripts/setup/`
   - `setup_ubuntu_24.04_gpu.sh` → already in `scripts/setup/`
   - `setup_venv.sh` → already in `scripts/setup/`

2. **Removed redundant test files**:
   - `test_h2o_sparkling_java11.py` → already in `tests/performance/`
   - `test_h2o_sparkling_minimal.py` → already in `tests/functional/`

3. **Added maintenance script**:
   - `final_cleanup.sh` to handle future cleanup needs

## Benefits

- **Clean repository structure**: No more redundant files
- **Reduced confusion**: Clear location for each file type
- **Improved maintainability**: Easier to find and work with files

## Testing

Verified that all deleted files already existed in their proper target locations:
- Confirmed setup scripts existed in `scripts/setup/` directory
- Confirmed test files existed in appropriate test directories

## Related PRs

This completes the repository cleanup work started in previous PRs.