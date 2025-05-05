# Clean up redundant files and add multi-GPU support

## PR Summary

This pull request:

1. **Removes redundant files** from the main directory:
   - Moves scripts to appropriate directories
   - Organizes test files into functional and performance categories
   - Removes duplicate code and consolidates functionality

2. **Adds multi-GPU support**:
   - Implements PyTorch neural network training on multiple GPUs
   - Enhances H2O Sparkling Water for multi-GPU XGBoost training
   - Adds automated setup scripts for multi-GPU environment

3. **Improves repository organization**:
   - Reorganizes notebooks into dedicated directories
   - Moves Colab-specific notebooks to notebook/colab
   - Creates proper directory structure for configuration and data

4. **Adds comprehensive documentation**:
   - MULTI_GPU_TESTING.md for multi-GPU capabilities
   - Test comparison documentation
   - Setup scripts with detailed instructions

## Benefits

- **Improved code organization**: All files now reside in logical directories
- **Enhanced GPU utilization**: Can now utilize multiple GPUs in parallel
- **Better maintainability**: Reduced duplication and clearer structure
- **Simplified environment setup**: Automated scripts for different environments

## Testing

The code has been tested with:
- PyTorch multi-GPU neural network training
- H2O Sparkling Water with XGBoost on multiple GPUs
- Various environment configurations (Java 11, Java 17)

## Implementation Details

1. **File cleanup**:
   - Moved redundant scripts to organized directories
   - Consolidated test files into functional and performance categories
   - Created helper scripts for environment setup

2. **Multi-GPU implementation**:
   - Added GPU isolation with CUDA_VISIBLE_DEVICES
   - Implemented thread-based GPU monitoring
   - Created visualization tools for performance analysis
   - Added parallel training capabilities across GPUs

This PR aligns with our goal of creating a clean, maintainable codebase with advanced GPU capabilities for crypto analysis.