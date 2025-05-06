# Clean up redundant files and add feature engineering modules

## PR Summary

This pull request:

1. **Removes redundant files** from the main directory:
   - Moves scripts to appropriate directories
   - Organizes test files into functional and performance categories
   - Removes duplicate code and consolidates functionality
   - Removes unused colab notebooks and redundant PR instruction files
   - Deletes backup and temporary files

2. **Adds comprehensive feature engineering modules**:
   - Creates new `feature/` directory with EDA, feature engineering, and feature selection
   - Adds exploratory data analysis tools with visualization capabilities
   - Implements polynomial feature generator using Spark and H2O
   - Adds feature selection module supporting correlation, AutoML, and permutation importance

3. **Improves repository organization**:
   - Reorganizes notebooks into dedicated directories
   - Removes redundant scripts in favor of v2 versions
   - Creates directory structure for feature output in external directories

4. **Enhances data processing capabilities**:
   - Adds support for distributed processing with Spark and H2O
   - Implements efficient polynomial feature generation for large datasets
   - Provides tools for analyzing feature importance

## Benefits

- **Improved code organization**: All files now reside in logical directories
- **Enhanced data processing**: New modules for feature engineering and selection
- **Better maintainability**: Reduced duplication and clearer structure
- **Modular architecture**: Separation of EDA, feature engineering, and selection

## Implementation Details

1. **File cleanup**:
   - Removed redundant colab_notebooks folder (duplicated in notebook/colab)
   - Cleared temporary data files in data/submissions and data/yiedl/extracted
   - Deleted backup test directories
   - Removed old script versions in favor of v2 versions
   - Deleted duplicate PR instruction files

2. **Feature engineering module implementation**:
   - Created modular architecture for feature engineering
   - Implemented exploratory data analysis with visualization
   - Added polynomial feature generation with Spark and H2O
   - Created feature selection module with multiple methods
   - Set up external directories for results and feature storage

This PR aligns with our goal of creating a clean, maintainable codebase with advanced feature engineering capabilities for cryptocurrency analysis.