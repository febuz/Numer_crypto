# NaN Value Replacement Fix

## Problem

The codebase was experiencing a significant performance issue where the system would hang for approximately 5 hours during NaN value replacement. The root cause was identified as the use of zero replacement for NaN values in large datasets.

## Solution

This fix implements a more statistically sound approach by replacing NaN values with column means rather than zeros. This provides several benefits:

1. **Statistical Validity**: Using column means preserves the statistical properties of the data better than using zeros
2. **Performance Improvement**: Prevents the code from hanging during large data processing operations
3. **Better Model Performance**: Models trained on data with mean-imputed values typically perform better than those trained with zero-imputed values

## Files Modified

The following files have been modified to implement this fix:

1. `scripts/create_pytorch_predictions.py`: Updated to replace NaN values with column means
2. `scripts/features/high_memory.py`: Updated the rolling feature and EWM feature generators to use column means

## New Files

Several new files have been added to support this fix:

1. `scripts/data/process_data_nan_fix.py`: A utility script to replace NaN values with column means in any dataset
2. `scripts/run_with_nan_means.py`: A wrapper script to run the full pipeline with NaN mean replacement
3. `fix_nan_values.sh`: A convenient shell script to fix NaN values in a dataset

## How to Use

### Option 1: Fix an existing dataset

```bash
./fix_nan_values.sh [input_file]
```

If no input file is provided, the script will look for data in standard locations.

### Option 2: Run the full pipeline with NaN mean replacement

```bash
python -m scripts.run_with_nan_means [--input INPUT_FILE] [--output OUTPUT_DIR] [--skip-processing] [--skip-features]
```

Options:
- `--input`: Path to input data file (optional)
- `--output`: Directory to save processed data (optional)
- `--skip-processing`: Skip data processing and only fix NaNs
- `--skip-features`: Skip feature generation

## Implementation Details

The NaN replacement logic follows these steps:

1. Identify columns with NaN values
2. For each column with NaNs:
   - Calculate the column mean (excluding NaN values)
   - Replace NaN values with the calculated mean
   - If all values in a column are NaN (resulting in a NaN mean), fall back to zero replacement

This approach is implemented consistently across all relevant parts of the codebase to ensure uniform behavior.

## Performance Impact

This fix should significantly reduce processing time for datasets with NaN values and eliminate the 5-hour hang issue previously encountered.