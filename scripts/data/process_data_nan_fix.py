#!/usr/bin/env python3
"""
process_data_nan_fix.py - A utility to fix NaN values in the dataset by replacing with column means
"""
import os
import sys
import logging
import argparse
import polars as pl
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_nan_values(input_file, output_file=None):
    """
    Replace NaN values with column means instead of zeros.
    
    Args:
        input_file: Path to the parquet file with NaN values
        output_file: Path to save the fixed file (defaults to input_file with '_fixed' suffix)
        
    Returns:
        Path to the fixed file
    """
    logger.info(f"Loading data from {input_file}")
    
    try:
        # Load the data with polars
        df = pl.read_parquet(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # If output file not specified, create one
        if output_file is None:
            # Get directory and filename
            input_path = Path(input_file)
            dirname = input_path.parent
            filename = input_path.stem
            extension = input_path.suffix
            
            # Create new filename with '_fixed' suffix
            output_file = os.path.join(dirname, f"{filename}_fixed{extension}")
        
        # Get columns that might need NaN fixing (exclude non-numeric and special columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target', 'asset']
        numeric_cols = []
        
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                numeric_cols.append(col)
        
        logger.info(f"Found {len(numeric_cols)} numeric columns that may contain NaN values")
        
        # Calculate means for each column and replace NaNs
        columns_with_nans = []
        total_nan_count = 0
        
        # Check how many columns have NaNs and count them
        for col in numeric_cols:
            # Get null count for the column
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count > 0:
                columns_with_nans.append((col, null_count))
                total_nan_count += null_count
        
        logger.info(f"Found {len(columns_with_nans)} columns with NaN values, total {total_nan_count} NaNs")
        
        # Process each column with NaNs - replace with mean instead of 0
        exprs = []
        for col, nan_count in columns_with_nans:
            # Calculate mean of the column (excluding NaNs)
            try:
                col_mean = df.select(pl.col(col).mean()).item()
                # If mean is NaN (all values are NaN), use 0 instead
                if col_mean is None or pl.Series([col_mean]).is_null().any():
                    logger.warning(f"Column {col} has all NaN values, using 0 as replacement")
                    col_mean = 0.0
                
                # Create expression to fill nulls with mean
                expr = pl.col(col).fill_null(col_mean).alias(col)
                exprs.append(expr)
                
                logger.info(f"Column {col}: replacing {nan_count} NaNs with mean value {col_mean}")
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}")
                # Skip this column
                continue
        
        # Apply all expressions at once for efficiency
        if exprs:
            df = df.with_columns(exprs)
            logger.info(f"Replaced NaNs in {len(exprs)} columns with their respective means")
        
        # Verify no more NaNs in numeric columns
        remaining_nans = 0
        for col in numeric_cols:
            null_count = df.select(pl.col(col).null_count()).item()
            remaining_nans += null_count
        
        if remaining_nans > 0:
            logger.warning(f"Still have {remaining_nans} NaNs after fixing - these may be in columns that had all NaN values")
        else:
            logger.info("Successfully replaced all NaN values")
        
        # Save the fixed data
        logger.info(f"Saving fixed data to {output_file}")
        df.write_parquet(output_file)
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error fixing NaN values: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='Fix NaN values in dataset by replacing with column means')
    parser.add_argument('--input', type=str, required=True, help='Input parquet file with NaN values')
    parser.add_argument('--output', type=str, help='Output file for fixed data')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    output_file = fix_nan_values(args.input, args.output)
    
    if output_file:
        logger.info(f"NaN fixing completed successfully: {output_file}")
        return 0
    else:
        logger.error("NaN fixing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())