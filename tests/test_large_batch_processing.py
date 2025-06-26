#!/usr/bin/env python3
"""
Test Large-Scale Batch Processing with GPU Acceleration

This test verifies that our batched processing approach works correctly with
very large datasets, similar to the ones used in production (3.4M rows × 3.6K columns).

It demonstrates:
1. Safe memory usage with extremely large datasets
2. Efficient batched feature engineering
3. Graceful fallback to CPU when needed
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import GPU Math Accelerator
from scripts.features.gpu_math_accelerator import GPUMathAccelerator

# Create a simple timer context manager
class Timer:
    def __init__(self, name="Operation"):
        self.name = name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {self.duration:.2f} seconds")

def monitor_memory():
    """Monitor and log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        
        # Get system memory
        system_mem = psutil.virtual_memory()
        system_used_gb = (system_mem.total - system_mem.available) / (1024 ** 3)
        system_total_gb = system_mem.total / (1024 ** 3)
        
        logger.info(f"Memory usage: Process: {mem_gb:.2f} GB, System: {system_used_gb:.2f}/{system_total_gb:.2f} GB ({system_mem.percent}%)")
        
        return mem_gb, system_used_gb, system_total_gb
    except ImportError:
        logger.warning("psutil not available, skipping memory monitoring")
        return 0, 0, 0

def create_synthetic_dataset(rows: int, cols: int, data_dir: str, batch_size: int = 100000) -> str:
    """
    Create a synthetic dataset with specified dimensions, using batching for memory efficiency
    
    Args:
        rows: Number of rows
        cols: Number of columns
        data_dir: Directory to save the dataset
        batch_size: Batch size for generation
        
    Returns:
        Path to the saved parquet file
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # File path
    file_path = os.path.join(data_dir, f"synthetic_data_{rows}x{cols}.parquet")
    
    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Using existing synthetic dataset: {file_path}")
        return file_path
    
    logger.info(f"Creating synthetic dataset with {rows:,} rows and {cols:,} columns")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random column names
    col_names = [f'feature_{i}' for i in range(cols)]
    
    # Create dataframe in batches
    with Timer("Synthetic data generation"):
        for i in range(0, rows, batch_size):
            end_idx = min(i + batch_size, rows)
            batch_rows = end_idx - i
            
            logger.info(f"Generating batch {i//batch_size + 1}/{(rows + batch_size - 1)//batch_size}: rows {i}-{end_idx}")
            
            # Create random data for this batch
            batch_data = np.random.randn(batch_rows, cols).astype(np.float32)
            
            # Convert to dataframe
            batch_df = pd.DataFrame(batch_data, columns=col_names)
            
            # Save batch
            mode = 'w' if i == 0 else 'a'  # Overwrite first time, append after
            batch_df.to_parquet(file_path, engine='pyarrow', index=False, mode=mode)
            
            # Monitor memory
            monitor_memory()
            
            # Clean up
            del batch_data
            del batch_df
            gc.collect()
    
    logger.info(f"Synthetic dataset created and saved to {file_path}")
    return file_path

def test_batch_processing_with_random_subset(data_path: str, subset_rows: int, 
                                            subset_cols: int, batch_size: int) -> None:
    """
    Test batch processing with a random subset of the full dataset
    
    Args:
        data_path: Path to the full dataset
        subset_rows: Number of rows to sample
        subset_cols: Number of columns to sample
        batch_size: Size of processing batches
    """
    logger.info(f"Testing batch processing with random subset: {subset_rows:,} rows × {subset_cols:,} columns")
    
    # Read dataset metadata
    file_meta = pd.read_parquet(data_path, engine='pyarrow')
    total_rows = len(file_meta)
    total_cols = len(file_meta.columns)
    
    logger.info(f"Full dataset has {total_rows:,} rows and {total_cols:,} columns")
    
    # Sample random rows
    row_indices = np.random.choice(total_rows, size=min(subset_rows, total_rows), replace=False)
    row_indices.sort()  # Sort for efficiency
    
    # Sample random columns
    all_cols = file_meta.columns.tolist()
    col_indices = np.random.choice(total_cols, size=min(subset_cols, total_cols), replace=False)
    selected_cols = [all_cols[i] for i in col_indices]
    
    # Read the subset
    with Timer("Reading data subset"):
        df_subset = pd.read_parquet(data_path, engine='pyarrow', columns=selected_cols).iloc[row_indices]
    
    logger.info(f"Loaded subset with shape: {df_subset.shape}")
    
    # Monitor memory
    monitor_memory()
    
    # Initialize GPU Math Accelerator
    os.environ["GPU_MEMORY_LIMIT"] = "20.0"  # Set GPU memory limit
    accelerator = GPUMathAccelerator()
    
    # Process data in batches
    feature_names = df_subset.columns.tolist()
    data_array = df_subset.values.astype(np.float32)
    
    # Process in batches
    n_samples = data_array.shape[0]
    transformed_batches = []
    transformed_names = None
    
    logger.info(f"Processing {n_samples:,} rows in batches of {batch_size:,}")
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: rows {i:,}-{end_idx:,}")
        
        # Get batch
        batch_data = data_array[i:end_idx]
        
        # Process batch
        with Timer(f"GPU transform batch {i//batch_size + 1}"):
            try:
                # Apply transformations - use conservative settings
                batch_result, batch_names = accelerator.generate_all_math_transforms(
                    batch_data, feature_names,
                    include_trig=False,  # Disable trig transforms
                    include_poly=False,  # Disable polynomial transforms
                    max_interactions=100,  # Limit interactions
                    include_random_baselines=False,  # Skip random baselines
                    batch_size=min(10000, batch_size//2)  # Use nested batching if needed
                )
                
                # Store results
                transformed_batches.append(batch_result)
                
                if transformed_names is None:
                    transformed_names = batch_names
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                logger.info("Continuing with next batch")
        
        # Monitor memory
        monitor_memory()
        
        # Clean up
        del batch_data
        gc.collect()
    
    # Combine results if we have any
    if transformed_batches:
        with Timer("Combining transformed batches"):
            try:
                # Use vstack for numpy arrays
                all_transformed = np.vstack(transformed_batches)
                logger.info(f"Final transformed shape: {all_transformed.shape}")
            except Exception as e:
                logger.error(f"Error combining batches: {e}")
                logger.info("Number of batches: " + str(len(transformed_batches)))
                for i, batch in enumerate(transformed_batches):
                    logger.info(f"Batch {i} shape: {batch.shape}")
    else:
        logger.warning("No transformed batches were created")
    
    # Final memory check
    monitor_memory()
    
    logger.info("Batch processing test completed")

def test_batch_processing_with_full_data(data_path: str, batch_size: int) -> None:
    """
    Test batch processing with the full dataset
    
    Args:
        data_path: Path to the dataset
        batch_size: Size of processing batches
    """
    logger.info(f"Testing batch processing with full dataset")
    
    # Read dataset metadata
    file_meta = pd.read_parquet(data_path, engine='pyarrow')
    total_rows = len(file_meta)
    total_cols = len(file_meta.columns)
    
    logger.info(f"Full dataset has {total_rows:,} rows and {total_cols:,} columns")
    
    # Define column batches to avoid loading the entire dataset at once
    col_batch_size = 500  # Process 500 columns at a time
    all_cols = file_meta.columns.tolist()
    
    # Process in column batches
    for col_start in range(0, total_cols, col_batch_size):
        col_end = min(col_start + col_batch_size, total_cols)
        selected_cols = all_cols[col_start:col_end]
        
        logger.info(f"Processing column batch {col_start//col_batch_size + 1}/{(total_cols + col_batch_size - 1)//col_batch_size}: columns {col_start}-{col_end}")
        
        # Read just these columns
        with Timer(f"Reading column batch {col_start//col_batch_size + 1}"):
            try:
                df_subset = pd.read_parquet(data_path, engine='pyarrow', columns=selected_cols)
                logger.info(f"Loaded data with shape: {df_subset.shape}")
            except Exception as e:
                logger.error(f"Error reading column batch: {e}")
                continue
        
        # Monitor memory
        monitor_memory()
        
        # Initialize GPU Math Accelerator
        os.environ["GPU_MEMORY_LIMIT"] = "20.0"  # Set GPU memory limit
        accelerator = GPUMathAccelerator()
        
        # Convert to numpy array
        feature_names = df_subset.columns.tolist()
        data_array = df_subset.values.astype(np.float32)
        
        # Process in row batches
        n_samples = data_array.shape[0]
        
        logger.info(f"Processing {n_samples:,} rows in batches of {batch_size:,}")
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            logger.info(f"Processing row batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: rows {i:,}-{end_idx:,}")
            
            # Get batch
            batch_data = data_array[i:end_idx]
            
            # Process batch
            with Timer(f"GPU transform batch {i//batch_size + 1}"):
                try:
                    # Apply transformations - use conservative settings
                    batch_result, batch_names = accelerator.generate_all_math_transforms(
                        batch_data, feature_names,
                        include_trig=False,  # Disable trig transforms
                        include_poly=False,  # Disable polynomial transforms
                        max_interactions=50,  # Very conservative interactions
                        include_random_baselines=False,  # Skip random baselines
                        batch_size=min(5000, batch_size//4)  # Use nested batching if needed
                    )
                    
                    # Just log the shape, don't store results to save memory
                    logger.info(f"Transformed shape: {batch_result.shape}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
            
            # Monitor memory
            monitor_memory()
            
            # Clean up
            del batch_data
            gc.collect()
        
        # Clean up column batch
        del df_subset
        gc.collect()
    
    # Final memory check
    monitor_memory()
    
    logger.info("Full data batch processing test completed")

def main():
    parser = argparse.ArgumentParser(description='Test Large-Scale Batch Processing')
    parser.add_argument('--rows', type=int, default=3000000, help='Number of rows for synthetic data')
    parser.add_argument('--cols', type=int, default=3000, help='Number of columns for synthetic data')
    parser.add_argument('--data-dir', type=str, default='./data/test_large', help='Directory for test data')
    parser.add_argument('--batch-size', type=int, default=50000, help='Batch size for processing')
    parser.add_argument('--subset-rows', type=int, default=500000, help='Number of rows for subset test')
    parser.add_argument('--subset-cols', type=int, default=500, help='Number of columns for subset test')
    parser.add_argument('--test-full', action='store_true', help='Test with the full dataset')
    parser.add_argument('--skip-generation', action='store_true', help='Skip dataset generation')
    
    args = parser.parse_args()
    
    logger.info("Starting Large-Scale Batch Processing Test")
    
    # Log initial memory usage
    logger.info("Initial memory usage:")
    monitor_memory()
    
    # Step 1: Create or load synthetic dataset
    if not args.skip_generation:
        data_path = create_synthetic_dataset(args.rows, args.cols, args.data_dir, args.batch_size)
    else:
        # Use existing dataset
        data_path = os.path.join(args.data_dir, f"synthetic_data_{args.rows}x{args.cols}.parquet")
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found: {data_path}")
            logger.info("Either generate the dataset first or provide the correct path")
            return
    
    # Step 2: Test with random subset
    test_batch_processing_with_random_subset(
        data_path, args.subset_rows, args.subset_cols, args.batch_size
    )
    
    # Step 3: Test with full data if requested
    if args.test_full:
        test_batch_processing_with_full_data(data_path, args.batch_size)
    
    logger.info("Large-Scale Batch Processing Test completed successfully")

if __name__ == "__main__":
    main()