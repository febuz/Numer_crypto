#!/usr/bin/env python3
"""
Memory optimization utilities for the Numerai Crypto pipeline.

This module provides functions to optimize memory usage, chunk large datasets,
and monitor memory consumption.
"""
import os
import sys
import gc
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Callable, Optional

# Try to import psutil, but allow fallback if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from config.settings import HARDWARE_CONFIG

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

# Parse total memory from hardware config
try:
    memory_str = HARDWARE_CONFIG.get('total_memory', '64g')
    if memory_str.endswith('g'):
        TOTAL_MEMORY_GB = int(memory_str[:-1])
    elif memory_str.endswith('m'):
        TOTAL_MEMORY_GB = int(memory_str[:-1]) / 1024
    else:
        TOTAL_MEMORY_GB = 64  # Default fallback
except (ValueError, AttributeError):
    TOTAL_MEMORY_GB = 64  # Default fallback

# Reserve 10% of memory for system operations
USABLE_MEMORY_GB = TOTAL_MEMORY_GB * 0.9

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage details.
    
    Returns:
        Dict[str, float]: Dictionary with memory usage metrics in GB
    """
    if not HAS_PSUTIL:
        # Fallback when psutil not available - provide estimate from Python interpreter
        import resource
        # Get memory usage of current process using resource module (Unix systems only)
        try:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            process_memory = rusage.ru_maxrss / (1024 * 1024)  # Convert KB to GB
        except (ImportError, AttributeError):
            process_memory = 0
            
        return {
            'process_gb': process_memory,
            'system_total_gb': TOTAL_MEMORY_GB,
            'system_used_gb': 0,
            'system_available_gb': TOTAL_MEMORY_GB,
            'system_percent': 0
        }
        
    # If psutil is available, use it for accurate information
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    
    system = psutil.virtual_memory()
    system_total = system.total / (1024 ** 3)
    system_used = system.used / (1024 ** 3)
    system_available = system.available / (1024 ** 3)
    system_percent = system.percent
    
    return {
        'process_gb': process_memory,
        'system_total_gb': system_total,
        'system_used_gb': system_used,
        'system_available_gb': system_available,
        'system_percent': system_percent
    }

def log_memory_usage(prefix: str = "") -> None:
    """
    Log current memory usage with optional prefix.
    
    Args:
        prefix (str): Optional prefix for the log message
    """
    mem = get_memory_usage()
    logger.info(f"{prefix} Memory Usage - Process: {mem['process_gb']:.2f} GB, "
                f"System: {mem['system_used_gb']:.2f}/{mem['system_total_gb']:.2f} GB "
                f"({mem['system_percent']}%)")

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a pandas DataFrame by downcasting numeric types.
    
    Args:
        df (pd.DataFrame): DataFrame to optimize
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    
    # Process each column
    for col in df.columns:
        col_dtype = df[col].dtype
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_dtype):
            # Integer columns
            if pd.api.types.is_integer_dtype(col_dtype):
                c_min, c_max = df[col].min(), df[col].max()
                
                # Check if unsigned int is suitable
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                # Signed int
                else:
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            
            # Float columns
            elif pd.api.types.is_float_dtype(col_dtype):
                # Downcasting may cause precision loss, careful with financial data
                c_min, c_max = df[col].min(), df[col].max()
                if not pd.isna(c_min) and not pd.isna(c_max):
                    if (np.abs(df[col]).max() < 65000) and (df[col].round(2) == df[col]).all():
                        df[col] = df[col].astype(np.float16)
                    else:
                        df[col] = df[col].astype(np.float32)
        
        # Convert object columns to categorical when cardinality is low
        elif col_dtype == 'object':
            unique_values = df[col].nunique()
            total_values = len(df[col])
            # Convert to category if column has low cardinality (less than 50% unique values)
            if unique_values < 0.5 * total_values and unique_values < 10000:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.info(f"Optimized DataFrame memory from {start_mem:.2f} MB to {end_mem:.2f} MB "
                f"({reduction:.2f}% reduction)")
    
    return df

def estimate_chunk_size(num_rows: int, num_cols: int, 
                       dtype_size: float = 8,
                       buffer_factor: float = 0.5) -> int:
    """
    Estimate optimal chunk size for processing large datasets.
    
    Args:
        num_rows (int): Total number of rows in the dataset
        num_cols (int): Number of columns in the dataset
        dtype_size (float): Average size of each element in bytes (default: 8 bytes for float64)
        buffer_factor (float): Buffer factor for additional operations (0.5 means reserve 2x memory)
        
    Returns:
        int: Recommended chunk size in rows
    """
    # Estimate bytes per row
    bytes_per_row = num_cols * dtype_size
    
    # Available memory for chunking (considering buffer factor)
    available_memory_bytes = (USABLE_MEMORY_GB * 1024 ** 3) * buffer_factor
    
    # Calculate chunk size
    chunk_size = int(available_memory_bytes / bytes_per_row)
    
    # Cap chunk size to total rows
    chunk_size = min(chunk_size, num_rows)
    
    # Ensure chunk size is at least 1000 rows, unless dataset is smaller
    chunk_size = max(chunk_size, min(1000, num_rows))
    
    logger.info(f"Estimated chunk size: {chunk_size:,} rows (dataset: {num_rows:,} rows, {num_cols} columns)")
    
    return chunk_size

def get_parallel_chunk_count() -> int:
    """
    Calculate optimal number of chunks to process in parallel.
    
    Returns:
        int: Number of chunks to process in parallel
    """
    # Determine based on CPU cores and memory
    cpu_count = os.cpu_count() or 8
    
    # Using a heuristic: each processing unit needs ~2GB plus overhead
    processing_units = max(1, int(USABLE_MEMORY_GB / 4))
    
    # Cap by CPU count
    parallel_chunks = min(cpu_count, processing_units)
    
    logger.info(f"Using {parallel_chunks} parallel chunks (CPUs: {cpu_count}, Memory: {USABLE_MEMORY_GB:.1f} GB)")
    
    return parallel_chunks

def clear_memory(full_gc: bool = False) -> None:
    """
    Clear memory by removing cached objects and running garbage collection.
    
    Args:
        full_gc (bool): Whether to perform a full garbage collection (more thorough but slower)
    """
    # Log memory before cleanup
    log_memory_usage("Before memory cleanup:")
    
    # Clear pandas cache
    pd.core.computation.expressions.clear_cache()
    
    # Run garbage collection
    if full_gc:
        logger.info("Running full garbage collection...")
        gc.collect(generation=2)  # Full collection
    else:
        gc.collect(generation=0)  # Collect youngest generation only
    
    # Log memory after cleanup
    log_memory_usage("After memory cleanup:")

def process_in_chunks(data: pd.DataFrame, 
                      process_func: Callable[[pd.DataFrame], pd.DataFrame],
                      chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Process a large DataFrame in chunks to avoid memory issues.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        process_func (Callable): Function to apply to each chunk
        chunk_size (int, optional): Size of each chunk in rows
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    num_rows, num_cols = data.shape
    
    # Determine chunk size if not provided
    if chunk_size is None:
        chunk_size = estimate_chunk_size(num_rows, num_cols)
    
    # If data is small enough to process at once
    if num_rows <= chunk_size:
        logger.info(f"Processing entire dataset at once ({num_rows} rows)")
        return process_func(data)
    
    # Process in chunks
    result_chunks = []
    num_chunks = (num_rows + chunk_size - 1) // chunk_size  # Ceiling division
    
    logger.info(f"Processing dataset in {num_chunks} chunks of size {chunk_size}")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_rows)
        
        logger.info(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx:,} to {end_idx:,})")
        
        # Process the chunk
        chunk = data.iloc[start_idx:end_idx].copy()
        result_chunk = process_func(chunk)
        
        # Append to results
        result_chunks.append(result_chunk)
        
        # Clear memory after each chunk
        del chunk
        clear_memory(full_gc=(i % 5 == 0))  # Full GC every 5 chunks
    
    # Combine chunks
    logger.info("Combining processed chunks...")
    combined_result = pd.concat(result_chunks, axis=0, ignore_index=True)
    
    # Clear memory
    del result_chunks
    clear_memory(full_gc=True)
    
    return combined_result

if __name__ == "__main__":
    # Test memory utilities
    log_memory_usage("Initial memory:")
    
    # Create a test DataFrame
    logger.info("Creating test DataFrame...")
    test_size = 1_000_000
    test_df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, size=test_size),
        'float_col': np.random.random(test_size),
        'large_float_col': np.random.random(test_size) * 1000000,
        'category_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=test_size),
        'many_cat_col': np.random.choice([f'Cat{i}' for i in range(1000)], size=test_size)
    })
    
    log_memory_usage("After DataFrame creation:")
    
    # Test memory optimization
    logger.info("Optimizing DataFrame memory...")
    optimized_df = optimize_dataframe_memory(test_df)
    
    log_memory_usage("After memory optimization:")
    
    # Test chunk size estimation
    chunk_size = estimate_chunk_size(test_size, len(test_df.columns))
    logger.info(f"Estimated chunk size: {chunk_size:,} rows")
    
    # Test parallel chunk count
    parallel_chunks = get_parallel_chunk_count()
    logger.info(f"Parallel chunk count: {parallel_chunks}")
    
    # Test memory cleanup
    clear_memory(full_gc=True)
    log_memory_usage("Final memory:")