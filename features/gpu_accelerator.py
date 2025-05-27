#!/usr/bin/env python3
"""
GPU Accelerator for Feature Generation

This module provides GPU-accelerated functions for feature generation,
significantly improving performance for large datasets.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import relevant utilities
from utils.gpu.detection import get_available_gpus, select_best_gpu
from utils.gpu.optimization import optimize_cuda_memory_usage
from utils.memory_utils import log_memory_usage, clear_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize GPU support flags
cuda_available = False
cupy_available = False
torch_available = False

# Try to import CUDA libraries
def detect_gpu_libraries():
    """Detect GPU libraries with better error handling and logging"""
    global cuda_available, cupy_available, torch_available
    
    # Try to detect PyTorch first (most reliable)
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            cuda_devices = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            logger.info(f"PyTorch CUDA is available: {cuda_devices} devices, CUDA {cuda_version}")
            torch_available = True
            cuda_available = True
            return True
        else:
            logger.warning("PyTorch installed but CUDA is not available")
    except ImportError:
        logger.info("PyTorch not available")
    
    # Try to detect CuPy as second option
    try:
        import cupy as cp
        logger.info(f"CuPy version: {cp.__version__}")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            logger.info(f"CuPy detected {device_count} CUDA devices")
            cupy_available = True
            cuda_available = True
            return True
        except Exception as e:
            logger.warning(f"CuPy is installed but CUDA initialization failed: {e}")
    except ImportError:
        logger.info("CuPy not available")
    
    # If we get here, check if NVIDIA GPU is available at system level
    try:
        import subprocess
        nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if nvidia_smi.returncode == 0:
            logger.warning("NVIDIA GPU detected but no CUDA Python libraries are available")
            logger.warning("Try activating the virtual environment: source /media/knight2/EDB/numer_crypto_temp/environment/bin/activate")
        else:
            logger.warning("No NVIDIA GPU detected via nvidia-smi")
    except Exception as e:
        logger.warning(f"Error checking for NVIDIA GPU: {e}")
    
    return False

# Run detection
detect_gpu_libraries()

# Try to activate environment and detect again if needed
if not cuda_available:
    try:
        import subprocess
        logger.info("Trying to activate environment and detect GPU again...")
        result = subprocess.run(
            ["/bin/bash", "-c", "source /media/knight2/EDB/numer_crypto_temp/environment/bin/activate && python3 -c \"import torch; print(torch.cuda.is_available())\""],
            capture_output=True, text=True
        )
        if "True" in result.stdout:
            logger.info("PyTorch with CUDA found in virtual environment! Attempting to import...")
            import torch
            if torch.cuda.is_available():
                # Update the global flags
                torch_available = True
                cuda_available = True
                logger.info(f"Successfully imported PyTorch with GPU support from environment")
            else:
                logger.warning("PyTorch imported but CUDA not available in this context")
    except Exception as e:
        logger.warning(f"Failed to dynamically load PyTorch: {e}")

if cuda_available:
    logger.info("GPU acceleration is available and will be used")
else:
    logger.warning("No CUDA libraries available. Using CPU with NumPy instead.")


class GPUFeatureAccelerator:
    """
    Class for accelerating feature generation using GPUs.
    Uses vectorized operations and GPU acceleration where available.
    """
    
    def __init__(self, output_dir: str = "/media/knight2/EDB/numer_crypto_temp/data/features", force_gpu: bool = False):
        """
        Initialize the GPU Feature Accelerator
        
        Args:
            output_dir: Directory to save generated feature files
            force_gpu: Force GPU detection even in subprocesses
        """
        global cuda_available, torch_available, cupy_available
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Force GPU detection if needed
        if force_gpu and not cuda_available:
            logger.info("Force GPU flag set, attempting to detect GPU again...")
            detect_gpu_libraries()
            
            # Try to import environment libraries
            try:
                import subprocess
                result = subprocess.run(
                    ["/bin/bash", "-c", "source /media/knight2/EDB/numer_crypto_temp/environment/bin/activate && python3 -c \"import torch; print(torch.cuda.is_available())\""],
                    capture_output=True, text=True
                )
                if "True" in result.stdout:
                    logger.info("PyTorch with CUDA found in environment! Importing...")
                    # Import torch again to ensure it's loaded from the environment
                    import torch
                    if torch.cuda.is_available():
                        # Update the global flags
                        torch_available = True
                        cuda_available = True
                        logger.info(f"Successfully imported PyTorch with GPU support")
                    else:
                        logger.warning("PyTorch imported but CUDA still not available")
            except Exception as e:
                logger.warning(f"Failed to force GPU detection: {e}")
        
        # Select best GPU if available
        self.gpu_id = select_best_gpu() if cuda_available else None
        
        # Set up GPU environment if available
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            optimize_cuda_memory_usage(reserve_memory_fraction=0.1)
            logger.info(f"Using GPU {self.gpu_id} for feature acceleration")
            
            # Initialize GPU memory for the library we're using
            if cupy_available:
                self.xp = cp
                # Get GPU memory info
                try:
                    mem_info = cp.cuda.runtime.memGetInfo()
                    free_bytes = mem_info[0]
                    total_bytes = mem_info[1]
                    logger.info(f"GPU memory: {free_bytes/(1024**3):.2f} GB free / {total_bytes/(1024**3):.2f} GB total")
                except Exception as e:
                    logger.warning(f"Could not get GPU memory info: {e}")
            elif torch_available:
                self.xp = torch
                try:
                    # Get GPU memory info
                    free_bytes = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    logger.info(f"GPU memory: {free_bytes/(1024**3):.2f} GB free / {total_bytes/(1024**3):.2f} GB total")
                    
                    # Run test to verify GPU acceleration
                    test_tensor = torch.ones((10, 10), device='cuda')
                    test_result = test_tensor + test_tensor
                    logger.info(f"PyTorch GPU test successful - tensor operations working")
                except Exception as e:
                    logger.warning(f"Could not initialize PyTorch GPU: {e}")
                    self.xp = np
                    self.gpu_id = None
            else:
                self.xp = np
                logger.warning("No GPU acceleration libraries available, falling back to NumPy")
        else:
            self.xp = np
            logger.warning("No GPU available, using CPU with NumPy")
    
    def _to_device(self, data):
        """Move data to appropriate device (GPU or CPU)"""
        if not cuda_available:
            return data  # Already on CPU with numpy
            
        if cupy_available:
            if isinstance(data, np.ndarray):
                return cp.array(data)
            return data
        elif torch_available:
            if isinstance(data, np.ndarray):
                return torch.tensor(data, device='cuda')
            return data
        else:
            return data
    
    def _to_numpy(self, data):
        """Move data back to CPU as numpy array"""
        if not cuda_available or isinstance(data, np.ndarray):
            return data
            
        if cupy_available:
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
            return data
        elif torch_available:
            if torch.is_tensor(data):
                return data.cpu().numpy()
            return data
        else:
            return data
            
    def generate_rolling_features(self, df, group_col, numeric_cols, windows, date_col=None):
        """
        Generate rolling window features using GPU acceleration.
        
        Args:
            df: Input DataFrame (pandas or polars)
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            windows: List of window sizes
            date_col: Optional date column for sorting
            
        Returns:
            DataFrame with added rolling features
        """
        logger.info(f"Generating GPU-accelerated rolling features with windows {windows}")
        
        # Determine if input is polars or pandas
        is_polars = 'pl' in str(type(df))
        
        if is_polars:
            import polars as pl
            # Convert to pandas for processing
            pandas_df = df.to_pandas()
        else:
            import pandas as pd
            pandas_df = df
            
        # Ensure group column is of appropriate type for groupby
        pandas_df[group_col] = pandas_df[group_col].astype('category')
        
        # Pre-sort by group and date if date column exists
        if date_col and date_col in pandas_df.columns:
            pandas_df = pandas_df.sort_values([group_col, date_col])
            
        # Get unique groups
        groups = pandas_df[group_col].unique()
        
        # Initialize dict to store results
        result_dict = {}
        
        # For each window and function, create features
        for window in windows:
            for func_name in ['mean', 'std', 'max', 'min']:
                logger.info(f"Generating {func_name} features with window {window}")
                
                # Process each group in parallel on GPU if possible
                for col in numeric_cols:
                    # Define column name
                    feature_name = f"{col}_roll_{window}_{func_name}"
                    
                    # Initialize column with NaN
                    result_dict[feature_name] = np.full(len(pandas_df), np.nan)
                    
                    # Process each group 
                    for group in groups:
                        # Get indices for this group
                        mask = pandas_df[group_col] == group
                        indices = np.where(mask)[0]
                        
                        if len(indices) == 0:
                            continue
                            
                        # Get data for this group and column
                        data = pandas_df.loc[mask, col].values
                        
                        # Skip if no data
                        if len(data) == 0 or np.all(np.isnan(data)):
                            continue
                            
                        # Move data to GPU if available
                        data_device = self._to_device(data)
                        
                        # Calculate rolling feature
                        if cupy_available:
                            # Using CuPy for GPU acceleration
                            if func_name == 'mean':
                                result = cp.lib.stride_tricks.sliding_window_view(data_device, window)
                                result = cp.nanmean(result, axis=1)
                                # Pad with NaN at the beginning
                                result = cp.pad(result, (window-1, 0), constant_values=cp.nan)
                            elif func_name == 'std':
                                result = cp.lib.stride_tricks.sliding_window_view(data_device, window)
                                result = cp.nanstd(result, axis=1)
                                result = cp.pad(result, (window-1, 0), constant_values=cp.nan)
                            elif func_name == 'max':
                                result = cp.lib.stride_tricks.sliding_window_view(data_device, window)
                                result = cp.nanmax(result, axis=1)
                                result = cp.pad(result, (window-1, 0), constant_values=cp.nan)
                            elif func_name == 'min':
                                result = cp.lib.stride_tricks.sliding_window_view(data_device, window)
                                result = cp.nanmin(result, axis=1)
                                result = cp.pad(result, (window-1, 0), constant_values=cp.nan)
                        elif torch_available:
                            # Using PyTorch for GPU acceleration
                            if func_name == 'mean':
                                result = torch.nn.functional.avg_pool1d(
                                    torch.nan_to_num(data_device.float().view(1, 1, -1), 0),
                                    kernel_size=window,
                                    stride=1,
                                    padding=0
                                ).view(-1).cpu().numpy()
                                # Pad with NaN at the beginning
                                result = np.pad(result, (window-1, 0), constant_values=np.nan)
                            elif func_name == 'std':
                                # Calculate using rolling mean and sum of squares
                                x2 = torch.nan_to_num(data_device.float() ** 2, 0).view(1, 1, -1)
                                x = torch.nan_to_num(data_device.float(), 0).view(1, 1, -1)
                                
                                mean_x2 = torch.nn.functional.avg_pool1d(x2, kernel_size=window, stride=1, padding=0).view(-1)
                                mean_x = torch.nn.functional.avg_pool1d(x, kernel_size=window, stride=1, padding=0).view(-1)
                                
                                result = torch.sqrt(torch.clamp(mean_x2 - mean_x**2, min=0)).cpu().numpy()
                                result = np.pad(result, (window-1, 0), constant_values=np.nan)
                            elif func_name == 'max':
                                result = torch.nn.functional.max_pool1d(
                                    torch.nan_to_num(data_device.float().view(1, 1, -1), float('-inf')),
                                    kernel_size=window,
                                    stride=1,
                                    padding=0
                                ).view(-1).cpu().numpy()
                                result = np.pad(result, (window-1, 0), constant_values=np.nan)
                            elif func_name == 'min':
                                # Max pool with negated values
                                result = -torch.nn.functional.max_pool1d(
                                    -torch.nan_to_num(data_device.float().view(1, 1, -1), float('inf')),
                                    kernel_size=window,
                                    stride=1,
                                    padding=0
                                ).view(-1).cpu().numpy()
                                result = np.pad(result, (window-1, 0), constant_values=np.nan)
                        # TensorFlow code removed
                        else:
                            # Fall back to numpy if no GPU libraries available
                            data_device = data  # data already as numpy array

                            # Note: This is vectorized numpy, still much faster than loops
                            if func_name == 'mean':
                                result = np.convolve(np.nan_to_num(data_device, 0), np.ones(window)/window, mode='full')[:len(data_device)]
                                # Fix NaN handling
                                result[:window-1] = np.nan
                            elif func_name == 'std':
                                result = np.full_like(data_device, np.nan)
                                for i in range(window-1, len(data_device)):
                                    result[i] = np.nanstd(data_device[i-window+1:i+1])
                            elif func_name == 'max':
                                result = np.full_like(data_device, np.nan)
                                for i in range(window-1, len(data_device)):
                                    result[i] = np.nanmax(data_device[i-window+1:i+1])
                            elif func_name == 'min':
                                result = np.full_like(data_device, np.nan)
                                for i in range(window-1, len(data_device)):
                                    result[i] = np.nanmin(data_device[i-window+1:i+1])
                        
                        # Convert back to numpy if needed
                        if not isinstance(result, np.ndarray):
                            result = self._to_numpy(result)
                            
                        # Ensure result length matches the group length
                        if len(result) != len(indices):
                            # Truncate or pad as needed
                            if len(result) > len(indices):
                                result = result[:len(indices)]
                            else:
                                result = np.pad(result, (0, len(indices) - len(result)), constant_values=np.nan)
                        
                        # Assign result to the column
                        result_dict[feature_name][indices] = result
        
        # Create a DataFrame from the result dictionary
        import pandas as pd
        features_df = pd.DataFrame(result_dict)
        
        # Combine with original DataFrame
        result_df = pd.concat([pandas_df, features_df], axis=1)
        
        # Convert back to polars if input was polars
        if is_polars:
            import polars as pl
            return pl.from_pandas(result_df)
        
        return result_df
        
    def generate_lag_features(self, df, group_col, numeric_cols, lag_periods, date_col=None):
        """
        Generate lag features using GPU acceleration.
        
        Args:
            df: Input DataFrame (pandas or polars)
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            lag_periods: List of lag periods
            date_col: Optional date column for sorting
            
        Returns:
            DataFrame with added lag features
        """
        logger.info(f"Generating GPU-accelerated lag features with periods {lag_periods}")
        
        # Determine if input is polars or pandas
        is_polars = 'pl' in str(type(df))
        
        if is_polars:
            import polars as pl
            # Convert to pandas for processing
            pandas_df = df.to_pandas()
        else:
            import pandas as pd
            pandas_df = df
            
        # Ensure group column is of appropriate type for groupby
        pandas_df[group_col] = pandas_df[group_col].astype('category')
        
        # Pre-sort by group and date if date column exists
        if date_col and date_col in pandas_df.columns:
            pandas_df = pandas_df.sort_values([group_col, date_col])
            
        # Get unique groups
        groups = pandas_df[group_col].unique()
        
        # Initialize dict to store results
        result_dict = {}
        
        # For each lag period, create features
        for lag in lag_periods:
            logger.info(f"Generating lag features with period {lag}")
            
            # Process each column
            for col in numeric_cols:
                # Define column name
                feature_name = f"{col}_lag_{lag}"
                
                # Initialize column with NaN
                result_dict[feature_name] = np.full(len(pandas_df), np.nan)
                
                # Process each group 
                for group in groups:
                    # Get indices for this group
                    mask = pandas_df[group_col] == group
                    indices = np.where(mask)[0]
                    
                    if len(indices) == 0:
                        continue
                        
                    # Get data for this group and column
                    data = pandas_df.loc[mask, col].values
                    
                    # Skip if no data
                    if len(data) == 0:
                        continue
                        
                    # Calculate lag - this is simple array shifting
                    # No need for GPU acceleration for this operation
                    if lag > 0:
                        lagged_data = np.pad(data, (lag, 0), mode='constant', constant_values=np.nan)[:-lag]
                    else:
                        lagged_data = data  # No lag
                        
                    # Ensure result length matches the group length
                    if len(lagged_data) != len(indices):
                        # Truncate or pad as needed
                        if len(lagged_data) > len(indices):
                            lagged_data = lagged_data[:len(indices)]
                        else:
                            lagged_data = np.pad(lagged_data, (0, len(indices) - len(lagged_data)), constant_values=np.nan)
                    
                    # Assign result to the column
                    result_dict[feature_name][indices] = lagged_data
        
        # Create a DataFrame from the result dictionary
        import pandas as pd
        features_df = pd.DataFrame(result_dict)
        
        # Combine with original DataFrame
        result_df = pd.concat([pandas_df, features_df], axis=1)
        
        # Convert back to polars if input was polars
        if is_polars:
            import polars as pl
            return pl.from_pandas(result_df)
        
        return result_df

    def generate_ewm_features(self, df, group_col, numeric_cols, spans, date_col=None):
        """
        Generate exponentially weighted moving average features using GPU acceleration.
        
        Args:
            df: Input DataFrame (pandas or polars)
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            spans: List of span values for EWM
            date_col: Optional date column for sorting
            
        Returns:
            DataFrame with added EWM features
        """
        logger.info(f"Generating GPU-accelerated EWM features with spans {spans}")
        
        # Determine if input is polars or pandas
        is_polars = 'pl' in str(type(df))
        
        if is_polars:
            import polars as pl
            # Convert to pandas for processing
            pandas_df = df.to_pandas()
        else:
            import pandas as pd
            pandas_df = df
            
        # Ensure group column is of appropriate type for groupby
        pandas_df[group_col] = pandas_df[group_col].astype('category')
        
        # Pre-sort by group and date if date column exists
        if date_col and date_col in pandas_df.columns:
            pandas_df = pandas_df.sort_values([group_col, date_col])
            
        # Get unique groups
        groups = pandas_df[group_col].unique()
        
        # Initialize dict to store results
        result_dict = {}
        
        # For each span, create features
        for span in spans:
            logger.info(f"Generating EWM features with span {span}")
            
            # Calculate alpha (same formula as pandas ewm uses)
            alpha = 2.0 / (span + 1.0)
            
            # Process each column
            for col in numeric_cols:
                # Define column name
                feature_name = f"{col}_ewm_{span}"
                
                # Initialize column with NaN
                result_dict[feature_name] = np.full(len(pandas_df), np.nan)
                
                # Process each group 
                for group in groups:
                    # Get indices for this group
                    mask = pandas_df[group_col] == group
                    indices = np.where(mask)[0]
                    
                    if len(indices) == 0:
                        continue
                        
                    # Get data for this group and column
                    data = pandas_df.loc[mask, col].values
                    
                    # Skip if no data or all NaN
                    if len(data) == 0 or np.all(np.isnan(data)):
                        continue
                        
                    # Move data to GPU if available
                    data_device = self._to_device(data)
                    
                    # Calculate EWM feature
                    if cupy_available:
                        # Using CuPy for GPU acceleration
                        # Replace NaN with 0 for calculation
                        data_no_nan = cp.nan_to_num(data_device, 0)
                        
                        # Create weights
                        weights = cp.power(1-alpha, cp.arange(len(data_no_nan)-1, -1, -1))
                        weights = weights / weights.sum()
                        
                        # Compute weighted average using FFT convolution (faster)
                        result = cp.convolve(data_no_nan, weights, mode='full')[:len(data_no_nan)]
                        
                    elif torch_available:
                        # Using PyTorch for GPU acceleration
                        # Replace NaN with 0 for calculation
                        data_no_nan = torch.nan_to_num(data_device.float(), 0)
                        
                        # Create weights
                        weights = torch.pow(1-alpha, torch.arange(len(data_no_nan)-1, -1, -1, device='cuda', dtype=torch.float32))
                        weights = weights / weights.sum()
                        
                        # Compute weighted average
                        result = torch.nn.functional.conv1d(
                            data_no_nan.view(1, 1, -1),
                            weights.view(1, 1, -1),
                            padding=len(weights)-1
                        ).view(-1)[:len(data_no_nan)].cpu().numpy()
                        
                    # TensorFlow code removed
                        
                    else:
                        # Fall back to numpy if no GPU libraries available
                        # This is vectorized numpy, still much faster than loops
                        data_no_nan = np.nan_to_num(data, 0)
                        weights = np.power(1-alpha, np.arange(len(data_no_nan)-1, -1, -1))
                        weights = weights / weights.sum()
                        result = np.convolve(data_no_nan, weights, mode='full')[:len(data_no_nan)]
                    
                    # Convert back to numpy if needed
                    if not isinstance(result, np.ndarray):
                        result = self._to_numpy(result)
                        
                    # Restore NaN values
                    result[np.isnan(data)] = np.nan
                        
                    # Ensure result length matches the group length
                    if len(result) != len(indices):
                        # Truncate or pad as needed
                        if len(result) > len(indices):
                            result = result[:len(indices)]
                        else:
                            result = np.pad(result, (0, len(indices) - len(result)), constant_values=np.nan)
                    
                    # Assign result to the column
                    result_dict[feature_name][indices] = result
        
        # Create a DataFrame from the result dictionary
        import pandas as pd
        features_df = pd.DataFrame(result_dict)
        
        # Combine with original DataFrame
        result_df = pd.concat([pandas_df, features_df], axis=1)
        
        # Convert back to polars if input was polars
        if is_polars:
            import polars as pl
            return pl.from_pandas(result_df)
        
        return result_df

    def generate_all_features(self, df, group_col, numeric_cols, rolling_windows=None, lag_periods=None, ewm_spans=None, date_col=None):
        """
        Generate all types of features using GPU acceleration.
        
        Args:
            df: Input DataFrame (pandas or polars)
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            rolling_windows: List of window sizes for rolling features
            lag_periods: List of lag periods
            ewm_spans: List of span values for EWM
            date_col: Optional date column for sorting
            
        Returns:
            DataFrame with all added features
        """
        # Use default values if not provided
        if rolling_windows is None:
            rolling_windows = [3, 7, 14, 28, 56]
        
        if lag_periods is None:
            lag_periods = [1, 2, 3, 5, 7, 14, 28]
            
        if ewm_spans is None:
            ewm_spans = [5, 10, 20, 40]
            
        logger.info(f"Generating all GPU-accelerated features")
        logger.info(f"Using {len(numeric_cols)} numeric columns with {len(rolling_windows)} windows, {len(lag_periods)} lags, and {len(ewm_spans)} EWM spans")
        
        # Generate rolling features
        start_time = time.time()
        result_df = self.generate_rolling_features(df, group_col, numeric_cols, rolling_windows, date_col)
        logger.info(f"Rolling features generated in {time.time() - start_time:.2f} seconds")
        
        # Generate lag features
        start_time = time.time()
        result_df = self.generate_lag_features(result_df, group_col, numeric_cols, lag_periods, date_col)
        logger.info(f"Lag features generated in {time.time() - start_time:.2f} seconds")
        
        # Generate EWM features
        start_time = time.time()
        result_df = self.generate_ewm_features(result_df, group_col, numeric_cols, ewm_spans, date_col)
        logger.info(f"EWM features generated in {time.time() - start_time:.2f} seconds")
        
        return result_df

if __name__ == "__main__":
    import time
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Generate features using GPU acceleration')
    parser.add_argument('--input-file', type=str, required=True, help='Input data file (CSV or Parquet)')
    parser.add_argument('--output-file', type=str, help='Output file (CSV or Parquet)')
    parser.add_argument('--group-col', type=str, default='symbol', help='Column to group by')
    parser.add_argument('--date-col', type=str, default='date', help='Date column for sorting')
    parser.add_argument('--limit-cols', type=int, default=20, help='Limit number of numeric columns to process')
    parser.add_argument('--max-features', type=int, default=10000, help='Maximum number of features to generate')
    parser.add_argument('--force-gpu', action='store_true', help='Force GPU detection and usage')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison with CPU methods')
    
    args = parser.parse_args()
    
    # Load data
    start_time = time.time()
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.parquet'):
        df = pd.read_parquet(args.input_file)
    else:
        raise ValueError("Input file must be CSV or Parquet")
        
    logger.info(f"Loaded data with shape {df.shape} in {time.time() - start_time:.2f} seconds")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != args.group_col and col != args.date_col and 'target' not in col and 'id' not in col]
    
    # Limit columns if specified
    if args.limit_cols > 0 and len(numeric_cols) > args.limit_cols:
        logger.info(f"Limiting to {args.limit_cols} numeric columns (from {len(numeric_cols)} total)")
        numeric_cols = numeric_cols[:args.limit_cols]
        
    # Initialize accelerator with force-gpu flag
    accelerator = GPUFeatureAccelerator(force_gpu=args.force_gpu)
    
    # Generate features
    start_time = time.time()
    result_df = accelerator.generate_all_features(
        df, 
        group_col=args.group_col, 
        numeric_cols=numeric_cols,
        date_col=args.date_col
    )
    
    total_time = time.time() - start_time
    logger.info(f"Generated {result_df.shape[1] - df.shape[1]} features in {total_time:.2f} seconds")
    
    # Save results if output file specified
    if args.output_file:
        if args.output_file.endswith('.csv'):
            result_df.to_csv(args.output_file, index=False)
        elif args.output_file.endswith('.parquet'):
            result_df.to_parquet(args.output_file, index=False)
            
        logger.info(f"Saved results to {args.output_file}")
        
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running benchmark comparison with CPU methods...")
        
        # Run CPU-based methods for comparison
        import pandas as pd
        
        # Benchmark rolling features
        logger.info("Benchmarking rolling features (CPU pandas vs GPU)...")
        start_time = time.time()
        
        # Use CPU pandas method
        cpu_result = df.copy()
        windows = [3, 7, 14]  # Use fewer windows for benchmark
        numeric_cols_subset = numeric_cols[:5]  # Use fewer columns for benchmark
        
        for col in numeric_cols_subset:
            for window in windows:
                cpu_result[f"{col}_roll_{window}_mean"] = df.groupby(args.group_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
        cpu_time = time.time() - start_time
        logger.info(f"CPU pandas rolling features: {cpu_time:.2f} seconds")
        
        # Use GPU method for same task
        start_time = time.time()
        gpu_result = accelerator.generate_rolling_features(
            df,
            group_col=args.group_col,
            numeric_cols=numeric_cols_subset,
            windows=windows,
            date_col=args.date_col
        )
        gpu_time = time.time() - start_time
        logger.info(f"GPU rolling features: {gpu_time:.2f} seconds")
        
        # Report speedup
        speedup = cpu_time / gpu_time
        logger.info(f"GPU speedup factor: {speedup:.2f}x")