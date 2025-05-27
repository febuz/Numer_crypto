#!/usr/bin/env python3
"""
Polars GPU Accelerator for Feature Generation

This module provides GPU-accelerated functions for feature generation using Polars,
significantly improving performance for large datasets.
"""

import os
import sys
import time
import logging
import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Handle environment variables directly instead of using dotenv
# This avoids the dependency on python-dotenv
def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
        return True
    return False

# Try to load environment variables
try:
    load_env_file()
    print("Loaded environment variables from .env file")
except Exception as e:
    print(f"Error loading environment variables: {e}")

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

# Force early detection with environment activation
# This helps with detecting CUDA when the script is run outside the virtual environment
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
            logger.warning("PyTorch is installed but CUDA is not available")
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
            logger.warning("Install PyTorch or CuPy to enable GPU acceleration")
        else:
            logger.warning("No NVIDIA GPU detected via nvidia-smi")
    except Exception as e:
        logger.warning(f"Error checking for NVIDIA GPU: {e}")
    
    return False

# Run detection
detect_gpu_libraries()

# Log GPU detection results
if cuda_available:
    logger.info("GPU acceleration is available and will be used")
else:
    logger.warning("No CUDA libraries available. Using CPU with NumPy instead.")
    logger.warning("To enable GPU acceleration, install one of the following packages:")
    logger.warning("- pytorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    logger.warning("- cupy: pip install cupy-cuda11x (replace 11x with your CUDA version)")


class PolarsGPUAccelerator:
    """
    Class for accelerating feature generation using GPUs with Polars.
    Uses vectorized operations and GPU acceleration where available.
    """
    
    def __init__(self, output_dir: str = "/media/knight2/EDB/numer_crypto_temp/data/features",
                 force_gpu: bool = True, use_all_gpus: bool = True):
        """
        Initialize the Polars GPU Feature Accelerator
        
        Args:
            output_dir: Directory to save generated feature files
            force_gpu: If True, will force detection of GPU even in subprocesses
            use_all_gpus: If True, will attempt to use all available GPUs
        """
        global cuda_available, torch_available, cupy_available
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_all_gpus = use_all_gpus
        self.available_gpus = []
        
        # Force GPU library detection again (helps with subprocess invocations)
        if force_gpu and not cuda_available:
            # Force new detection attempt
            detect_gpu_libraries()
        
        # Check if CUDA libraries are available
        if not cuda_available:
            logger.warning("No CUDA libraries available. Using CPU with NumPy instead.")
            logger.warning("To enable GPU acceleration, ensure PyTorch or CuPy are installed in the virtual environment")
            
            # Try to load PyTorch dynamically with environment activation
            try:
                import subprocess
                result = subprocess.run(
                    ["/bin/bash", "-c", f"source /media/knight2/EDB/numer_crypto_temp/environment/bin/activate && python3 -c \"import torch; print(torch.cuda.is_available())\""],
                    capture_output=True, text=True
                )
                if "True" in result.stdout:
                    logger.info("PyTorch with CUDA found in virtual environment! Attempting to import...")
                    try:
                        # Then try to import torch and check for CUDA
                        import torch
                        # Update globals after import
                        if torch.cuda.is_available():
                            torch_available = True
                            cuda_available = True
                            logger.info(f"Successfully imported PyTorch with GPU support from environment")
                        else:
                            logger.warning("PyTorch imported but CUDA not available in this context")
                    except Exception as e:
                        logger.warning(f"Error importing PyTorch: {e}")
            except Exception as e:
                logger.warning(f"Failed to dynamically load PyTorch: {e}")
            
            if not cuda_available:
                self.gpu_id = None
                self.xp = np
                return
        
        # Check environment variables first
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        
        # Check if we should use all available GPUs
        if self.use_all_gpus:
            # Get information about available GPUs
            self.available_gpus = get_available_gpus()
            
            if len(self.available_gpus) > 1:
                logger.info(f"Found {len(self.available_gpus)} GPUs, will try to use all of them")
                
                # If CUDA_VISIBLE_DEVICES is already set, check that it includes multiple GPUs
                if cuda_visible_devices and "," in cuda_visible_devices:
                    logger.info(f"Using GPUs specified in CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
                    # No need to modify environment variable
                else:
                    # Set environment variable to use all GPUs
                    gpu_ids = ",".join(str(gpu["index"]) for gpu in self.available_gpus)
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                    logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_ids} to use all GPUs")
                
                # Select first GPU as primary
                self.gpu_id = 0
            else:
                # Only one GPU or no GPUs, select best
                self.gpu_id = select_best_gpu()
        else:
            # Just select the best GPU
            self.gpu_id = select_best_gpu()
        
        # Set up GPU environment if available
        if self.gpu_id is not None:
            # If not using all GPUs, set CUDA_VISIBLE_DEVICES only if not already set
            if not self.use_all_gpus and not cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            
            # Optimize CUDA memory usage
            optimize_cuda_memory_usage(reserve_memory_fraction=0.1)
            logger.info(f"Primary GPU {self.gpu_id} selected for feature acceleration")
            
            # Initialize GPU memory for the library we're using
            if torch_available:  # Prefer PyTorch over CuPy as it's more reliable
                # Force re-import to ensure we have the environment context
                import torch
                self.xp = torch
                try:
                    # Get device count
                    device_count = torch.cuda.device_count()
                    logger.info(f"PyTorch sees {device_count} CUDA device(s)")
                    
                    # Get info for all devices
                    for i in range(device_count):
                        device_name = torch.cuda.get_device_name(i)
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        free_memory = total_memory - torch.cuda.memory_allocated(i)
                        logger.info(f"GPU {i}: {device_name}, {free_memory/(1024**3):.2f} GB free / {total_memory/(1024**3):.2f} GB total")
                    
                    # Test a small GPU operation to verify it works
                    test_tensor = torch.ones((10, 10), device='cuda')
                    test_result = test_tensor + test_tensor
                    logger.info(f"Verified PyTorch CUDA works with test operation")
                except Exception as e:
                    logger.warning(f"Could not initialize PyTorch GPU: {e}")
                    # Fall back to CPU
                    self.xp = np
                    self.gpu_id = None
            elif cupy_available:
                # Import CuPy in this context to ensure it's loaded in the environment
                import cupy as cp
                self.xp = cp
                # Get GPU memory info
                try:
                    # Get number of available GPUs
                    device_count = cp.cuda.runtime.getDeviceCount()
                    logger.info(f"CuPy sees {device_count} CUDA device(s)")
                    
                    # Get info for each device
                    for i in range(device_count):
                        cp.cuda.Device(i).use()
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free_bytes = mem_info[0]
                        total_bytes = mem_info[1]
                        device_name = cp.cuda.runtime.getDeviceProperties(i)["name"].decode("utf-8")
                        logger.info(f"GPU {i}: {device_name}, {free_bytes/(1024**3):.2f} GB free / {total_bytes/(1024**3):.2f} GB total")
                except Exception as e:
                    logger.warning(f"Could not get GPU memory info: {e}")
                    # Fall back to CPU
                    self.xp = np
                    self.gpu_id = None
            else:
                self.xp = np
                self.gpu_id = None
                logger.warning("No GPU acceleration libraries available, falling back to NumPy")
        else:
            self.xp = np
            logger.warning("No GPU available, using CPU with NumPy")
            logger.warning("Make sure you have NVIDIA GPU drivers installed and CUDA setup correctly.")
    
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
    
    def distribute_work_to_gpus(self, data_groups, process_fn, **kwargs):
        """
        Distribute work across available GPUs.
        
        Args:
            data_groups: List of data groups to process
            process_fn: Function to process each group
            **kwargs: Additional arguments to pass to process_fn
            
        Returns:
            List of results from each GPU
        """
        # Import here to avoid circular import
        import torch
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Set the multiprocessing start method to 'spawn' for CUDA compatibility
        # This is crucial for CUDA to work in subprocesses
        try:
            mp.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError:
            # Method already set, which is fine
            logger.info("Multiprocessing start method already set (likely to 'spawn')")
        
        # Determine the original mapping of CUDA_VISIBLE_DEVICES
        original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cuda_device_map = {}
        
        if original_cuda_devices:
            # Parse the CUDA_VISIBLE_DEVICES string to get the actual device mapping
            try:
                device_ids = original_cuda_devices.split(",")
                for i, device_id in enumerate(device_ids):
                    cuda_device_map[i] = int(device_id.strip())
                logger.info(f"CUDA device mapping: {cuda_device_map}")
            except Exception as e:
                logger.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
                cuda_device_map = {}
        
        # Check if we have multiple GPUs and should distribute
        device_count = torch.cuda.device_count() if torch_available else 0
        if device_count <= 1 or not self.use_all_gpus:
            logger.info(f"Not distributing work across GPUs (device count: {device_count}, use_all_gpus: {self.use_all_gpus})")
            # Process all groups in the current process
            return [process_fn(group, **kwargs) for group in data_groups]
        
        # Log information about work distribution
        logger.info(f"Distributing work across {device_count} GPUs")
        logger.info(f"Total groups to process: {len(data_groups)}")
        
        # Create chunks of work for each GPU
        chunk_size = len(data_groups) // device_count
        remainder = len(data_groups) % device_count
        
        chunks = []
        start_idx = 0
        for i in range(device_count):
            # Add one more group to some chunks to distribute remainder
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            # Get the actual physical GPU ID
            physical_gpu_id = cuda_device_map.get(i, i)
            
            chunks.append({
                'gpu_id': physical_gpu_id,
                'groups': data_groups[start_idx:end_idx]
            })
            
            start_idx = end_idx
        
        # Log chunk sizes
        for chunk in chunks:
            logger.info(f"GPU {chunk['gpu_id']}: {len(chunk['groups'])} groups")
        
        # Get function name based on the process_fn
        process_fn_name = process_fn.__name__ if hasattr(process_fn, "__name__") else "unknown"
        
        # Process chunks in parallel using ProcessPoolExecutor with 'spawn' method
        # This ensures each process has its own memory space and CUDA context
        all_results = []
        
        logger.info(f"Original CUDA_VISIBLE_DEVICES: {original_cuda_devices}")
        
        # Submit all workers in parallel to maximize throughput
        logger.info("Starting worker processes in parallel for maximum throughput")
        
        # Use the executor context manager for proper cleanup
        # Limit max_workers to device_count
        with ProcessPoolExecutor(max_workers=device_count,
                                 mp_context=mp.get_context('spawn')) as executor:
            # Submit all workers at once with dedicated environment variables
            futures = []
            
            # First create all worker environments - this is important to do before any submissions
            worker_environments = []
            for i, chunk in enumerate(chunks):
                worker_env = os.environ.copy()
                
                # Set each worker to see ONLY its assigned GPU
                # Each worker gets its own completely isolated CUDA environment
                worker_env["CUDA_VISIBLE_DEVICES"] = str(chunk['gpu_id'])
                
                # Additional isolation to prevent worker contention
                worker_env["OMP_NUM_THREADS"] = "1"  # Limit CPU threads per worker
                worker_env["CUDA_CACHE_DISABLE"] = "1"  # Disable CUDA cache for clean isolation
                worker_env["CUDA_CACHE_PATH"] = f"/tmp/cuda-cache-{chunk['gpu_id']}"  # Separate cache paths
                worker_env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Prevent memory fragmentation
                
                # Use PCI_BUS_ID for consistent device ordering
                worker_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                
                logger.info(f"Prepared environment for GPU {chunk['gpu_id']} worker with {len(chunk['groups'])} groups")
                worker_environments.append((chunk, worker_env))
            
            # Submit all workers at once in a batch to ensure they start simultaneously
            logger.info(f"Submitting all {len(worker_environments)} workers simultaneously")
            
            # Use a pre-submission delay to ensure clean startup
            import time
            time.sleep(0.5)  # Small delay before starting all workers
            
            # Now submit all workers at once
            for chunk, worker_env in worker_environments:
                logger.info(f"Submitting worker for physical GPU {chunk['gpu_id']} with CUDA_VISIBLE_DEVICES={worker_env['CUDA_VISIBLE_DEVICES']}")
                
                # Submit without waiting - critical for parallel execution
                future = executor.submit(
                    _gpu_worker_function, 
                    chunk, 
                    process_fn_name, 
                    kwargs,
                    worker_env  # Pass custom environment to the worker
                )
                
                futures.append((future, chunk))
            
            # Add a slight delay between submissions to avoid contention
            time.sleep(0.2)
            
            # Collect results as they complete
            for future, chunk in futures:
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    logger.info(f"GPU {chunk['gpu_id']} completed processing {len(chunk['groups'])} groups")
                except Exception as e:
                    logger.error(f"GPU {chunk['gpu_id']} failed with error: {e}")
                    logger.error(f"This is likely due to CUDA initialization issues in subprocesses")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue with other chunks, don't fail completely
        
        return all_results
            
    def generate_rolling_features(self, df: pl.DataFrame, group_col: str, 
                                  numeric_cols: List[str], windows: List[int],
                                  date_col: Optional[str] = None) -> pl.DataFrame:
        """
        Generate rolling window features using GPU acceleration with Polars.
        
        Args:
            df: Input Polars DataFrame
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            windows: List of window sizes
            date_col: Optional date column for sorting
            
        Returns:
            Polars DataFrame with added rolling features
        """
        logger.info(f"Generating GPU-accelerated rolling features with windows {windows}")
        
        # Make a copy to avoid modifying the input DataFrame
        result_df = df.clone()
        
        # Sort by group and date if date column exists
        if date_col and date_col in df.columns:
            sorted_df = result_df.sort([group_col, date_col])
        else:
            sorted_df = result_df
        
        # Get unique groups for parallel processing
        groups = sorted_df[group_col].unique().to_list()
        logger.info(f"Distributing rolling feature generation across {len(groups)} groups")
        
        # Distribute work across GPUs
        results = self.distribute_work_to_gpus(groups, process_rolling_group, 
                   df=sorted_df, 
                   numeric_cols=numeric_cols, 
                   windows=windows, 
                   date_col=date_col, 
                   group_col=group_col)
        
        # Collect results from all groups and merge them into DataFrame
        # We'll create a DataFrame for each feature and join them
        feature_dfs = []
        processed_features = set()
        
        for result in results:
            if not result or "features" not in result:
                continue
                
            features = result["features"]
            group = result["group"]
            
            # Skip empty features
            if len(features) <= 1:  # Only the group column
                continue
                
            # Convert each feature to DataFrame and collect
            for feature_name, values in features.items():
                if feature_name == group_col:
                    continue
                    
                if feature_name not in processed_features:
                    processed_features.add(feature_name)
                    try:
                        # Create feature DataFrame
                        if len(values) == 1:
                            # Single value for whole group
                            feature_df = pl.DataFrame({
                                group_col: [group],
                                feature_name: values
                            })
                        else:
                            # Value per row
                            group_mask = sorted_df[group_col] == group
                            group_indices = sorted_df.with_row_count().filter(group_mask)["row_nr"]
                            
                            # Add to data structure keyed by feature
                            feature_data = {
                                group_col: [group] * len(values),
                                feature_name: values,
                                "row_index": group_indices
                            }
                            feature_df = pl.DataFrame(feature_data)
                        
                        feature_dfs.append(feature_df)
                    except Exception as e:
                        logger.error(f"Error creating DataFrame for feature {feature_name}: {e}")
        
        # If no features were generated, return the original DataFrame
        if not feature_dfs:
            return result_df
            
        # Merge all feature DataFrames
        for feature_df in feature_dfs:
            try:
                # Check if feature DataFrame has row_index column
                if "row_index" in feature_df.columns:
                    # Join by row index for a precise match
                    result_df = result_df.with_row_count().join(
                        feature_df, 
                        left_on=["row_nr", group_col], 
                        right_on=["row_index", group_col],
                        how="left"
                    ).drop(["row_nr", "row_index"])
                else:
                    # Join by group column (will duplicate values)
                    result_df = result_df.join(
                        feature_df,
                        on=group_col,
                        how="left"
                    )
            except Exception as e:
                logger.error(f"Error joining feature DataFrame: {e}")
                
        return result_df
        
    def generate_lag_features(self, df: pl.DataFrame, group_col: str, 
                             numeric_cols: List[str], lag_periods: List[int], 
                             date_col: Optional[str] = None) -> pl.DataFrame:
        """
        Generate lag features with Polars using multiple GPUs.
        
        Args:
            df: Input Polars DataFrame
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            lag_periods: List of lag periods
            date_col: Optional date column for sorting
            
        Returns:
            Polars DataFrame with added lag features
        """
        logger.info(f"Generating lag features with periods {lag_periods}")
        
        # Make a copy to avoid modifying the input DataFrame
        result_df = df.clone()
        
        # Sort by group and date if date column exists
        if date_col and date_col in df.columns:
            sorted_df = result_df.sort([group_col, date_col])
        else:
            sorted_df = result_df
        
        # Get unique groups for parallel processing
        groups = sorted_df[group_col].unique().to_list()
        logger.info(f"Distributing lag feature generation across {len(groups)} groups")
        
        # Distribute work across GPUs
        results = self.distribute_work_to_gpus(groups, process_lag_group, 
                   df=sorted_df, 
                   numeric_cols=numeric_cols, 
                   lag_periods=lag_periods, 
                   date_col=date_col, 
                   group_col=group_col)
        
        # Collect results from all groups and merge them into DataFrame
        # We'll create a DataFrame for each feature and join them
        feature_dfs = []
        processed_features = set()
        
        for result in results:
            if not result or "features" not in result:
                continue
                
            features = result["features"]
            group = result["group"]
            
            # Skip empty features
            if len(features) <= 1:  # Only the group column
                continue
                
            # Convert each feature to DataFrame and collect
            for feature_name, values in features.items():
                if feature_name == group_col:
                    continue
                    
                if feature_name not in processed_features:
                    processed_features.add(feature_name)
                    try:
                        # Create feature DataFrame
                        if len(values) == 1:
                            # Single value for whole group
                            feature_df = pl.DataFrame({
                                group_col: [group],
                                feature_name: values
                            })
                        else:
                            # Value per row
                            group_mask = sorted_df[group_col] == group
                            group_indices = sorted_df.with_row_count().filter(group_mask)["row_nr"]
                            
                            # Add to data structure keyed by feature
                            feature_data = {
                                group_col: [group] * len(values),
                                feature_name: values,
                                "row_index": group_indices
                            }
                            feature_df = pl.DataFrame(feature_data)
                        
                        feature_dfs.append(feature_df)
                    except Exception as e:
                        logger.error(f"Error creating DataFrame for feature {feature_name}: {e}")
        
        # If no features were generated, return the original DataFrame
        if not feature_dfs:
            return result_df
            
        # Merge all feature DataFrames
        for feature_df in feature_dfs:
            try:
                # Check if feature DataFrame has row_index column
                if "row_index" in feature_df.columns:
                    # Join by row index for a precise match
                    result_df = result_df.with_row_count().join(
                        feature_df, 
                        left_on=["row_nr", group_col], 
                        right_on=["row_index", group_col],
                        how="left"
                    ).drop(["row_nr", "row_index"])
                else:
                    # Join by group column (will duplicate values)
                    result_df = result_df.join(
                        feature_df,
                        on=group_col,
                        how="left"
                    )
            except Exception as e:
                logger.error(f"Error joining feature DataFrame: {e}")
                
        return result_df
    
    def generate_ewm_features(self, df: pl.DataFrame, group_col: str,
                             numeric_cols: List[str], spans: List[int],
                             date_col: Optional[str] = None) -> pl.DataFrame:
        """
        Generate exponentially weighted moving average features using multiple GPUs.
        
        Args:
            df: Input Polars DataFrame
            group_col: Column name for grouping (e.g., 'symbol')
            numeric_cols: List of numeric columns to generate features for
            spans: List of span values for EWM
            date_col: Optional date column for sorting
            
        Returns:
            Polars DataFrame with added EWM features
        """
        logger.info(f"Generating GPU-accelerated EWM features with spans {spans}")
        
        # Make a copy to avoid modifying the input DataFrame
        result_df = df.clone()
        
        # Sort by group and date if date column exists
        if date_col and date_col in df.columns:
            sorted_df = result_df.sort([group_col, date_col])
        else:
            sorted_df = result_df
            logger.warning("No date column provided for EWM features. Results may not be accurate.")
        
        # Get unique groups for parallel processing
        groups = sorted_df[group_col].unique().to_list()
        logger.info(f"Distributing EWM feature generation across {len(groups)} groups")
        
        # Distribute work across GPUs
        results = self.distribute_work_to_gpus(groups, process_ewm_group, 
                   df=sorted_df, 
                   numeric_cols=numeric_cols, 
                   spans=spans, 
                   date_col=date_col, 
                   group_col=group_col)
        
        # Collect results from all groups and merge them into DataFrame
        # We'll create a DataFrame for each feature and join them
        feature_dfs = []
        processed_features = set()
        
        for result in results:
            if not result or "features" not in result:
                continue
                
            features = result["features"]
            group = result["group"]
            
            # Skip empty features
            if len(features) <= 1:  # Only the group column
                continue
                
            # Convert each feature to DataFrame and collect
            for feature_name, values in features.items():
                if feature_name == group_col:
                    continue
                    
                if feature_name not in processed_features:
                    processed_features.add(feature_name)
                    try:
                        # Create feature DataFrame
                        if len(values) == 1:
                            # Single value for whole group
                            feature_df = pl.DataFrame({
                                group_col: [group],
                                feature_name: values
                            })
                        else:
                            # Value per row
                            group_mask = sorted_df[group_col] == group
                            group_indices = sorted_df.with_row_count().filter(group_mask)["row_nr"]
                            
                            # Add to data structure keyed by feature
                            feature_data = {
                                group_col: [group] * len(values),
                                feature_name: values,
                                "row_index": group_indices
                            }
                            feature_df = pl.DataFrame(feature_data)
                        
                        feature_dfs.append(feature_df)
                    except Exception as e:
                        logger.error(f"Error creating DataFrame for feature {feature_name}: {e}")
        
        # If no features were generated, return the original DataFrame
        if not feature_dfs:
            return result_df
            
        # Merge all feature DataFrames
        for feature_df in feature_dfs:
            try:
                # Check if feature DataFrame has row_index column
                if "row_index" in feature_df.columns:
                    # Join by row index for a precise match
                    result_df = result_df.with_row_count().join(
                        feature_df, 
                        left_on=["row_nr", group_col], 
                        right_on=["row_index", group_col],
                        how="left"
                    ).drop(["row_nr", "row_index"])
                else:
                    # Join by group column (will duplicate values)
                    result_df = result_df.join(
                        feature_df,
                        on=group_col,
                        how="left"
                    )
            except Exception as e:
                logger.error(f"Error joining feature DataFrame: {e}")
                
        return result_df
    
    def generate_all_features(self, df, group_col, numeric_cols, rolling_windows=None, 
                             lag_periods=None, ewm_spans=None, date_col=None):
        """
        Generate all types of features using GPU acceleration with Polars.
        Distributes work across all available GPUs for maximum performance.
        
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
        
        # Log GPU information
        logger.info("=" * 50)
        logger.info("GPU ACCELERATION STATUS")
        logger.info("=" * 50)
        
        if cuda_available:
            if torch_available:
                import torch
                device_count = torch.cuda.device_count()
                logger.info(f"PyTorch detected {device_count} CUDA devices")
                
                # Log information about each GPU
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    free_memory = total_memory - torch.cuda.memory_allocated(i)
                    logger.info(f"GPU {i}: {device_name}, {free_memory/(1024**3):.2f} GB free / {total_memory/(1024**3):.2f} GB total")
                
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"PyTorch Version: {torch.__version__}")
                logger.info(f"Using multi-GPU distribution: {self.use_all_gpus}")
            elif cupy_available:
                import cupy as cp
                device_count = cp.cuda.runtime.getDeviceCount()
                logger.info(f"CuPy detected {device_count} CUDA devices")
                logger.info(f"CuPy Version: {cp.__version__}")
                logger.info(f"Using multi-GPU distribution: {self.use_all_gpus}")
            else:
                logger.info("CUDA is available but no GPU acceleration libraries (PyTorch/CuPy) detected")
        else:
            logger.warning("No GPU acceleration available. Using CPU with NumPy instead.")
        
        logger.info("=" * 50)
        logger.info(f"Generating all GPU-accelerated features with Polars")
        logger.info(f"Using {len(numeric_cols)} numeric columns with {len(rolling_windows)} windows, {len(lag_periods)} lags, and {len(ewm_spans)} EWM spans")
        
        # Record overall start time
        total_start_time = time.time()
        
        # Convert to Polars if input is pandas
        is_pandas = 'pandas' in str(type(df))
        if is_pandas:
            # Convert pandas to polars
            import polars as pl
            df = pl.from_pandas(df)
        
        # Make a copy to avoid modifying the input
        result_df = df.clone()
        
        # ====== Generate rolling features ======
        start_time = time.time()
        logger.info("Starting rolling feature generation across all GPUs...")
        result_df = self.generate_rolling_features(result_df, group_col, numeric_cols, rolling_windows, date_col)
        rolling_time = time.time() - start_time
        logger.info(f"Rolling features generated in {rolling_time:.2f} seconds")
        
        # Clear GPU memory between operations
        if cuda_available and torch_available:
            import torch
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after rolling features")
        
        # ====== Generate lag features ======
        start_time = time.time()
        logger.info("Starting lag feature generation across all GPUs...")
        result_df = self.generate_lag_features(result_df, group_col, numeric_cols, lag_periods, date_col)
        lag_time = time.time() - start_time
        logger.info(f"Lag features generated in {lag_time:.2f} seconds")
        
        # Clear GPU memory between operations
        if cuda_available and torch_available:
            import torch
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after lag features")
        
        # ====== Generate EWM features ======
        start_time = time.time()
        logger.info("Starting EWM feature generation across all GPUs...")
        result_df = self.generate_ewm_features(result_df, group_col, numeric_cols, ewm_spans, date_col)
        ewm_time = time.time() - start_time
        logger.info(f"EWM features generated in {ewm_time:.2f} seconds")
        
        # Calculate total time and log feature counts
        total_time = time.time() - total_start_time
        new_features_count = result_df.width - df.width
        
        logger.info("=" * 50)
        logger.info(f"FEATURE GENERATION COMPLETE")
        logger.info(f"Total features generated: {new_features_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"  - Rolling features: {rolling_time:.2f} seconds")
        logger.info(f"  - Lag features: {lag_time:.2f} seconds")
        logger.info(f"  - EWM features: {ewm_time:.2f} seconds")
        if cuda_available:
            logger.info(f"GPU acceleration active: {self.gpu_id is not None}")
            if self.use_all_gpus and torch_available:
                import torch
                logger.info(f"Multi-GPU processing: {torch.cuda.device_count()} devices")
        logger.info("=" * 50)
        
        # Convert back to pandas if input was pandas
        if is_pandas:
            result_df = result_df.to_pandas()
        
        return result_df


# Define worker function at module level for pickling compatibility
def _gpu_worker_function(chunk, process_fn_name, kwargs_dict, worker_env=None):
    """
    Worker function to process a chunk of data on a specific GPU.
    Defined at module level to allow pickling.
    
    Args:
        chunk: Data chunk to process
        process_fn_name: Name of the process function to call
        kwargs_dict: Additional arguments to pass to process_fn
        worker_env: Custom environment variables for this worker
        
    Returns:
        List of results
    """
    try:
        import os
        import logging
        import time
        
        logger = logging.getLogger(__name__)
        
        # Set custom environment variables if provided
        if worker_env is not None:
            for key, value in worker_env.items():
                os.environ[key] = value
        
        # Get the physical GPU ID from the chunk
        physical_gpu_id = chunk['gpu_id']
        
        # Set environment to use only this GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        logger.info(f"Worker GPU {physical_gpu_id} processing {len(chunk['groups'])} groups")
        
        # Import torch after setting environment variables
        import torch
        import numpy as np
        
        # Initialize GPU with detailed error reporting and enhanced operations
        try:
            # Get CUDA environment info for debugging
            logger.info(f"Worker {physical_gpu_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            
            # Check for CUDA availability
            if not torch.cuda.is_available():
                logger.error(f"Worker {physical_gpu_id} couldn't access CUDA! torch.cuda.is_available() returned False")
                # Try to diagnose the issue
                logger.error(f"PyTorch version: {torch.__version__}")
                raise RuntimeError(f"Worker {physical_gpu_id} couldn't access CUDA")
            
            # Check GPU devices and properties
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.error(f"Worker {physical_gpu_id} shows CUDA available but reports 0 devices!")
                raise RuntimeError(f"Worker {physical_gpu_id} reports 0 CUDA devices")
                
            logger.info(f"Worker {physical_gpu_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            logger.info(f"Worker {physical_gpu_id} sees {device_count} CUDA device(s)")
            
            # With CUDA_VISIBLE_DEVICES correctly set, the worker should always see the physical GPU as device 0
            device = torch.device("cuda:0")
            
            # Get device properties for detailed info
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            logger.info(f"Worker for GPU {physical_gpu_id} using device: {device_name} with {total_memory/(1024**3):.2f} GB total memory")
                
            # Force PyTorch to use this specific GPU - it should be device 0 in this worker's context
            torch.cuda.set_device(0)
            
            # Shared synchronization - wait for a moment to let all workers initialize in parallel
            logger.info(f"Worker for GPU {physical_gpu_id} initialized, waiting for coordination...")
            time.sleep(0.1)  # Short pause to allow all workers to reach this point
            
            # Progressive initialization for all GPUs
            logger.info(f"Starting progressive initialization for GPU {physical_gpu_id}...")
            
            # Simple GPU initialization - just verify it works
            logger.info(f"Initializing GPU {physical_gpu_id} with minimal warm-up...")
            
            # Quick warm-up to ensure GPU is ready
            try:
                # Single small test operation
                test_tensor = torch.rand(100, 100, device=device)
                test_result = torch.matmul(test_tensor, test_tensor)
                torch.cuda.synchronize()
                logger.info(f"GPU {physical_gpu_id} initialization test passed")
                
                # Clear the test tensors
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"GPU {physical_gpu_id} initialization test failed: {e}")
                raise
            
            # Report GPU memory usage
            free_memory = total_memory - torch.cuda.memory_allocated(0)
            logger.info(f"GPU {physical_gpu_id} initialized successfully with {free_memory/(1024**3):.2f} GB free memory")
            
            # Keep a small tensor allocated to maintain GPU activity
            persistent_tensor = torch.rand(50, 50, device=device)
            logger.info(f"GPU {physical_gpu_id} ready for processing")
            
            # Final coordination point - all GPUs should be ready now
            logger.info(f"GPU {physical_gpu_id} initialization complete and ready for processing")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU {physical_gpu_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with CPU fallback instead of completely failing
            logger.warning(f"Will attempt to continue without GPU acceleration for worker {physical_gpu_id}")
        
        # Get the process function by name
        if process_fn_name == "process_rolling_group":
            from features.polars_gpu_accelerator import process_rolling_group as process_fn
        elif process_fn_name == "process_lag_group":
            from features.polars_gpu_accelerator import process_lag_group as process_fn
        elif process_fn_name == "process_ewm_group":
            from features.polars_gpu_accelerator import process_ewm_group as process_fn
        else:
            # Default simple processing function
            def process_fn(group, **kwargs):
                return {"group": group, "features": {}}
        
        # Process all groups assigned to this GPU
        results = []
        
        # Set up reasonable batch sizes for progress logging
        batch_size = 10  # Report progress every 10 groups
        
        # Create small tensors for occasional GPU activity monitoring
        activity_tensor = torch.rand(200, 200, device=device)
        
        # Announce processing start with device information
        logger.info(f"GPU {physical_gpu_id} starting to process {len(chunk['groups'])} groups with batch_size={batch_size}")
        
        # Initialize progress tracking
        last_progress_time = time.time()
        progress_interval = 300  # 5 minutes in seconds
        last_reported_progress = 0
        
        # Process groups in balanced chunks
        for i, group in enumerate(chunk['groups']):
            current_time = time.time()
            current_progress = (i / len(chunk['groups'])) * 100
            
            # 5-minute progress updates
            if (current_time - last_progress_time >= progress_interval or 
                current_progress - last_reported_progress >= 10):  # Also update every 10%
                
                logger.info(f"GPU {physical_gpu_id} INTERMEDIATE PROGRESS: {i}/{len(chunk['groups'])} groups processed ({current_progress:.1f}%)")
                logger.info(f"GPU {physical_gpu_id} processing rate: {i / (current_time - start_time):.2f} groups/second")
                
                # Memory usage info
                allocated_memory = torch.cuda.memory_allocated(0)
                cached_memory = torch.cuda.memory_reserved(0)
                logger.info(f"GPU {physical_gpu_id} memory: {allocated_memory/(1024**3):.2f} GB allocated, {cached_memory/(1024**3):.2f} GB cached")
                
                last_progress_time = current_time
                last_reported_progress = current_progress
            
            # Log progress periodically
            if i % batch_size == 0:
                logger.info(f"GPU {physical_gpu_id} progress: {i}/{len(chunk['groups'])} groups processed ({current_progress:.1f}%)")
                
                # Simple GPU operation to maintain activity visibility
                if torch.cuda.is_available() and i % (batch_size * 2) == 0:
                    # Lightweight operation just to show GPU is active
                    activity_result = torch.matmul(activity_tensor, activity_tensor)
                    torch.cuda.synchronize()
                    allocated_memory = torch.cuda.memory_allocated(0)
                    logger.info(f"GPU {physical_gpu_id} memory: {allocated_memory/(1024**3):.2f} GB")
            
            # Process the current group
            start_time = time.time()
            try:
                logger.info(f"Processing {process_fn_name} for group {group} using GPU {physical_gpu_id}")
                result = process_fn(group, **kwargs_dict)
            except Exception as e:
                logger.error(f"Error processing group {group} on GPU {physical_gpu_id}: {e}")
                # Create an empty result with just the group to avoid breaking the pipeline
                result = {"group": group, "features": {kwargs_dict.get('group_col', 'group'): [group]}}
            
            elapsed = time.time() - start_time
            
            # Log detailed processing time at the following intervals:
            # 1. First few groups
            # 2. Periodically throughout
            # 3. Last few groups
            if (i < 3) or (i % (batch_size * 5) == 0) or (i >= len(chunk['groups']) - 3):
                logger.info(f"GPU {physical_gpu_id} processed group {i} in {elapsed:.4f} seconds")
            
            # Add to results
            results.append(result)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_memory = torch.cuda.memory_allocated(0)
            logger.info(f"GPU {physical_gpu_id} final memory used: {allocated_memory/(1024**3):.2f} GB")
        
        logger.info(f"GPU {physical_gpu_id} completed all {len(chunk['groups'])} groups")
        return results
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error in GPU worker {chunk['gpu_id']}: {e}")
        logger.error(traceback.format_exc())
        return []

# Define processing functions at module level for pickling
def process_rolling_group(group, df=None, numeric_cols=None, windows=None, date_col=None, group_col=None):
    """Process a group for rolling features. Defined at module level for pickling."""
    try:
        import polars as pl
        import numpy as np
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        
        if use_gpu:
            logger.info(f"Processing rolling features for group {group} using GPU")
        
        group_result = {}
        # Get data for this group
        if df is not None and group_col is not None:
            group_mask = df[group_col] == group
            group_data = df.filter(group_mask)
        else:
            # Return empty result if no dataframe
            return {"group": group, "features": {group_col: [group]}}
            
        # Initialize group result with the group column
        group_result[group_col] = [group]
        
        # Process each window, function, and column
        for window in windows:
            for func_name in ['mean', 'std', 'max', 'min']:
                for col in numeric_cols:
                    # Create feature name
                    feature_name = f"{col}_roll_{window}_{func_name}"
                    
                    # Calculate the rolling value
                    try:
                        if use_gpu:
                            # Get column data
                            col_data = group_data[col].to_numpy()
                            
                            # Convert to PyTorch tensor on GPU with mixed precision support
                            tensor_data = torch.tensor(col_data, dtype=torch.float16, device=device)  # Use half precision
                            
                            # Create result tensor with same shape
                            result = torch.zeros_like(tensor_data)
                            
                            # Calculate rolling window function using GPU with mixed precision
                            with torch.cuda.amp.autocast():
                                if func_name == 'mean':
                                    # Implement rolling mean with optimized computation
                                    for i in range(len(tensor_data)):
                                        if i < window:
                                            # For initial points with insufficient history
                                            start_idx = 0
                                            valid_points = i + 1
                                        else:
                                            start_idx = i - window + 1
                                            valid_points = window
                                            
                                        # Calculate mean of window
                                        window_sum = torch.sum(tensor_data[start_idx:i+1])
                                        result[i] = window_sum / valid_points
                                    
                            elif func_name == 'std':
                                # Implement rolling std with GPU
                                for i in range(len(tensor_data)):
                                    if i < window:
                                        start_idx = 0
                                        valid_points = i + 1
                                    else:
                                        start_idx = i - window + 1
                                        valid_points = window
                                        
                                    # Calculate std of window
                                    if valid_points > 1:
                                        window_data = tensor_data[start_idx:i+1]
                                        window_mean = torch.mean(window_data)
                                        window_var = torch.sum((window_data - window_mean)**2) / (valid_points - 1)
                                        result[i] = torch.sqrt(window_var)
                                    else:
                                        result[i] = torch.tensor(0.0, device=device)
                                    
                            elif func_name == 'max':
                                # Implement rolling max with GPU
                                for i in range(len(tensor_data)):
                                    if i < window:
                                        window_data = tensor_data[0:i+1]
                                    else:
                                        window_data = tensor_data[i-window+1:i+1]
                                        
                                    result[i] = torch.max(window_data)
                                    
                            elif func_name == 'min':
                                # Implement rolling min with GPU
                                for i in range(len(tensor_data)):
                                    if i < window:
                                        window_data = tensor_data[0:i+1]
                                    else:
                                        window_data = tensor_data[i-window+1:i+1]
                                        
                                    result[i] = torch.min(window_data)
                            
                            # Move result back to CPU and convert to numpy
                            feature_values = result.cpu().numpy()
                            
                            # Log GPU memory usage occasionally
                            if window == windows[0] and func_name == 'mean' and col == numeric_cols[0]:
                                mem_allocated = torch.cuda.memory_allocated(0)
                                logger.debug(f"GPU memory used for group {group}: {mem_allocated/(1024**3):.2f} GB")
                        else:
                            # Use Polars expressions if GPU not available
                            if func_name == 'mean':
                                expr = pl.col(col).rolling_mean(window_size=window, min_samples=1)
                            elif func_name == 'std':
                                expr = pl.col(col).rolling_std(window_size=window, min_samples=1)
                            elif func_name == 'max':
                                expr = pl.col(col).rolling_max(window_size=window, min_samples=1)
                            elif func_name == 'min':
                                expr = pl.col(col).rolling_min(window_size=window, min_samples=1)
                            
                            # Apply the expression to get feature values
                            feature_values = group_data.select(expr).to_numpy().flatten()
                        
                        # Add to group result
                        group_result[feature_name] = feature_values
                    except Exception as e:
                        logger.error(f"Error calculating {feature_name} for group {group}: {e}")
                        # Add NaN values as placeholder
                        group_result[feature_name] = [float('nan')] * len(group_data)
        
        # Free GPU memory when done with this group
        if use_gpu:
            torch.cuda.empty_cache()
            
        return {"group": group, "features": group_result}
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing group {group}: {e}")
        logger.error(traceback.format_exc())
        return {"group": group, "features": {group_col: [group]}}

def process_lag_group(group, df=None, numeric_cols=None, lag_periods=None, date_col=None, group_col=None):
    """Process a group for lag features. Defined at module level for pickling."""
    try:
        import polars as pl
        import numpy as np
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        
        if use_gpu:
            logger.info(f"Processing lag features for group {group} using GPU")
        
        group_result = {}
        # Get data for this group
        if df is not None and group_col is not None:
            group_mask = df[group_col] == group
            group_data = df.filter(group_mask)
        else:
            # Return empty result if no dataframe
            return {"group": group, "features": {group_col: [group]}}
            
        # Initialize group result with the group column
        group_result[group_col] = [group]
        
        # Process each lag and column
        for lag in lag_periods:
            for col in numeric_cols:
                # Create feature name
                feature_name = f"{col}_lag_{lag}"
                
                try:
                    if use_gpu:
                        # Get column data
                        col_data = group_data[col].to_numpy()
                        
                        # Convert to PyTorch tensor on GPU
                        tensor_data = torch.tensor(col_data, dtype=torch.float32, device=device)
                        
                        # Create result tensor with same shape
                        result = torch.zeros_like(tensor_data)
                        
                        # Calculate lag using GPU tensor operations
                        data_len = len(tensor_data)
                        for i in range(data_len):
                            if i < lag:  # Not enough history
                                result[i] = float('nan')
                            else:
                                result[i] = tensor_data[i - lag]
                        
                        # Log memory usage occasionally
                        if lag == lag_periods[0] and col == numeric_cols[0]:
                            mem_allocated = torch.cuda.memory_allocated(0)
                            logger.debug(f"GPU memory used for lag group {group}: {mem_allocated/(1024**3):.2f} GB")
                        
                        # Move result back to CPU and convert to numpy
                        feature_values = result.cpu().numpy()
                    else:
                        # Use Polars shift function if GPU not available
                        expr = pl.col(col).shift(lag)
                        feature_values = group_data.select(expr).to_numpy().flatten()
                    
                    # Add to group result
                    group_result[feature_name] = feature_values
                except Exception as e:
                    logger.error(f"Error calculating {feature_name} for group {group}: {e}")
                    # Add NaN values as placeholder
                    group_result[feature_name] = [float('nan')] * len(group_data)
        
        # Free GPU memory when done with this group
        if use_gpu:
            torch.cuda.empty_cache()
            
        return {"group": group, "features": group_result}
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing group {group}: {e}")
        logger.error(traceback.format_exc())
        return {"group": group, "features": {group_col: [group]}}

def process_ewm_group(group, df=None, numeric_cols=None, spans=None, date_col=None, group_col=None):
    """Process a group for EWM features. Defined at module level for pickling."""
    try:
        import numpy as np
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        
        if use_gpu:
            logger.info(f"Processing EWM features for group {group} using GPU")
        
        group_result = {}
        # Get data for this group
        if df is not None and group_col is not None:
            group_mask = df[group_col] == group
            group_data = df.filter(group_mask)
        else:
            # Return empty result if no dataframe
            return {"group": group, "features": {group_col: [group]}}
            
        # Initialize group result with the group column
        group_result[group_col] = [group]
        
        # Process each span and column
        for span in spans:
            # Calculate alpha (same formula as pandas ewm uses)
            alpha = 2.0 / (span + 1.0)
            
            for col in numeric_cols:
                # Create feature name
                feature_name = f"{col}_ewm_{span}"
                
                try:
                    if use_gpu:
                        # Get column data
                        col_data = group_data[col].to_numpy()
                        
                        # Skip if all NaN
                        if np.all(np.isnan(col_data)):
                            continue
                            
                        # Replace NaN with 0 for calculation
                        col_data_no_nan = np.nan_to_num(col_data, 0)
                        
                        # Convert to PyTorch tensor on GPU
                        tensor_data = torch.tensor(col_data_no_nan, dtype=torch.float32, device=device)
                        
                        # Create result tensor with same shape
                        result = torch.zeros_like(tensor_data)
                        
                        # Calculate EWM using GPU tensor operations
                        data_len = len(tensor_data)
                        
                        # Create decay weights on GPU
                        decay_weights = torch.pow(1-alpha, torch.arange(data_len-1, -1, -1, device=device))
                        decay_weights = decay_weights / decay_weights.sum()
                        
                        # Implement EWM calculation using GPU tensors
                        # Method 1: Manual implementation
                        for i in range(data_len):
                            if i == 0:
                                result[i] = tensor_data[i]
                            else:
                                # Calculate weighted sum up to this point
                                # Slice and normalize weights for varying length
                                window_data = tensor_data[:i+1]
                                window_weights = decay_weights[data_len-i-1:data_len]
                                window_weights = window_weights / window_weights.sum()
                                
                                # Weighted sum
                                result[i] = torch.sum(window_data * window_weights)
                        
                        # Method 2: Also try with convolution for better GPU utilization
                        if data_len > 100:  # Only use convolution for larger datasets
                            # Use convolution for EWM calculation (potentially faster on GPU)
                            # Reshape tensors for 1D convolution
                            x = tensor_data.view(1, 1, -1)
                            w = decay_weights.view(1, 1, -1)
                            
                            # Perform convolution and extract relevant part
                            conv_result = torch.nn.functional.conv1d(x, w, padding=data_len-1)
                            conv_result = conv_result.view(-1)[:data_len]
                            
                            # Use convolution result as it may be more efficient
                            result = conv_result
                        
                        # Log memory usage occasionally
                        if span == spans[0] and col == numeric_cols[0]:
                            mem_allocated = torch.cuda.memory_allocated(0)
                            logger.debug(f"GPU memory used for EWM group {group}: {mem_allocated/(1024**3):.2f} GB")
                        
                        # Move result back to CPU and convert to numpy
                        feature_values = result.cpu().numpy()
                    else:
                        # Use NumPy if GPU not available
                        col_data = group_data[col].to_numpy()
                        
                        # Skip if all NaN
                        if np.all(np.isnan(col_data)):
                            continue
                            
                        # Replace NaN with 0 for calculation
                        col_data_no_nan = np.nan_to_num(col_data, 0)
                        
                        # Create EWM weights and calculate
                        weights = np.power(1-alpha, np.arange(len(col_data_no_nan)-1, -1, -1))
                        weights = weights / weights.sum()
                        feature_values = np.convolve(col_data_no_nan, weights, mode='full')[:len(col_data_no_nan)]
                    
                    # Add the result to the group result
                    group_result[feature_name] = feature_values
                    
                except Exception as e:
                    logger.error(f"Error calculating {feature_name} for group {group}: {e}")
                    # Add NaN values as placeholder
                    group_result[feature_name] = [float('nan')] * len(group_data)
        
        # Free GPU memory when done with this group
        if use_gpu:
            torch.cuda.empty_cache()
            
        return {"group": group, "features": group_result}
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing group {group}: {e}")
        logger.error(traceback.format_exc())
        return {"group": group, "features": {group_col: [group]}}

if __name__ == "__main__":
    import argparse
    
    # Activate environment if needed
    import os
    import sys
    import subprocess
    
    # Check if we're running in the virtual environment
    venv_path = "/media/knight2/EDB/numer_crypto_temp/environment"
    if not os.environ.get('VIRTUAL_ENV') and os.path.exists(f"{venv_path}/bin/activate"):
        logger.info(f"Not running in virtual environment. Trying to activate {venv_path}")
        try:
            # Execute the script in the virtual environment
            env_python = f"{venv_path}/bin/python"
            if os.path.exists(env_python):
                # Get the current script path
                script_path = os.path.abspath(__file__)
                # Execute the script in the virtual environment
                subprocess.call([env_python, script_path] + sys.argv[1:])
                sys.exit(0)
        except Exception as e:
            logger.warning(f"Failed to activate virtual environment: {e}")
    
    # Check if CUDA libraries are available
    if not cuda_available:
        # Force another detection attempt with improved diagnostics
        logger.info("Performing enhanced GPU detection...")
        
        try:
            # First try subprocess to check if PyTorch CUDA is available
            result = subprocess.run(
                ["/bin/bash", "-c", f"source {venv_path}/bin/activate && python3 -c \"import torch; print('CUDA_AVAILABLE:' + str(torch.cuda.is_available())); print('DEVICE_COUNT:' + str(torch.cuda.device_count()))\""],
                capture_output=True, text=True
            )
            if "CUDA_AVAILABLE:True" in result.stdout:
                logger.info(f"PyTorch with CUDA is available through subprocess: {result.stdout}")
                logger.info("However, main process cannot access it. This suggests an environment issue.")
            else:
                logger.warning(f"PyTorch CUDA not available through subprocess: {result.stdout}")
        except Exception as e:
            logger.warning(f"Failed to check PyTorch CUDA through subprocess: {e}")
    
    parser = argparse.ArgumentParser(description='Generate features using GPU acceleration with Polars')
    parser.add_argument('--input-file', type=str, required=False, help='Input data file (CSV or Parquet)', default="/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet")
    parser.add_argument('--output-file', type=str, help='Output file (CSV or Parquet)', default="/media/knight2/EDB/numer_crypto_temp/data/features/gpu_features.parquet")
    parser.add_argument('--group-col', type=str, default='symbol', help='Column to group by')
    parser.add_argument('--date-col', type=str, default='date', help='Date column for sorting')
    parser.add_argument('--limit-cols', type=int, default=20, help='Limit number of numeric columns to process')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison with CPU methods')
    parser.add_argument('--check-gpu', action='store_true', help='Check GPU availability and exit')
    parser.add_argument('--force-gpu', action='store_true', help='Force GPU detection and usage')
    
    parser.add_argument('--use-all-gpus', action='store_true', help='Use all available GPUs for computation')
    args = parser.parse_args()
    
    # Check GPU availability if requested
    if args.check_gpu:
        logger.info("Checking GPU availability for Polars GPU acceleration...")
        
        # Check CUDA environment variables
        logger.info("\nChecking CUDA environment variables:")
        for env_var in ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH"]:
            value = os.environ.get(env_var, "Not set")
            logger.info(f"  {env_var}: {value}")
        
        # Try each GPU library
        logger.info("\nChecking for CUDA libraries:")
        
        try:
            import cupy as cp
            logger.info("✓ CuPy is available")
            
            # Get GPU details
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                logger.info(f"  Found {device_count} CUDA device(s)")
                
                for i in range(device_count):
                    device_properties = cp.cuda.runtime.getDeviceProperties(i)
                    name = device_properties["name"].decode("utf-8")
                    mem = device_properties["totalGlobalMem"]
                    logger.info(f"  GPU {i}: {name} with {mem/(1024**3):.1f} GB memory")
            except Exception as e:
                logger.warning(f"  Error getting GPU details: {e}")
        except ImportError:
            logger.warning("✗ CuPy is not installed")
            logger.info("  To install: pip install cupy-cuda11x (replace with your CUDA version)")
        
        try:
            import torch
            logger.info("✓ PyTorch is available")
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"  Found {device_count} CUDA device(s)")
                
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory
                    free_mem = mem - torch.cuda.memory_allocated(i)
                    logger.info(f"  GPU {i}: {name} with {free_mem/(1024**3):.1f} GB free / {mem/(1024**3):.1f} GB total")
                
                # Test multi-GPU availability
                if device_count > 1:
                    logger.info("\nTesting multi-GPU setup:")
                    logger.info("  Attempting to access all GPUs sequentially...")
                    
                    for i in range(device_count):
                        try:
                            # Try to create a tensor on each GPU
                            with torch.cuda.device(i):
                                x = torch.ones(10, 10, device=f"cuda:{i}")
                                # Do a small computation to verify it works
                                y = x + x
                                logger.info(f"  ✓ Successfully accessed GPU {i}")
                        except Exception as e:
                            logger.error(f"  ✗ Failed to access GPU {i}: {e}")
            else:
                logger.warning("  PyTorch is installed but CUDA is not available")
                
                # Try to diagnose why CUDA is not available
                logger.info("\nDiagnosing PyTorch CUDA issues:")
                try:
                    logger.info(f"  PyTorch version: {torch.__version__}")
                    build_info = torch._C._show_config()
                    logger.info(f"  CUDA in build config: {'CUDA' in build_info}")
                except Exception as e:
                    logger.warning(f"  Could not get PyTorch build info: {e}")
        except ImportError:
            logger.warning("✗ PyTorch is not installed")
            logger.info("  To install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        # Using PyTorch only, no TensorFlow support
            
        # Also check GPU monitoring utilities
        try:
            import py3nvml
            logger.info("✓ py3nvml is available for GPU monitoring")
        except ImportError:
            logger.warning("✗ py3nvml is not installed")
            logger.info("  To install: pip install py3nvml")
            
        try:
            import GPUtil
            logger.info("✓ GPUtil is available for GPU monitoring")
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    logger.info(f"  Found {len(gpus)} GPU(s)")
                    for i, gpu in enumerate(gpus):
                        logger.info(f"  GPU {i}: {gpu.name} with {gpu.memoryTotal} MB memory, Load: {gpu.load*100:.1f}%")
                else:
                    logger.warning("  No GPUs detected by GPUtil")
            except Exception as e:
                logger.warning(f"  Error getting GPU details: {e}")
        except ImportError:
            logger.warning("✗ GPUtil is not installed")
            logger.info("  To install: pip install GPUtil")
        
        # Check NVIDIA system tools
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("\nNVIDIA System Tools:")
                logger.info("✓ nvidia-smi is available")
                logger.info("  Output summary:")
                
                # Extract GPU information from nvidia-smi output
                output_lines = result.stdout.split('\n')
                gpu_info_lines = [line for line in output_lines if "MiB" in line and "%" in line]
                for line in gpu_info_lines:
                    logger.info(f"  {line.strip()}")
            else:
                logger.warning("✗ nvidia-smi failed to run")
        except Exception as e:
            logger.warning(f"✗ nvidia-smi not available: {e}")
        
        # Check if any CUDA libraries are available
        if not any([cuda_available, cupy_available, torch_available]):
            logger.warning("\nNo GPU acceleration libraries are available.")
            logger.info("To use GPU acceleration, install one of the following packages:")
            logger.info("1. PyTorch (recommended): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            logger.info("2. CuPy: pip install cupy-cuda11x (replace 11x with your CUDA version)")
            
            # Check for common CUDA issues
            logger.info("\nPossible GPU/CUDA issues to check:")
            logger.info("1. Verify NVIDIA drivers are properly installed (run nvidia-smi)")
            logger.info("2. Check that CUDA toolkit is installed (run nvcc --version)")
            logger.info("3. Make sure CUDA_HOME and LD_LIBRARY_PATH are set properly")
            logger.info("4. Ensure your Python environment has GPU-enabled packages")
        else:
            logger.info("\nAt least one GPU library is available. You can use GPU acceleration!")
            
            # Recommend using all GPUs if multiple detected
            if torch_available and torch.cuda.device_count() > 1:
                logger.info(f"\nMultiple GPUs detected ({torch.cuda.device_count()}). Recommended flags:")
                logger.info("  --use-all-gpus        Use all available GPUs for computation")
            
        sys.exit(0)
    
    # Set CUDA_VISIBLE_DEVICES to use all GPUs if available and requested
    # This is important to do before importing any CUDA libraries
    if args.use_all_gpus:
        # Try to get all available GPUs
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
                if gpu_count > 0:
                    gpu_ids = ",".join(str(i) for i in range(gpu_count))
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                    logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids} to use all {gpu_count} GPUs")
        except Exception as e:
            logger.warning(f"Failed to set CUDA_VISIBLE_DEVICES: {e}")

    # Load data
    start_time = time.time()
    if args.input_file.endswith('.csv'):
        import polars as pl
        df = pl.read_csv(args.input_file)
    elif args.input_file.endswith('.parquet'):
        import polars as pl
        df = pl.read_parquet(args.input_file)
    else:
        raise ValueError("Input file must be CSV or Parquet")
        
    logger.info(f"Loaded data with shape {df.shape} in {time.time() - start_time:.2f} seconds")
    
    # Get numeric columns
    excluded_cols = [args.group_col, args.date_col, 'era', 'target', 'id']
    numeric_cols = [col for col in df.columns 
                  if col not in excluded_cols and 
                  df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
    # Limit columns if specified
    if args.limit_cols > 0 and len(numeric_cols) > args.limit_cols:
        logger.info(f"Limiting to {args.limit_cols} numeric columns (from {len(numeric_cols)} total)")
        numeric_cols = numeric_cols[:args.limit_cols]
        
    # Initialize accelerator - use all GPUs if specified
    accelerator = PolarsGPUAccelerator(
        force_gpu=args.force_gpu,
        use_all_gpus=args.use_all_gpus
    )
    
    # Generate features
    start_time = time.time()
    result_df = accelerator.generate_all_features(
        df, 
        group_col=args.group_col, 
        numeric_cols=numeric_cols,
        date_col=args.date_col
    )
    
    total_time = time.time() - start_time
    logger.info(f"Generated {result_df.width - df.width} features in {total_time:.2f} seconds")
    
    # Save results if output file specified
    if args.output_file:
        if args.output_file.endswith('.csv'):
            result_df.write_csv(args.output_file)
        elif args.output_file.endswith('.parquet'):
            result_df.write_parquet(args.output_file)
            
        logger.info(f"Saved results to {args.output_file}")
        
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running benchmark comparison with CPU methods...")
        
        # Run CPU-based methods for comparison
        import polars as pl
        
        # Benchmark rolling features
        logger.info("Benchmarking rolling features (CPU vs GPU)...")
        
        # Subset of data for benchmark
        if len(numeric_cols) > 5:
            benchmark_cols = numeric_cols[:5]
        else:
            benchmark_cols = numeric_cols
            
        windows = [3, 7, 14]  # Use fewer windows for benchmark
        
        # CPU Polars method
        start_time = time.time()
        cpu_result = df.clone()
        
        for col in benchmark_cols:
            for window in windows:
                for func_name in ['mean']:  # Just do mean for comparison
                    feature_name = f"{col}_roll_{window}_{func_name}"
                    
                    if func_name == 'mean':
                        expr = pl.col(col).rolling_mean(window_size=window, min_samples=1).alias(feature_name)
                    
                    # Apply the expression
                    feature_df = cpu_result.sort([args.group_col, args.date_col]).group_by(args.group_col).agg([expr])
                    cpu_result = cpu_result.join(feature_df, on=args.group_col, how="left")
        
        cpu_time = time.time() - start_time
        logger.info(f"CPU Polars rolling features: {cpu_time:.2f} seconds")
        
        # GPU method
        start_time = time.time()
        gpu_result = accelerator.generate_rolling_features(
            df.clone(),
            group_col=args.group_col,
            numeric_cols=benchmark_cols,
            windows=windows,
            date_col=args.date_col
        )
        gpu_time = time.time() - start_time
        logger.info(f"GPU rolling features: {gpu_time:.2f} seconds")
        
        # Report speedup
        speedup = cpu_time / gpu_time
        logger.info(f"GPU speedup factor: {speedup:.2f}x")
        
        # Compare actual values to ensure correctness (sample 5 values)
        logger.info("\nVerifying GPU vs CPU results (sampling 5 values):")
        
        try:
            # Get the first feature column
            feature_cols = [col for col in cpu_result.columns if col.startswith(benchmark_cols[0])]
            if feature_cols:
                feature_col = feature_cols[0]
                
                # Get a few values from each result
                cpu_values = cpu_result[feature_col].head(5).to_list()
                gpu_values = gpu_result[feature_col].head(5).to_list()
                
                # Print the values
                logger.info(f"  Feature: {feature_col}")
                logger.info(f"  CPU values: {cpu_values}")
                logger.info(f"  GPU values: {gpu_values}")
                
                # Check if values are close
                import numpy as np
                diffs = [abs(c - g) for c, g in zip(cpu_values, gpu_values) if not (np.isnan(c) and np.isnan(g))]
                max_diff = max(diffs) if diffs else 0
                
                if max_diff < 1e-5:
                    logger.info("  ✓ CPU and GPU results match!")
                else:
                    logger.warning(f"  ✗ CPU and GPU results differ (max diff: {max_diff:.6f})")
        except Exception as e:
            logger.warning(f"  Error comparing results: {e}")
        
        # Provide hints if no speedup
        if speedup <= 1.0:
            logger.warning("\nNo speedup observed with GPU acceleration. Possible reasons:")
            logger.warning("1. No CUDA libraries available or properly configured")
            logger.warning("2. Dataset is too small to benefit from GPU acceleration")
            logger.warning("3. GPU overhead exceeds benefits for this workload")
            logger.warning("\nRecommendations:")
            logger.warning("- Run with --check-gpu to verify GPU availability")
            logger.warning("- Try with a larger dataset")
            logger.warning("- If multiple GPUs are available, use --use-all-gpus to distribute work")