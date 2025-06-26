#!/usr/bin/env python3
"""
GPU Memory Utilities

This module provides utilities for managing GPU memory, including estimation,
measurement, and memory clearing functions.
"""

import os
import sys
import gc
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize GPU support flags
cuda_available = False
cupy_available = False
torch_available = False

def detect_gpu_libraries():
    """Detect and initialize GPU libraries"""
    global cuda_available, cupy_available, torch_available
    
    # Try CuPy first (most efficient for array operations)
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"CuPy detected {device_count} CUDA devices")
        cupy_available = True
        cuda_available = True
        return True
    except ImportError:
        logger.info("CuPy not available")
    except Exception as e:
        logger.warning(f"CuPy failed: {e}")
    
    # Try PyTorch as fallback
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"PyTorch detected {device_count} CUDA devices")
            torch_available = True
            cuda_available = True
            return True
        else:
            logger.warning("PyTorch installed but CUDA is not available")
    except ImportError:
        logger.info("PyTorch not available")
    except Exception as e:
        logger.warning(f"PyTorch failed: {e}")
    
    logger.warning("No GPU support available, using CPU only")
    return False

# Run detection at module import
detect_gpu_libraries()

def clear_gpu_memory(force_full_clear=False):
    """
    Clear GPU memory
    
    Args:
        force_full_clear: If True, forces full garbage collection
    """
    if not cuda_available:
        return
        
    # Try to clear memory in CuPy
    if cupy_available:
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Clear memory pool
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            # Log memory usage after clearing
            free, total = cp.cuda.runtime.memGetInfo()
            logger.debug(f"GPU memory after CuPy clear: {free/1e9:.2f}GB free / {total/1e9:.2f}GB total")
        except Exception as e:
            logger.warning(f"Error clearing CuPy memory: {e}")
    
    # Try to clear memory in PyTorch
    if torch_available:
        try:
            import torch
            torch.cuda.empty_cache()
            
            # Log memory usage after clearing
            free_bytes = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            logger.debug(f"GPU memory after PyTorch clear: {free_bytes/1e9:.2f}GB free / {total_bytes/1e9:.2f}GB total")
        except Exception as e:
            logger.warning(f"Error clearing PyTorch memory: {e}")
    
    # Force Python garbage collection if requested
    if force_full_clear:
        gc.collect()

def get_gpu_memory_usage() -> Tuple[float, float, float]:
    """
    Get current GPU memory usage
    
    Returns:
        Tuple of (used_gb, free_gb, total_gb)
    """
    used_gb, free_gb, total_gb = 0.0, 0.0, 0.0
    
    if not cuda_available:
        return used_gb, free_gb, total_gb
    
    # Try to get memory info from CuPy
    if cupy_available:
        try:
            import cupy as cp
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            used_bytes = total_bytes - free_bytes
            
            # Convert to GB
            used_gb = used_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            
            return used_gb, free_gb, total_gb
        except Exception as e:
            logger.warning(f"Error getting CuPy memory info: {e}")
    
    # Try to get memory info from PyTorch
    if torch_available:
        try:
            import torch
            used_bytes = torch.cuda.memory_allocated()
            reserved_bytes = torch.cuda.memory_reserved()
            free_bytes = reserved_bytes - used_bytes
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            
            # Convert to GB
            used_gb = used_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            
            return used_gb, free_gb, total_gb
        except Exception as e:
            logger.warning(f"Error getting PyTorch memory info: {e}")
    
    return used_gb, free_gb, total_gb

def estimate_memory_usage(data_shape: Tuple[int, int], operation_type: str = "basic") -> float:
    """
    Estimate memory usage for a given operation and data shape
    
    Args:
        data_shape: Shape of the input data (rows, cols)
        operation_type: Type of operation ("basic", "polynomial", "interaction", etc.)
        
    Returns:
        Estimated memory usage in GB
    """
    rows, cols = data_shape
    
    # Base memory for input data (assuming float32)
    base_memory_gb = (rows * cols * 4) / (1024**3)
    
    # Estimate multiplier based on operation type
    if operation_type == "basic":
        # Basic operations add about 5x the original data size
        multiplier = 5.0
    elif operation_type == "polynomial":
        # Polynomial transforms can be very memory intensive
        multiplier = 8.0 * cols  # Scales with number of columns
    elif operation_type == "interaction":
        # Interaction features create n^2 combinations in worst case
        multiplier = min(10.0, 2.0 + (cols / 200)**2)  # Cap at 10x for very large datasets
    elif operation_type == "trigonometric":
        # Trigonometric functions typically add about 4x
        multiplier = 4.0
    else:
        # Default multiplier for unknown operations
        multiplier = 3.0
    
    # Add overhead factor (20%)
    overhead = 1.2
    
    # Final estimate
    estimated_gb = base_memory_gb * multiplier * overhead
    
    return estimated_gb

def check_gpu_memory_availability(data_shape: Tuple[int, int], operation_type: str = "basic") -> bool:
    """
    Check if there's enough GPU memory for an operation
    
    Args:
        data_shape: Shape of the input data (rows, cols)
        operation_type: Type of operation
        
    Returns:
        True if there's likely enough memory, False otherwise
    """
    if not cuda_available:
        return False
    
    # Get current memory usage
    used_gb, free_gb, total_gb = get_gpu_memory_usage()
    
    # Estimate required memory
    required_gb = estimate_memory_usage(data_shape, operation_type)
    
    # Check if we have enough free memory (with 1GB buffer)
    has_enough = free_gb > (required_gb + 1.0)
    
    logger.debug(f"Memory check for {operation_type}: need {required_gb:.2f}GB, have {free_gb:.2f}GB free - {'PASS' if has_enough else 'FAIL'}")
    
    return has_enough

def manage_memory(check_interval=5, threshold_gb=500):
    """
    Periodic memory management - can be run in a separate thread
    
    Args:
        check_interval: How often to check memory in seconds
        threshold_gb: Memory threshold in GB to trigger cleanup
    """
    if not cuda_available:
        return
        
    import time
    
    while True:
        used_gb, free_gb, total_gb = get_gpu_memory_usage()
        
        # If usage exceeds threshold, clear memory
        if used_gb > threshold_gb:
            logger.info(f"Memory usage ({used_gb:.2f}GB) exceeds threshold ({threshold_gb}GB), clearing GPU memory")
            clear_gpu_memory(force_full_clear=True)
        
        # Sleep for the specified interval
        time.sleep(check_interval)

# Setup periodic memory management in a background thread if needed
def start_memory_manager(check_interval=5, threshold_gb=500):
    """
    Start memory management in a background thread
    
    Args:
        check_interval: How often to check memory in seconds
        threshold_gb: Memory threshold in GB to trigger cleanup
    """
    if not cuda_available:
        return None
        
    import threading
    
    memory_thread = threading.Thread(
        target=manage_memory, 
        args=(check_interval, threshold_gb),
        daemon=True
    )
    memory_thread.start()
    
    return memory_thread

if __name__ == "__main__":
    # Test memory functions
    detect_gpu_libraries()
    
    if cuda_available:
        used_gb, free_gb, total_gb = get_gpu_memory_usage()
        print(f"GPU memory: {used_gb:.2f}GB used, {free_gb:.2f}GB free, {total_gb:.2f}GB total")
        
        # Test memory estimation
        test_shape = (1000000, 50)  # 1M rows, 50 cols
        for op_type in ["basic", "polynomial", "interaction", "trigonometric"]:
            est_gb = estimate_memory_usage(test_shape, op_type)
            print(f"Estimated memory for {op_type}: {est_gb:.2f}GB")
            
        # Test memory availability check
        for op_type in ["basic", "polynomial", "interaction"]:
            available = check_gpu_memory_availability(test_shape, op_type)
            print(f"Memory available for {op_type}: {available}")
    else:
        print("No GPU support available")