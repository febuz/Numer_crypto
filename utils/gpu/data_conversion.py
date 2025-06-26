#!/usr/bin/env python3
"""
GPU Data Conversion Utilities

This module provides utilities for converting data between CPU and GPU formats,
supporting various GPU libraries including CuPy and PyTorch.
"""

import os
import sys
import logging
import numpy as np
from typing import Union, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import local modules
from utils.gpu.memory_utils import (
    cuda_available, cupy_available, torch_available, 
    detect_gpu_libraries
)

# Run detection at module import if not already done
if not cuda_available:
    detect_gpu_libraries()

def to_gpu(data: np.ndarray) -> Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']:
    """
    Convert numpy array to GPU array using available libraries
    
    Args:
        data: NumPy array to convert
        
    Returns:
        GPU array or tensor, or original numpy array if GPU not available
    """
    if not cuda_available:
        return data
    
    # Handle empty or None data
    if data is None or data.size == 0:
        return data
    
    # Try CuPy first
    if cupy_available:
        try:
            import cupy as cp
            # Already a CuPy array
            if isinstance(data, cp.ndarray):
                return data
            
            # Convert to CuPy array
            return cp.asarray(data)
        except Exception as e:
            logger.warning(f"Error converting to CuPy array: {e}")
    
    # Try PyTorch as fallback
    if torch_available:
        try:
            import torch
            # Already a PyTorch tensor
            if torch.is_tensor(data):
                if data.device.type == 'cuda':
                    return data
                else:
                    return data.cuda()
            
            # Convert to PyTorch tensor
            return torch.tensor(data, device='cuda')
        except Exception as e:
            logger.warning(f"Error converting to PyTorch tensor: {e}")
    
    # Return original if conversion failed
    return data

def to_cpu(data: Any) -> np.ndarray:
    """
    Convert GPU array or tensor to numpy array
    
    Args:
        data: GPU array or tensor to convert
        
    Returns:
        NumPy array
    """
    # Handle empty or None data
    if data is None:
        return np.array([])
    
    # Already a NumPy array
    if isinstance(data, np.ndarray):
        return data
    
    # Handle CuPy array
    if cupy_available:
        try:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
        except Exception as e:
            logger.warning(f"Error converting from CuPy array: {e}")
    
    # Handle PyTorch tensor
    if torch_available:
        try:
            import torch
            if torch.is_tensor(data):
                return data.cpu().numpy()
        except Exception as e:
            logger.warning(f"Error converting from PyTorch tensor: {e}")
    
    # Try direct conversion as last resort
    try:
        return np.array(data)
    except Exception as e:
        logger.error(f"Failed to convert to NumPy array: {e}")
        return np.array([])

def chunk_array_for_gpu(data: np.ndarray, max_chunk_size: int = 1000000) -> list:
    """
    Split large arrays into chunks for GPU processing
    
    Args:
        data: Array to split
        max_chunk_size: Maximum rows per chunk
        
    Returns:
        List of array chunks
    """
    if data.shape[0] <= max_chunk_size:
        return [data]
    
    # Calculate number of chunks
    n_chunks = (data.shape[0] + max_chunk_size - 1) // max_chunk_size
    
    # Split array into chunks
    chunks = []
    for i in range(n_chunks):
        start_idx = i * max_chunk_size
        end_idx = min((i + 1) * max_chunk_size, data.shape[0])
        chunks.append(data[start_idx:end_idx])
    
    logger.info(f"Split array of shape {data.shape} into {len(chunks)} chunks")
    return chunks

def gpu_memmap(array_shape: tuple, dtype=np.float32, filename=None) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Create a memory-mapped array that can be used with GPU 
    
    Args:
        array_shape: Shape of the array
        dtype: Data type
        filename: File to map to, if None a temporary file is created
        
    Returns:
        Memory-mapped array compatible with GPU operations
    """
    if not cuda_available:
        # Create numpy memmap if GPU not available
        if filename is None:
            import tempfile
            filename = tempfile.mktemp(suffix='.dat')
        
        return np.memmap(
            filename, 
            dtype=dtype, 
            mode='w+', 
            shape=array_shape
        )
    
    # For CuPy, create a numpy memmap and then move to GPU in chunks
    if cupy_available:
        try:
            import cupy as cp
            
            # Create numpy memmap
            if filename is None:
                import tempfile
                filename = tempfile.mktemp(suffix='.dat')
            
            numpy_memmap = np.memmap(
                filename, 
                dtype=dtype, 
                mode='w+', 
                shape=array_shape
            )
            
            # Let CuPy manage the GPU array
            gpu_array = cp.zeros(array_shape, dtype=dtype)
            
            return gpu_array, numpy_memmap
        except Exception as e:
            logger.warning(f"Error creating CuPy memory-mapped array: {e}")
    
    # Fallback to numpy memmap
    if filename is None:
        import tempfile
        filename = tempfile.mktemp(suffix='.dat')
    
    return np.memmap(
        filename, 
        dtype=dtype, 
        mode='w+', 
        shape=array_shape
    )

if __name__ == "__main__":
    # Test data conversion functions
    detect_gpu_libraries()
    
    if cuda_available:
        # Create test array
        test_array = np.random.random((1000, 10)).astype(np.float32)
        
        # Convert to GPU
        gpu_array = to_gpu(test_array)
        print(f"Original array: {type(test_array)}, shape: {test_array.shape}")
        print(f"GPU array: {type(gpu_array)}, shape: {gpu_array.shape}")
        
        # Convert back to CPU
        cpu_array = to_cpu(gpu_array)
        print(f"Back to CPU: {type(cpu_array)}, shape: {cpu_array.shape}")
        
        # Test chunking
        chunks = chunk_array_for_gpu(np.random.random((2500000, 10)), max_chunk_size=1000000)
        print(f"Split into {len(chunks)} chunks with shapes: {[chunk.shape for chunk in chunks]}")
    else:
        print("No GPU support available")