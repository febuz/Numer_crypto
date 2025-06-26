#!/usr/bin/env python3
"""
GPU Acceleration Interface

This module provides a unified interface for GPU-accelerated mathematical transformations
and feature engineering operations, abstracting away the underlying GPU libraries.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import GPU utilities
from utils.gpu.memory_utils import (
    cuda_available, cupy_available, torch_available,
    detect_gpu_libraries, clear_gpu_memory, get_gpu_memory_usage,
    check_gpu_memory_availability, start_memory_manager
)
from utils.gpu.data_conversion import to_gpu, to_cpu, chunk_array_for_gpu
from utils.gpu.math_transforms import (
    basic_transforms, trigonometric_transforms,
    polynomial_transforms, interaction_transforms
)

class GPUAccelerator:
    """
    Unified interface for GPU-accelerated data processing and feature engineering
    """
    
    def __init__(self, device_id: int = 0, memory_pool_size: Optional[int] = None):
        """
        Initialize the GPU Accelerator
        
        Args:
            device_id: GPU device ID to use
            memory_pool_size: Memory pool size in MB (for CuPy)
        """
        # Check if GPU is available
        if not cuda_available:
            self.has_gpu = detect_gpu_libraries()
        else:
            self.has_gpu = True
        
        # Initialize GPU settings
        self.device_id = device_id if self.has_gpu else None
        
        # Set up device environment
        if self.has_gpu:
            # Set device for CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            
            # Set up memory pool for CuPy
            if cupy_available and memory_pool_size is not None:
                try:
                    import cupy as cp
                    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
                    cp.cuda.set_allocator(pool.malloc)
                    logger.info(f"CuPy memory pool initialized with size: {memory_pool_size} MB")
                except Exception as e:
                    logger.warning(f"Failed to initialize CuPy memory pool: {e}")
            
            # Start memory manager
            self.memory_manager = start_memory_manager(check_interval=10, threshold_gb=400)
            
            # Log GPU info
            self._log_gpu_info()
        else:
            logger.warning("No GPU support available. Using CPU fallback for all operations.")
    
    def _log_gpu_info(self):
        """Log information about the GPU"""
        if not self.has_gpu:
            logger.info("No GPU available")
            return
        
        # Log GPU memory
        used_gb, free_gb, total_gb = get_gpu_memory_usage()
        logger.info(f"GPU {self.device_id} memory: {used_gb:.2f}GB used, {free_gb:.2f}GB free, {total_gb:.2f}GB total")
        
        # Log GPU library info
        if cupy_available:
            import cupy as cp
            logger.info(f"Using CuPy {cp.__version__}")
            try:
                cuda_version = cp.cuda.runtime.runtimeGetVersion()
                logger.info(f"CUDA version: {cuda_version//1000}.{(cuda_version%1000)//10}")
            except:
                pass
        elif torch_available:
            import torch
            logger.info(f"Using PyTorch {torch.__version__}")
            if hasattr(torch.version, 'cuda'):
                logger.info(f"CUDA version: {torch.version.cuda}")
    
    def basic_transforms(self, data: np.ndarray, feature_names: List[str], 
                        max_chunk_size: int = 1000000) -> Tuple[np.ndarray, List[str]]:
        """
        Apply basic mathematical transformations to features
        
        Args:
            data: Input array of shape (n_samples, n_features)
            feature_names: Names of the input features
            max_chunk_size: Maximum chunk size for GPU processing
            
        Returns:
            Tuple of (transformed_data, transformed_feature_names)
        """
        return basic_transforms(data, feature_names, max_chunk_size)
    
    def trigonometric_transforms(self, data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply trigonometric transformations to features
        
        Args:
            data: Input array of shape (n_samples, n_features)
            feature_names: Names of the input features
            
        Returns:
            Tuple of (transformed_data, transformed_feature_names)
        """
        return trigonometric_transforms(data, feature_names)
    
    def polynomial_transforms(self, data: np.ndarray, feature_names: List[str], 
                             max_degree: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Apply polynomial transformations to features
        
        Args:
            data: Input array of shape (n_samples, n_features)
            feature_names: Names of the input features
            max_degree: Maximum polynomial degree
            
        Returns:
            Tuple of (transformed_data, transformed_feature_names)
        """
        return polynomial_transforms(data, feature_names, max_degree)
    
    def interaction_transforms(self, data: np.ndarray, feature_names: List[str], 
                              max_interactions: int = 2000) -> Tuple[np.ndarray, List[str]]:
        """
        Generate interaction features (pairwise products)
        
        Args:
            data: Input array of shape (n_samples, n_features)
            feature_names: Names of the input features
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            Tuple of (transformed_data, transformed_feature_names)
        """
        return interaction_transforms(data, feature_names, max_interactions)
    
    def clear_memory(self, force_full_clear=False):
        """
        Clear GPU memory
        
        Args:
            force_full_clear: If True, forces full garbage collection
        """
        clear_gpu_memory(force_full_clear)
    
    def to_gpu(self, data):
        """
        Move data to GPU
        
        Args:
            data: NumPy array to move
            
        Returns:
            GPU array or tensor
        """
        return to_gpu(data)
    
    def to_cpu(self, data):
        """
        Move data to CPU
        
        Args:
            data: GPU array or tensor
            
        Returns:
            NumPy array
        """
        return to_cpu(data)
    
    def get_memory_usage(self):
        """
        Get current GPU memory usage
        
        Returns:
            Tuple of (used_gb, free_gb, total_gb)
        """
        return get_gpu_memory_usage()
    
    def check_memory_availability(self, data_shape, operation_type="basic"):
        """
        Check if there's enough GPU memory for an operation
        
        Args:
            data_shape: Shape of the data
            operation_type: Type of operation
            
        Returns:
            True if there's enough memory, False otherwise
        """
        return check_gpu_memory_availability(data_shape, operation_type)
    
    def process_in_chunks(self, data, chunk_size, processing_func, *args, **kwargs):
        """
        Process large data in chunks
        
        Args:
            data: Input data array
            chunk_size: Maximum chunk size
            processing_func: Function to apply to each chunk
            *args, **kwargs: Arguments to pass to processing_func
            
        Returns:
            Processed data
        """
        # Check if chunking is needed
        if data.shape[0] <= chunk_size:
            return processing_func(data, *args, **kwargs)
        
        # Split into chunks
        chunks = chunk_array_for_gpu(data, chunk_size)
        
        # Process each chunk
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with shape {chunk.shape}")
            # Process chunk
            chunk_result = processing_func(chunk, *args, **kwargs)
            results.append(chunk_result)
            
            # Clear GPU memory
            self.clear_memory()
        
        # Combine results
        if isinstance(results[0], tuple):
            # If result is a tuple, combine the first element and keep the rest from the first result
            combined = (np.vstack([r[0] for r in results]),) + results[0][1:]
            return combined
        else:
            # If result is a single array
            return np.vstack(results)

# Create a singleton instance for easier imports
accelerator = GPUAccelerator()

if __name__ == "__main__":
    # Test the accelerator
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GPU Accelerator")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows in test data")
    parser.add_argument("--cols", type=int, default=20, help="Number of columns in test data")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    
    # Create test accelerator
    test_acc = GPUAccelerator(device_id=args.device)
    
    # Create test data
    np.random.seed(42)
    X = np.random.random((args.rows, args.cols)).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Test basic transforms
    X_basic, basic_names = test_acc.basic_transforms(X, feature_names)
    print(f"Basic transforms: {X.shape} -> {X_basic.shape}")
    
    # Test trigonometric transforms
    X_trig, trig_names = test_acc.trigonometric_transforms(X, feature_names)
    print(f"Trigonometric transforms: {X.shape} -> {X_trig.shape}")
    
    # Test polynomial transforms
    X_poly, poly_names = test_acc.polynomial_transforms(X, feature_names, max_degree=2)
    print(f"Polynomial transforms: {X.shape} -> {X_poly.shape}")
    
    # Test interaction transforms
    X_inter, inter_names = test_acc.interaction_transforms(X, feature_names, max_interactions=100)
    print(f"Interaction transforms: {X.shape} -> {X_inter.shape}")
    
    # Test chunked processing
    def simple_process(data):
        return data * 2
    
    X_chunked = test_acc.process_in_chunks(X, chunk_size=1000, processing_func=simple_process)
    print(f"Chunked processing: {X.shape} -> {X_chunked.shape}")
    
    # Print memory usage
    used_gb, free_gb, total_gb = test_acc.get_memory_usage()
    print(f"GPU memory: {used_gb:.2f}GB used, {free_gb:.2f}GB free, {total_gb:.2f}GB total")