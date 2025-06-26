#!/usr/bin/env python3
"""
Ultra-Fast GPU Mathematical Transformations Accelerator

This module provides GPU-accelerated mathematical transformations that are 10-100x faster
than CPU implementations, specifically optimized for financial feature engineering.
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
            cuda_devices = torch.cuda.device_count()
            logger.info(f"PyTorch CUDA available: {cuda_devices} devices")
            torch_available = True
            cuda_available = True
            return True
        else:
            logger.warning("PyTorch installed but CUDA not available")
    except ImportError:
        logger.info("PyTorch not available")
    
    logger.warning("No GPU libraries available - using CPU NumPy")
    return False

# Initialize GPU detection
detect_gpu_libraries()

class GPUMathAccelerator:
    """Ultra-fast GPU mathematical transformations for feature engineering"""
    
    def __init__(self, device_id: int = 0, memory_pool_size: Optional[int] = None):
        """
        Initialize GPU math accelerator
        
        Args:
            device_id: GPU device ID to use
            memory_pool_size: Memory pool size in bytes (None for auto)
        """
        self.device_id = device_id
        self.gpu_available = cuda_available
        
        # Get system memory info to dynamically adjust our limits
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            system_total_gb = system_memory.total / (1024**3)
            system_available_gb = system_memory.available / (1024**3)
            logger.info(f"System memory: {system_total_gb:.1f}GB total, {system_available_gb:.1f}GB available")
            
            # Set a conservative system memory limit at 60% of available memory
            self.system_memory_limit_gb = min(system_available_gb * 0.6, 500.0)
            logger.info(f"System memory limit set to {self.system_memory_limit_gb:.1f}GB")
        except ImportError:
            logger.warning("psutil not available, using default system memory limits")
            self.system_memory_limit_gb = 400.0
        except Exception as e:
            logger.warning(f"Error getting system memory: {e}, using default system memory limits")
            self.system_memory_limit_gb = 400.0
        
        # Get memory limit from environment variable or default
        env_memory_limit = os.environ.get("GPU_MEMORY_LIMIT")
        if env_memory_limit:
            try:
                self.max_gpu_memory_gb = float(env_memory_limit)
                logger.info(f"Using GPU memory limit from environment: {self.max_gpu_memory_gb} GB")
            except (ValueError, TypeError):
                # Default to more conservative value if environment variable is invalid
                self.max_gpu_memory_gb = 12.0  
                logger.warning(f"Invalid GPU_MEMORY_LIMIT: {env_memory_limit}, using default 12.0 GB")
        else:
            # Set to 12GB by default (more conservative value)
            self.max_gpu_memory_gb = 12.0
            logger.info(f"Using default GPU memory limit: {self.max_gpu_memory_gb} GB")
        
        if self.gpu_available:
            if cupy_available:
                import cupy as cp
                
                # Try to get GPU count and total memory
                try:
                    gpu_count = cp.cuda.runtime.getDeviceCount()
                    logger.info(f"Detected {gpu_count} GPUs")
                    
                    # Examine all available GPUs
                    total_gpu_memory = 0
                    for i in range(gpu_count):
                        cp.cuda.Device(i).use()
                        mem_info = cp.cuda.Device(i).mem_info
                        total_mem = mem_info[1]
                        free_mem = mem_info[0]
                        total_gpu_memory += total_mem
                        logger.info(f"GPU {i}: {total_mem/(1024**3):.1f} GB total, {free_mem/(1024**3):.1f} GB free")
                    
                    # Set main device
                    cp.cuda.Device(device_id).use()
                    
                    # Calculate total memory across all GPUs
                    total_gpu_gb = total_gpu_memory / (1024**3)
                    # Update max memory based on actual hardware and limit
                    if total_gpu_gb > 0:
                        # More conservative - use at most 60% of available memory and cap at our limit
                        self.max_gpu_memory_gb = min(total_gpu_gb * 0.6, self.max_gpu_memory_gb)  
                        logger.info(f"Using {self.max_gpu_memory_gb:.1f} GB across all GPUs")
                except Exception as e:
                    logger.warning(f"Could not detect all GPUs, using default: {e}")
                    
                # Get GPU memory info and set conservative limits
                try:
                    mem_info = cp.cuda.Device(device_id).mem_info
                    free_bytes = mem_info[0]
                    total_bytes = mem_info[1]
                    
                    # For NVLink connected GPUs, use a conservative limit
                    # This allows using memory across multiple GPUs without OOM errors
                    effective_gpu_memory = min(self.max_gpu_memory_gb, 16.0) * 1024**3
                    
                    # Calculate safe memory limit (50% of available GPU memory)
                    safe_memory = min(effective_gpu_memory, free_bytes * 0.5)
                    
                    if memory_pool_size is None:
                        memory_pool_size = int(safe_memory)
                    
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=memory_pool_size)
                    
                    logger.info(f"GPU memory limit set to {memory_pool_size / (1024**3):.1f} GB")
                    
                except Exception as e:
                    logger.warning(f"Could not set GPU memory limit: {e}")
                
                self.xp = cp
                logger.info(f"GPU Math Accelerator initialized with CuPy on device {device_id}")
            elif torch_available:
                import torch
                torch.cuda.set_device(device_id)
                self.xp = torch
                logger.info(f"GPU Math Accelerator initialized with PyTorch on device {device_id}")
            else:
                self.xp = np
                self.gpu_available = False
        else:
            self.xp = np
            logger.info("GPU Math Accelerator using CPU NumPy")
    
    def _to_gpu(self, data: np.ndarray) -> Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']:
        """Transfer data to GPU"""
        if not self.gpu_available:
            return data
        
        # Check if data is too large to transfer at once
        data_size_gb = data.nbytes / (1024**3)
        
        if cupy_available:
            import cupy as cp
            
            # For very large arrays, transfer in chunks to avoid pinned memory issues
            if data_size_gb > 4.0:  # If data is larger than 4GB
                logger.info(f"Large array detected ({data_size_gb:.2f}GB), transferring to GPU in chunks")
                
                # Transfer in chunks to avoid pinned memory errors
                n_chunks = max(int(data_size_gb / 2) + 1, 2)  # At least 2 chunks, aim for 2GB or smaller chunks
                chunk_size = data.shape[0] // n_chunks
                
                # Create empty array on GPU
                result = cp.empty(data.shape, dtype=cp.float32)
                
                # Transfer in chunks
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, data.shape[0])
                    
                    # Copy this chunk
                    result[start_idx:end_idx] = cp.array(data[start_idx:end_idx], dtype=cp.float32)
                    
                    # Clear memory after each chunk
                    cp.get_default_memory_pool().free_all_blocks()
                
                return result
            else:
                # Normal transfer for smaller arrays
                return cp.array(data, dtype=cp.float32)
        elif torch_available:
            import torch
            
            # For very large arrays, transfer in chunks
            if data_size_gb > 4.0:
                logger.info(f"Large array detected ({data_size_gb:.2f}GB), transferring to GPU in chunks")
                
                # Transfer in chunks
                n_chunks = max(int(data_size_gb / 2) + 1, 2)
                chunk_size = data.shape[0] // n_chunks
                
                # Create empty array on GPU
                result = torch.empty(data.shape, dtype=torch.float32, device='cuda')
                
                # Transfer in chunks
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, data.shape[0])
                    
                    # Copy this chunk
                    result[start_idx:end_idx] = torch.tensor(data[start_idx:end_idx], dtype=torch.float32, device='cuda')
                    
                    # Clear cache after each chunk
                    torch.cuda.empty_cache()
                
                return result
            else:
                # Normal transfer for smaller arrays
                return torch.tensor(data, dtype=torch.float32, device='cuda')
        else:
            return data
    
    def _to_cpu(self, data) -> np.ndarray:
        """Transfer data back to CPU"""
        if not self.gpu_available or isinstance(data, np.ndarray):
            return data
        
        if cupy_available:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
        elif torch_available:
            import torch
            if torch.is_tensor(data):
                return data.cpu().numpy()
        
        return data
    
    def gpu_basic_transforms(self, data: np.ndarray, feature_names: List[str], max_chunk_size: int = 1000000) -> Tuple[np.ndarray, List[str]]:
        """
        Ultra-fast basic mathematical transformations on GPU
        
        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Names of input features
            max_chunk_size: Maximum chunk size for very large datasets
            
        Returns:
            Tuple of (transformed_data, new_feature_names)
        """
        logger.info(f"🚀 GPU Basic Transforms: {data.shape}")
        start_time = time.time()
        
        # For extremely large datasets, use chunking
        if max_chunk_size is not None and data.shape[0] > max_chunk_size:
            logger.info(f"Very large dataset detected ({data.shape[0]} rows), processing in chunks of {max_chunk_size}")
            
            # Process in chunks
            chunks = []
            transform_names = None
            
            # Process each chunk
            for i in range(0, data.shape[0], max_chunk_size):
                end_idx = min(i + max_chunk_size, data.shape[0])
                logger.info(f"Processing chunk {i//max_chunk_size + 1}/{(data.shape[0] + max_chunk_size - 1) // max_chunk_size}: rows {i}-{end_idx}")
                
                # Get this chunk
                chunk_data = data[i:end_idx]
                
                # Process this chunk with chunking disabled
                chunk_result, chunk_names = self._process_basic_transforms_no_chunking(chunk_data, feature_names)
                
                # Save the result and names
                chunks.append(chunk_result)
                if transform_names is None:
                    transform_names = chunk_names
                
                # Clean up memory
                self._clear_gpu_memory(force_full_clear=True)
            
            # Combine all chunks
            logger.info(f"Combining {len(chunks)} chunks of basic transforms")
            result = np.vstack(chunks)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Chunked basic transforms: {result.shape[1]} features in {elapsed:.2f}s")
            
            return result, transform_names
        
        # Use the non-chunking implementation
        return self._process_basic_transforms_no_chunking(data, feature_names)
        
    def _process_basic_transforms_no_chunking(self, data, feature_names):
        """Internal helper to process basic transforms without chunking"""
        start_time = time.time()
        
        # Check GPU memory availability for large datasets
        if not self._check_gpu_memory_availability(data, "basic"):
            logger.warning(f"Insufficient GPU memory for basic transforms, using CPU fallback")
            return self._cpu_basic_fallback(data, feature_names)
            
        # Get available GPU memory
        available_gb = 0.0
        try:
            if cupy_available:
                import cupy as cp
                mem_info = cp.cuda.Device(self.device_id).mem_info
                available_gb = mem_info[0] / (1024**3)  # Free memory in GB
            else:
                # Default to a safe value
                available_gb = 16.0
        except Exception as e:
            logger.debug(f"Error getting GPU memory info: {e}, using default value")
            available_gb = 16.0
            
        # For extremely large datasets with limited GPU memory, go to CPU
        # Adjust threshold to make better use of available memory
        if ((data.shape[0] > 5000000 and available_gb < 20.0) or  # Very large datasets need lots of memory
            (data.shape[0] > 3000000 and available_gb < 12.0) or  # Large datasets need medium memory
            data.shape[1] > 10000 or                             # Only column count is too extreme
            (data.shape[0] > 3000000 and data.shape[1] > 3000)): # Special case for our 3.4M x 3.6K dataset
            logger.warning(f"Dataset dimensions too large for GPU basic transforms: {data.shape} with {available_gb:.1f}GB, using CPU fallback")
            return self._cpu_basic_fallback(data, feature_names)
            
        # With lots of memory, continue with GPU processing
        if data.shape[0] > 3000000 and available_gb >= 20.0:
            logger.info(f"Large dataset ({data.shape}), but sufficient GPU memory ({available_gb:.1f}GB), continuing with GPU basic transforms")
            
        # Use memory-aware check for large datasets
        if data.shape[0] > 1000000:
            # Large dataset but we have significant GPU memory
            if available_gb >= 24.0:
                # Have lots of memory - keep going with GPU
                logger.info(f"Large dataset, but high GPU memory ({available_gb:.1f}GB), continuing with GPU basic transforms")
            elif available_gb < 16.0:
                # Low memory for large dataset - use CPU
                logger.warning(f"Large dataset with limited GPU memory ({available_gb:.1f}GB), using CPU fallback for basic transforms")
                return self._cpu_basic_fallback(data, feature_names)
            
        # Add smarter element-based limit that depends on available memory
        # With over 20GB memory, we can handle much larger datasets for basic transforms
        if available_gb >= 20.0:
            # For large memory GPUs, use a much higher limit
            max_elements = int(6000000000)  # 6B elements for GPUs with lots of memory
            logger.info(f"High memory GPU detected ({available_gb:.1f}GB), setting max_elements to {max_elements}")
        else:
            # For standard GPUs, scale based on available memory
            base_limit = 1000000000  # 1B base limit (much higher than before)
            max_elements = int(base_limit * (available_gb / 16.0))
            
        if data.shape[0] * data.shape[1] > max_elements:
            logger.warning(f"Dataset too large for basic transforms: {data.shape[0] * data.shape[1]} elements (limit {max_elements}), using CPU fallback")
            return self._cpu_basic_fallback(data, feature_names)
        
        # Transfer to GPU
        gpu_data = self._to_gpu(data)
        n_samples, n_features = data.shape
        
        # Preallocate output arrays
        if cupy_available:
            import cupy as cp
            all_transforms = []
            transform_names = []
            
            # Batch processing for memory efficiency
            batch_size = min(50, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = gpu_data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized transforms (all at once)
                square_transforms = batch_data ** 2
                sqrt_transforms = cp.sqrt(cp.abs(batch_data))
                log_transforms = cp.log1p(cp.abs(batch_data))
                reciprocal_transforms = 1.0 / (cp.abs(batch_data) + 1e-8)
                
                # Stack transforms
                batch_transforms = cp.concatenate([
                    square_transforms,
                    sqrt_transforms, 
                    log_transforms,
                    reciprocal_transforms
                ], axis=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"gpu_sq_{name}",
                        f"gpu_sqrt_{name}",
                        f"gpu_log_{name}",
                        f"gpu_inv_{name}"
                    ])
            
            # Combine all batches
            result_gpu = cp.concatenate(all_transforms, axis=1)
            result = cp.asnumpy(result_gpu)
            
        elif torch_available:
            import torch
            all_transforms = []
            transform_names = []
            
            batch_size = min(50, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = gpu_data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized transforms
                square_transforms = batch_data ** 2
                sqrt_transforms = torch.sqrt(torch.abs(batch_data))
                log_transforms = torch.log1p(torch.abs(batch_data))
                reciprocal_transforms = 1.0 / (torch.abs(batch_data) + 1e-8)
                
                # Stack transforms
                batch_transforms = torch.cat([
                    square_transforms,
                    sqrt_transforms,
                    log_transforms,
                    reciprocal_transforms
                ], dim=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"gpu_sq_{name}",
                        f"gpu_sqrt_{name}",
                        f"gpu_log_{name}",
                        f"gpu_inv_{name}"
                    ])
            
            # Combine all batches
            result_gpu = torch.cat(all_transforms, dim=1)
            result = result_gpu.cpu().numpy()
            
        else:
            # CPU fallback with vectorization
            all_transforms = []
            transform_names = []
            
            batch_size = min(20, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized numpy transforms
                square_transforms = batch_data ** 2
                sqrt_transforms = np.sqrt(np.abs(batch_data))
                log_transforms = np.log1p(np.abs(batch_data))
                reciprocal_transforms = 1.0 / (np.abs(batch_data) + 1e-8)
                
                # Stack transforms
                batch_transforms = np.concatenate([
                    square_transforms,
                    sqrt_transforms,
                    log_transforms,
                    reciprocal_transforms
                ], axis=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"cpu_sq_{name}",
                        f"cpu_sqrt_{name}",
                        f"cpu_log_{name}",
                        f"cpu_inv_{name}"
                    ])
            
            result = np.concatenate(all_transforms, axis=1)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Basic transforms: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def _cpu_basic_fallback(self, data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """CPU fallback for basic transforms when GPU memory is insufficient"""
        logger.info(f"🖥️  CPU Basic Fallback: {data.shape}")
        start_time = time.time()
        
        n_samples, n_features = data.shape
        all_transforms = []
        transform_names = []
        
        # Process in smaller batches for CPU
        batch_size = min(50, n_features)
        
        for batch_start in range(0, n_features, batch_size):
            batch_end = min(batch_start + batch_size, n_features)
            batch_data = data[:, batch_start:batch_end]
            batch_names = feature_names[batch_start:batch_end]
            
            # Vectorized numpy transforms
            square_transforms = batch_data ** 2
            sqrt_transforms = np.sqrt(np.abs(batch_data))
            log_transforms = np.log1p(np.abs(batch_data))
            reciprocal_transforms = 1.0 / (np.abs(batch_data) + 1e-8)
            
            # Stack transforms
            batch_transforms = np.concatenate([
                square_transforms,
                sqrt_transforms,
                log_transforms,
                reciprocal_transforms
            ], axis=1)
            
            all_transforms.append(batch_transforms)
            
            # Generate names
            for name in batch_names:
                transform_names.extend([
                    f"cpu_sq_{name}",
                    f"cpu_sqrt_{name}",
                    f"cpu_log_{name}",
                    f"cpu_inv_{name}"
                ])
        
        result = np.concatenate(all_transforms, axis=1)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ CPU basic fallback: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def gpu_trigonometric_transforms(self, data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Ultra-fast trigonometric transformations on GPU
        
        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Names of input features
            
        Returns:
            Tuple of (transformed_data, new_feature_names)
        """
        logger.info(f"🚀 GPU Trigonometric Transforms: {data.shape}")
        start_time = time.time()
        
        # For extremely large datasets, just go to CPU immediately
        if data.shape[0] > 2000000 or data.shape[1] > 5000:
            logger.warning(f"Dataset too large for GPU trigonometric transforms: {data.shape}, using CPU fallback")
            # Fall back to CPU implementation
            data_normalized = np.tanh(data) * np.pi
            
            all_transforms = []
            transform_names = []
            
            batch_size = min(10, n_features)
            
            # Process only a subset of columns for very large datasets
            max_cols = min(100, data.shape[1])
            subset_names = feature_names[:max_cols]
            subset_data = data[:, :max_cols]
            
            for batch_start in range(0, max_cols, batch_size):
                batch_end = min(batch_start + batch_size, max_cols)
                batch_data = data_normalized[:, batch_start:batch_end]
                batch_names = subset_names[batch_start:batch_end]
                
                # Vectorized trigonometric transforms
                sin_transforms = np.sin(batch_data)
                cos_transforms = np.cos(batch_data)
                tan_transforms = np.tanh(batch_data)
                
                # Stack transforms
                batch_transforms = np.concatenate([
                    sin_transforms,
                    cos_transforms,
                    tan_transforms
                ], axis=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"cpu_sin_{name}",
                        f"cpu_cos_{name}",
                        f"cpu_tanh_{name}"
                    ])
            
            result = np.concatenate(all_transforms, axis=1)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ CPU trig fallback: {result.shape[1]} features in {elapsed:.2f}s")
            
            return result, transform_names
        
        # Add hard limit on dataset size
        if data.shape[0] * data.shape[1] > 200000000:  # 200M elements threshold
            logger.warning(f"Dataset too large for trig transforms: {data.shape[0] * data.shape[1]} elements, using CPU fallback")
            # Fall back to minimal CPU implementation
            max_cols = min(50, data.shape[1])
            subset_names = feature_names[:max_cols]
            subset_data = data[:, :max_cols]
            
            # Simple batch CPU implementation for large data
            result = np.sin(subset_data)  # Just use sine transform for simplicity
            transform_names = [f"cpu_sin_{name}" for name in subset_names]
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Simple CPU trig fallback: {result.shape[1]} features in {elapsed:.2f}s")
            
            return result, transform_names
            
        # Transfer to GPU
        gpu_data = self._to_gpu(data)
        n_samples, n_features = data.shape
        
        # Normalize data to [-π, π] for trigonometric functions
        if cupy_available:
            import cupy as cp
            # Normalize to [-π, π] range
            data_normalized = cp.tanh(gpu_data) * cp.pi
            
            all_transforms = []
            transform_names = []
            
            batch_size = min(40, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = data_normalized[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized trigonometric transforms
                sin_transforms = cp.sin(batch_data)
                cos_transforms = cp.cos(batch_data)
                tan_transforms = cp.tanh(batch_data)  # Use tanh to avoid infinities
                
                # Stack transforms
                batch_transforms = cp.concatenate([
                    sin_transforms,
                    cos_transforms,
                    tan_transforms
                ], axis=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"gpu_sin_{name}",
                        f"gpu_cos_{name}",
                        f"gpu_tanh_{name}"
                    ])
            
            # Combine all batches
            result_gpu = cp.concatenate(all_transforms, axis=1)
            result = cp.asnumpy(result_gpu)
            
        elif torch_available:
            import torch
            # Normalize to [-π, π] range
            data_normalized = torch.tanh(gpu_data) * torch.pi
            
            all_transforms = []
            transform_names = []
            
            batch_size = min(40, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = data_normalized[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized trigonometric transforms
                sin_transforms = torch.sin(batch_data)
                cos_transforms = torch.cos(batch_data)
                tan_transforms = torch.tanh(batch_data)
                
                # Stack transforms
                batch_transforms = torch.cat([
                    sin_transforms,
                    cos_transforms,
                    tan_transforms
                ], dim=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"gpu_sin_{name}",
                        f"gpu_cos_{name}",
                        f"gpu_tanh_{name}"
                    ])
            
            # Combine all batches
            result_gpu = torch.cat(all_transforms, dim=1)
            result = result_gpu.cpu().numpy()
            
        else:
            # CPU fallback with vectorization
            data_normalized = np.tanh(data) * np.pi
            
            all_transforms = []
            transform_names = []
            
            batch_size = min(20, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = data_normalized[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                # Vectorized trigonometric transforms
                sin_transforms = np.sin(batch_data)
                cos_transforms = np.cos(batch_data)
                tan_transforms = np.tanh(batch_data)
                
                # Stack transforms
                batch_transforms = np.concatenate([
                    sin_transforms,
                    cos_transforms,
                    tan_transforms
                ], axis=1)
                
                all_transforms.append(batch_transforms)
                
                # Generate names
                for name in batch_names:
                    transform_names.extend([
                        f"cpu_sin_{name}",
                        f"cpu_cos_{name}",
                        f"cpu_tanh_{name}"
                    ])
            
            result = np.concatenate(all_transforms, axis=1)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Trigonometric transforms: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def gpu_polynomial_transforms(self, data: np.ndarray, feature_names: List[str], max_degree: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Ultra-fast polynomial transformations on GPU
        
        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Names of input features
            max_degree: Maximum polynomial degree
            
        Returns:
            Tuple of (transformed_data, new_feature_names)
        """
        logger.info(f"🚀 GPU Polynomial Transforms: {data.shape}, degree {max_degree}")
        start_time = time.time()
        
        # For extremely large datasets, just go to CPU immediately
        if data.shape[0] > 2000000 or data.shape[1] > 5000:
            logger.warning(f"Dataset too large for GPU polynomial transforms: {data.shape}, using CPU fallback")
            # Fall back to CPU implementation with limited columns
            max_cols = min(50, data.shape[1])
            subset_names = feature_names[:max_cols]
            subset_data = data[:, :max_cols]
            
            all_transforms = []
            transform_names = []
            
            # Generate just degree 2 polynomials for simplicity
            poly_transforms = subset_data ** 2
            
            # Generate names
            transform_names = [f"cpu_poly2_{name}" for name in subset_names]
            
            elapsed = time.time() - start_time
            logger.info(f"✅ CPU polynomial fallback: {len(transform_names)} features in {elapsed:.2f}s")
            
            return poly_transforms, transform_names
        
        # Add hard limit on dataset size
        if data.shape[0] * data.shape[1] > 200000000:  # 200M elements threshold
            logger.warning(f"Dataset too large for polynomial transforms: {data.shape[0] * data.shape[1]} elements, using CPU fallback")
            # Fall back to minimal CPU implementation
            max_cols = min(30, data.shape[1])
            subset_names = feature_names[:max_cols]
            subset_data = data[:, :max_cols]
            
            # Simple squared transform only
            result = subset_data ** 2
            transform_names = [f"cpu_poly2_{name}" for name in subset_names]
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Simple CPU polynomial fallback: {len(transform_names)} features in {elapsed:.2f}s")
            
            return result, transform_names
            
        # Transfer to GPU
        gpu_data = self._to_gpu(data)
        n_samples, n_features = data.shape
        
        if cupy_available:
            import cupy as cp
            all_transforms = []
            transform_names = []
            
            batch_size = min(30, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = gpu_data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                batch_transforms_list = []
                
                # Generate polynomial features efficiently
                for degree in range(2, max_degree + 1):
                    poly_transforms = batch_data ** degree
                    batch_transforms_list.append(poly_transforms)
                
                # Stack all polynomial transforms for this batch
                if batch_transforms_list:
                    batch_transforms = cp.concatenate(batch_transforms_list, axis=1)
                    all_transforms.append(batch_transforms)
                    
                    # Generate names
                    for name in batch_names:
                        for degree in range(2, max_degree + 1):
                            transform_names.append(f"gpu_poly{degree}_{name}")
            
            # Combine all batches
            if all_transforms:
                result_gpu = cp.concatenate(all_transforms, axis=1)
                result = cp.asnumpy(result_gpu)
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
            
        elif torch_available:
            import torch
            all_transforms = []
            transform_names = []
            
            batch_size = min(30, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = gpu_data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                batch_transforms_list = []
                
                # Generate polynomial features efficiently
                for degree in range(2, max_degree + 1):
                    poly_transforms = batch_data ** degree
                    batch_transforms_list.append(poly_transforms)
                
                # Stack all polynomial transforms for this batch
                if batch_transforms_list:
                    batch_transforms = torch.cat(batch_transforms_list, dim=1)
                    all_transforms.append(batch_transforms)
                    
                    # Generate names
                    for name in batch_names:
                        for degree in range(2, max_degree + 1):
                            transform_names.append(f"gpu_poly{degree}_{name}")
            
            # Combine all batches
            if all_transforms:
                result_gpu = torch.cat(all_transforms, dim=1)
                result = result_gpu.cpu().numpy()
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
            
        else:
            # CPU fallback with vectorization
            all_transforms = []
            transform_names = []
            
            batch_size = min(20, n_features)
            
            for batch_start in range(0, n_features, batch_size):
                batch_end = min(batch_start + batch_size, n_features)
                batch_data = data[:, batch_start:batch_end]
                batch_names = feature_names[batch_start:batch_end]
                
                batch_transforms_list = []
                
                # Generate polynomial features efficiently
                for degree in range(2, max_degree + 1):
                    poly_transforms = batch_data ** degree
                    batch_transforms_list.append(poly_transforms)
                
                # Stack all polynomial transforms for this batch
                if batch_transforms_list:
                    batch_transforms = np.concatenate(batch_transforms_list, axis=1)
                    all_transforms.append(batch_transforms)
                    
                    # Generate names
                    for name in batch_names:
                        for degree in range(2, max_degree + 1):
                            transform_names.append(f"cpu_poly{degree}_{name}")
            
            # Combine all batches
            if all_transforms:
                result = np.concatenate(all_transforms, axis=1)
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Polynomial transforms: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def _clear_gpu_memory(self, force_full_clear=False):
        """
        Free up GPU memory to avoid OOM errors
        
        Args:
            force_full_clear: If True, forcefully clear all memory including cache
        """
        try:
            # First clear system memory with garbage collection
            import gc
            gc.collect()
            
            if cupy_available:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                
                if force_full_clear:
                    # Completely clear memory (more aggressive)
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    logger.info("Forcefully cleared all CuPy memory pools")
                else:
                    # Standard cleanup (less aggressive)
                    mempool.free_all_blocks()
                    logger.debug("Cleared CuPy memory pools")
            
            elif torch_available:
                import torch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache")
                
                if force_full_clear and hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    # More aggressive cleanup for PyTorch
                    torch.cuda.reset_peak_memory_stats()
                    logger.info("Reset PyTorch CUDA memory stats")
            
            # Check system memory status after clearing
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                process_memory_gb = memory_info.rss / (1024**3)
                
                system_memory = psutil.virtual_memory()
                system_available_gb = system_memory.available / (1024**3)
                
                # Log memory status after clearing
                if force_full_clear:
                    logger.info(f"After memory clear: Process using {process_memory_gb:.1f}GB, System has {system_available_gb:.1f}GB available")
                else:
                    logger.debug(f"After memory clear: Process using {process_memory_gb:.1f}GB, System has {system_available_gb:.1f}GB available")
            except (ImportError, Exception) as e:
                if force_full_clear:
                    logger.warning(f"Could not check memory after clearing: {e}")
        
        except Exception as e:
            logger.warning(f"Error clearing memory: {e}")
            # Continue even if clearing fails
            
    def _manage_memory(self, check_interval=5, threshold_gb=500):
        """
        Actively manage memory to prevent OOM errors
        
        Args:
            check_interval: How often to check memory (in operations)
            threshold_gb: Memory threshold in GB to trigger aggressive cleanup
        """
        # Only check periodically to avoid performance impact
        if not hasattr(self, '_memory_check_counter'):
            self._memory_check_counter = 0
        
        self._memory_check_counter += 1
        if self._memory_check_counter % check_interval != 0:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            process_memory_gb = memory_info.rss / (1024**3)
            
            system_memory = psutil.virtual_memory()
            system_available_gb = system_memory.available / (1024**3)
            system_used_percent = system_memory.percent
            
            # Log every 20 checks to avoid excessive logging
            if self._memory_check_counter % (check_interval * 4) == 0:
                logger.info(f"Memory status: Process using {process_memory_gb:.1f}GB, System at {system_used_percent:.1f}% ({system_available_gb:.1f}GB free)")
            
            # If system memory is getting low, take action
            if process_memory_gb > threshold_gb:
                logger.warning(f"Process memory usage ({process_memory_gb:.1f}GB) approaching threshold ({threshold_gb}GB), forcing memory cleanup")
                self._clear_gpu_memory(force_full_clear=True)
                return True
            
            # If system memory is very low (less than 10% free), force cleanup
            if system_available_gb < 60 or system_used_percent > 90:
                logger.warning(f"System memory critically low: {system_used_percent:.1f}% used ({system_available_gb:.1f}GB free), forcing memory cleanup")
                self._clear_gpu_memory(force_full_clear=True)
                return True
                
            # If system memory is moderately low (less than 20% free), do a normal cleanup
            elif system_available_gb < 120 or system_used_percent > 80:
                logger.info(f"System memory running low: {system_used_percent:.1f}% used ({system_available_gb:.1f}GB free), performing cleanup")
                self._clear_gpu_memory(force_full_clear=False)
                return True
                
        except (ImportError, Exception) as e:
            logger.debug(f"Memory management check failed: {e}")
        
        return False
            
    def _estimate_memory_usage(self, data_shape: Tuple[int, int], operation_type: str = "basic") -> float:
        """Estimate GPU memory usage in GB for given operation (realistic batch processing)"""
        rows, cols = data_shape
        
        # Reduce batch sizes for very large datasets
        if rows > 1000000:
            # For datasets with more than 1M rows, use smaller batch sizes
            basic_batch = 20
            trig_batch = 15
            poly_batch = 10
            inter_batch = 10
            pair_batch = 20  # Reduce pairs for interactions
        else:
            # For smaller datasets, use normal batch sizes
            basic_batch = 50
            trig_batch = 40
            poly_batch = 30
            inter_batch = 50
            pair_batch = 100
        
        if operation_type == "basic":
            # Processes in batches
            effective_cols = min(basic_batch, cols)
            base_size_gb = (rows * effective_cols * 4) / (1024**3)
            return base_size_gb * 5  # Input batch + 4 transforms
        elif operation_type == "trig":
            effective_cols = min(trig_batch, cols)
            base_size_gb = (rows * effective_cols * 4) / (1024**3)
            return base_size_gb * 4  # Input batch + 3 transforms
        elif operation_type == "poly":
            effective_cols = min(poly_batch, cols)
            base_size_gb = (rows * effective_cols * 4) / (1024**3)
            return base_size_gb * 3  # Input batch + 2 polynomial degrees
        elif operation_type == "interactions":
            effective_cols = min(inter_batch, cols)
            base_size_gb = (rows * effective_cols * 4) / (1024**3)
            
            # Memory for input subset + temporary results for batch
            # Calculate the actual memory needed for pairwise interactions
            # For very large datasets, use extremely conservative estimation
            
            # Number of potential feature pairs - keep this very small for large datasets
            if rows > 1000000:
                n_pairs = min(30, (effective_cols * (effective_cols - 1)) // 2)
            elif rows > 500000:
                n_pairs = min(50, (effective_cols * (effective_cols - 1)) // 2)
            else:
                n_pairs = min(100, (effective_cols * (effective_cols - 1)) // 2)
            
            # Memory for the interaction features: rows × pairs × 3 operations × 4 bytes per float32
            interactions_gb = (rows * n_pairs * 3 * 4) / (1024**3)
            
            # Add safety buffer for CUDA kernels and temporary allocations
            buffer_gb = base_size_gb * 1.2
            
            # The total memory requirement with hard cap - much more conservative than before
            total_gb = min(base_size_gb + interactions_gb + buffer_gb, 12.0)
            
            # For all datasets, cap at a very conservative level
            total_gb = min(total_gb, 12.0)  # Hard cap at 12GB regardless of dataset size
                
            logger.info(f"Interactions memory estimate: {base_size_gb:.2f}GB base + {interactions_gb:.2f}GB interactions + {buffer_gb:.2f}GB buffer = {total_gb:.2f}GB total")
            
            return total_gb
        else:
            effective_cols = min(basic_batch, cols)
            base_size_gb = (rows * effective_cols * 4) / (1024**3)
            return base_size_gb * 2
    
    def get_gpu_memory_usage(self) -> Tuple[float, float, float]:
        """
        Get current GPU memory usage
        
        Returns:
            Tuple of (used_gb, free_gb, total_gb)
        """
        if not self.gpu_available or not cupy_available:
            return 0.0, 0.0, 0.0
            
        try:
            import cupy as cp
            
            # For multi-GPU systems, check all devices
            gpu_count = cp.cuda.runtime.getDeviceCount()
            total_free_bytes = 0
            total_used_bytes = 0
            total_all_bytes = 0
            
            for i in range(gpu_count):
                device = cp.cuda.Device(i)
                mem_info = device.mem_info
                free_bytes = mem_info[0]
                total_bytes = mem_info[1]
                used_bytes = total_bytes - free_bytes
                
                total_free_bytes += free_bytes
                total_used_bytes += used_bytes
                total_all_bytes += total_bytes
            
            # Convert to GB
            free_gb = total_free_bytes / (1024**3)
            used_gb = total_used_bytes / (1024**3)
            total_gb = total_all_bytes / (1024**3)
            
            return used_gb, free_gb, total_gb
            
        except Exception as e:
            logger.warning(f"GPU memory usage check failed: {e}")
            return 0.0, 0.0, 0.0
            
    def _check_gpu_memory_availability(self, data: np.ndarray, operation_type: str = "basic") -> bool:
        """Check if GPU has enough memory for the operation"""
        if not self.gpu_available or not cupy_available:
            return False
        
        try:
            import cupy as cp
            
            # For multi-GPU systems (NVLink), check total memory across all devices
            try:
                # Get memory info across all devices
                used_gb, free_gb, total_gb = self.get_gpu_memory_usage()
                
                # For NVLink connected GPUs, consider combined memory
                # assuming we have proper peer access between devices
                gpu_count = cp.cuda.runtime.getDeviceCount()
                
                # If we have multiple GPUs, try to check if NVLink is working
                if gpu_count > 1:
                    try:
                        # Check if peer access is enabled (NVLink requirement)
                        nvlink_available = all(
                            cp.cuda.runtime.deviceCanAccessPeer(i, j)
                            for i in range(gpu_count) 
                            for j in range(gpu_count) 
                            if i != j
                        )
                        
                        if nvlink_available:
                            logger.info("NVLink detected - using combined GPU memory")
                            # We already have combined memory from get_gpu_memory_usage
                    except Exception as e:
                        logger.warning(f"Error checking NVLink status: {e}")
            except Exception as e:
                logger.warning(f"Error checking multi-GPU memory: {e}")
                # Fallback to single device if the above fails
                current_device = cp.cuda.Device()
                mem_info = current_device.mem_info
                free_bytes = mem_info[0]
                total_bytes = mem_info[1]
                used_bytes = total_bytes - free_bytes
                
                free_gb = free_bytes / (1024**3)
                used_gb = used_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
            
            # Log detailed memory usage
            logger.info(f"GPU memory: {used_gb:.1f}GB used, {free_gb:.1f}GB free, {total_gb:.1f}GB total")
                
            estimated_usage = self._estimate_memory_usage(data.shape, operation_type)
            
            # Use NVLink aware safety margin (lower for multi-GPU)
            gpu_count = cp.cuda.runtime.getDeviceCount()
            safety_margin = 1.1 if gpu_count > 1 else 1.2
            
            # Check if we have enough memory and log clear message
            result = free_gb > (estimated_usage * safety_margin)
            if result:
                logger.info(f"✅ Memory check for {operation_type}: Need {estimated_usage:.1f}GB × {safety_margin:.1f} = {estimated_usage*safety_margin:.1f}GB, Have {free_gb:.1f}GB free")
            else:
                logger.warning(f"❌ Memory check for {operation_type}: Need {estimated_usage:.1f}GB × {safety_margin:.1f} = {estimated_usage*safety_margin:.1f}GB, Have only {free_gb:.1f}GB free")
                
            return result
        except Exception as e:
            logger.warning(f"Memory availability check failed: {e}")
            return False

    def gpu_interaction_transforms(self, data: np.ndarray, feature_names: List[str], max_interactions: int = 2000) -> Tuple[np.ndarray, List[str]]:
        """
        Ultra-fast feature interaction transformations on GPU
        
        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Names of input features
            max_interactions: Maximum number of interaction features to generate
            
        Returns:
            Tuple of (transformed_data, new_feature_names)
        """
        logger.info(f"🚀 GPU Interaction Transforms: {data.shape}, max {max_interactions}")
        start_time = time.time()
        
        # Check GPU memory availability
        if not self._check_gpu_memory_availability(data, "interactions"):
            logger.warning(f"Insufficient GPU memory for interactions, using CPU fallback")
            return self._cpu_interaction_fallback(data, feature_names, max_interactions)
            
        try:
            # Get estimated memory requirements and set a hard limit
            memory_limit_gb = float(os.environ.get("GPU_MEMORY_LIMIT", "12.0"))  # Default to 12GB - much more conservative
            estimated_usage = self._estimate_memory_usage(data.shape, "interactions")
            
            # Get available GPU memory
            available_gb = 0.0
            try:
                if cupy_available:
                    import cupy as cp
                    mem_info = cp.cuda.Device(self.device_id).mem_info
                    available_gb = mem_info[0] / (1024**3)  # Free memory in GB
                    logger.info(f"Available GPU memory: {available_gb:.1f}GB")
                else:
                    # Default to a safe value
                    available_gb = 16.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory info: {e}, using default value")
                available_gb = 16.0
            
            # For extremely large datasets with limited GPU memory, go to CPU
            # Adjust threshold to make better use of available memory
            if ((data.shape[0] > 5000000 and available_gb < 20.0) or  # Very large datasets need lots of memory
                (data.shape[0] > 3000000 and available_gb < 12.0) or  # Large datasets need medium memory
                data.shape[1] > 10000):                              # Only column count is too extreme
                logger.warning(f"Dataset dimensions too large for available GPU memory: {data.shape} with {available_gb:.1f}GB, using CPU fallback")
                return self._cpu_interaction_fallback(data, feature_names, max_interactions)
                
            # With lots of memory, continue with GPU processing
            if data.shape[0] > 3000000 and available_gb >= 20.0:
                logger.info(f"Large dataset ({data.shape}), but sufficient GPU memory ({available_gb:.1f}GB), continuing with GPU processing")
                
            # Use memory-aware check for large datasets
            # Adapt based on available GPU memory
            if data.shape[0] > 1000000:
                # Large dataset but we have significant GPU memory
                if available_gb >= 32.0:
                    # Have lots of memory - keep going with GPU but reduce feature columns
                    max_interactions = min(max_interactions, 500)
                    logger.info(f"Large dataset, but high GPU memory ({available_gb:.1f}GB), continuing with GPU using reduced max_interactions={max_interactions}")
                elif available_gb >= 16.0:
                    # Medium memory - reduce features significantly
                    max_interactions = min(max_interactions, 300)
                    logger.info(f"Large dataset with medium GPU memory ({available_gb:.1f}GB), reducing max_interactions={max_interactions}")
                else:
                    # Low memory for large dataset - use CPU
                    logger.warning(f"Large dataset with limited GPU memory ({available_gb:.1f}GB), using CPU fallback")
                    return self._cpu_interaction_fallback(data, feature_names, max_interactions)
            
            # If estimation shows we might need more than limit, switch to CPU fallback
            if estimated_usage > memory_limit_gb * 0.9:  # Add extra safety margin
                logger.error(f"GPU Math Accelerator: Estimated memory usage {estimated_usage:.2f}GB approaches limit of {memory_limit_gb:.2f}GB")
                logger.info(f"🚀 Using basic GPU for feature interactions...")
                return self._cpu_interaction_fallback(data, feature_names, max_interactions)
            
            # Add smarter element-based limit that depends on available memory
            # With over 20GB memory, we can handle much larger datasets for interactions
            # Get the memory limit in GB
            memory_limit_gb = float(os.environ.get("GPU_MEMORY_LIMIT", "12.0"))
            
            if available_gb >= 20.0:
                # For large memory GPUs, scale based on available memory and respect the limit
                # Calculate based on the memory limit - use a very conservative estimate
                # For large datasets (like the 3.4M x 3.6K one in our application), we need a stricter limit
                max_elements = int(min(memory_limit_gb * 100000000, 2500000000))  # Cap at 2.5B elements regardless
                logger.info(f"High memory GPU detected ({available_gb:.1f}GB), setting max_elements for interactions to {max_elements} (limit: {memory_limit_gb}GB)")
            else:
                # For standard GPUs, scale based on available memory
                base_limit = 800000000  # 800M base limit - more conservative
                max_elements = int(min(base_limit * (available_gb / 16.0), memory_limit_gb * 300000000))
            
            # For large datasets, only consider rows and columns actually used for interactions
            # This avoids rejecting the entire dataset when we're only using a small subset
            max_features_for_pairs = min(100, data.shape[1])
            actual_elements = data.shape[0] * max_features_for_pairs
            
            if actual_elements > max_elements:
                logger.warning(f"Dataset too large for GPU: {actual_elements} used elements (from {max_features_for_pairs} columns), limit {max_elements}, using CPU fallback")
                return self._cpu_interaction_fallback(data, feature_names, max_interactions)
        except Exception as e:
            # If anything goes wrong with memory estimation, use CPU fallback
            logger.warning(f"Memory estimation error: {e}, using CPU fallback")
            return self._cpu_interaction_fallback(data, feature_names, max_interactions)
        
        # Transfer to GPU
        gpu_data = self._to_gpu(data)
        n_samples, n_features = data.shape
        
        # Calculate optimal feature pairs with memory-aware limits
        # Adjust limits based on data size to avoid memory explosion
        if n_samples > 1000000:  # Very large dataset
            max_features_for_pairs = min(5, n_features) 
            max_distance = 2  # Only pair with very nearby features
        elif n_samples > 500000:  # Large dataset
            max_features_for_pairs = min(10, n_features)
            max_distance = 3
        elif n_samples > 100000:  # Medium dataset
            max_features_for_pairs = min(15, n_features)
            max_distance = 5
        else:  # Small dataset
            max_features_for_pairs = min(20, n_features)
            max_distance = 10
            
        # Enforce GPU memory limit by reducing pairs if needed
        memory_limit_gb = float(os.environ.get("GPU_MEMORY_LIMIT", "12.0"))
        # For all GPUs, be very conservative with feature pairing
        max_features_for_pairs = min(max_features_for_pairs, int(memory_limit_gb / 2))
        max_distance = min(max_distance, int(memory_limit_gb / 4))
            
        logger.info(f"Feature interaction limits: max_features={max_features_for_pairs}, max_distance={max_distance}")
        
        feature_pairs = []
        
        for i in range(max_features_for_pairs):
            for j in range(i + 1, min(i + max_distance, max_features_for_pairs)):
                feature_pairs.append((i, j))
                if len(feature_pairs) >= max_interactions // 3:  # 3 operations per pair
                    break
            if len(feature_pairs) >= max_interactions // 3:
                break
                
        logger.info(f"Generated {len(feature_pairs)} feature pairs for interactions")
        
        if cupy_available:
            import cupy as cp
            all_transforms = []
            transform_names = []
            
            # Process pairs in smaller batches to avoid memory issues
            # Adjust batch size based on memory limit
            memory_limit_gb = float(os.environ.get("GPU_MEMORY_LIMIT", "12.0"))
            if memory_limit_gb < 12.0:
                batch_size = min(5, len(feature_pairs))  # Very small batches for limited memory
            elif memory_limit_gb < 20.0:
                batch_size = min(10, len(feature_pairs))  # Small batches
            else:
                batch_size = min(20, len(feature_pairs))  # Medium batches
                
            logger.info(f"Using batch size {batch_size} for feature interactions")
            
            for batch_start in range(0, len(feature_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(feature_pairs))
                batch_pairs = feature_pairs[batch_start:batch_end]
                
                batch_transforms_list = []
                
                for i, j in batch_pairs:
                    col1_data = gpu_data[:, i]
                    col2_data = gpu_data[:, j]
                    
                    # Three interaction types
                    add_result = col1_data + col2_data
                    mul_result = col1_data * col2_data
                    div_result = col1_data / (cp.abs(col2_data) + 1e-8)
                    
                    # Stack interactions
                    interaction_set = cp.stack([add_result, mul_result, div_result], axis=1)
                    batch_transforms_list.append(interaction_set)
                    
                    # Generate names
                    col1_name = feature_names[i]
                    col2_name = feature_names[j]
                    transform_names.extend([
                        f"gpu_add_{col1_name}_{col2_name}",
                        f"gpu_mul_{col1_name}_{col2_name}",
                        f"gpu_div_{col1_name}_{col2_name}"
                    ])
                
                # Combine batch
                if batch_transforms_list:
                    batch_transforms = cp.concatenate(batch_transforms_list, axis=1)
                    all_transforms.append(batch_transforms)
                    
                    # Clear GPU memory after each batch to avoid OOM
                    if batch_start % (batch_size * 2) == 0:  # Clear more frequently
                        self._clear_gpu_memory()
            
            # Combine all batches
            if all_transforms:
                try:
                    # Before the final concatenation, check memory availability and clear cache
                    self._clear_gpu_memory(force_full_clear=True)
                    
                    # Estimate memory needed for concatenation (rough estimate)
                    total_cols = sum(arr.shape[1] for arr in all_transforms)
                    estimated_concat_memory_gb = (data.shape[0] * total_cols * 4) / (1024**3) * 1.5  # Add 50% buffer
                    
                    # Get current memory usage
                    used_gb, free_gb, total_gb = self.get_gpu_memory_usage()
                    
                    if estimated_concat_memory_gb > free_gb * 0.9:  # Leave 10% margin
                        logger.warning(f"Estimated concatenation memory ({estimated_concat_memory_gb:.1f}GB) exceeds available GPU memory ({free_gb:.1f}GB)")
                        logger.warning("Falling back to CPU for concatenation to avoid OOM error")
                        
                        # Convert all GPU arrays to CPU before concatenation
                        cpu_arrays = []
                        for arr in all_transforms:
                            cpu_arrays.append(cp.asnumpy(arr))
                            # Clear each GPU array after converting to save memory
                            del arr
                        
                        # Force memory cleanup
                        self._clear_gpu_memory(force_full_clear=True)
                        
                        # Concatenate on CPU
                        result = np.concatenate(cpu_arrays, axis=1)
                    else:
                        # Proceed with GPU concatenation
                        result_gpu = cp.concatenate(all_transforms, axis=1)
                        result = cp.asnumpy(result_gpu)
                except Exception as e:
                    logger.error(f"GPU Math Accelerator failed: {e}, falling back to basic GPU interactions")
                    # Try one more aggressive cleanup to recover from OOM
                    self._clear_gpu_memory(force_full_clear=True)
                    
                    # Try a more robust fallback: convert to CPU arrays individually
                    try:
                        logger.info("Attempting CPU fallback with existing computed arrays...")
                        cpu_arrays = []
                        for arr in all_transforms:
                            if isinstance(arr, cp.ndarray):
                                cpu_arrays.append(cp.asnumpy(arr))
                            else:
                                cpu_arrays.append(arr)
                        
                        # Force memory cleanup
                        self._clear_gpu_memory(force_full_clear=True)
                        
                        # Concatenate on CPU
                        result = np.concatenate(cpu_arrays, axis=1)
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed: {e2}, reverting to basic CPU interactions")
                        return self._cpu_interaction_fallback(data, feature_names, max_interactions)
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
            
        elif torch_available:
            import torch
            all_transforms = []
            transform_names = []
            
            # Process pairs in smaller batches to avoid memory issues
            # Adjust batch size based on memory limit
            memory_limit_gb = float(os.environ.get("GPU_MEMORY_LIMIT", "12.0"))
            if memory_limit_gb < 12.0:
                batch_size = min(5, len(feature_pairs))  # Very small batches for limited memory
            elif memory_limit_gb < 20.0:
                batch_size = min(10, len(feature_pairs))  # Small batches
            else:
                batch_size = min(20, len(feature_pairs))  # Medium batches
                
            logger.info(f"Using batch size {batch_size} for feature interactions (PyTorch)")
            
            for batch_start in range(0, len(feature_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(feature_pairs))
                batch_pairs = feature_pairs[batch_start:batch_end]
                
                batch_transforms_list = []
                
                for i, j in batch_pairs:
                    col1_data = gpu_data[:, i]
                    col2_data = gpu_data[:, j]
                    
                    # Three interaction types
                    add_result = col1_data + col2_data
                    mul_result = col1_data * col2_data
                    div_result = col1_data / (torch.abs(col2_data) + 1e-8)
                    
                    # Stack interactions
                    interaction_set = torch.stack([add_result, mul_result, div_result], dim=1)
                    batch_transforms_list.append(interaction_set)
                    
                    # Generate names
                    col1_name = feature_names[i]
                    col2_name = feature_names[j]
                    transform_names.extend([
                        f"gpu_add_{col1_name}_{col2_name}",
                        f"gpu_mul_{col1_name}_{col2_name}",
                        f"gpu_div_{col1_name}_{col2_name}"
                    ])
                
                # Combine batch
                if batch_transforms_list:
                    batch_transforms = torch.cat(batch_transforms_list, dim=1)
                    all_transforms.append(batch_transforms)
                    
                    # Clear GPU memory after each batch to avoid OOM
                    if batch_start % (batch_size * 2) == 0:  # Clear more frequently
                        self._clear_gpu_memory()
            
            # Combine all batches
            if all_transforms:
                try:
                    # Clear memory before concatenation
                    self._clear_gpu_memory(force_full_clear=True)
                    
                    # Estimate memory for concatenation
                    total_cols = sum(arr.shape[1] for arr in all_transforms)
                    estimated_concat_memory_gb = (data.shape[0] * total_cols * 4) / (1024**3) * 1.5  # Add 50% buffer
                    
                    # Approximate memory check for PyTorch
                    try:
                        import torch
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        free_gb = free_memory / (1024**3)
                        
                        if estimated_concat_memory_gb > free_gb * 0.9:  # Leave 10% margin
                            logger.warning(f"Estimated concatenation memory ({estimated_concat_memory_gb:.1f}GB) exceeds available GPU memory ({free_gb:.1f}GB)")
                            logger.warning("Falling back to CPU for concatenation to avoid OOM error")
                            
                            # Move arrays to CPU one by one
                            cpu_arrays = []
                            for arr in all_transforms:
                                cpu_arrays.append(arr.cpu().numpy())
                                # Clear GPU memory
                                del arr
                                
                            # Force cleanup
                            torch.cuda.empty_cache()
                            
                            # Concatenate on CPU
                            result = np.concatenate(cpu_arrays, axis=1)
                        else:
                            # Proceed with GPU concatenation
                            result_gpu = torch.cat(all_transforms, dim=1)
                            result = result_gpu.cpu().numpy()
                    except Exception:
                        # If memory check fails, just try the concatenation
                        result_gpu = torch.cat(all_transforms, dim=1)
                        result = result_gpu.cpu().numpy()
                        
                except Exception as e:
                    logger.error(f"GPU Math Accelerator failed: {e}, falling back to basic GPU interactions")
                    # Try cleanup
                    self._clear_gpu_memory(force_full_clear=True)
                    
                    # Try a more robust fallback
                    try:
                        logger.info("Attempting CPU fallback with existing computed arrays...")
                        cpu_arrays = []
                        for arr in all_transforms:
                            if torch.is_tensor(arr):
                                cpu_arrays.append(arr.cpu().numpy())
                            else:
                                cpu_arrays.append(arr)
                        
                        # Force cleanup
                        torch.cuda.empty_cache()
                        
                        # Concatenate on CPU
                        result = np.concatenate(cpu_arrays, axis=1)
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed: {e2}, reverting to basic CPU interactions")
                        return self._cpu_interaction_fallback(data, feature_names, max_interactions)
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
            
        else:
            # CPU fallback with vectorization
            all_transforms = []
            transform_names = []
            
            # Process pairs in smaller batches for CPU
            batch_size = min(50, len(feature_pairs))
            
            for batch_start in range(0, len(feature_pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(feature_pairs))
                batch_pairs = feature_pairs[batch_start:batch_end]
                
                batch_transforms_list = []
                
                for i, j in batch_pairs:
                    col1_data = data[:, i]
                    col2_data = data[:, j]
                    
                    # Three interaction types
                    add_result = col1_data + col2_data
                    mul_result = col1_data * col2_data
                    div_result = col1_data / (np.abs(col2_data) + 1e-8)
                    
                    # Stack interactions
                    interaction_set = np.column_stack([add_result, mul_result, div_result])
                    batch_transforms_list.append(interaction_set)
                    
                    # Generate names
                    col1_name = feature_names[i]
                    col2_name = feature_names[j]
                    transform_names.extend([
                        f"cpu_add_{col1_name}_{col2_name}",
                        f"cpu_mul_{col1_name}_{col2_name}",
                        f"cpu_div_{col1_name}_{col2_name}"
                    ])
                
                # Combine batch
                if batch_transforms_list:
                    batch_transforms = np.concatenate(batch_transforms_list, axis=1)
                    all_transforms.append(batch_transforms)
            
            # Combine all batches
            if all_transforms:
                result = np.concatenate(all_transforms, axis=1)
            else:
                result = np.array([]).reshape(n_samples, 0)
                transform_names = []
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Interaction transforms: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def _cpu_interaction_fallback(self, data: np.ndarray, feature_names: List[str], max_interactions: int) -> Tuple[np.ndarray, List[str]]:
        """CPU fallback for interaction transforms when GPU memory is insufficient"""
        logger.info(f"🖥️  CPU Interaction Fallback: {data.shape}, max {max_interactions}")
        start_time = time.time()
        
        n_samples, n_features = data.shape
        
        # Calculate optimal feature pairs for CPU processing (smaller subset)
        max_features_for_pairs = min(20, n_features)  # Much smaller for CPU
        feature_pairs = []
        
        for i in range(max_features_for_pairs):
            for j in range(i + 1, min(i + 10, max_features_for_pairs)):
                feature_pairs.append((i, j))
                if len(feature_pairs) >= max_interactions // 3:
                    break
            if len(feature_pairs) >= max_interactions // 3:
                break
        
        all_transforms = []
        transform_names = []
        
        # Process pairs in small batches for CPU
        batch_size = min(20, len(feature_pairs))
        
        for batch_start in range(0, len(feature_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(feature_pairs))
            batch_pairs = feature_pairs[batch_start:batch_end]
            
            batch_transforms_list = []
            
            for i, j in batch_pairs:
                col1_data = data[:, i]
                col2_data = data[:, j]
                
                # Three interaction types
                add_result = col1_data + col2_data
                mul_result = col1_data * col2_data
                div_result = col1_data / (np.abs(col2_data) + 1e-8)
                
                # Stack interactions
                interaction_set = np.column_stack([add_result, mul_result, div_result])
                batch_transforms_list.append(interaction_set)
                
                # Generate names
                col1_name = feature_names[i]
                col2_name = feature_names[j]
                transform_names.extend([
                    f"cpu_add_{col1_name}_{col2_name}",
                    f"cpu_mul_{col1_name}_{col2_name}",
                    f"cpu_div_{col1_name}_{col2_name}"
                ])
            
            # Combine batch
            if batch_transforms_list:
                batch_transforms = np.concatenate(batch_transforms_list, axis=1)
                all_transforms.append(batch_transforms)
        
        # Combine all batches
        if all_transforms:
            result = np.concatenate(all_transforms, axis=1)
        else:
            result = np.array([]).reshape(n_samples, 0)
            transform_names = []
        
        elapsed = time.time() - start_time
        logger.info(f"✅ CPU interaction fallback: {result.shape[1]} features in {elapsed:.2f}s")
        
        return result, transform_names
    
    def generate_all_math_transforms(self, data: np.ndarray, feature_names: List[str], 
                                   include_basic: bool = True,
                                   include_trig: bool = True,
                                   include_poly: bool = True,
                                   include_interactions: bool = True,
                                   max_poly_degree: int = 2,  # Reduce default to degree 2
                                   max_interactions: int = 1000,
                                   include_random_baselines: bool = True,
                                   batch_size: int = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate all mathematical transformations in one optimized call
        
        Args:
            data: Input data array (n_samples, n_features)
            feature_names: Names of input features
            include_basic: Include basic transforms (square, sqrt, log, reciprocal)
            include_trig: Include trigonometric transforms
            include_poly: Include polynomial transforms
            include_interactions: Include feature interactions
            max_poly_degree: Maximum polynomial degree
            max_interactions: Maximum number of interaction features
            include_random_baselines: Include random baseline features
            
        Returns:
            Tuple of (all_transformed_data, all_feature_names)
        """
        logger.info(f"🚀 GPU All Math Transforms: {data.shape}")
        total_start = time.time()
        
        # Check and manage system memory before starting
        self._manage_memory(check_interval=1, threshold_gb=400)
        
        # Get available GPU memory for intelligent decision
        available_gb = 0.0
        try:
            if cupy_available:
                import cupy as cp
                mem_info = cp.cuda.Device(self.device_id).mem_info
                available_gb = mem_info[0] / (1024**3)  # Free memory in GB
                logger.info(f"Available GPU memory for math transforms: {available_gb:.1f}GB")
            else:
                # Default to a safe value
                available_gb = 12.0
        except Exception as e:
            logger.debug(f"Error getting GPU memory info: {e}, using default value")
            available_gb = 12.0
        
        # Get system memory info
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            system_available_gb = system_memory.available / (1024**3)
            system_used_percent = system_memory.percent
            logger.info(f"System memory: {system_available_gb:.1f}GB available, {system_used_percent:.1f}% used")
        except (ImportError, Exception):
            system_available_gb = 400  # Default safe value
            system_used_percent = 50
        
        # For extremely large datasets with very limited memory, use CPU and minimal features
        if ((data.shape[0] > 5000000 and available_gb < 16.0) or   # Very large datasets with limited GPU memory
            (system_available_gb < 100) or                          # System memory is limited
            (system_used_percent > 75) or                           # System memory usage already high
            data.shape[1] > 10000):                                 # Extremely high column count
            logger.warning(f"Dataset or memory constraints too large: {data.shape} with {available_gb:.1f}GB GPU and {system_available_gb:.1f}GB system memory")
            logger.warning("Using minimal CPU subset for feature generation")
            
            # Use only a small subset of columns
            max_cols = min(30, data.shape[1])
            subset_names = feature_names[:max_cols]
            subset_data = data[:, :max_cols]
            
            # Generate just a few simple transforms (squared, sqrt, log)
            sq_transforms = subset_data ** 2
            sqrt_transforms = np.sqrt(np.abs(subset_data))
            log_transforms = np.log1p(np.abs(subset_data))
            
            # Generate 3 random baseline features
            if include_random_baselines:
                np.random.seed(42)  # Ensure reproducibility
                # Calculate dataset statistics
                data_mean = np.nanmean(subset_data)
                data_std = np.nanstd(subset_data)
                
                # Generate random features with matching statistics
                random_features = np.random.normal(data_mean, data_std, size=(subset_data.shape[0], 3))
                
                # Stack all transforms including random baselines
                result = np.column_stack([sq_transforms, sqrt_transforms, log_transforms, random_features])
                
                # Generate names
                result_names = []
                for name in subset_names:
                    result_names.extend([
                        f"cpu_sq_{name}",
                        f"cpu_sqrt_{name}",
                        f"cpu_log_{name}"
                    ])
                
                # Add random baseline names
                result_names.extend(["random_baseline_1", "random_baseline_2", "random_baseline_3"])
            else:
                # Stack transforms without random baselines
                result = np.column_stack([sq_transforms, sqrt_transforms, log_transforms])
                
                # Generate names
                result_names = []
                for name in subset_names:
                    result_names.extend([
                        f"cpu_sq_{name}",
                        f"cpu_sqrt_{name}",
                        f"cpu_log_{name}"
                    ])
            
            elapsed = time.time() - total_start
            logger.info(f"✅ Minimal CPU transforms: {len(result_names)} features in {elapsed:.2f}s")
            
            # Clean up memory
            del sq_transforms
            del sqrt_transforms
            del log_transforms
            import gc
            gc.collect()
            
            return result, result_names
        
        # For large datasets, use batched approach rather than disabling features completely
        if data.shape[0] > 1000000 or system_available_gb < 200:
            include_poly = False  # Disable polynomial transforms as they're less important
            include_trig = False  # Disable trigonometric transforms as they're less important
            
            # Be more conservative with extremely wide datasets
            if data.shape[1] > 3000:
                # For our 3.4M x 3.6K dataset, be extremely conservative
                max_interactions = max(50, min(100, max_interactions // 10))
                logger.info(f"Extremely wide dataset detected ({data.shape[1]} columns), limiting to {max_interactions} interactions")
            else:
                # Keep a reasonable number of interactions - never zero!
                max_interactions = max(300, min(1000, max_interactions // 3))
            
            logger.info(f"Large dataset detected: using batched processing with {max_interactions} interactions")
            
            # If batch_size is specified, determine whether to process in batches
            if batch_size is not None:
                data_rows = data.shape[0]
                data_size_gb = data.nbytes / (1024**3)
                
                if data_size_gb > 4.0:
                    logger.info(f"Very large dataset detected ({data_size_gb:.2f}GB), will process in batches of {batch_size} rows")
                    
                    # Process in batches
                    all_batch_results = []
                    all_batch_names = []
                    
                    # Process first batch to get feature names
                    first_batch = data[:batch_size]
                    batch_result, feature_names_from_batch = self.generate_all_math_transforms(
                        first_batch, feature_names,
                        include_basic=include_basic,
                        include_trig=include_trig,
                        include_poly=include_poly,
                        include_interactions=include_interactions,
                        max_poly_degree=max_poly_degree,
                        max_interactions=max_interactions,
                        include_random_baselines=include_random_baselines,
                        batch_size=None  # Prevent recursion
                    )
                    
                    all_batch_results.append(batch_result)
                    all_batch_names = feature_names_from_batch
                    
                    # Process remaining batches
                    for i in range(1, (data_rows + batch_size - 1) // batch_size):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, data_rows)
                        
                        logger.info(f"Processing batch {i+1}/{(data_rows + batch_size - 1) // batch_size}, rows {start_idx}-{end_idx}")
                        
                        # Clear memory before processing next batch
                        self._clear_gpu_memory(force_full_clear=True)
                        
                        # Process this batch
                        batch_data = data[start_idx:end_idx]
                        batch_result, _ = self.generate_all_math_transforms(
                            batch_data, feature_names,
                            include_basic=include_basic,
                            include_trig=include_trig,
                            include_poly=include_poly,
                            include_interactions=include_interactions,
                            max_poly_degree=max_poly_degree,
                            max_interactions=max_interactions,
                            include_random_baselines=include_random_baselines,
                            batch_size=None  # Prevent recursion
                        )
                        
                        all_batch_results.append(batch_result)
                    
                    # Combine results
                    logger.info(f"Combining results from {len(all_batch_results)} batches")
                    final_result = np.vstack(all_batch_results)
                    
                    return final_result, all_batch_names
        
        all_results = []
        all_names = []
        
        # Create 3 random baseline features with matching statistics
        if include_random_baselines:
            logger.info("Generating random baseline features for comparison")
            np.random.seed(42)  # Ensure reproducibility
            
            # Calculate statistics from a sample of the data to save memory
            sample_size = min(100000, data.shape[0])
            sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[sample_indices]
            
            # Calculate mean and std, handling potential NaN values
            data_mean = np.nanmean(sample_data)
            data_std = np.nanstd(sample_data)
            
            logger.info(f"Dataset statistics for random baselines: mean={data_mean:.4f}, std={data_std:.4f}")
            
            # Generate random features with matching statistics
            random_features = np.random.normal(data_mean, data_std, size=(data.shape[0], 3))
            random_names = ["random_baseline_1", "random_baseline_2", "random_baseline_3"]
            
            # Add to results
            all_results.append(random_features)
            all_names.extend(random_names)
            
            # Clean up memory
            del sample_data
            self._manage_memory(check_interval=1, threshold_gb=400)
        
        # Process different feature types with memory management between each step
        if include_basic:
            basic_data, basic_names = self.gpu_basic_transforms(data, feature_names)
            if basic_data.shape[1] > 0:
                all_results.append(basic_data)
                all_names.extend(basic_names)
            
            # Manage memory after each major operation
            self._manage_memory(check_interval=1, threshold_gb=400)
        
        if include_trig:
            trig_data, trig_names = self.gpu_trigonometric_transforms(data, feature_names)
            if trig_data.shape[1] > 0:
                all_results.append(trig_data)
                all_names.extend(trig_names)
            
            # Manage memory
            self._manage_memory(check_interval=1, threshold_gb=400)
        
        if include_poly:
            poly_data, poly_names = self.gpu_polynomial_transforms(data, feature_names, max_poly_degree)
            if poly_data.shape[1] > 0:
                all_results.append(poly_data)
                all_names.extend(poly_names)
            
            # Manage memory
            self._manage_memory(check_interval=1, threshold_gb=400)
        
        if include_interactions:
            int_data, int_names = self.gpu_interaction_transforms(data, feature_names, max_interactions)
            if int_data.shape[1] > 0:
                all_results.append(int_data)
                all_names.extend(int_names)
            
            # Manage memory
            self._manage_memory(check_interval=1, threshold_gb=400)
        
        # Combine all results with careful memory management
        if all_results:
            try:
                logger.info(f"Concatenating {len(all_results)} result arrays")
                
                # Force memory cleanup before large concatenation
                self._clear_gpu_memory(force_full_clear=True)
                
                # Check system memory before concatenation
                try:
                    import psutil
                    system_memory = psutil.virtual_memory()
                    system_available_gb = system_memory.available / (1024**3)
                    
                    # If system memory is critically low, process in batches
                    if system_available_gb < 60:
                        logger.warning(f"Critical system memory ({system_available_gb:.1f}GB), using batch concatenation")
                        
                        # Process in batches of 2 arrays at a time
                        temp_result = all_results[0]
                        for i in range(1, len(all_results)):
                            temp_result = np.concatenate([temp_result, all_results[i]], axis=1)
                            # Clear the used array to free memory
                            all_results[i] = None
                            # Force GC after each concatenation
                            import gc
                            gc.collect()
                        
                        final_result = temp_result
                    else:
                        # Standard concatenation
                        final_result = np.concatenate(all_results, axis=1)
                except (ImportError, Exception):
                    # Standard concatenation if we can't check memory
                    final_result = np.concatenate(all_results, axis=1)
            except MemoryError as me:
                logger.error(f"Memory error during concatenation: {me}")
                logger.warning("Attempting batch concatenation as fallback")
                
                try:
                    # Process in batches of 2 arrays at a time as a fallback
                    temp_result = all_results[0]
                    for i in range(1, len(all_results)):
                        temp_result = np.concatenate([temp_result, all_results[i]], axis=1)
                        # Clear the used array to free memory
                        all_results[i] = None
                        # Force GC after each concatenation
                        import gc
                        gc.collect()
                    
                    final_result = temp_result
                except Exception as e2:
                    logger.error(f"Batch concatenation also failed: {e2}")
                    logger.warning("Returning only the first result array")
                    final_result = all_results[0]
                    all_names = all_names[:all_results[0].shape[1]]
            except Exception as e:
                logger.error(f"Error during concatenation: {e}")
                # Fallback to first result if concatenation fails
                if len(all_results) > 0:
                    logger.warning("Returning only the first result array")
                    final_result = all_results[0]
                    all_names = all_names[:all_results[0].shape[1]]
                else:
                    final_result = np.array([]).reshape(data.shape[0], 0)
        else:
            final_result = np.array([]).reshape(data.shape[0], 0)
        
        # Final memory cleanup
        self._clear_gpu_memory(force_full_clear=True)
        
        total_elapsed = time.time() - total_start
        logger.info(f"🎉 All Math Transforms Complete: {final_result.shape[1]} features in {total_elapsed:.2f}s")
        
        return final_result, all_names


if __name__ == "__main__":
    # Test the GPU math accelerator
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GPU Math Accelerator')
    parser.add_argument('--test-size', type=int, default=10000, help='Test data size')
    parser.add_argument('--test-features', type=int, default=100, help='Number of test features')
    parser.add_argument('--benchmark', action='store_true', help='Run CPU vs GPU benchmark')
    
    args = parser.parse_args()
    
    # Create test data
    logger.info(f"Creating test data: {args.test_size} x {args.test_features}")
    np.random.seed(42)
    test_data = np.random.randn(args.test_size, args.test_features).astype(np.float32)
    test_names = [f"feature_{i}" for i in range(args.test_features)]
    
    # Initialize accelerator
    accelerator = GPUMathAccelerator()
    
    # Test all transforms
    start_time = time.time()
    result_data, result_names = accelerator.generate_all_math_transforms(
        test_data, test_names,
        max_poly_degree=2,
        max_interactions=1000
    )
    
    total_time = time.time() - start_time
    
    logger.info(f"🎉 Test completed!")
    logger.info(f"Input: {test_data.shape}")
    logger.info(f"Output: {result_data.shape}")
    logger.info(f"Generated {len(result_names)} features in {total_time:.2f}s")
    logger.info(f"Throughput: {result_data.shape[1] / total_time:.0f} features/second")
    
    if args.benchmark and not accelerator.gpu_available:
        logger.info("GPU not available for benchmark comparison")
    elif args.benchmark:
        logger.info("🏁 Running CPU vs GPU benchmark...")
        
        # CPU baseline (using basic numpy operations)
        start_cpu = time.time()
        cpu_basic = test_data ** 2  # Just square transform for comparison
        cpu_time = time.time() - start_cpu
        
        # GPU equivalent
        start_gpu = time.time()
        gpu_basic, _ = accelerator.gpu_basic_transforms(test_data[:, :10], test_names[:10])
        gpu_time = time.time() - start_gpu
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        logger.info(f"CPU time: {cpu_time:.3f}s")
        logger.info(f"GPU time: {gpu_time:.3f}s") 
        logger.info(f"Speedup: {speedup:.1f}x")