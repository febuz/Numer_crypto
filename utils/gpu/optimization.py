#!/usr/bin/env python3
"""
GPU optimization utilities for multi-GPU training.

This module provides functions for distributing workloads across multiple GPUs,
parallelizing model training, and optimizing memory usage on GPUs.
"""
import os
import sys
import logging
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from utils.gpu.detection import get_available_gpus, select_best_gpu
from config.settings import HARDWARE_CONFIG

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

def distribute_data_to_gpus(data: np.ndarray, labels: Optional[np.ndarray] = None) -> List[Dict[str, np.ndarray]]:
    """
    Distribute data evenly across available GPUs.
    
    Args:
        data (np.ndarray): Input data array
        labels (np.ndarray, optional): Input labels array
        
    Returns:
        List[Dict[str, np.ndarray]]: List of data chunks for each GPU
    """
    # Get available GPUs
    gpus = get_available_gpus()
    num_gpus = len(gpus)
    
    if num_gpus == 0:
        logger.warning("No GPUs available. Cannot distribute data.")
        return [{'data': data, 'labels': labels, 'gpu_id': None}]
    
    logger.info(f"Distributing data across {num_gpus} GPUs")
    
    # Calculate chunk size
    n_samples = data.shape[0]
    chunk_size = n_samples // num_gpus
    remainder = n_samples % num_gpus
    
    # Create data chunks
    data_chunks = []
    start_idx = 0
    
    for i, gpu in enumerate(gpus):
        # Adjust chunk size to account for remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # Create chunk
        chunk = {
            'data': data[start_idx:end_idx],
            'gpu_id': gpu['index']
        }
        
        # Add labels if provided
        if labels is not None:
            chunk['labels'] = labels[start_idx:end_idx]
        
        data_chunks.append(chunk)
        start_idx = end_idx
    
    # Log distribution
    for i, chunk in enumerate(data_chunks):
        if labels is not None:
            logger.info(f"GPU {chunk['gpu_id']}: {chunk['data'].shape[0]} samples")
        else:
            logger.info(f"GPU {chunk['gpu_id']}: {chunk['data'].shape[0]} samples")
    
    return data_chunks

def parallel_gpu_function(func: Callable, data_chunks: List[Dict[str, Any]], **kwargs) -> List[Any]:
    """
    Run a function in parallel on multiple GPUs.
    
    Args:
        func (Callable): Function to run on each GPU
        data_chunks (List[Dict]): List of data chunks for each GPU
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List[Any]: List of results from each GPU
    """
    # Check if we have any GPUs
    if not data_chunks or all(chunk.get('gpu_id') is None for chunk in data_chunks):
        logger.warning("No GPU chunks available. Running on CPU instead.")
        # Run on CPU
        return [func(data_chunks[0], **kwargs)]
    
    # Function to run on each GPU
    def gpu_worker(chunk, queue, worker_id, **kwargs):
        try:
            # Set CUDA_VISIBLE_DEVICES to use only this GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(chunk['gpu_id'])
            logger.info(f"Worker {worker_id} using GPU {chunk['gpu_id']}")
            
            # Run the function
            result = func(chunk, **kwargs)
            
            # Put the result in the queue
            queue.put((worker_id, result))
            
        except Exception as e:
            logger.error(f"Error in GPU worker {worker_id}: {e}")
            queue.put((worker_id, None))
    
    # Create a queue for results
    queue = mp.Queue()
    
    # Create processes for each GPU
    processes = []
    for i, chunk in enumerate(data_chunks):
        if chunk.get('gpu_id') is not None:
            p = mp.Process(target=gpu_worker, args=(chunk, queue, i), kwargs=kwargs)
            processes.append(p)
    
    # Start processes
    for p in processes:
        p.start()
    
    # Get results
    results = [None] * len(processes)
    for _ in range(len(processes)):
        worker_id, result = queue.get()
        results[worker_id] = result
    
    # Join processes
    for p in processes:
        p.join()
    
    return results

def get_multi_gpu_model_params(model_type: str, num_gpus: Optional[int] = None) -> Dict[str, Any]:
    """
    Get model parameters optimized for multi-GPU training.
    
    Args:
        model_type (str): Model type ('xgboost', 'lightgbm', 'h2o')
        num_gpus (int, optional): Number of GPUs to use (if None, use all available)
        
    Returns:
        Dict[str, Any]: Model parameters for multi-GPU training
    """
    # Get available GPUs
    gpus = get_available_gpus()
    
    if not gpus:
        logger.warning("No GPUs available. Using CPU parameters.")
        # Return CPU parameters
        if model_type.lower() == 'xgboost':
            return {
                'tree_method': 'hist',
                'n_jobs': -1,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif model_type.lower() == 'lightgbm':
            return {
                'device': 'cpu',
                'n_jobs': -1,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 255,
                'max_depth': 8,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5
            }
        elif model_type.lower() == 'h2o':
            return {
                'backend': 'cpu',
                'nthreads': -1,
                'max_runtime_secs': 3600,
                'max_models': 20,
                'seed': 42
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # Determine number of GPUs to use
    if num_gpus is None:
        num_gpus = len(gpus)
    else:
        num_gpus = min(num_gpus, len(gpus))
    
    logger.info(f"Configuring for {num_gpus} GPUs")
    
    # Get GPU IDs
    gpu_ids = [gpu['index'] for gpu in gpus[:num_gpus]]
    
    # Set model-specific parameters
    if model_type.lower() == 'xgboost':
        # For XGBoost
        if num_gpus == 1:
            # Single GPU
            return {
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': gpu_ids[0],
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 12,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:
            # Multi-GPU
            return {
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'n_gpus': num_gpus,
                'gpu_id': 0,  # Will be overridden in distributed training
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 12,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
    
    elif model_type.lower() == 'lightgbm':
        # For LightGBM
        return {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,  # Will be set per worker
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 255,
            'max_depth': 12,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
    
    elif model_type.lower() == 'h2o':
        # For H2O
        return {
            'backend': 'gpu',
            'gpu_id': gpu_ids,
            'nthreads': -1,
            'max_runtime_secs': 3600,
            'max_models': 20,
            'seed': 42
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def optimize_cuda_memory_usage(reserve_memory_fraction: float = 0.1) -> None:
    """
    Optimize CUDA memory usage by setting TensorFlow/PyTorch memory growth.
    
    Args:
        reserve_memory_fraction (float): Fraction of GPU memory to reserve
    """
    # Try to set TensorFlow memory growth
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # Set memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optionally set memory limit
                if reserve_memory_fraction > 0:
                    memory_limit = int(HARDWARE_CONFIG.get('gpu_memory', '24g').replace('g', ''))
                    memory_limit_mb = int(memory_limit * 1024 * (1 - reserve_memory_fraction))
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
            
            logger.info(f"Set TensorFlow memory growth for {len(gpus)} GPUs")
    except ImportError:
        logger.debug("TensorFlow not available")
    except Exception as e:
        logger.warning(f"Error setting TensorFlow memory growth: {e}")
    
    # Try to set PyTorch memory caching
    try:
        import torch
        if torch.cuda.is_available():
            # Set to not cache allocations (slower but uses less memory)
            torch.cuda.set_per_process_memory_fraction(1 - reserve_memory_fraction)
            torch.cuda.empty_cache()
            
            logger.info(f"Set PyTorch memory fraction to {1 - reserve_memory_fraction}")
    except ImportError:
        logger.debug("PyTorch not available")
    except Exception as e:
        logger.warning(f"Error setting PyTorch memory options: {e}")

def setup_multi_gpu_environment(num_gpus: Optional[int] = None) -> List[int]:
    """
    Set up the environment for multi-GPU training.
    
    Args:
        num_gpus (int, optional): Number of GPUs to use (if None, use all available)
        
    Returns:
        List[int]: List of selected GPU IDs
    """
    # Get available GPUs
    gpus = get_available_gpus()
    
    if not gpus:
        logger.warning("No GPUs available. Using CPU only.")
        return []
    
    # Determine number of GPUs to use
    if num_gpus is None:
        num_gpus = len(gpus)
    else:
        num_gpus = min(num_gpus, len(gpus))
    
    # Get GPU IDs
    gpu_ids = [gpu['index'] for gpu in gpus[:num_gpus]]
    
    # Set CUDA_VISIBLE_DEVICES to use selected GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Optimize CUDA memory usage
    optimize_cuda_memory_usage()
    
    logger.info(f"Set up multi-GPU environment with {num_gpus} GPUs: {gpu_ids}")
    
    return gpu_ids

if __name__ == "__main__":
    # Test GPU optimization utilities
    gpu_ids = setup_multi_gpu_environment()
    
    if gpu_ids:
        # Test multi-GPU parameters
        xgb_params = get_multi_gpu_model_params('xgboost')
        lgb_params = get_multi_gpu_model_params('lightgbm')
        h2o_params = get_multi_gpu_model_params('h2o')
        
        logger.info("XGBoost multi-GPU parameters:")
        for k, v in xgb_params.items():
            logger.info(f"  {k}: {v}")
        
        logger.info("LightGBM multi-GPU parameters:")
        for k, v in lgb_params.items():
            logger.info(f"  {k}: {v}")
        
        logger.info("H2O multi-GPU parameters:")
        for k, v in h2o_params.items():
            logger.info(f"  {k}: {v}")
        
        # Test data distribution
        X = np.random.random((1000, 10))
        y = np.random.randint(0, 2, 1000)
        
        chunks = distribute_data_to_gpus(X, y)
        logger.info(f"Distributed data into {len(chunks)} chunks")
    else:
        logger.info("No GPUs available for testing")