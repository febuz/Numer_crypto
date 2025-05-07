#!/usr/bin/env python3
"""
GPU setup and configuration utilities.

This module provides functions for setting up GPU training
and configuring machine learning models for GPU usage.
"""
import os
import sys
import logging
import threading
import time

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from utils.gpu.detection import get_available_gpus, get_gpu_utilization, select_best_gpu
from config.settings import XGBOOST_PARAMS, LIGHTGBM_PARAMS

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

def setup_gpu_training(gpu_id=None):
    """
    Set up the environment for GPU training.
    
    Args:
        gpu_id: The GPU ID to use. If None, selects the best GPU.
        
    Returns:
        int: The selected GPU ID, or None if no GPUs are available
    """
    # Auto-select GPU if not specified
    if gpu_id is None:
        gpu_id = select_best_gpu()
    
    # If no GPUs available, return None
    if gpu_id is None:
        logger.info("No GPUs available, using CPU")
        return None
    
    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    return gpu_id

def monitor_gpu_usage(interval=5, gpu_id=None):
    """
    Start a background thread to monitor GPU usage.
    
    Args:
        interval: Polling interval in seconds
        gpu_id: The GPU ID to monitor. If None, monitors all GPUs.
        
    Returns:
        Thread: The monitoring thread
    """
    try:
        def _monitor_thread():
            logger.info(f"Started GPU monitoring thread (interval: {interval}s)")
            while True:
                utils = get_gpu_utilization()
                if not utils:
                    logger.warning("No GPU utilization data available")
                else:
                    for util in utils:
                        if gpu_id is None or util['index'] == gpu_id:
                            logger.info(f"GPU {util['index']}: {util.get('gpu_util', 'N/A')} util, {util.get('memory_util', 'N/A')} mem")
                time.sleep(interval)
        
        # Start monitoring thread
        thread = threading.Thread(target=_monitor_thread, daemon=True)
        thread.start()
        return thread
    except Exception as e:
        logger.error(f"Error starting GPU monitoring: {e}")
        return None

def get_gpu_model_params(model_type='xgboost', gpu_id=None):
    """
    Get model parameters optimized for GPU training.
    
    Args:
        model_type (str): The type of model ('xgboost' or 'lightgbm')
        gpu_id (int): The GPU ID to use, or None to auto-select
    
    Returns:
        dict: Model parameters for GPU training
    """
    # Check if GPUs are available
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        logger.warning(f"No GPUs available. Using CPU parameters for {model_type}")
        if model_type.lower() == 'xgboost':
            params = XGBOOST_PARAMS.copy()
            # Remove GPU-specific params
            params.pop('tree_method', None)
            params.pop('predictor', None)
            params.pop('gpu_id', None)
            params.pop('backend', None)
            return params
        else:  # lightgbm
            params = LIGHTGBM_PARAMS.copy()
            params['device'] = 'cpu'
            params.pop('gpu_platform_id', None)
            params.pop('gpu_device_id', None)
            return params
    
    # Auto-select GPU if not specified
    if gpu_id is None:
        gpu_id = select_best_gpu()
    
    # Ensure valid GPU ID
    max_gpu_id = max(gpu['index'] for gpu in available_gpus)
    if gpu_id > max_gpu_id:
        logger.warning(f"Requested GPU ID {gpu_id} exceeds available GPUs. Using GPU {max_gpu_id} instead.")
        gpu_id = max_gpu_id
    
    # Set environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create model parameters
    if model_type.lower() == 'xgboost':
        params = XGBOOST_PARAMS.copy()
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
        params['gpu_id'] = 0  # Always use 0 since we set CUDA_VISIBLE_DEVICES
        params['backend'] = 'gpu'
        return params
    
    elif model_type.lower() == 'lightgbm':
        params = LIGHTGBM_PARAMS.copy()
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0  # Always use 0 since we set CUDA_VISIBLE_DEVICES
        return params
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Test GPU setup
    gpu_id = setup_gpu_training()
    print(f"Selected GPU ID: {gpu_id}")
    
    if gpu_id is not None:
        # Get model parameters
        xgb_params = get_gpu_model_params('xgboost', gpu_id)
        lgb_params = get_gpu_model_params('lightgbm', gpu_id)
        
        print("\nXGBoost GPU Parameters:")
        for k, v in xgb_params.items():
            print(f"  {k}: {v}")
        
        print("\nLightGBM GPU Parameters:")
        for k, v in lgb_params.items():
            print(f"  {k}: {v}")
        
        # Start monitoring
        monitor_thread = monitor_gpu_usage(interval=2, gpu_id=gpu_id)
        
        # Wait for a few intervals
        time.sleep(8)
    else:
        print("No GPUs available for testing")