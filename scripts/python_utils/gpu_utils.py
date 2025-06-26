#!/usr/bin/env python3
"""
GPU utility functions for Numerai Crypto Pipeline
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clear_gpu_cache() -> None:
    """Clear GPU memory cache for PyTorch and CuPy if available"""
    try:
        import torch
        torch.cuda.empty_cache()
        logger.info('Cleared PyTorch CUDA cache')
    except ImportError:
        logger.info('PyTorch not available for cache clearing')
    except Exception as e:
        logger.warning(f'Error clearing PyTorch cache: {e}')
    
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        logger.info('Cleared CuPy memory pool')
    except ImportError:
        logger.info('CuPy not available for cache clearing')
    except Exception as e:
        logger.warning(f'Error clearing CuPy cache: {e}')

def get_gpu_info() -> Dict[str, Union[int, List[str]]]:
    """Get information about available GPUs using PyTorch"""
    result = {
        'gpu_count': 0,
        'gpu_names': []
    }
    
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        result['gpu_count'] = gpu_count
        
        if gpu_count > 0:
            for i in range(gpu_count):
                result['gpu_names'].append(torch.cuda.get_device_name(i))
            logger.info(f'Found {gpu_count} GPUs: {", ".join(result["gpu_names"])}')
        else:
            logger.info('No GPUs detected by PyTorch')
    except ImportError:
        logger.info('PyTorch not installed - GPU check skipped')
    except Exception as e:
        logger.warning(f'Error during GPU check: {e}')
    
    return result

def set_gpu_environment_variables(gpu_indices: str, 
                                 memory_limit_gb: int = 0,
                                 enable_azure_synapse: bool = True) -> None:
    """Set environment variables for GPU usage"""
    # Set basic GPU environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
    
    # Set thread limits to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4" 
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    
    # Set memory limits if provided
    if memory_limit_gb > 0:
        # Convert to bytes for XGBoost
        os.environ["XGB_GPU_MEMORY_LIMIT"] = str(memory_limit_gb * 1024 * 1024 * 1024)
        # Convert to MB for LightGBM
        os.environ["LIGHTGBM_GPU_MEMORY_MB"] = str(memory_limit_gb * 1024)
    
    # Configure Azure Synapse LightGBM if enabled
    if enable_azure_synapse:
        os.environ["USE_AZURE_SYNAPSE_LIGHTGBM"] = "1"
        os.environ["LIGHTGBM_SYNAPSE_MODE"] = "1"
        os.environ["LIGHTGBM_USE_SYNAPSE"] = "1"
    else:
        os.environ["USE_AZURE_SYNAPSE_LIGHTGBM"] = "0"
        os.environ["LIGHTGBM_SYNAPSE_MODE"] = "0" 
        os.environ["LIGHTGBM_USE_SYNAPSE"] = "0"

if __name__ == "__main__":
    # Simple command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU utilities for Numerai Crypto Pipeline")
    parser.add_argument("--clear-cache", action="store_true", help="Clear GPU memory cache")
    parser.add_argument("--get-info", action="store_true", help="Get information about available GPUs")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_gpu_cache()
    
    if args.get_info:
        gpu_info = get_gpu_info()
        print(f"GPU count: {gpu_info['gpu_count']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"GPU {i}: {name}")