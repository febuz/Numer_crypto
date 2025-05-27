#!/usr/bin/env python3
"""
GPU Optimization Utilities
"""

import os
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def optimize_cuda_memory_usage(reserve_memory_fraction: float = 0.1):
    """Optimize CUDA memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if supported
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            
            logger.info(f"Optimized CUDA memory for {device_count} devices")
            return True
    except ImportError:
        logger.warning("PyTorch not available for CUDA optimization")
    except Exception as e:
        logger.warning(f"CUDA optimization failed: {e}")
    
    return False


def clear_gpu_memory():
    """Clear GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            return True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU memory clear failed: {e}")
    
    return False


def set_gpu_environment(gpu_ids: Optional[str] = None):
    """Set GPU environment variables"""
    if gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_ids}")
    
    # Other optimization environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    return True