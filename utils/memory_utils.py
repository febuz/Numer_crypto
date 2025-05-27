#!/usr/bin/env python3
"""
Memory Management Utilities
"""

import gc
import os
import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def log_memory_usage(label: str = ""):
    """Log current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        system_memory = psutil.virtual_memory()
        available_mb = system_memory.available / 1024 / 1024
        
        logger.info(f"Memory usage {label}: {memory_mb:.1f} MB process, {available_mb:.1f} MB available")
        
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return 0


def clear_memory():
    """Clear Python memory"""
    gc.collect()
    
    # Try to clear GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not clear GPU memory: {e}")


def get_memory_info():
    """Get detailed memory information"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'system_total_mb': system_memory.total / 1024 / 1024,
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_used_percent': system_memory.percent
        }
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return {}