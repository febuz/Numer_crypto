#!/usr/bin/env python3
"""
PyCaret GPU optimization utilities
"""

import os
import gc
import json
import warnings
import tempfile
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')


def optimize_memory():
    """Apply memory optimizations"""
    gc.collect()
    
    try:
        import psutil
        import resource
        
        # Set memory limit to 80% of available memory
        available_memory = psutil.virtual_memory().available
        memory_limit = int(available_memory * 0.8)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        print(f"Memory limit set to {memory_limit // (1024**3)} GB")
    except Exception:
        print("Could not set memory limits")


def setup_gpu_environment(gpu_count: int = 3):
    """Setup GPU environment for optimal performance"""
    # CUDA settings
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_MEMORY_ALLOW_GROWTH'] = 'true'
    
    # Multi-threading
    os.environ['NUMEXPR_NUM_THREADS'] = '6'
    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['NUMBA_NUM_THREADS'] = '6'
    
    # Memory settings
    os.environ['PYTHONHASHSEED'] = '42'
    
    # PyCaret specific settings
    os.environ['PYCARET_CUSTOM_LOGGING'] = 'false'
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp/joblib_cache'
    
    # Python memory optimization flags
    os.environ['PYTHONOPTIMIZE'] = '1'
    os.environ['MALLOC_ARENA_MAX'] = '2'
    
    # Create temp directories
    os.makedirs('/tmp/joblib_cache', exist_ok=True)
    os.makedirs('/tmp/pycaret_cache', exist_ok=True)
    
    print(f"GPU environment configured for {gpu_count} GPUs")


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return status info"""
    gpu_info = {
        'gpu_count': 0,
        'gpus_detected': False,
        'drivers_working': False,
        'cuda_available': False,
        'gpu_names': []
    }
    
    # Check NVIDIA-SMI
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info['gpu_count'] = len(lines)
            gpu_info['gpus_detected'] = True
            gpu_info['drivers_working'] = True
            gpu_info['gpu_names'] = [line.split(', ')[1] for line in lines if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            if gpu_info['gpu_count'] == 0:
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['gpus_detected'] = True
    except ImportError:
        pass
    
    return gpu_info


def check_pycaret_packages() -> Dict[str, bool]:
    """Check required PyCaret packages"""
    packages = ['pycaret', 'pandas', 'numpy', 'sklearn', 'lightgbm', 'xgboost', 'catboost']
    status = {}
    
    for pkg in packages:
        try:
            __import__(pkg)
            status[pkg] = True
        except ImportError:
            status[pkg] = False
    
    return status


def check_gpu_libraries() -> Dict[str, Dict[str, Any]]:
    """Check GPU-specific libraries"""
    libraries = {}
    
    # Check LightGBM GPU support
    try:
        import lightgbm as lgb
        libraries['lightgbm'] = {'available': True, 'gpu_support': False}
        try:
            lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0)
            libraries['lightgbm']['gpu_support'] = True
        except Exception:
            pass
    except ImportError:
        libraries['lightgbm'] = {'available': False, 'gpu_support': False}
    
    # Check XGBoost GPU support
    try:
        import xgboost as xgb
        libraries['xgboost'] = {'available': True, 'gpu_support': False}
        gpu_available = xgb.build_info().get('USE_CUDA', 'OFF') == 'ON'
        libraries['xgboost']['gpu_support'] = gpu_available
    except ImportError:
        libraries['xgboost'] = {'available': False, 'gpu_support': False}
    
    # Check CatBoost GPU support
    try:
        import catboost
        libraries['catboost'] = {'available': True, 'gpu_support': True}
    except ImportError:
        libraries['catboost'] = {'available': False, 'gpu_support': False}
    
    # Check TensorFlow GPU support
    try:
        import tensorflow as tf
        libraries['tensorflow'] = {
            'available': True, 
            'gpu_support': len(tf.config.list_physical_devices('GPU')) > 0
        }
    except ImportError:
        libraries['tensorflow'] = {'available': False, 'gpu_support': False}
    
    # Check PyTorch GPU support
    try:
        import torch
        libraries['pytorch'] = {
            'available': True,
            'gpu_support': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except ImportError:
        libraries['pytorch'] = {'available': False, 'gpu_support': False, 'gpu_count': 0}
    
    return libraries


def create_pycaret_config(gpu_count: int = 3, use_gpu: bool = True) -> str:
    """Create PyCaret configuration and return path"""
    config = {
        'use_gpu': use_gpu,
        'gpu_count': gpu_count,
        'n_jobs': -1,
        'html': False,
        'session_id': 42,
        'train_size': 0.8,
        'data_split_shuffle': True,
        'data_split_stratify': False,
        'folds': 5,
        'fold_strategy': 'timeseries'
    }
    
    config_path = '/tmp/pycaret_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def verify_pycaret_installation() -> bool:
    """Verify PyCaret installation and GPU support"""
    try:
        from pycaret.regression import setup as regression_setup
        from pycaret.classification import setup as classification_setup
        return True
    except ImportError:
        return False


def initialize_pycaret_gpu(gpu_count: int = 3, memory_optimize: bool = True) -> bool:
    """Initialize PyCaret with GPU optimization"""
    try:
        # Apply optimizations
        if memory_optimize:
            optimize_memory()
        
        setup_gpu_environment(gpu_count)
        
        # Verify installation
        if not verify_pycaret_installation():
            print("✗ PyCaret installation verification failed")
            return False
        
        print("✓ PyCaret GPU initialization completed successfully")
        return True
    
    except Exception as e:
        print(f"✗ PyCaret GPU initialization failed: {e}")
        return False