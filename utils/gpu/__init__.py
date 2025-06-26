"""
GPU utilities for the Numerai Crypto project.
"""
from utils.gpu.detection import (
    get_available_gpus,
    get_gpu_utilization,
    select_best_gpu,
    print_gpu_status,
    is_package_available
)
from utils.gpu.setup import (
    setup_gpu_training,
    get_gpu_model_params,
    monitor_gpu_usage
)

__all__ = [
    'get_available_gpus',
    'get_gpu_utilization',
    'select_best_gpu',
    'setup_gpu_training',
    'get_gpu_model_params',
    'monitor_gpu_usage',
    'print_gpu_status',
    'is_package_available'
]
# Import LightGBM wrapper
try:
    from .lightgbm_wrapper import *
except ImportError:
    pass
