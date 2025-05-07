#!/usr/bin/env python3
"""
GPU detection and monitoring utilities.

This module provides functions for detecting and monitoring GPUs,
as well as selecting the optimal GPU for training.
"""
import os
import sys
import logging
import subprocess
import importlib.util

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from config.settings import HARDWARE_CONFIG

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

def is_package_available(package_name):
    """
    Check if a package is available/installed.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None

def get_available_gpus():
    """
    Detect available GPUs using various methods.
    
    Returns:
        list: List of available GPU information
    """
    gpu_info = []
    
    # Method 1: Try using nvidia-smi
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader", 
                                           shell=True, universal_newlines=True)
        lines = nvidia_smi.strip().split('\n')
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total': parts[2],
                    'memory_free': parts[3],
                    'source': 'nvidia-smi'
                })
        
        if gpu_info:
            return gpu_info
            
    except Exception as e:
        logger.debug(f"Could not get GPU info via nvidia-smi: {e}")
    
    # Method 2: Try using py3nvml
    try:
        if is_package_available('py3nvml'):
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            
            try:
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info.append({
                        'index': i,
                        'name': name,
                        'memory_total': f"{memory.total / 1024**2:.0f} MiB",
                        'memory_free': f"{memory.free / 1024**2:.0f} MiB",
                        'source': 'py3nvml'
                    })
                
                nvml.nvmlShutdown()
                
                if gpu_info:
                    return gpu_info
            except Exception as e:
                logger.debug(f"Error in py3nvml GPU detection: {e}")
                if 'nvml' in locals():
                    try:
                        nvml.nvmlShutdown()
                    except:
                        pass
    except ImportError:
        logger.debug("py3nvml not available")
    
    # Method 3: Try using GPUtil
    try:
        if is_package_available('GPUtil'):
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                gpu_info.append({
                    'index': i,
                    'name': gpu.name,
                    'memory_total': f"{gpu.memoryTotal} MiB",
                    'memory_free': f"{gpu.memoryFree} MiB",
                    'source': 'GPUtil'
                })
            
            if gpu_info:
                return gpu_info
    except ImportError:
        logger.debug("GPUtil not available")
    
    # Fallback to configuration if no GPUs detected
    if not gpu_info and HARDWARE_CONFIG.get('gpu_count', 0) > 0:
        logger.warning("No GPUs detected via tools, but configuration specifies GPU availability")
        for i in range(HARDWARE_CONFIG.get('gpu_count', 0)):
            gpu_info.append({
                'index': i,
                'name': HARDWARE_CONFIG.get('gpu_model', 'Unknown'),
                'memory_total': HARDWARE_CONFIG.get('gpu_memory', 'Unknown'),
                'memory_free': 'Unknown',
                'source': 'config'
            })
    
    return gpu_info

def get_gpu_utilization():
    """
    Get current GPU utilization.
    
    Returns:
        list: List of GPU utilization information
    """
    util_info = []
    
    # Method 1: nvidia-smi for utilization
    try:
        nvidia_smi = subprocess.check_output(
            "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader",
            shell=True, universal_newlines=True)
        
        lines = nvidia_smi.strip().split('\n')
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                util_info.append({
                    'index': int(parts[0]),
                    'gpu_util': parts[1],
                    'memory_util': parts[2],
                    'temperature': parts[3],
                    'source': 'nvidia-smi'
                })
        
        if util_info:
            return util_info
    except Exception as e:
        logger.debug(f"Could not get GPU utilization via nvidia-smi: {e}")
    
    # Method 2: GPUtil for utilization
    try:
        if is_package_available('GPUtil'):
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                util_info.append({
                    'index': gpu.id,
                    'gpu_util': f"{gpu.load * 100:.1f} %",
                    'memory_util': f"{(gpu.memoryTotal - gpu.memoryFree) / gpu.memoryTotal * 100:.1f} %",
                    'temperature': f"{gpu.temperature} C",
                    'source': 'GPUtil'
                })
            
            if util_info:
                return util_info
    except Exception as e:
        logger.debug(f"Could not get GPU utilization via GPUtil: {e}")
    
    return util_info

def select_best_gpu():
    """
    Select the best GPU for training based on memory and utilization.
    
    Returns:
        int: The index of the best GPU to use, or None if no GPUs are available
    """
    gpus = get_available_gpus()
    
    if not gpus:
        logger.warning("No GPUs available for selection")
        return None
    
    # If we have utilization data, use it
    utils = get_gpu_utilization()
    
    # Match utilization data to GPU data
    if utils:
        for gpu in gpus:
            for util in utils:
                if gpu['index'] == util['index']:
                    gpu.update(util)
    
    # Sort GPUs by free memory (if available) and lowest utilization
    for gpu in gpus:
        # Parse free memory if available
        try:
            if 'memory_free' in gpu and isinstance(gpu['memory_free'], str) and 'MiB' in gpu['memory_free']:
                gpu['memory_free_mib'] = float(gpu['memory_free'].split()[0])
            elif 'memory_free' in gpu and isinstance(gpu['memory_free'], (int, float)):
                gpu['memory_free_mib'] = float(gpu['memory_free'])
        except (ValueError, IndexError) as e:
            gpu['memory_free_mib'] = 0
        
        # Parse GPU utilization if available
        try:
            if 'gpu_util' in gpu and isinstance(gpu['gpu_util'], str) and '%' in gpu['gpu_util']:
                gpu['gpu_util_pct'] = float(gpu['gpu_util'].split()[0])
            elif 'gpu_util' in gpu and isinstance(gpu['gpu_util'], (int, float)):
                gpu['gpu_util_pct'] = float(gpu['gpu_util'])
        except (ValueError, IndexError) as e:
            gpu['gpu_util_pct'] = 100  # Assume worst case
    
    # First try to sort by free memory
    if all('memory_free_mib' in gpu for gpu in gpus):
        sorted_gpus = sorted(gpus, key=lambda g: g['memory_free_mib'], reverse=True)
        logger.info(f"Selected GPU {sorted_gpus[0]['index']} with {sorted_gpus[0]['memory_free']} free memory")
        return sorted_gpus[0]['index']
    
    # Otherwise by utilization
    if all('gpu_util_pct' in gpu for gpu in gpus):
        sorted_gpus = sorted(gpus, key=lambda g: g['gpu_util_pct'])
        logger.info(f"Selected GPU {sorted_gpus[0]['index']} with {sorted_gpus[0]['gpu_util']} utilization")
        return sorted_gpus[0]['index']
    
    # If no sorting criteria available, just take the first one
    logger.info(f"Selected GPU {gpus[0]['index']} (no sorting criteria available)")
    return gpus[0]['index']

def print_gpu_status():
    """
    Print current GPU status information to the console.
    """
    gpus = get_available_gpus()
    
    if not gpus:
        print("No GPUs detected.")
        return
    
    print(f"\n=== {len(gpus)} GPUs Detected ===")
    
    # Get utilization data
    utils = get_gpu_utilization()
    util_map = {u['index']: u for u in utils} if utils else {}
    
    # Print information
    for gpu in gpus:
        index = gpu['index']
        print(f"GPU {index}: {gpu['name']}")
        print(f"  Memory: {gpu['memory_total']}")
        
        if index in util_map:
            print(f"  Utilization: {util_map[index]['gpu_util']}")
            print(f"  Memory Utilization: {util_map[index]['memory_util']}")
            print(f"  Temperature: {util_map[index]['temperature']}")
    
    # Print environment variables
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"\nCUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

if __name__ == "__main__":
    # Print GPU information when run as a script
    gpus = get_available_gpus()
    
    if gpus:
        print(f"Found {len(gpus)} GPUs:")
        for gpu in gpus:
            print(f"  GPU {gpu['index']}: {gpu.get('name', 'Unknown')}")
            print(f"    Memory: {gpu.get('memory_total', 'Unknown')}")
            print(f"    Free Memory: {gpu.get('memory_free', 'Unknown')}")
            print(f"    Source: {gpu.get('source', 'Unknown')}")
        
        # Print utilization
        utils = get_gpu_utilization()
        if utils:
            print("\nGPU Utilization:")
            for util in utils:
                print(f"  GPU {util['index']}:")
                print(f"    GPU Utilization: {util.get('gpu_util', 'Unknown')}")
                print(f"    Memory Utilization: {util.get('memory_util', 'Unknown')}")
                print(f"    Temperature: {util.get('temperature', 'Unknown')}")
        
        # Select best GPU
        best_gpu = select_best_gpu()
        print(f"\nRecommended GPU: {best_gpu}")
    else:
        print("No GPUs detected!")