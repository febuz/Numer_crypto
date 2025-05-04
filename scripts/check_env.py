#!/usr/bin/env python
"""
Quick environment check script for Numer_crypto.

This script provides a fast check of the environment, hardware, and dependencies
without running the full suite of tests. It's a good first step when setting up
a new environment or diagnosing issues.
"""
import os
import sys
import platform
import importlib
from pathlib import Path

# Add the project root to the Python path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def check_python_version():
    """Check Python version."""
    print(f"\n=== Python Environment ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Platform: {platform.platform()}")
    
    # Check for minimum required version
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"✗ Python version {current_version} is below the required version {required_version}")
    else:
        print(f"✓ Python version is sufficient")

def check_core_packages():
    """Check if core packages are installed."""
    print(f"\n=== Core Packages ===")
    
    core_packages = [
        'numerapi', 'polars', 'pyspark', 'h2o', 
        'pysparkling', 'xgboost', 'lightgbm'
    ]
    
    all_installed = True
    for package in core_packages:
        try:
            # Try to import the package
            module = importlib.import_module(package.replace('-', '_'))
            
            # Get the version if available
            version = getattr(module, '__version__', 'unknown version')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package} not found")
            all_installed = False
    
    if all_installed:
        print("✓ All core packages installed")
    else:
        print("✗ Some core packages are missing, run 'pip install -r requirements.txt'")

def check_gpu_setup():
    """Check GPU setup."""
    print(f"\n=== GPU Setup ===")
    
    # Check for CUDA environment variables
    if 'CUDA_HOME' in os.environ:
        print(f"CUDA_HOME: {os.environ['CUDA_HOME']}")
    elif 'CUDA_PATH' in os.environ:
        print(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
    else:
        print("No CUDA environment variables found")
    
    # Try to use the GPU utils
    try:
        from numer_crypto.utils.gpu_utils import get_available_gpus
        
        gpus = get_available_gpus()
        if gpus:
            print(f"✓ {len(gpus)} GPUs found:")
            for gpu in gpus:
                print(f"  - GPU {gpu['index']}: {gpu.get('name', 'Unknown')}")
                if 'memory_total' in gpu:
                    print(f"    Memory: {gpu['memory_total']}")
        else:
            print("✗ No GPUs detected")
            
    except ImportError as e:
        print(f"✗ Error importing GPU utilities: {e}")
    except Exception as e:
        print(f"✗ Error detecting GPUs: {e}")

def check_memory():
    """Check available memory."""
    print(f"\n=== Memory ===")
    
    try:
        import psutil
        
        # Get system memory info
        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)
        available_gb = mem_info.available / (1024**3)
        
        print(f"Total system memory: {total_gb:.1f} GB")
        print(f"Available memory: {available_gb:.1f} GB")
        
        # Compare with configuration
        try:
            from numer_crypto.config.settings import HARDWARE_CONFIG
            
            configured_memory = HARDWARE_CONFIG.get('total_memory', '0g')
            if configured_memory.endswith('g'):
                configured_gb = float(configured_memory[:-1])
                print(f"Configured memory in settings: {configured_gb:.1f} GB")
                
                if total_gb < configured_gb * 0.8:  # Allow 20% margin
                    print(f"! Warning: Actual memory ({total_gb:.1f} GB) is less than 80% of configured memory ({configured_gb:.1f} GB)")
                else:
                    print(f"✓ Memory configuration looks appropriate")
        except ImportError:
            print("! Could not load HARDWARE_CONFIG")
        
    except ImportError:
        print("psutil not installed, cannot check memory")

def main():
    """Main function."""
    print("=" * 50)
    print("Numer_crypto Quick Environment Check")
    print("=" * 50)
    
    # Run all checks
    check_python_version()
    check_core_packages()
    check_gpu_setup()
    check_memory()
    
    print("\n" + "=" * 50)
    print("For a more thorough test, run:")
    print("python scripts/test_hardware.py --full")
    print("=" * 50)

if __name__ == "__main__":
    main()