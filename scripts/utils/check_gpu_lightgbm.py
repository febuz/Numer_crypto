#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Set environment variables
os.environ['USE_AZURE_SYNAPSE_LIGHTGBM'] = '1'
os.environ['LIGHTGBM_SYNAPSE_MODE'] = '1'
os.environ['LIGHTGBM_USE_SYNAPSE'] = '1'

def log(message, level="INFO"):
    levels = {
        "INFO": "\033[0;34m[INFO]\033[0m",
        "SUCCESS": "\033[0;32m[SUCCESS]\033[0m",
        "WARNING": "\033[1;33m[WARNING]\033[0m",
        "ERROR": "\033[0;31m[ERROR]\033[0m"
    }
    print(f"{levels.get(level, levels['INFO'])} {message}")

def check_lightgbm_installation():
    try:
        import lightgbm as lgb
        log(f"LightGBM version: {lgb.__version__}")
        
        # Check if tree_learner supports gpu
        params = lgb.LGBMModel().get_params()
        log(f"Available tree_learner types: {params.get('tree_learner')}")
        
        # Try to create a model with gpu tree_learner
        try:
            model = lgb.LGBMRegressor(tree_learner='gpu')
            log("Created model with tree_learner='gpu'", "SUCCESS")
        except Exception as e:
            log(f"Failed to create model with tree_learner='gpu': {e}", "ERROR")
            
        # Check if the compiled version has GPU support
        import inspect
        import importlib.util
        
        lgbm_path = Path(importlib.util.find_spec('lightgbm').origin).parent
        log(f"LightGBM package path: {lgbm_path}")
        
        # Check for CUDA/GPU files in package
        cuda_files = list(lgbm_path.glob("**/*cuda*"))
        gpu_files = list(lgbm_path.glob("**/*gpu*"))
        log(f"CUDA-related files found: {len(cuda_files)}")
        log(f"GPU-related files found: {len(gpu_files)}")
        
        if cuda_files or gpu_files:
            log("Found GPU-related files in the LightGBM package", "SUCCESS")
            for f in cuda_files + gpu_files:
                log(f"  - {f.relative_to(lgbm_path)}")
        else:
            log("No GPU-related files found in the LightGBM package", "WARNING")
            
    except ImportError:
        log("LightGBM is not installed", "ERROR")
        return False
    
    return True

def check_gpu_availability():
    log("Checking GPU availability...")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        log(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    # Try nvidia-smi
    try:
        nvidia_smi = subprocess.run(["nvidia-smi"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if nvidia_smi.returncode == 0:
            log("nvidia-smi command successful")
            # Extract GPU info
            for line in nvidia_smi.stdout.split("\n"):
                if "NVIDIA" in line and "%" in line:
                    log(f"GPU detected: {line.strip()}")
            return True
        else:
            log("nvidia-smi command failed", "WARNING")
    except:
        log("nvidia-smi command not available", "WARNING")
    
    # Try using NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        log(f"GPU count from NVML: {device_count}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            log(f"GPU {i}: {name}")
        pynvml.nvmlShutdown()
        return True
    except:
        log("Failed to check GPUs using NVML", "WARNING")
    
    return False

def install_gpu_lightgbm():
    log("Starting installation of LightGBM with GPU support...")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    log(f"Working in temporary directory: {temp_dir}")
    
    try:
        # Clone LightGBM repository
        subprocess.run(["git", "clone", "--recursive", "https://github.com/microsoft/LightGBM.git"], 
                     cwd=temp_dir, check=True)
        
        # Build with GPU support
        build_dir = os.path.join(temp_dir, "LightGBM", "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run CMake with GPU support
        subprocess.run([
            "cmake", "..",
            "-DUSE_GPU=ON",
            "-DBOOST_ROOT=",
            "-DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so",
            "-DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/"
        ], cwd=build_dir, check=True)
        
        # Build
        subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
        
        # Install the Python package
        python_dir = os.path.join(temp_dir, "LightGBM", "python-package")
        subprocess.run(["pip", "install", "--no-binary", "lightgbm", "."], cwd=python_dir, check=True)
        
        log("Installation of LightGBM with GPU support completed", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Installation failed: {e}", "ERROR")
        return False
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def check_gpu_train():
    log("Testing LightGBM training with GPU...")
    
    try:
        import numpy as np
        import lightgbm as lgb
        from sklearn.datasets import make_regression
        
        # Create a small dataset
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        
        # Configure LightGBM parameters with GPU
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'tree_learner': 'serial',
            'verbose': 0
        }
        
        # Create dataset
        lgb_data = lgb.Dataset(X, label=y)
        
        # Train model
        try:
            model = lgb.train(params, lgb_data, num_boost_round=10)
            log("Successfully trained model using GPU", "SUCCESS")
            return True
        except Exception as e:
            log(f"Failed to train model using GPU: {e}", "ERROR")
            
            # Try fallback to CPU
            log("Trying fallback to CPU training...")
            try:
                params['device'] = 'cpu'
                params.pop('tree_learner', None)
                params.pop('gpu_platform_id', None)
                params.pop('gpu_device_id', None)
                model = lgb.train(params, lgb_data, num_boost_round=10)
                log("Successfully trained model using CPU", "SUCCESS")
            except Exception as e2:
                log(f"Failed to train model using CPU: {e2}", "ERROR")
            
            return False
        
    except ImportError as e:
        log(f"Missing required packages: {e}", "ERROR")
        return False

def main():
    log("Checking LightGBM GPU support...")
    
    # Check current installation
    if not check_lightgbm_installation():
        log("LightGBM is not properly installed", "ERROR")
        return
    
    # Check GPU availability
    has_gpu = check_gpu_availability()
    if not has_gpu:
        log("No GPUs detected on this system", "WARNING")
    
    # Try GPU training
    gpu_train_successful = check_gpu_train()
    
    # Provide recommendations
    if not gpu_train_successful and has_gpu:
        log("\nRecommendations:", "INFO")
        log("1. Your LightGBM installation does not support GPU or is not correctly configured")
        log("2. Consider reinstalling LightGBM with GPU support using:")
        log("   pip uninstall -y lightgbm")
        log("   pip install lightgbm --config-settings=use_gpu=1")
        log("3. If that fails, try building from source:")
        log("   git clone --recursive https://github.com/microsoft/LightGBM.git")
        log("   cd LightGBM")
        log("   mkdir build")
        log("   cd build")
        log("   cmake -DUSE_GPU=ON ..")
        log("   make -j4")
        log("   cd ../python-package")
        log("   pip install --no-binary lightgbm .")
        
        answer = input("\nWould you like to attempt automatic installation of LightGBM with GPU support? (y/n): ")
        if answer.lower() == 'y':
            install_gpu_lightgbm()
    elif gpu_train_successful:
        log("Your LightGBM installation appears to support GPU training correctly", "SUCCESS")
    else:
        log("LightGBM is working but no GPU is available for acceleration", "INFO")

if __name__ == "__main__":
    main()