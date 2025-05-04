#!/usr/bin/env python
"""
Hardware capabilities test script for Numer_crypto.

This script tests the available hardware resources and package installations,
ensuring that all required dependencies are correctly set up and that the 
system can utilize high-memory operations and GPU acceleration if available.
"""
import os
import sys
import time
import argparse
import importlib
import numpy as np
import subprocess
from pathlib import Path

# Add the project root to the Python path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import configurations (will be tested)
from numer_crypto.config.settings import HARDWARE_CONFIG, SPARK_CONFIG, H2O_CONFIG
from numer_crypto.utils.spark_utils import (
    get_system_resources, 
    create_spark_session, 
    init_h2o, 
    select_gpu
)

def parse_args():
    """Parse command line arguments for the test script."""
    parser = argparse.ArgumentParser(
        description='Test hardware capabilities and package installation for Numer_crypto'
    )
    parser.add_argument('--full', action='store_true', 
                        help='Run a full test suite including small model training')
    parser.add_argument('--no-gpu', action='store_true', 
                        help='Skip GPU tests even if GPUs are available')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Print verbose output')
    return parser.parse_args()

def check_package_availability():
    """Check that all required packages are installed and importable."""
    print("\n=== Checking Required Packages ===")
    
    required_packages = [
        # Core data processing
        'numerapi', 'pandas', 'polars', 'pyspark', 'pyarrow',
        # Machine learning  
        'h2o', 'pysparkling', 'xgboost', 'lightgbm', 'scikit-learn',
        # Utilities
        'matplotlib', 'numpy', 'psutil', 'cloudpickle',
        # Optional GPU packages (will warn, not fail)
        'nvidia_ml_py', 'py3nvml', 'GPUtil'
    ]
    
    all_packages_available = True
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError as e:
            if package in ['nvidia_ml_py', 'py3nvml', 'GPUtil']:
                print(f"! {package} is not installed - GPU monitoring capabilities may be limited")
            else:
                print(f"✗ {package} is not installed: {e}")
                all_packages_available = False
    
    if all_packages_available:
        print("\nAll essential packages are installed.")
    else:
        print("\nSome required packages are missing. Please install them using:")
        print("pip install -r requirements.txt")
    
    return all_packages_available

def test_memory_capabilities():
    """Test available system memory."""
    print("\n=== Testing Memory Capabilities ===")
    import psutil
    
    # Get system memory info
    mem_info = psutil.virtual_memory()
    total_gb = mem_info.total / (1024**3)
    available_gb = mem_info.available / (1024**3)
    
    print(f"Total system memory: {total_gb:.1f} GB")
    print(f"Available memory: {available_gb:.1f} GB")
    
    # Check against configuration
    configured_memory = HARDWARE_CONFIG.get('total_memory', '0g')
    if configured_memory.endswith('g'):
        configured_gb = float(configured_memory[:-1])
        print(f"Configured memory in settings: {configured_gb:.1f} GB")
        
        if total_gb < configured_gb * 0.9:  # Allow 10% margin
            print(f"! Warning: Actual memory ({total_gb:.1f} GB) is less than configured ({configured_gb:.1f} GB)")
        else:
            print(f"✓ Memory configuration looks correct")
    
    # Test large array allocation if we have enough memory
    if available_gb > 10:  # Only test if at least 10GB available
        try:
            print("Testing large memory allocation...")
            # Try to allocate a 4GB array
            size_gb = min(4, available_gb * 0.5)  # Use at most half of available memory
            size_bytes = int(size_gb * 1024**3 / 8)  # Size in number of float64 values
            
            start_time = time.time()
            large_array = np.random.random(size_bytes)
            elapsed = time.time() - start_time
            
            print(f"✓ Successfully allocated {size_gb:.1f} GB array in {elapsed:.2f} seconds")
            # Force cleanup
            large_array = None
        except MemoryError as e:
            print(f"✗ Failed to allocate large array: {e}")
            return False
    
    return True

def test_gpu_capabilities(no_gpu=False):
    """Test GPU capabilities if available."""
    print("\n=== Testing GPU Capabilities ===")
    
    if no_gpu:
        print("GPU testing disabled via command line flag")
        return None
    
    # Check GPU count from configuration
    gpu_count = HARDWARE_CONFIG.get('gpu_count', 0)
    print(f"Configured GPU count: {gpu_count}")
    
    if gpu_count == 0:
        print("No GPUs configured in settings - skipping GPU tests")
        return None
    
    # Try different methods to detect GPUs
    detected_gpus = 0
    gpu_info = {}
    
    # Method 1: Try using nvidia-smi
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader", shell=True)
        nvidia_smi = nvidia_smi.decode('utf-8').strip().split('\n')
        detected_gpus = len(nvidia_smi)
        
        print(f"Detected {detected_gpus} GPUs via nvidia-smi:")
        for i, gpu_data in enumerate(nvidia_smi):
            print(f"  GPU {i}: {gpu_data}")
            gpu_info[i] = gpu_data
    except Exception as e:
        print(f"nvidia-smi not available: {e}")
    
    # Method 2: Try using py3nvml
    if detected_gpus == 0:
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            detected_gpus = nvml.nvmlDeviceGetCount()
            print(f"Detected {detected_gpus} GPUs via py3nvml")
            
            for i in range(detected_gpus):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"  GPU {i}: {name.decode('utf-8')}, Memory: {memory.total / 1024**3:.1f} GB")
                gpu_info[i] = name
            
            nvml.nvmlShutdown()
        except Exception as e:
            print(f"py3nvml not available: {e}")
    
    # Method 3: Try using GPUtil
    if detected_gpus == 0:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            detected_gpus = len(gpus)
            print(f"Detected {detected_gpus} GPUs via GPUtil")
            
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}, Memory: {gpu.memoryTotal} MB")
                gpu_info[i] = gpu.name
        except Exception as e:
            print(f"GPUtil not available: {e}")
    
    if detected_gpus == 0:
        print("No GPUs detected. Skipping GPU tests.")
        return None
    
    # Check if detected GPU count matches configuration
    if detected_gpus != gpu_count:
        print(f"! Warning: Detected {detected_gpus} GPUs, but configured for {gpu_count} GPUs")
    else:
        print(f"✓ Detected GPU count matches configuration: {detected_gpus}")
    
    # Test XGBoost GPU capability
    try:
        import xgboost as xgb
        print("\nTesting XGBoost GPU support...")
        
        # Create a small dataset
        X = np.random.random((1000, 50))
        y = np.random.random(1000)
        dtrain = xgb.DMatrix(X, label=y)
        
        # Configure for GPU
        gpu_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0
        }
        
        try:
            # Try to train a simple model on GPU
            start_time = time.time()
            bst = xgb.train(gpu_params, dtrain, num_boost_round=10)
            elapsed = time.time() - start_time
            print(f"✓ Successfully trained XGBoost model with GPU in {elapsed:.2f} seconds")
            return detected_gpus
        except Exception as e:
            print(f"✗ XGBoost GPU training failed: {e}")
            print("  Attempting to train with CPU as fallback...")
            
            # Try CPU fallback
            cpu_params = gpu_params.copy()
            cpu_params['tree_method'] = 'hist'
            del cpu_params['gpu_id']
            
            start_time = time.time()
            bst = xgb.train(cpu_params, dtrain, num_boost_round=10)
            elapsed = time.time() - start_time
            print(f"✓ Successfully trained XGBoost model with CPU in {elapsed:.2f} seconds")
            
    except Exception as e:
        print(f"✗ XGBoost testing failed: {e}")
    
    # Also test LightGBM if GPU count > 0
    if detected_gpus > 0:
        try:
            import lightgbm as lgb
            print("\nTesting LightGBM GPU support...")
            
            # Create a small dataset
            X = np.random.random((1000, 50))
            y = np.random.random(1000)
            lgb_train = lgb.Dataset(X, y)
            
            # Configure for GPU
            gpu_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            }
            
            try:
                # Try to train a simple model on GPU
                start_time = time.time()
                bst = lgb.train(gpu_params, lgb_train, num_boost_round=10)
                elapsed = time.time() - start_time
                print(f"✓ Successfully trained LightGBM model with GPU in {elapsed:.2f} seconds")
            except Exception as e:
                print(f"✗ LightGBM GPU training failed: {e}")
                print("  Attempting to train with CPU as fallback...")
                
                # Try CPU fallback
                cpu_params = gpu_params.copy()
                cpu_params['device'] = 'cpu'
                del cpu_params['gpu_platform_id']
                del cpu_params['gpu_device_id']
                
                start_time = time.time()
                bst = lgb.train(cpu_params, lgb_train, num_boost_round=10)
                elapsed = time.time() - start_time
                print(f"✓ Successfully trained LightGBM model with CPU in {elapsed:.2f} seconds")
                
        except Exception as e:
            print(f"✗ LightGBM testing failed: {e}")
    
    return detected_gpus

def test_spark_capabilities():
    """Test Spark capabilities."""
    print("\n=== Testing Spark Capabilities ===")
    
    try:
        # Create Spark session with dynamic configuration
        print("Initializing Spark...")
        spark = create_spark_session(dynamic_config=True)
        
        print(f"✓ Successfully created Spark session: {spark.version}")
        
        # Test with a simple Spark operation
        print("Testing Spark DataFrame operations...")
        test_data = [(i, f"val_{i}") for i in range(1000)]
        df = spark.createDataFrame(test_data, ["id", "value"])
        
        # Perform a simple aggregation
        start_time = time.time()
        result = df.groupBy().count().collect()
        elapsed = time.time() - start_time
        
        print(f"✓ Successfully executed Spark DataFrame operation in {elapsed:.2f} seconds")
        
        # Test executor configuration
        conf = spark.sparkContext.getConf().getAll()
        print("\nSpark configuration:")
        
        # Extract and display key configurations
        configs_to_check = ['spark.driver.memory', 'spark.executor.memory', 
                           'spark.executor.cores', 'spark.default.parallelism']
        
        for key in configs_to_check:
            for k, v in conf:
                if k == key:
                    print(f"  {k}: {v}")
        
        # Verify that Spark can utilize a large part of the available memory
        import psutil
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        driver_memory = None
        executor_memory = None
        for k, v in conf:
            if k == 'spark.driver.memory' and v.endswith('g'):
                driver_memory = float(v[:-1])
            elif k == 'spark.executor.memory' and v.endswith('g'):
                executor_memory = float(v[:-1])
        
        if driver_memory and executor_memory:
            total_claimed = driver_memory + executor_memory
            print(f"  Total claimed memory (driver + executor): {total_claimed:.1f} GB")
            print(f"  System memory: {system_memory_gb:.1f} GB")
            
            # Check if the configured memory is reasonable
            if total_claimed > system_memory_gb:
                print(f"! Warning: Configured Spark memory ({total_claimed:.1f} GB) exceeds system memory ({system_memory_gb:.1f} GB)")
            elif total_claimed < 0.5 * system_memory_gb:
                print(f"! Warning: Configured Spark memory ({total_claimed:.1f} GB) is less than 50% of system memory ({system_memory_gb:.1f} GB)")
            else:
                print(f"✓ Spark memory configuration looks appropriate")
        
        # Stop Spark
        spark.stop()
        return True
    
    except Exception as e:
        print(f"✗ Spark capability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_h2o_capabilities():
    """Test H2O capabilities."""
    print("\n=== Testing H2O Capabilities ===")
    
    try:
        # Initialize H2O
        print("Initializing H2O...")
        h2o_instance = init_h2o(dynamic_config=True)
        
        print(f"✓ Successfully initialized H2O version {h2o_instance.version()}")
        
        # Check cluster info and available memory
        h2o_instance.cluster().show_status()
        
        # Test with a simple H2O frame
        print("Testing H2O frame operations...")
        
        # Create a test frame
        test_data = np.random.random((1000, 10))
        start_time = time.time()
        frame = h2o_instance.H2OFrame(test_data)
        elapsed = time.time() - start_time
        
        print(f"✓ Successfully created H2O frame of shape {frame.shape} in {elapsed:.2f} seconds")
        
        # Test a simple operation
        start_time = time.time()
        summary = frame.describe()
        elapsed = time.time() - start_time
        
        print(f"✓ Successfully executed H2O frame operation in {elapsed:.2f} seconds")
        
        # Try to train a simple model
        print("\nTesting simple H2O GBM model...")
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        
        # Create a test dataset
        x = list(range(frame.ncol))
        y = 0  # Use first column as target
        
        train, valid = frame.split_frame(ratios=[0.8])
        
        # Train a simple GBM model
        start_time = time.time()
        gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=3, learn_rate=0.1)
        gbm.train(x=x[1:], y=y, training_frame=train, validation_frame=valid)
        elapsed = time.time() - start_time
        
        print(f"✓ Successfully trained H2O GBM model in {elapsed:.2f} seconds")
        print(f"  Model performance (RMSE): {gbm.rmse()}")
        
        # Shutdown H2O
        h2o_instance.shutdown(prompt=False)
        return True
    
    except Exception as e:
        print(f"✗ H2O capability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_test_suite(no_gpu=False):
    """Run a full test suite including small model training."""
    print("\n=== Running Full Test Suite ===")
    
    # Get system resources
    resources = get_system_resources()
    print("System Resources:")
    for key, value in resources.items():
        print(f"  {key}: {value}")
        
    # Create a small sample dataset
    print("\nGenerating test dataset...")
    
    X = np.random.random((5000, 100))
    y = np.random.random(5000)
    
    results = {}
    
    # Test XGBoost
    try:
        print("\nTesting XGBoost...")
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X, label=y)
        cpu_params = {
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0
        }
        
        start_time = time.time()
        bst_cpu = xgb.train(cpu_params, dtrain, num_boost_round=100)
        cpu_time = time.time() - start_time
        print(f"  CPU training time: {cpu_time:.2f} seconds")
        results['xgboost_cpu'] = cpu_time
        
        # Test GPU if available and not disabled
        if resources['gpu_count'] > 0 and not no_gpu:
            gpu_params = cpu_params.copy()
            gpu_params['tree_method'] = 'gpu_hist'
            gpu_params['gpu_id'] = 0
            
            try:
                start_time = time.time()
                bst_gpu = xgb.train(gpu_params, dtrain, num_boost_round=100)
                gpu_time = time.time() - start_time
                print(f"  GPU training time: {gpu_time:.2f} seconds")
                results['xgboost_gpu'] = gpu_time
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"  ✓ GPU provides a {speedup:.1f}x speedup!")
                else:
                    print(f"  ! GPU is slower than CPU for this test case")
            except Exception as e:
                print(f"  ✗ XGBoost GPU training failed: {e}")
    except Exception as e:
        print(f"✗ XGBoost testing failed: {e}")
    
    # Test LightGBM
    try:
        print("\nTesting LightGBM...")
        import lightgbm as lgb
        
        lgb_train = lgb.Dataset(X, y)
        cpu_params = {
            'device': 'cpu',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1
        }
        
        start_time = time.time()
        bst_cpu = lgb.train(cpu_params, lgb_train, num_boost_round=100)
        cpu_time = time.time() - start_time
        print(f"  CPU training time: {cpu_time:.2f} seconds")
        results['lightgbm_cpu'] = cpu_time
        
        # Test GPU if available and not disabled
        if resources['gpu_count'] > 0 and not no_gpu:
            gpu_params = cpu_params.copy()
            gpu_params['device'] = 'gpu'
            gpu_params['gpu_platform_id'] = 0
            gpu_params['gpu_device_id'] = 0
            
            try:
                start_time = time.time()
                bst_gpu = lgb.train(gpu_params, lgb_train, num_boost_round=100)
                gpu_time = time.time() - start_time
                print(f"  GPU training time: {gpu_time:.2f} seconds")
                results['lightgbm_gpu'] = gpu_time
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"  ✓ GPU provides a {speedup:.1f}x speedup!")
                else:
                    print(f"  ! GPU is slower than CPU for this test case")
            except Exception as e:
                print(f"  ✗ LightGBM GPU training failed: {e}")
    except Exception as e:
        print(f"✗ LightGBM testing failed: {e}")
    
    # Test H2O XGBoost if available
    try:
        print("\nTesting H2O XGBoost...")
        import h2o
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        
        # Initialize H2O
        h2o_instance = init_h2o(dynamic_config=True)
        
        # Convert to H2O frame
        train = h2o_instance.H2OFrame(np.column_stack([y, X]))
        x_cols = list(range(1, train.ncol))
        y_col = 0
        
        # Train CPU model
        start_time = time.time()
        h2o_xgb = H2OXGBoostEstimator(
            ntrees=100,
            max_depth=6,
            learn_rate=0.1,
            distribution="gaussian"
        )
        h2o_xgb.train(x=x_cols, y=y_col, training_frame=train)
        cpu_time = time.time() - start_time
        print(f"  CPU training time: {cpu_time:.2f} seconds")
        results['h2o_xgboost_cpu'] = cpu_time
        
        # Try GPU if available
        if resources['gpu_count'] > 0 and not no_gpu:
            try:
                start_time = time.time()
                h2o_xgb_gpu = H2OXGBoostEstimator(
                    ntrees=100,
                    max_depth=6,
                    learn_rate=0.1,
                    distribution="gaussian",
                    backend="gpu",
                    gpu_id=0
                )
                h2o_xgb_gpu.train(x=x_cols, y=y_col, training_frame=train)
                gpu_time = time.time() - start_time
                print(f"  GPU training time: {gpu_time:.2f} seconds")
                results['h2o_xgboost_gpu'] = gpu_time
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"  ✓ GPU provides a {speedup:.1f}x speedup!")
                else:
                    print(f"  ! GPU is slower than CPU for this test case")
            except Exception as e:
                print(f"  ✗ H2O XGBoost GPU training failed: {e}")
        
        # Shutdown H2O
        h2o_instance.shutdown(prompt=False)
    except Exception as e:
        print(f"✗ H2O XGBoost testing failed: {e}")
    
    # Summary of results
    if results:
        print("\n=== Performance Summary ===")
        for model, time_taken in results.items():
            print(f"{model}: {time_taken:.2f} seconds")
    
    return results

def main():
    """Run the hardware capability tests."""
    args = parse_args()
    
    print("=" * 50)
    print("Numer_crypto Hardware and Package Test")
    print("=" * 50)
    
    # Run all tests
    packages_ok = check_package_availability()
    memory_ok = test_memory_capabilities()
    gpu_count = test_gpu_capabilities(args.no_gpu)
    spark_ok = test_spark_capabilities()
    h2o_ok = test_h2o_capabilities()
    
    # Optionally run full test suite
    if args.full:
        full_test_results = run_full_test_suite(args.no_gpu)
    
    # Summarize results
    print("\n" + "=" * 50)
    print("Hardware and Package Test Summary")
    print("=" * 50)
    print(f"Required packages: {'✓' if packages_ok else '✗'}")
    print(f"Memory capabilities: {'✓' if memory_ok else '✗'}")
    print(f"GPU capabilities: {'✓' if gpu_count else '—'} ({gpu_count or 0} GPUs detected)")
    print(f"Spark capabilities: {'✓' if spark_ok else '✗'}")
    print(f"H2O capabilities: {'✓' if h2o_ok else '✗'}")
    
    if all([packages_ok, memory_ok, spark_ok, h2o_ok]):
        print("\n✓ All essential tests passed!")
        if gpu_count:
            print(f"✓ {gpu_count} GPUs are available and usable")
        else:
            print("! No GPUs detected - the system will use CPU only")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())