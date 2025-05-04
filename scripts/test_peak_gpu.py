#!/usr/bin/env python3
"""
Peak GPU Utilization Test
Tests the maximum GPU utilization for XGBoost, LightGBM, and H2O.
Uses smaller datasets for quick results.
"""

import os
import sys
import time
import subprocess
import threading
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Function to get GPU utilization info
def get_gpu_info():
    """Get GPU info from nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 4:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used_mb': float(parts[2]),
                    'utilization_pct': float(parts[3])
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

# Thread function to monitor GPU utilization
def monitor_gpus(stop_event, results_list, interval=0.25):
    """Monitor GPU utilization in a separate thread"""
    max_utilization = 0
    max_memory = 0
    
    while not stop_event.is_set():
        gpu_info = get_gpu_info()
        timestamp = time.time()
        
        for gpu in gpu_info:
            results_list.append({
                'timestamp': timestamp,
                'gpu_index': gpu['index'],
                'name': gpu['name'],
                'utilization': gpu['utilization_pct'],
                'memory_used': gpu['memory_used_mb']
            })
            
            max_utilization = max(max_utilization, gpu['utilization_pct'])
            max_memory = max(max_memory, gpu['memory_used_mb'])
        
        time.sleep(interval)
    
    return max_utilization, max_memory

def test_xgboost():
    """Test XGBoost GPU peak utilization"""
    try:
        import xgboost as xgb
        print(f"\nTesting XGBoost {xgb.__version__} peak GPU utilization")
        
        # Create dataset
        n_samples = 100000  # Smaller dataset for quick test
        n_features = 20
        print(f"Creating dataset with {n_samples} samples and {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            random_state=42
        )
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # XGBoost 3.0+ uses 'device' parameter
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cuda',  # Use GPU
            'max_depth': 8,    # Deeper trees to use more GPU
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results, 0.1)  # Check every 0.1 seconds
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Train XGBoost model
        print("Training XGBoost on GPU...")
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=20,  # Fewer rounds for quick test
        )
        training_time = time.time() - start_time
        
        # Let monitoring run a bit longer to catch peak values
        time.sleep(0.5)
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Find peak utilization across all GPUs
        max_util = 0
        max_mem = 0
        max_gpu_name = ""
        
        for result in gpu_results:
            if result['utilization'] > max_util:
                max_util = result['utilization']
                max_gpu_name = result['name']
            if result['memory_used'] > max_mem:
                max_mem = result['memory_used']
        
        print(f"XGBoost completed in {training_time:.2f} seconds")
        print(f"Peak GPU Utilization: {max_util:.1f}% on {max_gpu_name}")
        print(f"Peak GPU Memory: {max_mem:.1f} MB")
        
        return {
            'library': 'xgboost',
            'version': xgb.__version__,
            'training_time': training_time,
            'peak_utilization': max_util,
            'peak_memory': max_mem,
            'gpu_name': max_gpu_name
        }
    
    except Exception as e:
        print(f"Error testing XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return {
            'library': 'xgboost',
            'error': str(e)
        }

def test_lightgbm():
    """Test LightGBM GPU peak utilization"""
    try:
        import lightgbm as lgb
        print(f"\nTesting LightGBM {lgb.__version__} peak GPU utilization")
        
        # Create dataset
        n_samples = 100000  # Smaller dataset for quick test
        n_features = 20
        print(f"Creating dataset with {n_samples} samples and {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            random_state=42
        )
        
        # Convert to LightGBM Dataset
        train_data = lgb.Dataset(X, label=y)
        
        # LightGBM GPU parameters
        params = {
            'objective': 'binary',
            'device': 'gpu',  # Use GPU
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 8,
            'learning_rate': 0.1,
            'verbose': -1,
            'seed': 42
        }
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results, 0.1)  # Check every 0.1 seconds
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Train LightGBM model
        print("Training LightGBM on GPU...")
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            num_boost_round=20,  # Fewer rounds for quick test
        )
        training_time = time.time() - start_time
        
        # Let monitoring run a bit longer to catch peak values
        time.sleep(0.5)
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Find peak utilization across all GPUs
        max_util = 0
        max_mem = 0
        max_gpu_name = ""
        
        for result in gpu_results:
            if result['utilization'] > max_util:
                max_util = result['utilization']
                max_gpu_name = result['name']
            if result['memory_used'] > max_mem:
                max_mem = result['memory_used']
        
        print(f"LightGBM completed in {training_time:.2f} seconds")
        print(f"Peak GPU Utilization: {max_util:.1f}% on {max_gpu_name}")
        print(f"Peak GPU Memory: {max_mem:.1f} MB")
        
        return {
            'library': 'lightgbm',
            'version': lgb.__version__,
            'training_time': training_time,
            'peak_utilization': max_util,
            'peak_memory': max_mem,
            'gpu_name': max_gpu_name
        }
    
    except Exception as e:
        print(f"Error testing LightGBM: {e}")
        import traceback
        traceback.print_exc()
        return {
            'library': 'lightgbm',
            'error': str(e)
        }

def test_h2o():
    """Test H2O XGBoost GPU peak utilization"""
    try:
        import h2o
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        print(f"\nTesting H2O {h2o.__version__} with XGBoost GPU")
        
        # Initialize H2O with less memory
        h2o.init(max_mem_size="1g", nthreads=-1, strict_version_check=False)
        
        # Create dataset
        n_samples = 50000  # Smaller dataset for H2O
        n_features = 10
        print(f"Creating dataset with {n_samples} samples and {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            random_state=42
        )
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        # Convert to H2O Frame
        print("Converting to H2O Frame...")
        frame = h2o.H2OFrame(df)
        
        # Convert target to categorical
        frame['target'] = frame['target'].asfactor()
        
        # Split data
        train, test = frame.split_frame(ratios=[0.8], seed=42)
        
        # Get features and target
        features = [f'feature_{i}' for i in range(n_features)]
        target = 'target'
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results, 0.1)  # Check every 0.1 seconds
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Train H2O XGBoost model
        print("Training H2O XGBoost on GPU...")
        try:
            start_time = time.time()
            model = H2OXGBoostEstimator(
                ntrees=20,          # Fewer trees for quick test
                max_depth=6,
                learn_rate=0.1,
                backend='gpu',      # Use GPU
                gpu_id=0
            )
            model.train(x=features, y=target, training_frame=train, validation_frame=test)
            training_time = time.time() - start_time
            h2o_gpu_success = True
        except Exception as e:
            print(f"Error training H2O XGBoost on GPU: {e}")
            training_time = time.time() - start_time
            h2o_gpu_success = False
        
        # Let monitoring run a bit longer to catch peak values
        time.sleep(0.5)
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Find peak utilization across all GPUs
        max_util = 0
        max_mem = 0
        max_gpu_name = ""
        
        for result in gpu_results:
            if result['utilization'] > max_util:
                max_util = result['utilization']
                max_gpu_name = result['name']
            if result['memory_used'] > max_mem:
                max_mem = result['memory_used']
        
        # Shutdown H2O
        h2o.cluster().shutdown()
        
        if h2o_gpu_success:
            print(f"H2O XGBoost completed in {training_time:.2f} seconds")
        else:
            print(f"H2O XGBoost failed after {training_time:.2f} seconds")
        
        print(f"Peak GPU Utilization: {max_util:.1f}% on {max_gpu_name}")
        print(f"Peak GPU Memory: {max_mem:.1f} MB")
        
        return {
            'library': 'h2o_xgboost',
            'version': h2o.__version__,
            'success': h2o_gpu_success,
            'training_time': training_time,
            'peak_utilization': max_util,
            'peak_memory': max_mem,
            'gpu_name': max_gpu_name
        }
    
    except Exception as e:
        print(f"Error testing H2O: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure H2O is shut down
        try:
            h2o.cluster().shutdown()
        except:
            pass
            
        return {
            'library': 'h2o_xgboost',
            'error': str(e)
        }

def main():
    """Main function"""
    print("=" * 80)
    print("PEAK GPU UTILIZATION TEST")
    print("=" * 80)
    
    # Create reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Store results
    results = []
    
    # Run XGBoost test
    print("\n" + "="*80)
    print("TESTING XGBOOST")
    print("="*80)
    results.append(test_xgboost())
    
    # Run LightGBM test
    print("\n" + "="*80)
    print("TESTING LIGHTGBM")
    print("="*80)
    results.append(test_lightgbm())
    
    # Run H2O test
    print("\n" + "="*80)
    print("TESTING H2O")
    print("="*80)
    results.append(test_h2o())
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF PEAK GPU UTILIZATION")
    print("="*80)
    
    for result in results:
        if 'error' in result:
            print(f"\n{result['library'].upper()}: Error - {result['error']}")
        else:
            print(f"\n{result['library'].upper()} (v{result.get('version', 'unknown')}):")
            print(f"  Training time: {result.get('training_time', 'N/A'):.2f} seconds")
            print(f"  Peak GPU utilization: {result.get('peak_utilization', 0):.1f}%")
            print(f"  Peak GPU memory: {result.get('peak_memory', 0):.1f} MB")
            print(f"  GPU: {result.get('gpu_name', 'unknown')}")
    
    # Save detailed results to file
    timestamp = int(time.time())
    result_file = os.path.join(reports_dir, f"peak_gpu_utilization_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())