#!/usr/bin/env python3
"""
Direct GPU Utilization Test for ML Libraries
Tests maximum GPU utilization with XGBoost, LightGBM, and H2O.
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
def monitor_gpus(stop_event, results_list, interval=0.5):
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

def test_xgboost_gpu():
    """Test XGBoost GPU utilization"""
    try:
        import xgboost as xgb
        print(f"\nTesting XGBoost version {xgb.__version__} GPU utilization")
        
        # Create a large dataset to stress the GPU
        print("Creating large dataset for XGBoost...")
        X, y = make_classification(
            n_samples=500000,  # 500K samples
            n_features=50,     # 50 features
            n_informative=40,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Free up memory
        del X, y
        
        # Convert to DMatrix
        print("Converting to DMatrix...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set GPU parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',  # Use GPU
            'max_depth': 10,            # Deeper trees to stress GPU
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results)
        )
        monitor_thread.start()
        
        # Train model
        print("\nTraining XGBoost model on GPU...")
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,  # More rounds to stress GPU
            evals=[(dtest, 'test')],
            verbose_eval=10
        )
        training_time = time.time() - start_time
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Get peak utilization
        max_util = max([result['utilization'] for result in gpu_results]) if gpu_results else 0
        max_mem = max([result['memory_used'] for result in gpu_results]) if gpu_results else 0
        
        print(f"XGBoost GPU Training completed in {training_time:.2f} seconds")
        print(f"Peak GPU Utilization: {max_util:.1f}%")
        print(f"Peak GPU Memory Used: {max_mem:.1f} MB")
        
        return {
            'success': True,
            'training_time': training_time,
            'peak_utilization': max_util,
            'peak_memory': max_mem
        }
        
    except Exception as e:
        print(f"Error testing XGBoost GPU: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_lightgbm_gpu():
    """Test LightGBM GPU utilization"""
    try:
        import lightgbm as lgb
        print(f"\nTesting LightGBM version {lgb.__version__} GPU utilization")
        
        # Create a large dataset to stress the GPU
        print("Creating large dataset for LightGBM...")
        X, y = make_classification(
            n_samples=500000,  # 500K samples
            n_features=50,     # 50 features
            n_informative=40,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Free up memory
        del X, y
        
        # Convert to LightGBM Dataset
        print("Converting to LightGBM Dataset...")
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Set GPU parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'device': 'gpu',  # Use GPU
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 10,  # Deeper trees to stress GPU
            'num_leaves': 127,
            'learning_rate': 0.1,
            'verbosity': -1,
            'seed': 42
        }
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results)
        )
        monitor_thread.start()
        
        # Train model
        print("\nTraining LightGBM model on GPU...")
        start_time = time.time()
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,  # More rounds to stress GPU
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation(period=10)]
        )
        training_time = time.time() - start_time
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Get peak utilization
        max_util = max([result['utilization'] for result in gpu_results]) if gpu_results else 0
        max_mem = max([result['memory_used'] for result in gpu_results]) if gpu_results else 0
        
        print(f"LightGBM GPU Training completed in {training_time:.2f} seconds")
        print(f"Peak GPU Utilization: {max_util:.1f}%")
        print(f"Peak GPU Memory Used: {max_mem:.1f} MB")
        
        return {
            'success': True,
            'training_time': training_time,
            'peak_utilization': max_util,
            'peak_memory': max_mem
        }
        
    except Exception as e:
        print(f"Error testing LightGBM GPU: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_h2o_xgboost_gpu():
    """Test H2O XGBoost GPU utilization"""
    try:
        import h2o
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        
        print(f"\nTesting H2O XGBoost GPU utilization")
        
        # Initialize H2O
        h2o.init(max_mem_size="2g", strict_version_check=False)
        
        # Create a large dataset to stress the GPU
        print("Creating large dataset for H2O XGBoost...")
        X, y = make_classification(
            n_samples=200000,  # 200K samples (smaller as H2O has more overhead)
            n_features=20,     # 20 features
            n_informative=15,
            random_state=42
        )
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Split the data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Free up memory
        del X, y, df
        
        # Convert to H2O frames
        print("Converting to H2O frames...")
        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
        
        # Convert target to categorical for classification
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # Features and target
        features = train_df.columns.tolist()
        target = 'target'
        features.remove(target)
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_gpus,
            args=(stop_event, gpu_results)
        )
        monitor_thread.start()
        
        # Train model
        print("\nTraining H2O XGBoost model on GPU...")
        try:
            start_time = time.time()
            model = H2OXGBoostEstimator(
                ntrees=100,       # More trees to stress GPU
                max_depth=10,     # Deeper trees to stress GPU
                learn_rate=0.1,
                seed=42,
                backend='gpu',    # Use GPU
                gpu_id=0
            )
            model.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
            training_time = time.time() - start_time
            
            # Get model performance
            perf = model.model_performance(test_h2o)
            auc = perf.auc()
            
            h2o_success = True
            
        except Exception as e:
            print(f"Error training H2O XGBoost on GPU: {e}")
            training_time = time.time() - start_time
            auc = None
            h2o_success = False
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Get peak utilization
        max_util = max([result['utilization'] for result in gpu_results]) if gpu_results else 0
        max_mem = max([result['memory_used'] for result in gpu_results]) if gpu_results else 0
        
        # Shutdown H2O
        h2o.cluster().shutdown()
        
        if h2o_success:
            print(f"H2O XGBoost GPU Training completed in {training_time:.2f} seconds")
            print(f"AUC: {auc}")
        else:
            print(f"H2O XGBoost GPU Training failed after {training_time:.2f} seconds")
            
        print(f"Peak GPU Utilization: {max_util:.1f}%")
        print(f"Peak GPU Memory Used: {max_mem:.1f} MB")
        
        return {
            'success': h2o_success,
            'training_time': training_time,
            'auc': auc if h2o_success else None,
            'peak_utilization': max_util,
            'peak_memory': max_mem
        }
        
    except Exception as e:
        print(f"Error testing H2O GPU: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure H2O is shut down
        try:
            h2o.cluster().shutdown()
        except:
            pass
            
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to run GPU utilization tests"""
    print("=" * 80)
    print("DIRECT GPU UTILIZATION TEST")
    print("=" * 80)
    
    # Print system info
    print("\nSystem Information:")
    try:
        # Get CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read().split('\n')
        cpu_model = [line for line in cpu_info if 'model name' in line]
        if cpu_model:
            print(f"CPU: {cpu_model[0].split(':')[1].strip()}")
        
        # Get memory info
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read().split('\n')
        total_mem = [line for line in mem_info if 'MemTotal' in line]
        if total_mem:
            mem_kb = int(total_mem[0].split(':')[1].strip().split()[0])
            print(f"Memory: {mem_kb / 1024 / 1024:.1f} GB")
    except Exception as e:
        print(f"Error getting system info: {e}")
    
    # Get GPU info
    print("\nAvailable GPUs:")
    gpus = get_gpu_info()
    for gpu in gpus:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Current utilization: {gpu['utilization_pct']}%")
        print(f"  Current memory usage: {gpu['memory_used_mb']} MB")
    
    if not gpus:
        print("No GPUs detected. Exiting.")
        return 1
    
    # Check libraries
    print("\nChecking libraries:")
    libraries = {}
    
    # Check XGBoost
    try:
        import xgboost as xgb
        libraries['xgboost'] = {'available': True, 'version': xgb.__version__}
        print(f"XGBoost: Available, version {xgb.__version__}")
    except ImportError:
        libraries['xgboost'] = {'available': False}
        print("XGBoost: Not available")
    
    # Check LightGBM
    try:
        import lightgbm as lgb
        libraries['lightgbm'] = {'available': True, 'version': lgb.__version__}
        print(f"LightGBM: Available, version {lgb.__version__}")
    except ImportError:
        libraries['lightgbm'] = {'available': False}
        print("LightGBM: Not available")
    
    # Check H2O
    try:
        import h2o
        libraries['h2o'] = {'available': True, 'version': h2o.__version__}
        print(f"H2O: Available, version {h2o.__version__}")
    except ImportError:
        libraries['h2o'] = {'available': False}
        print("H2O: Not available")
    
    # Run tests
    results = {}
    
    # Test XGBoost
    if libraries.get('xgboost', {}).get('available', False):
        print("\n" + "="*80)
        print("TESTING XGBOOST GPU UTILIZATION")
        print("="*80)
        results['xgboost'] = test_xgboost_gpu()
    
    # Test LightGBM
    if libraries.get('lightgbm', {}).get('available', False):
        print("\n" + "="*80)
        print("TESTING LIGHTGBM GPU UTILIZATION")
        print("="*80)
        results['lightgbm'] = test_lightgbm_gpu()
    
    # Test H2O
    if libraries.get('h2o', {}).get('available', False):
        print("\n" + "="*80)
        print("TESTING H2O XGBOOST GPU UTILIZATION")
        print("="*80)
        results['h2o_xgboost'] = test_h2o_xgboost_gpu()
    
    # Print summary
    print("\n" + "="*80)
    print("GPU UTILIZATION SUMMARY")
    print("="*80)
    
    for lib, res in results.items():
        if res.get('success', False):
            print(f"\n{lib.upper()}:")
            print(f"  Training time: {res.get('training_time', 'N/A'):.2f} seconds")
            print(f"  Peak GPU utilization: {res.get('peak_utilization', 0):.1f}%")
            print(f"  Peak GPU memory usage: {res.get('peak_memory', 0):.1f} MB")
        else:
            print(f"\n{lib.upper()}: Failed - {res.get('error', 'Unknown error')}")
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "reports", f"gpu_utilization_results_{int(time.time())}.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'libraries': libraries,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())