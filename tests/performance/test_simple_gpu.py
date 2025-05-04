#!/usr/bin/env python3
"""
Simple GPU Utilization Test

This script tests the basic GPU utilization of XGBoost and LightGBM
on multiple GPUs. It's a simplified version that doesn't require
H2O Sparkling Water.
"""

import os
import sys
import time
import json
import threading
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

# GPU monitoring utilities
def get_gpu_info():
    """Get GPU utilization and memory information using nvidia-smi"""
    try:
        # Run nvidia-smi command and parse output
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in nvidia_smi_output.strip().split('\n'):
            values = line.split(', ')
            if len(values) == 4:
                gpu_info.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'utilization_pct': float(values[2]),
                    'memory_used_mb': float(values[3])
                })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def monitor_gpus(stop_event, results_list, interval=0.1):
    """Monitor GPU utilization in a separate thread"""
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
        time.sleep(interval)

def create_synthetic_dataset(n_rows=100000, n_cols=20, random_seed=42):
    """Create synthetic dataset for model training"""
    print(f"Creating dataset with {n_rows} samples and {n_cols} features")
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_cols,
        n_informative=int(n_cols * 0.8),
        n_redundant=int(n_cols * 0.1),
        n_classes=2,
        random_state=random_seed
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    return X_train, X_test, y_train, y_test

def train_xgboost_on_gpu(gpu_id, X_train, X_test, y_train, y_test, model_name=None):
    """Train XGBoost model on specific GPU"""
    model_name = model_name or f"XGBoost_Model_GPU{gpu_id}"
    print(f"\n=== Training {model_name} on GPU {gpu_id} ===")
    
    try:
        # Set specific GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Import XGBoost
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = Event()
        monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_event, gpu_results))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Set parameters for GPU training
        params = {
            'device': 'cuda',  # Use GPU
            'tree_method': 'hist',  # GPU-accelerated histogram 
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train model
        start_time = time.time()
        print(f"Training XGBoost model on GPU {gpu_id}...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=100,
            evals=[(dtest, 'test')],
            verbose_eval=25
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        pred_probs = model.predict(dtest)
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Calculate metrics
        peak_util = 0
        peak_memory = 0
        avg_util = 0
        
        if gpu_results:
            df = pd.DataFrame(gpu_results)
            peak_util = df['utilization'].max()
            peak_memory = df['memory_used'].max()
            avg_util = df['utilization'].mean()
        
        print(f"GPU {gpu_id} Results:")
        print(f"  - Training Time: {training_time:.2f} seconds")
        print(f"  - Peak GPU Utilization: {peak_util:.2f}%")
        print(f"  - Average GPU Utilization: {avg_util:.2f}%")
        print(f"  - Peak GPU Memory: {peak_memory:.2f} MB")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'library': 'xgboost',
            'success': True,
            'training_time': training_time,
            'peak_utilization': peak_util,
            'avg_utilization': avg_util,
            'peak_memory': peak_memory,
            'monitoring_data': gpu_results
        }
        
    except Exception as e:
        print(f"Error training XGBoost on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'library': 'xgboost',
            'success': False,
            'error': str(e)
        }

def train_lightgbm_on_gpu(gpu_id, X_train, X_test, y_train, y_test, model_name=None):
    """Train LightGBM model on specific GPU"""
    model_name = model_name or f"LightGBM_Model_GPU{gpu_id}"
    print(f"\n=== Training {model_name} on GPU {gpu_id} ===")
    
    try:
        # Set specific GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Import LightGBM
        import lightgbm as lgb
        print(f"LightGBM version: {lgb.__version__}")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = Event()
        monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_event, gpu_results))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Set parameters for GPU training
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,  # Uses 0 because we set CUDA_VISIBLE_DEVICES
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Train model
        start_time = time.time()
        print(f"Training LightGBM model on GPU {gpu_id}...")
        callbacks = []
        if hasattr(lgb, 'early_stopping'):
            # LightGBM 4.0+
            callbacks.append(lgb.early_stopping(10))
        if hasattr(lgb, 'log_evaluation'):
            # LightGBM 4.0+
            callbacks.append(lgb.log_evaluation(25))
            
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=callbacks
        )
        training_time = time.time() - start_time
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Calculate metrics
        peak_util = 0
        peak_memory = 0
        avg_util = 0
        
        if gpu_results:
            df = pd.DataFrame(gpu_results)
            peak_util = df['utilization'].max()
            peak_memory = df['memory_used'].max()
            avg_util = df['utilization'].mean()
        
        print(f"GPU {gpu_id} Results:")
        print(f"  - Training Time: {training_time:.2f} seconds")
        print(f"  - Peak GPU Utilization: {peak_util:.2f}%")
        print(f"  - Average GPU Utilization: {avg_util:.2f}%")
        print(f"  - Peak GPU Memory: {peak_memory:.2f} MB")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'library': 'lightgbm',
            'success': True,
            'training_time': training_time,
            'peak_utilization': peak_util,
            'avg_utilization': avg_util,
            'peak_memory': peak_memory,
            'monitoring_data': gpu_results
        }
        
    except Exception as e:
        print(f"Error training LightGBM on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'library': 'lightgbm',
            'success': False,
            'error': str(e)
        }

def plot_results(results, output_dir='../reports'):
    """Generate visualizations for test results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        print("No successful training results to plot")
        return
    
    # Group results by library
    libraries = {}
    for result in successful_results:
        lib = result['library']
        if lib not in libraries:
            libraries[lib] = []
        libraries[lib].append(result)
    
    # Plot per-GPU utilization for each library
    for lib, lib_results in libraries.items():
        plt.figure(figsize=(12, 8))
        
        # Plot utilization bars
        gpu_ids = [r['gpu_id'] for r in lib_results]
        peak_utils = [r['peak_utilization'] for r in lib_results]
        avg_utils = [r['avg_utilization'] for r in lib_results]
        
        x = range(len(gpu_ids))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], peak_utils, width, label='Peak Utilization')
        plt.bar([i + width/2 for i in x], avg_utils, width, label='Avg Utilization')
        
        plt.xlabel('GPU ID')
        plt.ylabel('Utilization (%)')
        plt.title(f'{lib} GPU Utilization by GPU')
        plt.xticks(x, [f'GPU {gpu_id}' for gpu_id in gpu_ids])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(peak_utils):
            plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
        for i, v in enumerate(avg_utils):
            plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
        
        plt.savefig(f"{output_dir}/{lib}_gpu_utilization_{timestamp}.png")
    
    # Create comparison plot across libraries
    if len(libraries) > 1:
        plt.figure(figsize=(14, 10))
        
        # Organize data
        lib_names = []
        lib_peak_utils = []
        lib_avg_utils = []
        lib_train_times = []
        
        for lib, lib_results in libraries.items():
            # Average across GPUs for each library
            avg_peak = sum(r['peak_utilization'] for r in lib_results) / len(lib_results)
            avg_util = sum(r['avg_utilization'] for r in lib_results) / len(lib_results)
            avg_time = sum(r['training_time'] for r in lib_results) / len(lib_results)
            
            lib_names.append(lib)
            lib_peak_utils.append(avg_peak)
            lib_avg_utils.append(avg_util)
            lib_train_times.append(avg_time)
        
        # Plot
        plt.subplot(2, 1, 1)
        x = range(len(lib_names))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], lib_peak_utils, width, label='Peak Utilization')
        plt.bar([i + width/2 for i in x], lib_avg_utils, width, label='Avg Utilization')
        
        plt.xlabel('Library')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization Comparison Between Libraries')
        plt.xticks(x, lib_names)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(lib_peak_utils):
            plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
        for i, v in enumerate(lib_avg_utils):
            plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
        
        # Plot training times
        plt.subplot(2, 1, 2)
        bars = plt.bar(lib_names, lib_train_times)
        plt.xlabel('Library')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/library_comparison_{timestamp}.png")
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main function to run GPU tests"""
    parser = argparse.ArgumentParser(description='Simple Multi-GPU Test')
    parser.add_argument('--rows', type=int, default=100000, help='Number of rows in the dataset')
    parser.add_argument('--cols', type=int, default=20, help='Number of features in the dataset')
    parser.add_argument('--output-dir', type=str, default='../reports', help='Output directory for results')
    parser.add_argument('--library', type=str, choices=['xgboost', 'lightgbm', 'both'], default='both',
                      help='Which library to test')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SIMPLE MULTI-GPU TEST")
    print("=" * 80)
    
    # Create reports directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get available GPUs
    gpus = get_gpu_info()
    if not gpus:
        print("No GPUs detected. Exiting.")
        return 1
    
    print(f"Detected {len(gpus)} GPUs:")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Current utilization: {gpu['utilization_pct']}%")
        print(f"    Current memory usage: {gpu['memory_used_mb']} MB")
    
    # Create dataset (shared across all tests)
    X_train, X_test, y_train, y_test = create_synthetic_dataset(args.rows, args.cols)
    
    # Run tests
    results = []
    
    # Configure parallel execution - limit based on system resources
    max_parallel = min(len(gpus), 3)  # Limit to 3 GPUs at once to avoid overloading
    
    if args.library in ['xgboost', 'both']:
        print("\n" + "="*80)
        print("TESTING XGBOOST ON MULTIPLE GPUS")
        print("="*80)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for gpu in gpus:
                gpu_id = gpu['index']
                future = executor.submit(
                    train_xgboost_on_gpu,
                    gpu_id,
                    X_train, X_test, y_train, y_test,
                    f"XGBoost_Model_GPU{gpu_id}"
                )
                futures[future] = gpu_id
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error running XGBoost on GPU {gpu_id}: {e}")
    
    if args.library in ['lightgbm', 'both']:
        print("\n" + "="*80)
        print("TESTING LIGHTGBM ON MULTIPLE GPUS")
        print("="*80)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for gpu in gpus:
                gpu_id = gpu['index']
                future = executor.submit(
                    train_lightgbm_on_gpu,
                    gpu_id,
                    X_train, X_test, y_train, y_test,
                    f"LightGBM_Model_GPU{gpu_id}"
                )
                futures[future] = gpu_id
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error running LightGBM on GPU {gpu_id}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("GPU TEST RESULTS")
    print("="*80)
    
    successful_count = sum(1 for r in results if r.get('success', False))
    failed_count = len(results) - successful_count
    
    print(f"Successful models: {successful_count}")
    print(f"Failed models: {failed_count}")
    
    # Group by library
    lib_results = {}
    for result in results:
        if result.get('success', False):
            lib = result['library']
            if lib not in lib_results:
                lib_results[lib] = []
            lib_results[lib].append(result)
    
    # Print results by library
    for lib, lib_res in lib_results.items():
        print(f"\n{lib.upper()} Results:")
        for res in lib_res:
            gpu_id = res['gpu_id']
            print(f"  GPU {gpu_id}:")
            print(f"    Training time: {res['training_time']:.2f} seconds")
            print(f"    Peak GPU Utilization: {res['peak_utilization']:.2f}%")
            print(f"    Avg GPU Utilization: {res['avg_utilization']:.2f}%")
            print(f"    Peak Memory Usage: {res['peak_memory']:.2f} MB")
    
    # Generate plots
    if successful_count > 0:
        plot_results(results, args.output_dir)
    
    # Save detailed results to file
    timestamp = int(time.time())
    result_file = os.path.join(args.output_dir, f"simple_gpu_test_{timestamp}.json")
    
    # Prepare metrics for saving (without raw monitoring data)
    results_for_saving = []
    for res in results:
        # Create a copy without the large monitoring data
        res_copy = {k: v for k, v in res.items() if k != 'monitoring_data'}
        if 'monitoring_data' in res:
            res_copy['monitoring_data_points'] = len(res['monitoring_data'])
        results_for_saving.append(res_copy)
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_count': len(gpus),
            'dataset_rows': args.rows,
            'dataset_cols': args.cols,
            'results': results_for_saving
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())