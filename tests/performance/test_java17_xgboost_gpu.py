#!/usr/bin/env python3
"""
Java 17 XGBoost GPU Test

This script tests XGBoost GPU performance with Java 17,
monitoring GPU utilization across all available GPUs.
"""

import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def monitor_gpus(stop_event, results_dict, interval=0.1):
    """Monitor all GPUs in a separate thread"""
    while not stop_event.is_set():
        gpu_info = get_gpu_info()
        timestamp = time.time()
        
        for gpu in gpu_info:
            gpu_idx = gpu['index']
            # Initialize list for this GPU if it doesn't exist
            if gpu_idx not in results_dict:
                results_dict[gpu_idx] = []
                
            results_dict[gpu_idx].append({
                'timestamp': timestamp,
                'gpu_index': gpu_idx,
                'name': gpu['name'],
                'utilization': gpu['utilization_pct'],
                'memory_used': gpu['memory_used_mb']
            })
        time.sleep(interval)

def train_xgboost_on_gpu(gpu_id, X_train, X_test, y_train, y_test, model_name=None):
    """Train XGBoost model on specific GPU"""
    model_name = model_name or f"XGBoost_Model_GPU{gpu_id}"
    print(f"\n=== Training {model_name} on GPU {gpu_id} ===")
    
    try:
        # We'll set CUDA_VISIBLE_DEVICES below to constrain to specific GPU
        
        # Import XGBoost
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters for GPU training - XGBoost 3.0 requires using only 'device' for GPU selection
        # Set CUDA_VISIBLE_DEVICES to constrain available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        params = {
            'device': 'cuda:0',  # Use GPU - since we set CUDA_VISIBLE_DEVICES, this will be the correct GPU
            'tree_method': 'hist',  # Modern XGBoost 3.0+ syntax (gpu_hist is deprecated)
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 12,  # Deeper tree to stress GPU more
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'nthread': 16  # Use more CPU threads to help with GPU data transfer
        }
        
        # Train model
        start_time = time.time()
        print(f"Training XGBoost model on GPU {gpu_id}...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=200,  # More rounds to stress GPU
            evals=[(dtest, 'test')],
            verbose_eval=50
        )
        training_time = time.time() - start_time
        
        print(f"GPU {gpu_id} - Training completed in {training_time:.2f} seconds")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': True,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"Error training XGBoost on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to run GPU tests with Java 17"""
    # Print Java version to confirm Java 17
    try:
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
        print(f"Java version:\n{java_version}")
        
        if "17" not in java_version:
            print("Warning: Not using Java 17. Please run with Java 17.")
    except Exception as e:
        print(f"Error checking Java version: {e}")
    
    print("=" * 80)
    print("XGBOOST GPU TEST WITH JAVA 17")
    print("=" * 80)
    
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
    
    # Create synthetic dataset (larger to stress GPU)
    print("\nCreating synthetic dataset...")
    X, y = make_classification(
        n_samples=500000,  # 500K samples
        n_features=50,     # 50 features
        n_informative=40,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start GPU monitoring
    gpu_metrics = {}
    monitor_stop_event = Event()
    monitor_thread = threading.Thread(
        target=monitor_gpus,
        args=(monitor_stop_event, gpu_metrics)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Train on all GPUs in parallel
    print("\n" + "="*80)
    print(f"TRAINING XGBOOST ON {len(gpus)} GPUs IN PARALLEL")
    print("="*80)
    
    results = []
    
    # Use ThreadPoolExecutor to run on all GPUs in parallel
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
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
    
    # Stop monitoring
    monitor_stop_event.set()
    monitor_thread.join()
    
    # Print summary
    print("\n" + "="*80)
    print("GPU UTILIZATION SUMMARY")
    print("="*80)
    
    successful_count = sum(1 for r in results if r.get('success', False))
    failed_count = len(results) - successful_count
    
    print(f"Successful models: {successful_count}")
    print(f"Failed models: {failed_count}")
    
    # GPU utilization summary
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        peak_util = df['utilization'].max()
        avg_util = df['utilization'].mean()
        peak_memory = df['memory_used'].max()
        
        print(f"\nGPU {gpu_id}:")
        print(f"  Peak utilization: {peak_util:.2f}%")
        print(f"  Average utilization: {avg_util:.2f}%")
        print(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    # Create GPU utilization charts
    print("\nSaving GPU utilization charts...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Plot utilization over time for each GPU
    plt.figure(figsize=(12, 8))
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        start_time = df['timestamp'].min()
        df['seconds'] = df['timestamp'] - start_time
        
        plt.plot(df['seconds'], df['utilization'], label=f'GPU {gpu_id}')
    
    plt.title('GPU Utilization Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(reports_dir / f"gpu_utilization_java17_{timestamp}.png")
    
    # Save results to JSON
    result_file = reports_dir / f"xgboost_java17_gpu_test_{timestamp}.json"
    
    # Prepare metrics for saving (without raw monitoring data)
    gpu_summary = {}
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        gpu_summary[str(gpu_id)] = {
            'peak_utilization': float(df['utilization'].max()),
            'avg_utilization': float(df['utilization'].mean()),
            'peak_memory': float(df['memory_used'].max()),
            'samples_count': len(df)
        }
    
    with open(result_file, 'w') as f:
        json.dump({
            'java_version': java_version.strip(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_count': len(gpus),
            'successful_models': successful_count,
            'failed_models': failed_count,
            'training_results': results,
            'gpu_summary': gpu_summary
        }, f, indent=2)
    
    print(f"Results saved to: {result_file}")
    
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())