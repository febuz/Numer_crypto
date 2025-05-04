#!/usr/bin/env python3
"""
Multi-GPU Test for H2O Sparkling Water

This script tests the ability of H2O Sparkling Water to utilize multiple
GPUs simultaneously. It creates multiple H2O XGBoost models running on 
different GPUs and measures the utilization and performance.

Requirements:
- Multiple NVIDIA GPUs
- CUDA support
- H2O Sparkling Water
- Java 11 or 17
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
    """Get detailed GPU information for all available GPUs"""
    try:
        # Run nvidia-smi command with more detailed information
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', 
             '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in nvidia_smi_output.strip().split('\n'):
            values = line.split(', ')
            if len(values) >= 6:
                gpu_info.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'utilization_pct': float(values[2]),
                    'memory_used_mb': float(values[3]),
                    'memory_total_mb': float(values[4]),
                    'temperature_c': float(values[5])
                })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def monitor_all_gpus(stop_event, results_dict, interval=0.1):
    """Monitor all GPUs in separate thread"""
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
                'memory_used': gpu['memory_used_mb'],
                'memory_total': gpu['memory_total_mb'],
                'temperature': gpu['temperature_c']
            })
        time.sleep(interval)

def create_synthetic_dataset(n_rows=100000, n_features=20, random_seed=42):
    """Create synthetic dataset for model training"""
    print(f"Creating dataset with {n_rows} samples and {n_features} features")
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=2,
        random_state=random_seed
    )
    
    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def setup_java(java_version=11):
    """Set up Java environment for testing"""
    if java_version == 11:
        java_home = "/usr/lib/jvm/java-11-openjdk-amd64"
    elif java_version == 17:
        java_home = "/usr/lib/jvm/java-17-openjdk-amd64"
    else:
        raise ValueError(f"Unsupported Java version: {java_version}")
    
    # Set environment variables
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
    
    # Add Java 17 module options if needed
    if java_version == 17:
        os.environ["_JAVA_OPTIONS"] = (
            "--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED "
            "--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED "
            "--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED "
            "--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED "
            "--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/sun.net=ALL-UNNAMED"
        )
    
    # Verify Java version
    try:
        java_version_output = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, encoding='utf-8'
        )
        print(f"Using Java:\n{java_version_output.strip()}")
        return True
    except Exception as e:
        print(f"Error setting up Java {java_version}: {e}")
        return False

def train_h2o_on_gpu(gpu_id, data_df, port_offset=0, model_name=None):
    """Train H2O XGBoost model on specific GPU"""
    model_name = model_name or f"XGBoost_Model_GPU{gpu_id}"
    print(f"\n=== Training {model_name} on GPU {gpu_id} ===")
    
    try:
        # Set specific GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Split data
        train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
        
        # Import required libraries
        from pyspark.sql import SparkSession
        import pysparkling
        from pyspark.sql.types import DoubleType, StructType, StructField
        from pyspark.ml.feature import VectorAssembler
        
        # Create Spark session with unique app name and port
        unique_app_name = f"H2OSparklingGPU_{gpu_id}_{int(time.time())}"
        unique_port = 54321 + port_offset  # Each instance needs a different port
        
        builder = SparkSession.builder \
            .appName(unique_app_name) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.jars.repositories", "https://h2oai.jfrog.io/artifactory/h2o-releases") \
            .config("spark.driver.port", str(unique_port)) \
            .config("spark.ui.port", str(10000 + port_offset))
        
        # Add Java module options if using Java 17
        if "add-opens" in os.environ.get("_JAVA_OPTIONS", ""):
            java_opts = os.environ["_JAVA_OPTIONS"]
            builder = builder \
                .config("spark.driver.extraJavaOptions", java_opts) \
                .config("spark.executor.extraJavaOptions", java_opts)
        
        spark = builder.getOrCreate()
        
        # Initialize H2O with unique port
        h2o_port = unique_port + 1000
        from pysparkling import H2OContext
        h2o_conf = pysparkling.H2OConf(spark)
        h2o_conf.set_internal_port_offset(h2o_port - 54321)
        h2o_context = H2OContext.getOrCreate(h2o_conf)
        
        # Convert pandas DataFrames to Spark DataFrames
        print(f"Converting data to Spark DataFrame (GPU {gpu_id})...")
        train_spark = spark.createDataFrame(train_df)
        test_spark = spark.createDataFrame(test_df)
        
        # Prepare features vector
        feature_cols = [col for col in train_df.columns if col != 'target']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_spark = assembler.transform(train_spark)
        test_spark = assembler.transform(test_spark)
        
        # Convert to H2O Frame for XGBoost
        print(f"Converting to H2O Frames (GPU {gpu_id})...")
        train_h2o = h2o_context.asH2OFrame(train_spark)
        test_h2o = h2o_context.asH2OFrame(test_spark)
        
        # Convert target to categorical for classification
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # Time the training process
        start_time = time.time()
        
        # Create and train XGBoost model with this GPU
        print(f"Training H2O XGBoost on GPU {gpu_id}...")
        from pysparkling.ml import H2OXGBoostEstimator
        
        estimator = H2OXGBoostEstimator(
            featuresCols=["features"],
            labelCol="target",
            tree_method="gpu_hist",
            gpu_id=0,  # Use 0 because we've set CUDA_VISIBLE_DEVICES to make the chosen GPU appear as 0
            ntrees=50,
            max_depth=10,
            learn_rate=0.1
        )
        
        model = estimator.fit(train_spark)
        
        training_time = time.time() - start_time
        
        # Evaluate model
        predictions = model.transform(test_spark)
        import h2o
        predictions_h2o = h2o_context.asH2OFrame(predictions)
        
        # Get AUC for binary classification
        from h2o.utils.metrics import Metrics
        perf = Metrics.make_metrics(
            predictions_h2o[predictions_h2o.names[-1]], 
            test_h2o['target'], 
            'binomial'
        )
        auc = perf.auc()
        
        print(f"GPU {gpu_id} Results:")
        print(f"  - Training Time: {training_time:.2f} seconds")
        print(f"  - AUC: {auc:.4f}")
        
        # Clean up
        h2o_context.stop()
        spark.stop()
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': True,
            'training_time': training_time,
            'auc': float(auc)
        }
            
    except Exception as e:
        print(f"Error training on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            h2o_context.stop()
            spark.stop()
        except:
            pass
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

def plot_multi_gpu_results(results, gpu_metrics, output_dir='../reports'):
    """Generate visualizations for multi-GPU test results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Training times comparison
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        print("No successful training results to plot")
        return
    
    # Plot training times
    plt.figure(figsize=(10, 6))
    gpu_ids = [r['gpu_id'] for r in successful_results]
    times = [r['training_time'] for r in successful_results]
    aucs = [r['auc'] for r in successful_results]
    
    bars = plt.bar(gpu_ids, times, color=['blue', 'green', 'red', 'purple'])
    plt.title('Training Time by GPU')
    plt.xlabel('GPU ID')
    plt.ylabel('Training Time (seconds)')
    
    # Add values on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        plt.annotate(f'{height:.2f}s\nAUC: {auc:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/multi_gpu_training_times_{timestamp}.png")
    
    # 2. GPU utilization over time
    # Plot each GPU's utilization on the same chart with different colors
    plt.figure(figsize=(14, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for idx, (gpu_id, metrics) in enumerate(gpu_metrics.items()):
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        start_time = df['timestamp'].min()
        df['seconds'] = df['timestamp'] - start_time
        
        color = colors[idx % len(colors)]
        plt.plot(df['seconds'], df['utilization'], 
                 label=f'GPU {gpu_id} ({df["name"].iloc[0]})', 
                 color=color)
    
    plt.title('GPU Utilization Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/multi_gpu_utilization_{timestamp}.png")
    
    # 3. Memory usage over time
    plt.figure(figsize=(14, 8))
    
    for idx, (gpu_id, metrics) in enumerate(gpu_metrics.items()):
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        start_time = df['timestamp'].min()
        df['seconds'] = df['timestamp'] - start_time
        
        color = colors[idx % len(colors)]
        plt.plot(df['seconds'], df['memory_used'], 
                 label=f'GPU {gpu_id} ({df["name"].iloc[0]})', 
                 color=color)
    
    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Used (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/multi_gpu_memory_{timestamp}.png")
    
    # 4. Aggregate metrics (peak utilization, peak memory)
    peaks = {}
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        peaks[gpu_id] = {
            'peak_util': df['utilization'].max(),
            'peak_memory': df['memory_used'].max(),
            'avg_util': df['utilization'].mean(),
            'avg_memory': df['memory_used'].mean(),
            'gpu_name': df['name'].iloc[0]
        }
    
    # Plot peak utilization
    plt.figure(figsize=(12, 6))
    gpu_ids = list(peaks.keys())
    peak_utils = [peaks[gpu_id]['peak_util'] for gpu_id in gpu_ids]
    avg_utils = [peaks[gpu_id]['avg_util'] for gpu_id in gpu_ids]
    
    x = np.arange(len(gpu_ids))
    width = 0.35
    
    plt.bar(x - width/2, peak_utils, width, label='Peak Utilization', color='blue')
    plt.bar(x + width/2, avg_utils, width, label='Average Utilization', color='green')
    
    plt.xlabel('GPU ID')
    plt.ylabel('Utilization (%)')
    plt.title('Peak vs Average GPU Utilization')
    plt.xticks(x, gpu_ids)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/multi_gpu_peak_utilization_{timestamp}.png")
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main function to run multi-GPU tests"""
    parser = argparse.ArgumentParser(description='Multi-GPU Test for H2O Sparkling Water')
    parser.add_argument('--rows', type=int, default=100000, help='Number of rows in the dataset')
    parser.add_argument('--cols', type=int, default=20, help='Number of features in the dataset')
    parser.add_argument('--java-version', type=int, default=11, choices=[11, 17], help='Java version to use')
    parser.add_argument('--output-dir', type=str, default='../reports', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTI-GPU TEST FOR H2O SPARKLING WATER")
    print("=" * 80)
    
    # Create reports directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up Java environment
    if not setup_java(args.java_version):
        print("Failed to set up Java environment. Exiting.")
        return 1
    
    # Get available GPUs
    gpus = get_gpu_info()
    if not gpus:
        print("No GPUs detected. Exiting.")
        return 1
    
    print(f"Detected {len(gpus)} GPUs:")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_total_mb']} MB")
        print(f"    Current utilization: {gpu['utilization_pct']}%")
    
    # Create synthetic dataset (shared across all GPU tests)
    data_df = create_synthetic_dataset(args.rows, args.cols)
    
    # Start GPU monitoring in a separate thread
    gpu_metrics = {}  # Dictionary to store metrics for each GPU
    monitor_stop_event = Event()
    monitor_thread = threading.Thread(
        target=monitor_all_gpus, 
        args=(monitor_stop_event, gpu_metrics)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Train on all available GPUs in parallel
    print("\n" + "="*80)
    print(f"TRAINING H2O SPARKLING WATER MODELS ON {len(gpus)} GPUs")
    print("="*80)
    
    results = []
    
    # Configure parallel execution - limit based on system resources
    max_parallel = min(len(gpus), 3)  # Limit to 3 GPUs at once to avoid overloading
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit a task for each GPU
        futures = {}
        for i, gpu in enumerate(gpus):
            gpu_id = gpu['index']
            model_name = f"XGBoost_Model_GPU{gpu_id}"
            port_offset = i * 100  # Ensure unique ports
            
            future = executor.submit(
                train_h2o_on_gpu, 
                gpu_id, 
                data_df, 
                port_offset, 
                model_name
            )
            futures[future] = gpu_id
        
        # Collect results as they complete
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed training on GPU {gpu_id}")
            except Exception as e:
                print(f"Error training on GPU {gpu_id}: {e}")
                results.append({
                    'gpu_id': gpu_id,
                    'success': False,
                    'error': str(e)
                })
    
    # Stop monitoring
    monitor_stop_event.set()
    monitor_thread.join()
    
    # Print summary results
    print("\n" + "="*80)
    print("MULTI-GPU TEST RESULTS")
    print("="*80)
    
    successful_count = sum(1 for r in results if r.get('success', False))
    failed_count = len(results) - successful_count
    
    print(f"Successful models: {successful_count}")
    print(f"Failed models: {failed_count}")
    
    # Print results for each GPU
    for result in results:
        gpu_id = result['gpu_id']
        if result.get('success', False):
            print(f"\nGPU {gpu_id}:")
            print(f"  Model: {result['model_name']}")
            print(f"  Training time: {result['training_time']:.2f} seconds")
            print(f"  AUC: {result['auc']:.4f}")
        else:
            print(f"\nGPU {gpu_id}: Failed - {result.get('error', 'Unknown error')}")
    
    # Print peak utilization for each GPU
    print("\nPeak GPU Utilization:")
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        peak_util = df['utilization'].max()
        peak_memory = df['memory_used'].max()
        avg_util = df['utilization'].mean()
        
        print(f"  GPU {gpu_id}:")
        print(f"    Peak: {peak_util:.2f}%")
        print(f"    Average: {avg_util:.2f}%")
        print(f"    Peak Memory: {peak_memory:.2f} MB")
    
    # Generate plots
    plot_multi_gpu_results(results, gpu_metrics, args.output_dir)
    
    # Save detailed results to file
    timestamp = int(time.time())
    result_file = os.path.join(args.output_dir, f"multi_gpu_test_{timestamp}.json")
    
    # Prepare metrics for saving (without raw monitoring data)
    metrics_summary = {}
    for gpu_id, metrics in gpu_metrics.items():
        if not metrics:
            continue
            
        df = pd.DataFrame(metrics)
        metrics_summary[str(gpu_id)] = {
            'gpu_name': df['name'].iloc[0],
            'samples_count': len(df),
            'peak_utilization': float(df['utilization'].max()),
            'avg_utilization': float(df['utilization'].mean()),
            'peak_memory': float(df['memory_used'].max()),
            'avg_memory': float(df['memory_used'].mean()),
            'peak_temperature': float(df['temperature'].max())
        }
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_count': len(gpus),
            'java_version': args.java_version,
            'results': results,
            'metrics_summary': metrics_summary,
            'test_parameters': vars(args)
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    # Success if at least one model trained successfully
    return 0 if successful_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())