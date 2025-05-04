#!/usr/bin/env python3
"""
Java 11 vs Java 17 GPU Performance Comparison

This script compares the GPU performance between Java 11 and Java 17
when running H2O Sparkling Water with XGBoost. It measures:
1. Peak GPU utilization
2. Training time
3. Memory usage
4. Accuracy metrics

Requirements:
- NVIDIA GPU with CUDA support
- Java 11 and Java 17 installed
- H2O Sparkling Water
- XGBoost
"""

import os
import sys
import time
import json
import subprocess
import threading
import argparse
from pathlib import Path
from datetime import datetime
from threading import Event

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
    
    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_cols)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_seed)
    
    return train_df, test_df

def setup_java(java_version):
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

def test_h2o_sparkling_gpu(java_version, train_df, test_df, gpu_id=0):
    """Test H2O Sparkling Water with GPU acceleration using specific Java version"""
    print(f"\n=== Testing H2O Sparkling Water GPU with Java {java_version} ===")
    
    # Set up Java environment
    if not setup_java(java_version):
        return {
            'java_version': java_version,
            'success': False,
            'error': 'Failed to set up Java environment'
        }
    
    try:
        # Import required libraries
        from pyspark.sql import SparkSession
        import pysparkling
        from pyspark.sql.types import DoubleType, StructType, StructField
        from pyspark.ml.feature import VectorAssembler
        
        print("Initializing Spark and H2O...")
        
        # Create Spark session with appropriate Java config
        builder = SparkSession.builder \
            .appName(f"H2OSparklingGPUTest_Java{java_version}") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g")
        
        # Add Java 17 specific configurations if needed
        if java_version == 17:
            java_opts = (
                "--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED "
                "--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED "
                "--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED "
                "--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED "
                "--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED "
                "--add-opens=java.base/java.net=ALL-UNNAMED "
                "--add-opens=java.base/sun.net=ALL-UNNAMED"
            )
            builder = builder \
                .config("spark.driver.extraJavaOptions", java_opts) \
                .config("spark.executor.extraJavaOptions", java_opts)
        
        spark = builder.getOrCreate()
            
        # Initialize H2O Sparkling with GPU
        from pysparkling import H2OContext
        h2o_context = H2OContext.getOrCreate()
        
        # Start GPU monitoring
        gpu_results = []
        stop_event = Event()
        monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_event, gpu_results))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Convert pandas DataFrames to Spark DataFrames
        print("Converting data to Spark DataFrame...")
        train_spark = spark.createDataFrame(train_df)
        test_spark = spark.createDataFrame(test_df)
        
        # Prepare features vector
        feature_cols = [col for col in train_df.columns if col != 'target']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_spark = assembler.transform(train_spark)
        test_spark = assembler.transform(test_spark)
        
        # Convert to H2O Frame for XGBoost
        print("Converting to H2O Frames...")
        train_h2o = h2o_context.asH2OFrame(train_spark)
        test_h2o = h2o_context.asH2OFrame(test_spark)
        
        # Convert target to categorical for classification
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # Time the training process
        start_time = time.time()
        
        # Create and train XGBoost model with GPU through H2O
        print(f"Training H2O XGBoost with GPU acceleration (Java {java_version})...")
        from pysparkling.ml import H2OXGBoostEstimator
        
        estimator = H2OXGBoostEstimator(
            featuresCols=["features"],
            labelCol="target",
            tree_method="gpu_hist",
            gpu_id=gpu_id,
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
            train_h2o['target'], 
            'binomial'
        )
        auc = perf.auc()
        
        # Stop GPU monitoring
        stop_event.set()
        monitor_thread.join()
        
        # Calculate peak utilization and memory
        df = pd.DataFrame(gpu_results)
        if not df.empty:
            peak_utilization = df['utilization'].max()
            peak_memory = df['memory_used'].max()
            avg_utilization = df['utilization'].mean()
            
            print(f"Java {java_version} H2O Sparkling GPU Results:")
            print(f"  - Peak GPU Utilization: {peak_utilization:.2f}%")
            print(f"  - Avg GPU Utilization: {avg_utilization:.2f}%")
            print(f"  - Peak GPU Memory: {peak_memory:.2f} MB")
            print(f"  - Training Time: {training_time:.2f} seconds")
            print(f"  - AUC: {auc:.4f}")
            
            # Clean up
            h2o_context.stop()
            spark.stop()
            
            return {
                'java_version': java_version,
                'success': True,
                'training_time': training_time,
                'peak_utilization': peak_utilization,
                'avg_utilization': avg_utilization,
                'peak_memory': peak_memory,
                'auc': float(auc),
                'monitoring_data': gpu_results
            }
        else:
            print("No GPU monitoring data collected")
            h2o_context.stop()
            spark.stop()
            return {
                'java_version': java_version,
                'success': False,
                'error': 'No GPU monitoring data collected'
            }
            
    except Exception as e:
        print(f"Error testing H2O Sparkling GPU with Java {java_version}: {e}")
        import traceback
        traceback.print_exc()
        try:
            h2o_context.stop()
            spark.stop()
        except:
            pass
        return {
            'java_version': java_version,
            'success': False,
            'error': str(e)
        }

def plot_comparison(java11_results, java17_results, output_dir='../reports'):
    """Plot comparison between Java 11 and Java 17 GPU performance"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Only proceed if we have successful results for both versions
    if (not java11_results.get('success', False) or 
        not java17_results.get('success', False)):
        print("Cannot generate plots: missing successful results")
        return
    
    # 1. Bar chart comparison of metrics
    metrics = {
        'Training Time (s)': [java11_results['training_time'], java17_results['training_time']],
        'Peak GPU Utilization (%)': [java11_results['peak_utilization'], java17_results['peak_utilization']],
        'Avg GPU Utilization (%)': [java11_results['avg_utilization'], java17_results['avg_utilization']],
        'Peak Memory Usage (MB)': [java11_results['peak_memory'], java17_results['peak_memory']],
        'AUC': [java11_results['auc'], java17_results['auc']]
    }
    
    # Create comparison plots
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    
    for i, (metric, values) in enumerate(metrics.items()):
        ax = axs[i]
        bars = ax.bar(['Java 11', 'Java 17'], values, color=['blue', 'green'])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/java_gpu_comparison_{timestamp}.png")
    
    # 2. GPU utilization over time plot
    java11_df = pd.DataFrame(java11_results['monitoring_data'])
    java17_df = pd.DataFrame(java17_results['monitoring_data'])
    
    # Convert timestamp to seconds from start
    java11_start = java11_df['timestamp'].min()
    java11_df['seconds'] = java11_df['timestamp'] - java11_start
    
    java17_start = java17_df['timestamp'].min()
    java17_df['seconds'] = java17_df['timestamp'] - java17_start
    
    # Plot utilization over time
    plt.figure(figsize=(12, 6))
    plt.plot(java11_df['seconds'], java11_df['utilization'], label='Java 11')
    plt.plot(java17_df['seconds'], java17_df['utilization'], label='Java 17')
    plt.title('GPU Utilization Over Time: Java 11 vs Java 17')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/java_gpu_utilization_time_{timestamp}.png")
    
    # 3. Memory usage over time
    plt.figure(figsize=(12, 6))
    plt.plot(java11_df['seconds'], java11_df['memory_used'], label='Java 11')
    plt.plot(java17_df['seconds'], java17_df['memory_used'], label='Java 17')
    plt.title('GPU Memory Usage Over Time: Java 11 vs Java 17')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/java_gpu_memory_time_{timestamp}.png")
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main function to run Java GPU comparison tests"""
    parser = argparse.ArgumentParser(description='Java 11 vs Java 17 GPU Performance Comparison')
    parser.add_argument('--rows', type=int, default=100000, help='Number of rows in the dataset')
    parser.add_argument('--cols', type=int, default=20, help='Number of features in the dataset')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use for testing')
    parser.add_argument('--output-dir', type=str, default='../reports', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("JAVA 11 VS JAVA 17 GPU PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Create reports directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create synthetic dataset (reused for both tests)
    train_df, test_df = create_synthetic_dataset(args.rows, args.cols)
    
    # Run tests with Java 11
    print("\n" + "="*80)
    print("TESTING WITH JAVA 11")
    print("="*80)
    java11_results = test_h2o_sparkling_gpu(11, train_df, test_df, args.gpu_id)
    
    # Run tests with Java 17
    print("\n" + "="*80)
    print("TESTING WITH JAVA 17")
    print("="*80)
    java17_results = test_h2o_sparkling_gpu(17, train_df, test_df, args.gpu_id)
    
    # Generate comparison report
    print("\n" + "="*80)
    print("JAVA GPU PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    # Print results
    if java11_results.get('success', False):
        print("\nJava 11 Results:")
        print(f"  Training Time: {java11_results['training_time']:.2f} seconds")
        print(f"  Peak GPU Utilization: {java11_results['peak_utilization']:.2f}%")
        print(f"  Avg GPU Utilization: {java11_results['avg_utilization']:.2f}%")
        print(f"  Peak GPU Memory: {java11_results['peak_memory']:.2f} MB")
        print(f"  AUC: {java11_results['auc']:.4f}")
    else:
        print(f"\nJava 11 test failed: {java11_results.get('error', 'Unknown error')}")
    
    if java17_results.get('success', False):
        print("\nJava 17 Results:")
        print(f"  Training Time: {java17_results['training_time']:.2f} seconds")
        print(f"  Peak GPU Utilization: {java17_results['peak_utilization']:.2f}%")
        print(f"  Avg GPU Utilization: {java17_results['avg_utilization']:.2f}%")
        print(f"  Peak GPU Memory: {java17_results['peak_memory']:.2f} MB")
        print(f"  AUC: {java17_results['auc']:.4f}")
    else:
        print(f"\nJava 17 test failed: {java17_results.get('error', 'Unknown error')}")
    
    # Compare results if both were successful
    if java11_results.get('success', False) and java17_results.get('success', False):
        time_diff_pct = ((java17_results['training_time'] - java11_results['training_time']) / 
                         java11_results['training_time'] * 100)
        util_diff_pct = ((java17_results['peak_utilization'] - java11_results['peak_utilization']) / 
                         java11_results['peak_utilization'] * 100)
        mem_diff_pct = ((java17_results['peak_memory'] - java11_results['peak_memory']) / 
                        java11_results['peak_memory'] * 100)
        
        print("\nComparison (Java 17 vs Java 11):")
        print(f"  Training Time: {time_diff_pct:.2f}% ({'faster' if time_diff_pct < 0 else 'slower'})")
        print(f"  Peak GPU Utilization: {util_diff_pct:.2f}% ({'higher' if util_diff_pct > 0 else 'lower'})")
        print(f"  Peak GPU Memory: {mem_diff_pct:.2f}% ({'higher' if mem_diff_pct > 0 else 'lower'})")
        
        # Create plots
        plot_comparison(java11_results, java17_results, args.output_dir)
    
    # Save detailed results to file
    timestamp = int(time.time())
    result_file = os.path.join(args.output_dir, f"java_gpu_comparison_{timestamp}.json")
    
    # Remove monitoring data from JSON (too large)
    if 'monitoring_data' in java11_results:
        java11_results['monitoring_data_length'] = len(java11_results['monitoring_data'])
        java11_results['monitoring_data_sample'] = java11_results['monitoring_data'][:5]
        del java11_results['monitoring_data']
    
    if 'monitoring_data' in java17_results:
        java17_results['monitoring_data_length'] = len(java17_results['monitoring_data'])
        java17_results['monitoring_data_sample'] = java17_results['monitoring_data'][:5]
        del java17_results['monitoring_data']
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'java11_results': java11_results,
            'java17_results': java17_results,
            'test_parameters': vars(args)
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())