#!/usr/bin/env python3
"""
Test GPU Capabilities for Numerai Crypto Ensemble

This script tests the GPU capabilities of the system for Numerai Crypto ensemble:
- Checks GPU availability and specifications
- Tests RAPIDS (cuDF, cuML) data processing capabilities
- Tests LightGBM GPU acceleration
- Tests H2O XGBoost GPU acceleration
- Tests PySpark with RAPIDS acceleration
- Benchmarks performance against CPU

Usage:
    python test_gpu_capabilities.py [--full] [--output-dir DIR]
"""
import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test GPU Capabilities for Numerai Crypto')
    parser.add_argument('--full', action='store_true', 
                       help='Run full test suite (including long-running tests)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for test results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def check_gpu_availability():
    """Check GPU availability and specifications"""
    print("\n=== GPU Availability ===")
    
    # Check if CUDA is available via nvidia-smi
    try:
        import subprocess
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], 
                                                  stderr=subprocess.PIPE,
                                                  encoding='utf-8')
        print("NVIDIA GPUs detected!")
        print(nvidia_smi_output)
        
        # Get GPU details
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,cuda_version', 
             '--format=csv,noheader'],
            encoding='utf-8'
        )
        
        gpus = []
        for line in gpu_info.strip().split('\n'):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory': parts[2],
                    'driver': parts[3],
                    'cuda': parts[4]
                })
        
        return {
            'gpu_available': True,
            'gpu_count': len(gpus),
            'gpus': gpus
        }
    
    except (subprocess.SubprocessError, FileNotFoundError):
        print("No NVIDIA GPUs detected via nvidia-smi.")
    
    # Try checking via Python packages
    try:
        # Try importing torch to check CUDA
        import torch
        if torch.cuda.is_available():
            print("PyTorch CUDA is available!")
            gpu_count = torch.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            
            gpus = []
            for i in range(gpu_count):
                gpus.append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i)
                })
            
            return {
                'gpu_available': True,
                'gpu_count': gpu_count,
                'gpus': gpus
            }
    except ImportError:
        print("PyTorch not available.")
    
    try:
        # Try importing tensorflow to check CUDA
        import tensorflow as tf
        if tf.test.is_gpu_available():
            print("TensorFlow can access GPU!")
            return {
                'gpu_available': True,
                'gpu_count': 1,  # TF doesn't easily expose count
                'gpus': [{'index': 0, 'name': 'GPU detected via TensorFlow'}]
            }
    except ImportError:
        print("TensorFlow not available.")
    
    print("No GPUs detected.")
    return {
        'gpu_available': False,
        'gpu_count': 0,
        'gpus': []
    }

def test_rapids():
    """Test RAPIDS (cuDF, cuML) capability"""
    print("\n=== RAPIDS Capability Test ===")
    results = {'available': False, 'components': {}}
    
    try:
        import cudf
        print(f"✅ cuDF available (version: {cudf.__version__})")
        results['available'] = True
        results['components']['cudf'] = {
            'available': True,
            'version': cudf.__version__
        }
        
        # Test simple cuDF operations
        print("Testing cuDF operations...")
        start_time = time.time()
        
        # Create a DataFrame
        df = cudf.DataFrame({
            'a': np.random.rand(1_000_000),
            'b': np.random.rand(1_000_000),
            'c': np.random.rand(1_000_000)
        })
        
        # Simple operations
        df['d'] = df['a'] + df['b'] * df['c']
        df['e'] = df['d'].log()
        result = df.groupby('e').agg({'a': 'mean', 'b': 'sum'})
        
        end_time = time.time()
        
        print(f"cuDF operations completed in {end_time - start_time:.4f} seconds")
        results['components']['cudf']['performance'] = end_time - start_time
        
    except ImportError:
        print("❌ cuDF not available - GPU-accelerated DataFrames not supported")
        results['components']['cudf'] = {
            'available': False
        }
    
    try:
        import cuml
        print(f"✅ cuML available (version: {cuml.__version__})")
        results['available'] = True
        results['components']['cuml'] = {
            'available': True,
            'version': cuml.__version__
        }
        
        # Test simple cuML operations
        print("Testing cuML operations...")
        try:
            start_time = time.time()
            
            # Create random data
            X = np.random.rand(10000, 100).astype(np.float32)
            y = np.random.randint(0, 2, 10000).astype(np.int32)
            
            # Convert to cuDF
            X_cudf = cudf.DataFrame(X)
            y_cudf = cudf.Series(y)
            
            # Train a model
            from cuml.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X_cudf, y_cudf)
            
            # Make predictions
            preds = model.predict(X_cudf)
            
            end_time = time.time()
            
            print(f"cuML operations completed in {end_time - start_time:.4f} seconds")
            results['components']['cuml']['performance'] = end_time - start_time
            
        except Exception as e:
            print(f"Error in cuML test: {e}")
            results['components']['cuml']['error'] = str(e)
        
    except ImportError:
        print("❌ cuML not available - GPU-accelerated ML not supported")
        results['components']['cuml'] = {
            'available': False
        }
    
    try:
        import rapids.spark
        print("✅ RAPIDS Accelerator for Apache Spark available")
        results['available'] = True
        results['components']['rapids_spark'] = {
            'available': True
        }
        
    except ImportError:
        print("❌ RAPIDS Accelerator for Apache Spark not available")
        results['components']['rapids_spark'] = {
            'available': False
        }
    
    return results

def test_lightgbm_gpu(full_test=False):
    """Test LightGBM GPU acceleration"""
    print("\n=== LightGBM GPU Capability Test ===")
    results = {'available': False}
    
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM available (version: {lgb.__version__})")
        results['available'] = True
        results['version'] = lgb.__version__
        
        # Test GPU support
        try:
            # Create synthetic dataset
            print("Creating synthetic dataset...")
            n_samples = 50000 if full_test else 10000
            n_features = 100
            
            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)
            
            # Try CPU training first for comparison
            print("Training on CPU...")
            start_time = time.time()
            
            # Create dataset
            cpu_data = lgb.Dataset(X, label=y)
            
            # CPU parameters
            cpu_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            
            # Train model
            cpu_model = lgb.train(
                params=cpu_params,
                train_set=cpu_data,
                num_boost_round=50
            )
            
            cpu_time = time.time() - start_time
            print(f"CPU training completed in {cpu_time:.4f} seconds")
            results['cpu_performance'] = cpu_time
            
            # Now try GPU training
            print("Training on GPU...")
            start_time = time.time()
            
            # GPU parameters
            gpu_params = cpu_params.copy()
            gpu_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
            
            # Train model
            gpu_model = lgb.train(
                params=gpu_params,
                train_set=cpu_data,
                num_boost_round=50
            )
            
            gpu_time = time.time() - start_time
            print(f"GPU training completed in {gpu_time:.4f} seconds")
            results['gpu_performance'] = gpu_time
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print(f"GPU speedup: {speedup:.2f}x")
            results['speedup'] = speedup
            results['gpu_works'] = True
            
        except Exception as e:
            print(f"Error in LightGBM GPU test: {e}")
            results['gpu_works'] = False
            results['error'] = str(e)
    
    except ImportError:
        print("❌ LightGBM not available")
    
    return results

def test_h2o_xgboost(full_test=False):
    """Test H2O XGBoost GPU acceleration"""
    print("\n=== H2O XGBoost GPU Capability Test ===")
    results = {'available': False}
    
    try:
        import h2o
        print(f"✅ H2O available")
        results['available'] = True
        
        # Initialize H2O
        h2o.init(nthreads=-1, max_mem_size="4G")
        
        # Get H2O version
        version = h2o.version()
        print(f"H2O version: {version}")
        results['version'] = version
        
        # Test XGBoost support
        try:
            from h2o.estimators.xgboost import H2OXGBoostEstimator
            
            # Check if H2O's XGBoost is available
            print("H2O XGBoost is available")
            results['xgboost_available'] = True
            
            # Create synthetic dataset
            print("Creating synthetic dataset...")
            n_samples = 50000 if full_test else 10000
            n_features = 20
            
            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)
            
            # Create pandas DataFrame
            data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
            data['target'] = y
            
            # Convert to H2O frame
            h2o_data = h2o.H2OFrame(data)
            h2o_data['target'] = h2o_data['target'].asfactor()
            
            # Split data
            train, valid = h2o_data.split_frame(ratios=[0.8])
            
            # First test CPU mode
            print("Training XGBoost on CPU...")
            start_time = time.time()
            
            cpu_model = H2OXGBoostEstimator(
                ntrees=50,
                max_depth=5,
                learn_rate=0.1,
                stopping_rounds=5,
                score_each_iteration=True,
                seed=1234
            )
            
            cpu_model.train(
                x=[f'feature_{i}' for i in range(n_features)],
                y="target",
                training_frame=train,
                validation_frame=valid
            )
            
            cpu_time = time.time() - start_time
            print(f"CPU training completed in {cpu_time:.4f} seconds")
            results['cpu_performance'] = cpu_time
            
            # Now try GPU training
            print("Training XGBoost on GPU...")
            start_time = time.time()
            
            gpu_model = H2OXGBoostEstimator(
                ntrees=50,
                max_depth=5,
                learn_rate=0.1,
                stopping_rounds=5,
                score_each_iteration=True,
                seed=1234,
                tree_method="gpu_hist",
                gpu_id=0
            )
            
            gpu_model.train(
                x=[f'feature_{i}' for i in range(n_features)],
                y="target",
                training_frame=train,
                validation_frame=valid
            )
            
            gpu_time = time.time() - start_time
            print(f"GPU training completed in {gpu_time:.4f} seconds")
            results['gpu_performance'] = gpu_time
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print(f"GPU speedup: {speedup:.2f}x")
            results['speedup'] = speedup
            results['gpu_works'] = True
            
            # Shutdown H2O
            h2o.shutdown(prompt=False)
            
        except Exception as e:
            print(f"Error in H2O XGBoost GPU test: {e}")
            results['xgboost_available'] = False
            results['error'] = str(e)
            try:
                h2o.shutdown(prompt=False)
            except:
                pass
    
    except ImportError:
        print("❌ H2O not available")
    
    return results

def test_pyspark_rapids(full_test=False):
    """Test PySpark with RAPIDS acceleration"""
    print("\n=== PySpark RAPIDS Capability Test ===")
    results = {'available': False}
    
    try:
        from pyspark.sql import SparkSession
        print("✅ PySpark available")
        results['available'] = True
        
        # Check for RAPIDS Accelerator for Spark
        try:
            import importlib.util
            rapids_spark_available = importlib.util.find_spec('rapids.spark') is not None
            
            if rapids_spark_available:
                print("✅ RAPIDS Accelerator for Apache Spark is available")
                results['rapids_available'] = True
                
                # Find RAPIDS jars
                rapids_jars_path = None
                if 'SPARK_RAPIDS_DIR' in os.environ:
                    rapids_jars_path = os.environ['SPARK_RAPIDS_DIR']
                elif 'CONDA_PREFIX' in os.environ:
                    import glob
                    pattern = f"{os.environ['CONDA_PREFIX']}/lib/python*/site-packages/rapids/jars/*"
                    jars = glob.glob(pattern)
                    if jars:
                        rapids_jars_path = os.path.dirname(jars[0])
                
                if rapids_jars_path:
                    print(f"RAPIDS jars found at: {rapids_jars_path}")
                    results['rapids_jars_path'] = rapids_jars_path
                    
                    # Create Spark session with RAPIDS
                    print("Creating Spark session with RAPIDS...")
                    builder = SparkSession.builder \
                        .appName("RAPIDS_Test") \
                        .config("spark.executor.memory", "4g") \
                        .config("spark.driver.memory", "4g") \
                        .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
                        .config("spark.rapids.sql.enabled", "true") \
                        .config("spark.rapids.sql.explain", "ALL")
                    
                    spark = builder.getOrCreate()
                    
                    # Check Spark version
                    version = spark.version
                    print(f"Spark version: {version}")
                    results['spark_version'] = version
                    
                    # Test simple operations
                    print("Testing Spark operations with RAPIDS...")
                    
                    # Create synthetic dataset
                    n_samples = 500000 if full_test else 100000
                    
                    # CPU Test first
                    print("Testing with standard Spark (CPU)...")
                    spark.conf.set("spark.rapids.sql.enabled", "false")
                    
                    start_time = time.time()
                    df_schema = "id INT, value1 DOUBLE, value2 DOUBLE, value3 DOUBLE"
                    
                    cpu_df = spark.range(n_samples) \
                        .selectExpr(
                            "id", 
                            "rand() as value1", 
                            "rand() as value2", 
                            "rand() as value3"
                        )
                    
                    # Force execution
                    cpu_df.cache()
                    cpu_count = cpu_df.count()
                    
                    # Calculate aggregates
                    result = cpu_df.groupBy((cpu_df.value1 * 10).cast("int").alias("group")) \
                        .agg(
                            {"value1": "avg", "value2": "sum", "value3": "max"}
                        )
                    
                    result.collect()
                    cpu_time = time.time() - start_time
                    print(f"CPU operation completed in {cpu_time:.4f} seconds")
                    results['cpu_performance'] = cpu_time
                    
                    # Now try with RAPIDS enabled
                    print("Testing with RAPIDS-accelerated Spark (GPU)...")
                    spark.conf.set("spark.rapids.sql.enabled", "true")
                    
                    start_time = time.time()
                    
                    gpu_df = spark.range(n_samples) \
                        .selectExpr(
                            "id", 
                            "rand() as value1", 
                            "rand() as value2", 
                            "rand() as value3"
                        )
                    
                    # Force execution
                    gpu_df.cache()
                    gpu_count = gpu_df.count()
                    
                    # Calculate aggregates
                    result = gpu_df.groupBy((gpu_df.value1 * 10).cast("int").alias("group")) \
                        .agg(
                            {"value1": "avg", "value2": "sum", "value3": "max"}
                        )
                    
                    result.collect()
                    gpu_time = time.time() - start_time
                    print(f"GPU operation completed in {gpu_time:.4f} seconds")
                    results['gpu_performance'] = gpu_time
                    
                    # Calculate speedup
                    speedup = cpu_time / gpu_time
                    print(f"GPU speedup: {speedup:.2f}x")
                    results['speedup'] = speedup
                    results['gpu_works'] = True
                    
                    # Stop Spark session
                    spark.stop()
                
                else:
                    print("RAPIDS jars not found. Cannot test RAPIDS with Spark.")
                    results['error'] = "RAPIDS jars not found"
            
            else:
                print("❌ RAPIDS Accelerator for Apache Spark is not available")
                results['rapids_available'] = False
        
        except Exception as e:
            print(f"Error in PySpark RAPIDS test: {e}")
            results['error'] = str(e)
            try:
                if 'spark' in locals():
                    spark.stop()
            except:
                pass
    
    except ImportError:
        print("❌ PySpark not available")
    
    return results

def main():
    """Main function to run all GPU capability tests"""
    args = parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, 'reports', f'gpu_tests_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test results will be saved to: {output_dir}")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Store all results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'python_version': sys.version,
            'os': os.name
        },
        'tests': {}
    }
    
    # Check GPU availability
    results['gpu_info'] = check_gpu_availability()
    
    # Test RAPIDS
    results['tests']['rapids'] = test_rapids()
    
    # Test LightGBM
    results['tests']['lightgbm'] = test_lightgbm_gpu(args.full)
    
    # Test H2O XGBoost
    results['tests']['h2o_xgboost'] = test_h2o_xgboost(args.full)
    
    # Test PySpark RAPIDS
    if args.full:
        results['tests']['pyspark_rapids'] = test_pyspark_rapids(args.full)
    
    # Save results
    results_file = os.path.join(output_dir, 'gpu_capability_results.json')
    with open(results_file, 'w') as f:
        # Convert any non-serializable objects to strings
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            elif isinstance(obj, (datetime, np.datetime64)):
                return obj.isoformat()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return str(obj)
        
        serialized_results = convert_to_serializable(results)
        json.dump(serialized_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("GPU CAPABILITY TEST SUMMARY")
    print("="*80)
    
    gpu_info = results['gpu_info']
    if gpu_info['gpu_available']:
        print(f"✅ GPU available: {gpu_info['gpu_count']} found")
        for gpu in gpu_info.get('gpus', []):
            name = gpu.get('name', 'Unknown')
            memory = gpu.get('memory', '')
            print(f"  - GPU {gpu['index']}: {name} {memory}")
    else:
        print("❌ No GPUs detected")
    
    print("\nFramework Support:")
    
    rapids = results['tests']['rapids']
    if rapids['available']:
        print("✅ RAPIDS available")
        components = rapids.get('components', {})
        if components.get('cudf', {}).get('available'):
            print(f"  - cuDF: {components['cudf'].get('version', 'Unknown')}")
        if components.get('cuml', {}).get('available'):
            print(f"  - cuML: {components['cuml'].get('version', 'Unknown')}")
        if components.get('rapids_spark', {}).get('available'):
            print(f"  - RAPIDS for Spark: Available")
    else:
        print("❌ RAPIDS not available")
    
    lightgbm = results['tests']['lightgbm']
    if lightgbm.get('available'):
        if lightgbm.get('gpu_works'):
            speedup = lightgbm.get('speedup', 0)
            print(f"✅ LightGBM GPU: Working ({speedup:.2f}x speedup)")
        else:
            print("❌ LightGBM GPU: Not working")
    else:
        print("❌ LightGBM: Not available")
    
    h2o = results['tests']['h2o_xgboost']
    if h2o.get('available'):
        if h2o.get('xgboost_available'):
            if h2o.get('gpu_works'):
                speedup = h2o.get('speedup', 0)
                print(f"✅ H2O XGBoost GPU: Working ({speedup:.2f}x speedup)")
            else:
                print("❌ H2O XGBoost GPU: Not working")
        else:
            print("❌ H2O XGBoost: Not available")
    else:
        print("❌ H2O: Not available")
    
    if args.full and 'pyspark_rapids' in results['tests']:
        pyspark = results['tests']['pyspark_rapids']
        if pyspark.get('available'):
            if pyspark.get('rapids_available'):
                if pyspark.get('gpu_works'):
                    speedup = pyspark.get('speedup', 0)
                    print(f"✅ PySpark RAPIDS: Working ({speedup:.2f}x speedup)")
                else:
                    print("❌ PySpark RAPIDS: Not working")
            else:
                print("❌ RAPIDS for Spark: Not available")
        else:
            print("❌ PySpark: Not available")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create recommendations
    print("\nRECOMMENDATIONS:")
    
    if not gpu_info['gpu_available']:
        print("- No GPUs detected. Consider using a system with NVIDIA GPUs for better performance.")
    elif gpu_info['gpu_count'] > 0:
        if not rapids['available']:
            print("- Install RAPIDS for GPU-accelerated data processing:")
            print("  conda install -c rapidsai -c conda-forge -c nvidia rapids=23.12 python=3.10 cuda=11.8")
        
        if not lightgbm.get('gpu_works', False):
            print("- Install/configure LightGBM with GPU support:")
            print("  pip install lightgbm --install-option=--gpu")
        
        if not h2o.get('gpu_works', False) and h2o.get('available', False):
            print("- Configure H2O XGBoost to use GPU:")
            print("  Use tree_method='gpu_hist' in XGBoostEstimator parameters")
        
        if args.full and 'pyspark_rapids' in results['tests']:
            if not results['tests']['pyspark_rapids'].get('gpu_works', False):
                print("- Configure Spark to use RAPIDS:")
                print("  1. Set SPARK_RAPIDS_DIR environment variable")
                print("  2. Add Spark configuration for RAPIDS")
    
    print("\nGPU capability testing completed!")

if __name__ == "__main__":
    main()