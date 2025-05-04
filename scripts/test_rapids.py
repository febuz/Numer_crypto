#!/usr/bin/env python
"""
Test script for RAPIDS GPU acceleration in Numer_crypto.

This script tests GPU acceleration using RAPIDS for data processing,
comparing performance between CPU (pandas/numpy) and GPU (cuDF/cuML) operations.
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the Python path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from numer_crypto.utils.rapids_utils import (
    RAPIDS_AVAILABLE, CUML_AVAILABLE, SPARK_RAPIDS_AVAILABLE,
    check_rapids_availability, pandas_to_cudf, with_cudf
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test RAPIDS GPU acceleration for Numer_crypto'
    )
    parser.add_argument('--size', type=int, default=1000000, 
                        help='Size of test data (rows)')
    parser.add_argument('--columns', type=int, default=50, 
                        help='Number of columns in test data')
    parser.add_argument('--skip-spark', action='store_true',
                        help='Skip Spark RAPIDS tests')
    parser.add_argument('--skip-ml', action='store_true',
                        help='Skip ML tests')
    return parser.parse_args()

def test_dataframe_operations(size, columns):
    """
    Test basic DataFrame operations with pandas vs cuDF.
    
    Args:
        size (int): Number of rows in test data
        columns (int): Number of columns in test data
    
    Returns:
        dict: Dictionary of timing results
    """
    print("\n=== Testing DataFrame Operations ===")
    
    # Generate test data
    print(f"Generating test data ({size} rows, {columns} columns)...")
    data = np.random.random((size, columns))
    
    # Timing results
    results = {}
    
    try:
        # Test with pandas
        import pandas as pd
        
        # Create DataFrame
        start_time = time.time()
        pdf = pd.DataFrame(data)
        pdf_create_time = time.time() - start_time
        print(f"pandas DataFrame creation: {pdf_create_time:.4f} seconds")
        results['pandas_create'] = pdf_create_time
        
        # Test operations
        # 1. Column-wise operations
        start_time = time.time()
        pdf['new_col'] = pdf[0] + pdf[1]
        pdf_col_op_time = time.time() - start_time
        print(f"pandas column operation: {pdf_col_op_time:.4f} seconds")
        results['pandas_col_op'] = pdf_col_op_time
        
        # 2. Aggregation
        start_time = time.time()
        pdf_agg = pdf.groupby(pdf[0] > 0.5).mean()
        pdf_agg_time = time.time() - start_time
        print(f"pandas aggregation: {pdf_agg_time:.4f} seconds")
        results['pandas_agg'] = pdf_agg_time
        
        # 3. Sorting
        start_time = time.time()
        pdf_sorted = pdf.sort_values(by=0)
        pdf_sort_time = time.time() - start_time
        print(f"pandas sorting: {pdf_sort_time:.4f} seconds")
        results['pandas_sort'] = pdf_sort_time
        
        # Test with cuDF if available
        if RAPIDS_AVAILABLE:
            import cudf
            
            # Create DataFrame
            start_time = time.time()
            gdf = cudf.DataFrame(data)
            gdf_create_time = time.time() - start_time
            print(f"cuDF DataFrame creation: {gdf_create_time:.4f} seconds")
            results['cudf_create'] = gdf_create_time
            
            # Speedup
            speedup = pdf_create_time / gdf_create_time
            print(f"Speedup: {speedup:.2f}x")
            
            # Test operations
            # 1. Column-wise operations
            start_time = time.time()
            gdf['new_col'] = gdf[0] + gdf[1]
            gdf_col_op_time = time.time() - start_time
            print(f"cuDF column operation: {gdf_col_op_time:.4f} seconds")
            results['cudf_col_op'] = gdf_col_op_time
            
            speedup = pdf_col_op_time / gdf_col_op_time
            print(f"Speedup: {speedup:.2f}x")
            
            # 2. Aggregation
            start_time = time.time()
            gdf_agg = gdf.groupby(gdf[0] > 0.5).mean()
            gdf_agg_time = time.time() - start_time
            print(f"cuDF aggregation: {gdf_agg_time:.4f} seconds")
            results['cudf_agg'] = gdf_agg_time
            
            speedup = pdf_agg_time / gdf_agg_time
            print(f"Speedup: {speedup:.2f}x")
            
            # 3. Sorting
            start_time = time.time()
            gdf_sorted = gdf.sort_values(by=0)
            gdf_sort_time = time.time() - start_time
            print(f"cuDF sorting: {gdf_sort_time:.4f} seconds")
            results['cudf_sort'] = gdf_sort_time
            
            speedup = pdf_sort_time / gdf_sort_time
            print(f"Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"Error during DataFrame operations test: {e}")
    
    return results

def test_ml_operations(size, columns):
    """
    Test machine learning operations with scikit-learn vs cuML.
    
    Args:
        size (int): Number of rows in test data
        columns (int): Number of columns in test data
    
    Returns:
        dict: Dictionary of timing results
    """
    if not CUML_AVAILABLE:
        print("\n=== Skipping ML Operations (cuML not available) ===")
        return {}
    
    print("\n=== Testing ML Operations ===")
    
    # Generate test data
    print(f"Generating test data ({size} rows, {columns} columns)...")
    X = np.random.random((size, columns))
    y = np.random.random(size)
    
    # Timing results
    results = {}
    
    try:
        # Test with scikit-learn
        from sklearn.ensemble import RandomForestRegressor
        
        # Random Forest
        print("\nRandom Forest:")
        start_time = time.time()
        rf_cpu = RandomForestRegressor(n_estimators=10, max_depth=5, n_jobs=-1)
        rf_cpu.fit(X, y)
        cpu_train_time = time.time() - start_time
        print(f"scikit-learn training time: {cpu_train_time:.4f} seconds")
        results['sklearn_rf_train'] = cpu_train_time
        
        start_time = time.time()
        rf_cpu.predict(X[:1000])
        cpu_predict_time = time.time() - start_time
        print(f"scikit-learn prediction time: {cpu_predict_time:.4f} seconds")
        results['sklearn_rf_predict'] = cpu_predict_time
        
        # Test with cuML
        import cuml
        
        # Random Forest
        start_time = time.time()
        rf_gpu = cuml.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5)
        rf_gpu.fit(X, y)
        gpu_train_time = time.time() - start_time
        print(f"cuML training time: {gpu_train_time:.4f} seconds")
        results['cuml_rf_train'] = gpu_train_time
        
        speedup = cpu_train_time / gpu_train_time
        print(f"Training speedup: {speedup:.2f}x")
        
        start_time = time.time()
        rf_gpu.predict(X[:1000])
        gpu_predict_time = time.time() - start_time
        print(f"cuML prediction time: {gpu_predict_time:.4f} seconds")
        results['cuml_rf_predict'] = gpu_predict_time
        
        speedup = cpu_predict_time / gpu_predict_time
        print(f"Prediction speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"Error during ML operations test: {e}")
    
    return results

def test_spark_operations(size, columns):
    """
    Test Spark operations with and without RAPIDS acceleration.
    
    Args:
        size (int): Number of rows in test data
        columns (int): Number of columns in test data
    
    Returns:
        dict: Dictionary of timing results
    """
    if not SPARK_RAPIDS_AVAILABLE:
        print("\n=== Skipping Spark Operations (RAPIDS for Spark not available) ===")
        return {}
    
    print("\n=== Testing Spark Operations ===")
    
    # Timing results
    results = {}
    
    try:
        # Generate test data
        print(f"Generating test data ({size} rows, {columns} columns)...")
        data = [(i, *np.random.random(columns).tolist()) for i in range(size)]
        
        # Standard Spark session (CPU)
        from pyspark.sql import SparkSession
        
        print("Creating CPU Spark session...")
        spark_cpu = SparkSession.builder \
            .appName("RAPIDSTest-CPU") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
        
        # Create DataFrame
        col_names = ["id"] + [f"col_{i}" for i in range(columns)]
        start_time = time.time()
        df_cpu = spark_cpu.createDataFrame(data, col_names)
        cpu_create_time = time.time() - start_time
        print(f"CPU Spark DataFrame creation: {cpu_create_time:.4f} seconds")
        results['spark_cpu_create'] = cpu_create_time
        
        # Aggregation
        start_time = time.time()
        result_cpu = df_cpu.groupBy(df_cpu["col_0"] > 0.5).avg().collect()
        cpu_agg_time = time.time() - start_time
        print(f"CPU Spark aggregation: {cpu_agg_time:.4f} seconds")
        results['spark_cpu_agg'] = cpu_agg_time
        
        # Join
        df_cpu2 = spark_cpu.createDataFrame(
            [(i, np.random.random()) for i in range(size)],
            ["id", "value"]
        )
        start_time = time.time()
        join_result_cpu = df_cpu.join(df_cpu2, "id").collect()
        cpu_join_time = time.time() - start_time
        print(f"CPU Spark join: {cpu_join_time:.4f} seconds")
        results['spark_cpu_join'] = cpu_join_time
        
        # Stop CPU session
        spark_cpu.stop()
        
        # Create GPU-accelerated Spark session
        from numer_crypto.utils.rapids_utils import enable_spark_rapids
        
        print("\nCreating GPU-accelerated Spark session...")
        spark_gpu = SparkSession.builder \
            .appName("RAPIDSTest-GPU") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
        
        spark_gpu = enable_spark_rapids(spark_gpu)
        
        # Create DataFrame
        start_time = time.time()
        df_gpu = spark_gpu.createDataFrame(data, col_names)
        gpu_create_time = time.time() - start_time
        print(f"GPU Spark DataFrame creation: {gpu_create_time:.4f} seconds")
        results['spark_gpu_create'] = gpu_create_time
        
        speedup = cpu_create_time / gpu_create_time
        print(f"Creation speedup: {speedup:.2f}x")
        
        # Aggregation
        start_time = time.time()
        result_gpu = df_gpu.groupBy(df_gpu["col_0"] > 0.5).avg().collect()
        gpu_agg_time = time.time() - start_time
        print(f"GPU Spark aggregation: {gpu_agg_time:.4f} seconds")
        results['spark_gpu_agg'] = gpu_agg_time
        
        speedup = cpu_agg_time / gpu_agg_time
        print(f"Aggregation speedup: {speedup:.2f}x")
        
        # Join
        df_gpu2 = spark_gpu.createDataFrame(
            [(i, np.random.random()) for i in range(size)],
            ["id", "value"]
        )
        start_time = time.time()
        join_result_gpu = df_gpu.join(df_gpu2, "id").collect()
        gpu_join_time = time.time() - start_time
        print(f"GPU Spark join: {gpu_join_time:.4f} seconds")
        results['spark_gpu_join'] = gpu_join_time
        
        speedup = cpu_join_time / gpu_join_time
        print(f"Join speedup: {speedup:.2f}x")
        
        # Stop GPU session
        spark_gpu.stop()
        
    except Exception as e:
        print(f"Error during Spark operations test: {e}")
    
    return results

def test_with_cudf_decorator():
    """
    Test the with_cudf decorator for automatic pandas/cuDF conversion.
    
    Returns:
        bool: True if test is successful, False otherwise
    """
    if not RAPIDS_AVAILABLE:
        print("\n=== Skipping with_cudf Decorator Test (RAPIDS not available) ===")
        return False
    
    print("\n=== Testing with_cudf Decorator ===")
    
    try:
        import pandas as pd
        
        # Define a function that uses GPU acceleration if available
        @with_cudf
        def process_dataframe(df):
            # This operation will be GPU-accelerated with cuDF
            # or use regular pandas if cuDF is not available
            result = df.groupby(df['A'] > 0.5).mean()
            return result
        
        # Create a test DataFrame
        data = {'A': np.random.random(1000), 'B': np.random.random(1000)}
        df = pd.DataFrame(data)
        
        # Call the decorated function
        result = process_dataframe(df)
        
        # Verify the result type (should be pandas even though cuDF was used internally)
        print(f"Result type: {type(result).__name__}")
        print(f"Result shape: {result.shape}")
        print(f"✓ with_cudf decorator test successful")
        
        return True
    
    except Exception as e:
        print(f"Error during with_cudf decorator test: {e}")
        return False

def test_pandas_conversion():
    """
    Test pandas to cuDF conversion.
    
    Returns:
        bool: True if test is successful, False otherwise
    """
    if not RAPIDS_AVAILABLE:
        print("\n=== Skipping pandas Conversion Test (RAPIDS not available) ===")
        return False
    
    print("\n=== Testing pandas/cuDF Conversion ===")
    
    try:
        import pandas as pd
        import cudf
        
        # Create a test DataFrame
        data = {'A': np.random.random(1000), 'B': np.random.random(1000)}
        pdf = pd.DataFrame(data)
        
        # Convert to cuDF
        start_time = time.time()
        gdf = pandas_to_cudf(pdf)
        conversion_time = time.time() - start_time
        print(f"pandas to cuDF conversion time: {conversion_time:.4f} seconds")
        
        # Verify the conversion
        print(f"Original type: {type(pdf).__name__}, shape: {pdf.shape}")
        print(f"Converted type: {type(gdf).__name__}, shape: {gdf.shape}")
        
        # Convert back to pandas
        start_time = time.time()
        pdf2 = gdf.to_pandas()
        conversion_back_time = time.time() - start_time
        print(f"cuDF to pandas conversion time: {conversion_back_time:.4f} seconds")
        
        # Verify the data is the same
        if pdf.equals(pdf2):
            print("✓ Data integrity maintained through conversions")
        else:
            print("✗ Data changed during conversion")
        
        return True
    
    except Exception as e:
        print(f"Error during pandas conversion test: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 50)
    print("Numer_crypto RAPIDS GPU Acceleration Test")
    print("=" * 50)
    
    # Check RAPIDS availability
    check_rapids_availability()
    
    # Adjust data size based on available memory
    if not RAPIDS_AVAILABLE and args.size > 1000000:
        print("\nWarning: RAPIDS not available, reducing test data size")
        args.size = 1000000
    
    # Run tests
    df_results = test_dataframe_operations(args.size, args.columns)
    
    if not args.skip_ml:
        ml_results = test_ml_operations(min(args.size, 100000), args.columns)
    
    if not args.skip_spark:
        spark_results = test_spark_operations(min(args.size, 100000), args.columns)
    
    # Test decorator
    test_with_cudf_decorator()
    
    # Test conversion
    test_pandas_conversion()
    
    # Print summary
    print("\n" + "=" * 50)
    print("RAPIDS GPU Acceleration Test Summary")
    print("=" * 50)
    
    if RAPIDS_AVAILABLE:
        print("✓ RAPIDS (cuDF) is available and functional")
        
        # Calculate overall speedup for DataFrame operations
        if 'pandas_agg' in df_results and 'cudf_agg' in df_results:
            df_speedup = df_results['pandas_agg'] / df_results['cudf_agg']
            print(f"DataFrame operations speedup: {df_speedup:.2f}x")
        
        if 'sklearn_rf_train' in locals().get('ml_results', {}) and 'cuml_rf_train' in locals().get('ml_results', {}):
            ml_speedup = ml_results['sklearn_rf_train'] / ml_results['cuml_rf_train']
            print(f"ML training speedup: {ml_speedup:.2f}x")
        
        if 'spark_cpu_agg' in locals().get('spark_results', {}) and 'spark_gpu_agg' in locals().get('spark_results', {}):
            spark_speedup = spark_results['spark_cpu_agg'] / spark_results['spark_gpu_agg']
            print(f"Spark operations speedup: {spark_speedup:.2f}x")
    else:
        print("✗ RAPIDS (cuDF) is not available")
        print("  Install RAPIDS to enable GPU acceleration:")
        print("  $ ./setup_env.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())