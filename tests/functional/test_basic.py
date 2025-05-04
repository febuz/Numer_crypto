#!/usr/bin/env python
"""
Basic test script for ML algorithms in Numer_crypto.

This script provides a basic test of the ML functionality, testing both CPU and
GPU versions of XGBoost and LightGBM (if available).
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the Python path if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test ML algorithms for Numer_crypto'
    )
    parser.add_argument('--rows', type=int, default=100000, 
                        help='Number of rows in test data')
    parser.add_argument('--cols', type=int, default=50, 
                        help='Number of columns in test data')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU tests even if GPUs are available')
    return parser.parse_args()

def check_gpu_availability():
    """Check if GPU is available for ML libraries."""
    gpu_available = False
    
    # Check environment variable set by activation script
    if os.environ.get('XGBOOST_GPU_SUPPORT') == '1':
        gpu_available = True
    
    # Also try to check directly with XGBoost
    try:
        import xgboost as xgb
        param = {'gpu_id': 0, 'tree_method': 'gpu_hist'}
        try:
            # Create a small dataset and test
            data = np.random.rand(50, 10)
            label = np.random.randint(2, size=50)
            dtrain = xgb.DMatrix(data, label=label)
            xgb.train(param, dtrain, num_boost_round=1)
            gpu_available = True
            print("XGBoost GPU support confirmed by test")
        except Exception as e:
            print(f"XGBoost GPU test failed: {e}")
    except ImportError:
        print("XGBoost not installed")
    
    # Try checking with LightGBM too
    try:
        import lightgbm as lgb
        if 'gpu' in ','.join(lgb.get_device_type()).lower():
            gpu_available = True
            print("LightGBM reports GPU support")
    except (ImportError, AttributeError):
        pass
    
    # Check for NVIDIA GPUs with system tools
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            print(f"Found {int(result.stdout.strip())} NVIDIA GPUs")
            gpu_available = True
    except (FileNotFoundError, ValueError):
        pass
    
    return gpu_available

def generate_synthetic_data(n_rows, n_cols):
    """Generate synthetic data for testing."""
    print(f"Generating synthetic data ({n_rows} rows, {n_cols} features)...")
    X = np.random.rand(n_rows, n_cols).astype(np.float32)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.1 > 1).astype(np.float32)
    return X, y

def test_xgboost(X, y, use_gpu=True):
    """Test XGBoost performance."""
    try:
        import xgboost as xgb
        print("\n=== Testing XGBoost ===")
        
        # Split data
        train_rows = int(0.8 * X.shape[0])
        X_train, y_train = X[:train_rows], y[:train_rows]
        X_test, y_test = X[train_rows:], y[train_rows:]
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # CPU training
        cpu_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 8,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }
        
        print("Training with CPU...")
        start_time = time.time()
        cpu_model = xgb.train(cpu_params, dtrain, num_boost_round=100)
        cpu_time = time.time() - start_time
        print(f"CPU training time: {cpu_time:.2f} seconds")
        
        # GPU training if requested and available
        if use_gpu:
            gpu_params = cpu_params.copy()
            gpu_params['tree_method'] = 'gpu_hist'
            gpu_params['gpu_id'] = 0
            
            try:
                print("\nTraining with GPU...")
                start_time = time.time()
                gpu_model = xgb.train(gpu_params, dtrain, num_boost_round=100)
                gpu_time = time.time() - start_time
                print(f"GPU training time: {gpu_time:.2f} seconds")
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"✓ GPU provides a {speedup:.1f}x speedup!")
                else:
                    print(f"! GPU is slower than CPU for this dataset")
                
                # Verify models give similar results
                cpu_preds = cpu_model.predict(dtest)
                gpu_preds = gpu_model.predict(dtest)
                correlation = np.corrcoef(cpu_preds, gpu_preds)[0, 1]
                print(f"Prediction correlation between CPU and GPU: {correlation:.4f}")
                
            except Exception as e:
                print(f"Error during GPU training: {e}")
        
        return True
    except ImportError:
        print("XGBoost not installed")
        return False
    except Exception as e:
        print(f"Error in XGBoost test: {e}")
        return False

def test_lightgbm(X, y, use_gpu=True):
    """Test LightGBM performance."""
    try:
        import lightgbm as lgb
        print("\n=== Testing LightGBM ===")
        
        # Split data
        train_rows = int(0.8 * X.shape[0])
        X_train, y_train = X[:train_rows], y[:train_rows]
        X_test, y_test = X[train_rows:], y[train_rows:]
        
        # Convert to Dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # CPU training
        cpu_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'device': 'cpu',
            'num_leaves': 64,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1
        }
        
        print("Training with CPU...")
        start_time = time.time()
        cpu_model = lgb.train(cpu_params, train_data, num_boost_round=100)
        cpu_time = time.time() - start_time
        print(f"CPU training time: {cpu_time:.2f} seconds")
        
        # GPU training if requested and available
        if use_gpu:
            gpu_params = cpu_params.copy()
            gpu_params['device'] = 'gpu'
            
            try:
                print("\nTraining with GPU...")
                start_time = time.time()
                gpu_model = lgb.train(gpu_params, train_data, num_boost_round=100)
                gpu_time = time.time() - start_time
                print(f"GPU training time: {gpu_time:.2f} seconds")
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"✓ GPU provides a {speedup:.1f}x speedup!")
                else:
                    print(f"! GPU is slower than CPU for this dataset")
                
                # Verify models give similar results
                cpu_preds = cpu_model.predict(X_test)
                gpu_preds = gpu_model.predict(X_test)
                correlation = np.corrcoef(cpu_preds, gpu_preds)[0, 1]
                print(f"Prediction correlation between CPU and GPU: {correlation:.4f}")
                
            except Exception as e:
                print(f"Error during GPU training: {e}")
        
        return True
    except ImportError:
        print("LightGBM not installed")
        return False
    except Exception as e:
        print(f"Error in LightGBM test: {e}")
        return False

def test_pandas_vs_polars():
    """Compare pandas vs polars performance."""
    try:
        import pandas as pd
        import polars as pl
        
        print("\n=== Testing DataFrame Libraries ===")
        
        # Generate test data
        rows = 1000000
        print(f"Generating {rows} rows of test data...")
        data = {
            'A': np.random.rand(rows),
            'B': np.random.rand(rows),
            'C': np.random.rand(rows),
            'D': np.random.randint(0, 100, size=rows)
        }
        
        # Pandas operations
        print("\nTesting pandas...")
        start_time = time.time()
        pdf = pd.DataFrame(data)
        pandas_create_time = time.time() - start_time
        print(f"pandas DataFrame creation: {pandas_create_time:.4f} seconds")
        
        start_time = time.time()
        result_pd = pdf.groupby('D').agg({'A': 'mean', 'B': 'sum', 'C': 'std'})
        pandas_agg_time = time.time() - start_time
        print(f"pandas aggregation: {pandas_agg_time:.4f} seconds")
        
        # Polars operations
        print("\nTesting polars...")
        start_time = time.time()
        pldf = pl.DataFrame(data)
        polars_create_time = time.time() - start_time
        print(f"polars DataFrame creation: {polars_create_time:.4f} seconds")
        
        start_time = time.time()
        result_pl = pldf.group_by('D').agg([
            pl.col('A').mean(),
            pl.col('B').sum(),
            pl.col('C').std()
        ])
        polars_agg_time = time.time() - start_time
        print(f"polars aggregation: {polars_agg_time:.4f} seconds")
        
        # Compare performance
        create_speedup = pandas_create_time / polars_create_time
        agg_speedup = pandas_agg_time / polars_agg_time
        print(f"\nPolars vs pandas speedup:")
        print(f"  DataFrame creation: {create_speedup:.2f}x")
        print(f"  Aggregation: {agg_speedup:.2f}x")
        
        return True
        
    except ImportError as e:
        print(f"Could not run pandas vs polars test: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 50)
    print("Numer_crypto ML Algorithm Test")
    print("=" * 50)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    print(f"GPU available: {gpu_available}")
    
    # Generate data
    X, y = generate_synthetic_data(args.rows, args.cols)
    
    # Run tests
    use_gpu = gpu_available and not args.no_gpu
    test_xgboost(X, y, use_gpu=use_gpu)
    test_lightgbm(X, y, use_gpu=use_gpu)
    test_pandas_vs_polars()
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()