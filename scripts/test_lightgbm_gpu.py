#!/usr/bin/env python3
"""
Test LightGBM with GPU acceleration
This script tests LightGBM's GPU acceleration capabilities using a synthetic dataset
and compares performance between CPU and GPU modes.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import lightgbm
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False
    print("LightGBM is not installed. Installing with pip...")
    import subprocess
    subprocess.run(["pip", "install", "lightgbm>=3.3.0", "--quiet"], check=True)
    try:
        import lightgbm as lgb
        lightgbm_available = True
        print("LightGBM installed successfully!")
    except ImportError:
        print("Failed to install LightGBM. Exiting.")
        sys.exit(1)

# Import project settings if available
try:
    from config.settings import HARDWARE_CONFIG
    custom_config = True
    gpu_count = HARDWARE_CONFIG.get('gpu_count', 0)
except ImportError:
    custom_config = False
    gpu_count = 0  # Default if no config

def create_synthetic_dataset(n_samples=5000, n_features=20):
    """Create a synthetic regression dataset"""
    print(f"Creating synthetic dataset with {n_samples} samples and {n_features} features...")
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        random_state=42
    )
    
    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Dataset created. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def check_gpu_availability():
    """Check if LightGBM can use GPU"""
    try:
        # Create a small dataset for testing
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        train_data = lgb.Dataset(X, label=y)
        
        # Try to train a model with GPU
        params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
        model = lgb.train(params, train_data, num_boost_round=1)
        return True
    except Exception as e:
        print(f"GPU not available for LightGBM: {e}")
        return False

def train_lightgbm(X_train, y_train, X_test, y_test, use_gpu=False):
    """Train a LightGBM model with CPU or GPU"""
    # Base parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbosity': -1
    }
    
    # Add GPU parameters if requested
    if use_gpu:
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model and measure time
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,  # Reduce number of rounds
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), 
                  lgb.log_evaluation(period=20)]
    )
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, rmse, train_time

def main():
    """Main function to run tests"""
    print("\n" + "="*80)
    print("LIGHTGBM GPU ACCELERATION TEST")
    print("="*80)
    
    if custom_config:
        print(f"Using custom hardware configuration: {HARDWARE_CONFIG}")
        print(f"GPU count: {gpu_count}")
    else:
        print("Using default configuration")
    
    # Check if LightGBM can use GPU
    gpu_available = check_gpu_availability()
    print(f"LightGBM GPU support: {'Available' if gpu_available else 'Not available'}")
    
    # Create synthetic dataset
    train_df, test_df = create_synthetic_dataset()
    
    # Extract features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Train with CPU
    print("\n" + "-"*80)
    print("Training LightGBM with CPU")
    print("-"*80)
    cpu_model, cpu_rmse, cpu_time = train_lightgbm(X_train, y_train, X_test, y_test, use_gpu=False)
    print(f"CPU training completed in {cpu_time:.2f} seconds")
    print(f"CPU RMSE: {cpu_rmse:.6f}")
    
    # Train with GPU if available
    if gpu_available:
        print("\n" + "-"*80)
        print("Training LightGBM with GPU")
        print("-"*80)
        gpu_model, gpu_rmse, gpu_time = train_lightgbm(X_train, y_train, X_test, y_test, use_gpu=True)
        print(f"GPU training completed in {gpu_time:.2f} seconds")
        print(f"GPU RMSE: {gpu_rmse:.6f}")
        
        # Speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\nGPU Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"LightGBM CPU training time: {cpu_time:.2f} seconds")
    if gpu_available:
        print(f"LightGBM GPU training time: {gpu_time:.2f} seconds")
        print(f"GPU speedup: {speedup:.2f}x")
    else:
        print("LightGBM GPU: Not available")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())