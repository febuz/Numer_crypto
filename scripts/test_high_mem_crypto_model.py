#!/usr/bin/env python3
"""
Test script for the high memory crypto model.
This script verifies that the environment is properly configured 
and the model components are working correctly.
"""
import os
import sys
import argparse
import platform
import psutil
import numpy as np
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Function to print system information
def print_system_info():
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU count: {os.cpu_count()}")
    
    # Memory information
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    
    # GPU information
    try:
        gpu_info = []
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(
                "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
                shell=True, universal_newlines=True
            )
            for line in nvidia_smi.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpu_info.append({
                        'index': parts[0],
                        'name': parts[1],
                        'memory': parts[2]
                    })
        except:
            pass
        
        if gpu_info:
            print("\n=== GPU Information ===")
            for gpu in gpu_info:
                print(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory']})")
        else:
            print("\nNo GPUs detected via nvidia-smi")
    except:
        print("\nCould not retrieve GPU information")

# Function to check dependencies
def check_dependencies():
    print("\n=== Checking Dependencies ===")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('torch', 'PyTorch'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('h2o', 'H2O'),
        ('sklearn', 'scikit-learn')
    ]
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                print(f"✓ {display_name}: {module.__version__}")
            else:
                print(f"✓ {display_name}: installed (no version info)")
                
            # Check PyTorch CUDA availability
            if module_name == 'torch':
                cuda_available = module.cuda.is_available()
                print(f"  - CUDA available: {'Yes' if cuda_available else 'No'}")
                if cuda_available:
                    print(f"  - CUDA version: {module.version.cuda}")
                    device_count = module.cuda.device_count()
                    print(f"  - GPU count: {device_count}")
                    for i in range(device_count):
                        print(f"  - GPU {i}: {module.cuda.get_device_name(i)}")
            
            # Check H2O version
            if module_name == 'h2o':
                print(f"  - H2O Version: {module.__version__}")
                if module.__version__ != "3.46.0.6":
                    print(f"  - WARNING: H2O version is not 3.46.0.6 (required for Sparkling Water)")
                
        except ImportError:
            print(f"✗ {display_name}: not installed")
        except Exception as e:
            print(f"✗ {display_name}: error checking - {str(e)}")

# Function to create and test synthetic data
def test_synthetic_data(n_samples=1000, n_features=20):
    print(f"\n=== Testing Synthetic Data Creation (n={n_samples}, features={n_features}) ===")
    
    try:
        # Create random training data
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, size=n_samples)
        
        X_test = np.random.randn(n_samples // 10, n_features)
        
        # Convert to pandas
        df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(n_features)])
        df_train['target'] = y_train
        
        df_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(n_features)])
        
        print(f"Training data shape: {df_train.shape}")
        print(f"Testing data shape: {df_test.shape}")
        
        # Test simple feature engineering
        df_train['feature_0_squared'] = df_train['feature_0'] ** 2
        df_train['feature_0_1_product'] = df_train['feature_0'] * df_train['feature_1']
        
        print("Sample feature engineering successful")
        
        return df_train, df_test
    except Exception as e:
        print(f"Error creating synthetic data: {str(e)}")
        return None, None

# Function to test LightGBM model
def test_lightgbm(df_train, feature_cols, use_gpu=False):
    print(f"\n=== Testing LightGBM (GPU={use_gpu}) ===")
    
    try:
        import lightgbm as lgb
        
        # Prepare data
        X = df_train[feature_cols].values
        y = df_train['target'].values
        
        # Create parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42
        }
        
        # Add GPU parameters if requested
        if use_gpu:
            try:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
                print("GPU parameters added")
            except:
                print("Could not add GPU parameters")
        
        # Create dataset
        lgb_train = lgb.Dataset(X, y)
        
        # Train a small model
        print("Training LightGBM model...")
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=10  # Just a few iterations for testing
        )
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        print(f"Feature importance calculated (max: {max(importance) if importance.size > 0 else 'N/A'})")
        
        print("LightGBM test successful!")
        return True
    except Exception as e:
        print(f"Error testing LightGBM: {str(e)}")
        return False

# Function to test PyTorch model
def test_pytorch(df_train, feature_cols, use_gpu=False):
    print(f"\n=== Testing PyTorch (GPU={use_gpu}) ===")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # Check if CUDA is available when GPU is requested
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA not available, falling back to CPU")
            use_gpu = False
        
        device = torch.device('cuda' if use_gpu else 'cpu')
        print(f"Using device: {device}")
        
        # Prepare data
        X = df_train[feature_cols].values
        y = df_train['target'].values
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        # Create a simple network
        class SimpleNet(nn.Module):
            def __init__(self, input_dim):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x.squeeze()
        
        # Create model, loss function, and optimizer
        model = SimpleNet(len(feature_cols)).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Train for a few epochs
        print("Training PyTorch model...")
        for epoch in range(5):  # Just a few epochs for testing
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
        
        print("PyTorch test successful!")
        return True
    except Exception as e:
        print(f"Error testing PyTorch: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test the high memory crypto model components')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU tests')
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    
    print("\n===== HIGH MEMORY CRYPTO MODEL TEST =====\n")
    print(f"GPU tests: {'Disabled' if args.no_gpu else 'Enabled'}")
    
    # Print system information
    print_system_info()
    
    # Check dependencies
    check_dependencies()
    
    # Create synthetic data
    df_train, df_test = test_synthetic_data()
    
    if df_train is not None:
        # Get feature columns
        feature_cols = [col for col in df_train.columns if 'feature_' in col]
        
        # Test LightGBM
        lgb_success = test_lightgbm(df_train, feature_cols, use_gpu=use_gpu)
        
        # Test PyTorch
        pt_success = test_pytorch(df_train, feature_cols, use_gpu=use_gpu)
        
        # Print summary
        print("\n===== TEST SUMMARY =====")
        print(f"LightGBM: {'PASSED' if lgb_success else 'FAILED'}")
        print(f"PyTorch: {'PASSED' if pt_success else 'FAILED'}")
        
        if lgb_success and pt_success:
            print("\nAll tests passed. The environment is ready for the high memory crypto model.")
            return 0
        else:
            print("\nSome tests failed. Please check the output for details.")
            return 1
    else:
        print("\nFailed to create synthetic data. Cannot proceed with model tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())