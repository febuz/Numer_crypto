#!/usr/bin/env python
"""
Script to check ML libraries (XGBoost and LightGBM).
"""
import sys
import time
import platform
import warnings
warnings.filterwarnings('ignore')

print(f"Python version: {platform.python_version()}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Generate test data
    print("\nGenerating test data...")
    n_rows = 10000
    n_cols = 20
    X = np.random.rand(n_rows, n_cols).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1).astype(np.int32)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Check scikit-learn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        print(f"\nUsing scikit-learn for RandomForest")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Evaluate
        score = rf.score(X_test, y_test)
        print(f"  Training time: {elapsed:.4f} seconds")
        print(f"  Test accuracy: {score:.4f}")
    except ImportError:
        print("scikit-learn not installed")
    
    # Check XGBoost
    try:
        import xgboost as xgb
        print(f"\nXGBoost version: {xgb.__version__}")
        
        # Split data
        X_train, X_test = X[:8000], X[8000:]
        y_train, y_test = y[:8000], y[8000:]
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Define parameters
        params = {
            'max_depth': 4,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'tree_method': 'hist'  # CPU version
        }
        
        # Train model
        print("  Training XGBoost model (CPU)...")
        start_time = time.time()
        model = xgb.train(params, dtrain, num_boost_round=10)
        elapsed = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = (y_pred_binary == y_test).mean()
        
        print(f"  Training time: {elapsed:.4f} seconds")
        print(f"  Test accuracy: {accuracy:.4f}")
        
        # Try GPU version if system supports it
        try:
            # Check GPU availability
            gpu_params = params.copy()
            gpu_params['tree_method'] = 'gpu_hist'
            gpu_params['gpu_id'] = 0
            
            # Small test to see if GPU is available
            mini_train = xgb.DMatrix(X_train[:100], label=y_train[:100])
            try:
                xgb.train(gpu_params, mini_train, num_boost_round=1)
                
                # If we get here, GPU training works
                print("\n  GPU support detected! Training XGBoost model (GPU)...")
                start_time = time.time()
                gpu_model = xgb.train(gpu_params, dtrain, num_boost_round=10)
                gpu_elapsed = time.time() - start_time
                
                # Evaluate
                gpu_y_pred = gpu_model.predict(dtest)
                gpu_y_pred_binary = (gpu_y_pred > 0.5).astype(int)
                gpu_accuracy = (gpu_y_pred_binary == y_test).mean()
                
                print(f"  GPU training time: {gpu_elapsed:.4f} seconds")
                print(f"  GPU test accuracy: {gpu_accuracy:.4f}")
                
                # Compare
                if elapsed > gpu_elapsed:
                    speedup = elapsed / gpu_elapsed
                    print(f"  GPU is {speedup:.2f}x faster than CPU!")
                else:
                    slowdown = gpu_elapsed / elapsed
                    print(f"  GPU is {slowdown:.2f}x slower than CPU")
                
            except Exception as e:
                print(f"  GPU support not available: {e}")
        except Exception as e:
            print(f"  Could not test GPU support: {e}")
            
    except ImportError:
        print("XGBoost not installed")
    
    # Check LightGBM
    try:
        import lightgbm as lgb
        print(f"\nLightGBM version: {lgb.__version__}")
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Define parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'max_depth': 4,
            'learning_rate': 0.1,
            'device': 'cpu'  # CPU version
        }
        
        # Train model
        print("  Training LightGBM model (CPU)...")
        start_time = time.time()
        model = lgb.train(params, train_data, num_boost_round=10)
        elapsed = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = (y_pred_binary == y_test).mean()
        
        print(f"  Training time: {elapsed:.4f} seconds")
        print(f"  Test accuracy: {accuracy:.4f}")
        
        # Try GPU version if system supports it
        try:
            # Check if GPU is available
            if 'gpu' in lgb.get_device_type():
                print("\n  GPU support detected! Training LightGBM model (GPU)...")
                
                gpu_params = params.copy()
                gpu_params['device'] = 'gpu'
                
                start_time = time.time()
                gpu_model = lgb.train(gpu_params, train_data, num_boost_round=10)
                gpu_elapsed = time.time() - start_time
                
                # Evaluate
                gpu_y_pred = gpu_model.predict(X_test)
                gpu_y_pred_binary = (gpu_y_pred > 0.5).astype(int)
                gpu_accuracy = (gpu_y_pred_binary == y_test).mean()
                
                print(f"  GPU training time: {gpu_elapsed:.4f} seconds")
                print(f"  GPU test accuracy: {gpu_accuracy:.4f}")
                
                # Compare
                if elapsed > gpu_elapsed:
                    speedup = elapsed / gpu_elapsed
                    print(f"  GPU is {speedup:.2f}x faster than CPU!")
                else:
                    slowdown = gpu_elapsed / elapsed
                    print(f"  GPU is {slowdown:.2f}x slower than CPU")
            else:
                print("  GPU support not available")
        except Exception as e:
            print(f"  Could not test GPU support: {e}")
            
    except ImportError:
        print("LightGBM not installed")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error during test: {e}")
    
print("\nTests completed!")