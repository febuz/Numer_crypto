#!/usr/bin/env python
"""
Minimal test script for ML algorithms in Numer_crypto.
"""
import os
import sys
import time
import numpy as np

def check_versions():
    """Print versions of installed packages."""
    print("=== Package Versions ===")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except ImportError:
        print("pandas: Not installed")
        
    try:
        import polars as pl
        print(f"Polars: {pl.__version__}")
    except ImportError:
        print("Polars: Not installed")
    
    try:
        import xgboost as xgb
        print(f"XGBoost: {xgb.__version__}")
    except ImportError:
        print("XGBoost: Not installed")
    
    try:
        import lightgbm as lgb
        print(f"LightGBM: {lgb.__version__}")
    except ImportError:
        print("LightGBM: Not installed")
    
    try:
        import pyspark
        print(f"PySpark: {pyspark.__version__}")
    except ImportError:
        print("PySpark: Not installed")

def test_pandas_vs_polars():
    """Simple test to compare pandas vs polars."""
    try:
        import pandas as pd
        import polars as pl
        
        print("\n=== DataFrame Libraries Simple Test ===")
        
        # Generate small test data
        rows = 100000
        print(f"Generating {rows} rows of test data...")
        data = {
            'A': np.random.rand(rows),
            'B': np.random.rand(rows),
            'C': np.random.rand(rows)
        }
        
        # Pandas operations
        print("\nTesting pandas...")
        start_time = time.time()
        pdf = pd.DataFrame(data)
        result_pd = pdf.describe()
        pandas_time = time.time() - start_time
        print(f"pandas time: {pandas_time:.4f} seconds")
        
        # Polars operations
        print("\nTesting polars...")
        start_time = time.time()
        pldf = pl.DataFrame(data)
        result_pl = pldf.describe()
        polars_time = time.time() - start_time
        print(f"polars time: {polars_time:.4f} seconds")
        
        # Compare performance
        speedup = pandas_time / polars_time
        print(f"\nPolars vs pandas speedup: {speedup:.2f}x")
        return True
        
    except ImportError as e:
        print(f"Could not run pandas vs polars test: {e}")
        return False

def test_basic_ml():
    """Basic ML test with small dataset."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        print("\n=== Basic ML Test ===")
        
        # Generate small test data
        print("Generating small test data...")
        X = np.random.rand(1000, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Train a simple model
        print("Training RandomForest model...")
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        elapsed = time.time() - start_time
        
        print(f"Training completed in {elapsed:.4f} seconds")
        print(f"Model score: {model.score(X, y):.4f}")
        
        # Try XGBoost if available
        try:
            import xgboost as xgb
            print("\nTraining XGBoost model...")
            dtrain = xgb.DMatrix(X, label=y)
            params = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic'}
            
            start_time = time.time()
            xgb_model = xgb.train(params, dtrain, num_boost_round=10)
            elapsed = time.time() - start_time
            
            print(f"XGBoost training completed in {elapsed:.4f} seconds")
        except ImportError:
            print("XGBoost not available for testing")
        
        return True
    
    except Exception as e:
        print(f"Error in basic ML test: {e}")
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("Numer_crypto Minimal Test")
    print("=" * 50)
    
    # Check package versions
    check_versions()
    
    # Run basic tests
    test_pandas_vs_polars()
    test_basic_ml()
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()