#!/usr/bin/env python
"""
Simple test script for H2O functionality.

This script tests:
1. H2O initialization
2. Loading a basic dataset
3. Training a simple model
"""
import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Directory for test data
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
os.makedirs(TEST_DIR, exist_ok=True)

def create_test_dataset():
    """Create a simple test dataset."""
    print("Creating test dataset...")
    
    # Generate synthetic data
    np.random.seed(1234)
    n_rows = 5000
    n_cols = 10
    
    # Features
    X = np.random.randn(n_rows, n_cols)
    
    # Target (binary classification)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_rows) * 0.1 > 0).astype(int)
    
    # Combine into a single array
    data = np.column_stack([y, X])
    
    # Save to CSV
    csv_path = os.path.join(TEST_DIR, 'test_dataset.csv')
    header = ['target'] + [f'x{i}' for i in range(n_cols)]
    
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in data:
            f.write(','.join(str(val) for val in row) + '\n')
    
    print(f"Dataset saved to {csv_path}")
    return csv_path

def test_h2o():
    """Test basic H2O functionality."""
    print("\n=== Testing H2O ===")
    
    try:
        import h2o
        from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
        print(f"H2O version: {h2o.__version__}")
        
        # Initialize H2O
        print("Initializing H2O...")
        h2o.init(max_mem_size="2G")
        
        # Create dataset
        csv_path = create_test_dataset()
        
        # Import dataset
        print("Importing dataset...")
        data = h2o.import_file(csv_path)
        print(f"Dataset shape: {data.shape}")
        
        # Display data summary
        print("Dataset summary:")
        print(data.describe())
        
        # Split data into train and test
        train, test = data.split_frame(ratios=[0.8])
        
        # Define features and target
        y = "target"
        x = [col for col in data.columns if col != y]
        
        print(f"Target: {y}")
        print(f"Features: {x}")
        
        # Train a simple model
        print("\nTraining Random Forest model...")
        model = H2ORandomForestEstimator(
            ntrees=10,
            max_depth=5,
            seed=1234
        )
        
        start_time = time.time()
        model.train(x=x, y=y, training_frame=train)
        training_time = time.time() - start_time
        
        print(f"Model trained in {training_time:.2f} seconds")
        
        # Show model performance
        print("\nModel performance:")
        perf = model.model_performance(test)
        print(f"Metrics: {perf.metric_json()}")
        try:
            print(f"AUC: {perf.auc()}")
        except:
            print("AUC not available")
        try:
            print(f"Accuracy: {1 - perf.mean_per_class_error()}")
        except:
            print("Accuracy not available")
        
        # Try GBM model
        print("\nTraining GBM model...")
        gbm = H2OGradientBoostingEstimator(
            ntrees=10,
            max_depth=3,
            learn_rate=0.1,
            seed=1234
        )
        
        start_time = time.time()
        gbm.train(x=x, y=y, training_frame=train)
        training_time = time.time() - start_time
        
        print(f"GBM trained in {training_time:.2f} seconds")
        
        # Show model performance
        print("\nGBM performance:")
        perf = gbm.model_performance(test)
        print(f"AUC: {perf.auc()}")
        print(f"Accuracy: {1 - perf.mean_per_class_error()}")
        
        # Try AutoML if available
        try:
            from h2o.automl import H2OAutoML
            print("\nRunning AutoML (limited to 30 seconds)...")
            
            aml = H2OAutoML(
                max_runtime_secs=30,
                seed=1234
            )
            
            start_time = time.time()
            aml.train(x=x, y=y, training_frame=train)
            training_time = time.time() - start_time
            
            print(f"AutoML completed in {training_time:.2f} seconds")
            print(f"Best model: {aml.leader.model_id}")
            
            # Show leaderboard
            print("\nLeaderboard:")
            print(aml.leaderboard.head(5))
            
            # Show performance
            print("\nBest model performance:")
            perf = aml.leader.model_performance(test)
            print(f"AUC: {perf.auc()}")
            print(f"Accuracy: {1 - perf.mean_per_class_error()}")
            
        except ImportError:
            print("H2O AutoML not available")
        
        # Shutdown H2O
        h2o.cluster().shutdown()
        
        return True
    except ImportError:
        print("H2O not installed")
        return False
    except Exception as e:
        print(f"Error in H2O test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("H2O Basic Functionality Test")
    print("=" * 60)
    
    # Test H2O
    result = test_h2o()
    
    print("\n" + "=" * 60)
    print(f"Test completed with {'success' if result else 'failure'}!")
    print("=" * 60)

if __name__ == "__main__":
    main()