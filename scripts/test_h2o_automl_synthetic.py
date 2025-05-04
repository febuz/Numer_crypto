#!/usr/bin/env python3
"""
Test H2O AutoML and Sparkling Water with a synthetic dataset
This script avoids relying on external data sources by creating a synthetic dataset
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import h2o - handle ImportError if not installed
try:
    import h2o
    from h2o.automl import H2OAutoML
    h2o_available = True
except ImportError:
    h2o_available = False
    print("H2O is not installed. Installing with pip...")
    import subprocess
    subprocess.run(["pip", "install", "h2o>=3.40.0.1", "--quiet"], check=True)
    try:
        import h2o
        from h2o.automl import H2OAutoML
        h2o_available = True
        print("H2O installed successfully!")
    except ImportError:
        print("Failed to install H2O. Exiting.")
        sys.exit(1)

# Try to import pyspark and h2o sparkling
try:
    from pyspark.sql import SparkSession
    import pysparkling
    from pysparkling.ml import H2OAutoML as H2OSparklingAutoML
    sparkling_available = True
except ImportError:
    sparkling_available = False
    print("PySparkling is not available, will test only standalone H2O")

# Import project settings if available
try:
    from config.settings import HARDWARE_CONFIG, H2O_CONFIG
    custom_config = True
    max_mem = HARDWARE_CONFIG.get('total_memory', '4g')
    # Strip the 'g' and convert to int, then take 80%
    if 'g' in max_mem:
        h2o_mem = str(int(int(max_mem.replace('g', '')) * 0.8)) + 'g'
    else:
        h2o_mem = '4g'  # Default
except ImportError:
    custom_config = False
    h2o_mem = '4g'  # Default to 4GB if no config

def create_synthetic_dataset(n_samples=10000, n_features=20, n_classes=2):
    """Create a synthetic classification dataset"""
    print(f"Creating synthetic dataset with {n_samples} samples and {n_features} features...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=n_classes,
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

def test_standalone_h2o(train_df, test_df):
    """Test H2O AutoML with a synthetic dataset"""
    print("\n" + "="*80)
    print("TESTING STANDALONE H2O AUTOML")
    print("="*80)
    
    # Initialize H2O
    start_time = time.time()
    h2o.init(max_mem_size=h2o_mem)
    init_time = time.time() - start_time
    print(f"H2O initialized in {init_time:.2f} seconds")
    
    # Convert pandas DataFrames to H2O frames
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    
    # Set feature names and target
    features = train_df.columns.tolist()
    target = 'target'
    features.remove(target)
    
    # Print info about the frames
    print(f"Training data shape: {train_h2o.shape}")
    print(f"Test data shape: {test_h2o.shape}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    
    # Run AutoML
    try:
        start_time = time.time()
        aml = H2OAutoML(
            max_models=5,
            seed=42,
            max_runtime_secs=120  # Limit to 2 minutes for testing
        )
        aml.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
        train_time = time.time() - start_time
        
        # Get the leaderboard
        lb = aml.leaderboard
        print("\nH2O AutoML Leaderboard (top 5 models):")
        print(lb.head(5))
        
        # Get the best model
        best_model = aml.leader
        print(f"\nBest model: {best_model}")
        
        # Model performance
        try:
            perf = best_model.model_performance(test_h2o)
            metrics = perf.metric_json()
            print("\nModel performance metrics:")
            if 'AUC' in metrics:
                print(f"AUC: {metrics['AUC']}")
            if 'logloss' in metrics:
                print(f"Logloss: {metrics['logloss']}")
            if 'accuracy' in metrics:
                print(f"Accuracy: {metrics['accuracy']}")
            print(f"Training completed in {train_time:.2f} seconds")
            success = True
        except Exception as e:
            print(f"Error getting model performance: {e}")
            success = False
            
    except Exception as e:
        print(f"Error in H2O AutoML: {e}")
        success = False
    
    # Shutdown H2O
    h2o.cluster().shutdown()
    return success

def test_sparkling_water(train_df, test_df):
    """Test H2O Sparkling Water with a synthetic dataset"""
    if not sparkling_available:
        print("PySparkling is not available, skipping Sparkling Water test")
        return False
        
    print("\n" + "="*80)
    print("TESTING H2O SPARKLING WATER")
    print("="*80)
    
    try:
        # Create a Spark session
        spark = SparkSession.builder \
            .appName("H2OSparklingTest") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
            
        # Convert pandas DataFrames to Spark DataFrames
        train_spark = spark.createDataFrame(train_df)
        test_spark = spark.createDataFrame(test_df)
        
        # Initialize H2O Sparkling
        import h2o
        from pysparkling import H2OContext
        hc = H2OContext.getOrCreate()
        
        # Set feature names and target
        features = train_df.columns.tolist()
        target = 'target'
        features.remove(target)
        
        # Run H2O Sparkling AutoML
        try:
            start_time = time.time()
            automl = H2OSparklingAutoML(
                maxModels=5,
                seed=42,
                maxRuntimeSecs=120  # Limit to 2 minutes for testing
            )
            
            # Set the feature and target columns
            automl.setFeaturesCols(features)
            automl.setLabelCol(target)
            
            # Train the model
            model = automl.fit(train_spark)
            train_time = time.time() - start_time
            
            # Make predictions
            predictions = model.transform(test_spark)
            print(f"Predictions shape: {predictions.count()} rows")
            
            # Show sample predictions
            print("\nSample predictions:")
            predictions.select("prediction", target).show(5)
            
            print(f"Sparkling Water training completed in {train_time:.2f} seconds")
            success = True
            
        except Exception as e:
            print(f"Error in H2O Sparkling AutoML: {e}")
            success = False
            
        # Stop the Spark session
        spark.stop()
        return success
        
    except Exception as e:
        print(f"Error setting up Sparkling Water: {e}")
        return False

def main():
    """Main function to run tests"""
    print("\n" + "="*80)
    print("H2O AND SPARKLING WATER TEST WITH SYNTHETIC DATA")
    print("="*80)
    
    if custom_config:
        print(f"Using custom hardware configuration: {HARDWARE_CONFIG}")
        print(f"H2O memory allocation: {h2o_mem}")
    else:
        print("Using default configuration")
    
    # Create synthetic dataset
    train_df, test_df = create_synthetic_dataset()
    
    # Test standalone H2O
    h2o_success = test_standalone_h2o(train_df, test_df)
    
    # Test Sparkling Water if available
    sparkling_success = test_sparkling_water(train_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Standalone H2O AutoML: {'SUCCESS' if h2o_success else 'FAILED'}")
    if sparkling_available:
        print(f"H2O Sparkling Water: {'SUCCESS' if sparkling_success else 'FAILED'}")
    else:
        print("H2O Sparkling Water: NOT TESTED (dependencies missing)")
    
    return 0 if (h2o_success or sparkling_success) else 1

if __name__ == "__main__":
    sys.exit(main())