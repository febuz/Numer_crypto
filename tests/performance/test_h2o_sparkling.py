#!/usr/bin/env python3
"""
Test H2O Sparkling Water in the test_env virtual environment
This script tests both standalone H2O and H2O Sparkling Water with a synthetic dataset
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Print Python and Environment information
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Import H2O
try:
    import h2o
    print("H2O version:", h2o.__version__)
    h2o_available = True
except ImportError:
    print("H2O not available")
    h2o_available = False

# Import PySpark and PySparkling
try:
    from pyspark.sql import SparkSession
    import pyspark
    print("PySpark version:", pyspark.__version__)
    
    # Check if h2o_pysparkling_3.5 is installed 
    try:
        # Try importing the correct pysparkling module
        import h2o_pysparkling_3_5
        from h2o_pysparkling_3_5 import H2OContext
        print("PySparkling version:", h2o_pysparkling_3_5.__version__)
        sparkling_available = True
    except ImportError:
        try:
            # Try the standard import path as a fallback
            from pysparkling import H2OContext
            import pysparkling
            print("PySparkling version:", pysparkling.__version__)
            sparkling_available = True
        except ImportError:
            print("PySparkling not available")
            sparkling_available = False
except ImportError:
    print("PySpark not available")
    sparkling_available = False

def create_synthetic_dataset(n_samples=1000, n_features=10, n_classes=2):
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
    df['target'] = y.astype(int)  # Ensure integer targets for classification
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Dataset created. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def test_standalone_h2o(train_df, test_df):
    """Test standalone H2O with a synthetic dataset"""
    if not h2o_available:
        print("H2O not available, skipping standalone H2O test")
        return False
        
    print("\n" + "="*80)
    print("TESTING STANDALONE H2O")
    print("="*80)
    
    # Initialize H2O with smaller memory allocation
    start_time = time.time()
    try:
        h2o.init(max_mem_size="1g")
    except h2o.exceptions.H2OConnectionError as e:
        # Try with strict_version_check=False if there's a version mismatch
        if "Version mismatch" in str(e):
            print("Version mismatch detected. Trying with strict_version_check=False")
            h2o.init(max_mem_size="1g", strict_version_check=False)
        else:
            raise
    init_time = time.time() - start_time
    print(f"H2O initialized in {init_time:.2f} seconds")
    
    # Convert pandas DataFrames to H2O frames
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    
    # Convert target to categorical for classification
    train_h2o['target'] = train_h2o['target'].asfactor()
    test_h2o['target'] = test_h2o['target'].asfactor()
    
    # Set feature names and target
    features = train_df.columns.tolist()
    target = 'target'
    features.remove(target)
    
    # Print info about the frames
    print(f"Training data shape: {train_h2o.shape}")
    print(f"Test data shape: {test_h2o.shape}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    
    # Try Random Forest as a simple model
    try:
        from h2o.estimators import H2ORandomForestEstimator
        start_time = time.time()
        
        model = H2ORandomForestEstimator(
            ntrees=10,
            max_depth=5,
            seed=42
        )
        
        model.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
        train_time = time.time() - start_time
        
        # Model performance
        perf = model.model_performance(test_h2o)
        print("\nModel performance metrics:")
        print(f"AUC: {perf.auc()}")
        print(f"Logloss: {perf.logloss()}")
        print(f"Accuracy: {perf.accuracy()[0][1]}")
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Variable importance
        print("\nVariable importance:")
        varimp = model.varimp(use_pandas=True)
        if varimp is not None:
            print(varimp.head())
        
        h2o_success = True
    except Exception as e:
        print(f"Error in H2O Random Forest: {e}")
        h2o_success = False
    
    # Shutdown H2O
    h2o.cluster().shutdown()
    return h2o_success

def test_h2o_sparkling(train_df, test_df):
    """Test H2O Sparkling Water with a synthetic dataset"""
    if not sparkling_available:
        print("PySparkling not available, skipping Sparkling Water test")
        return False
        
    print("\n" + "="*80)
    print("TESTING H2O SPARKLING WATER")
    print("="*80)
    
    try:
        # Create a Spark session
        spark = SparkSession.builder \
            .appName("H2OSparklingTest") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()
            
        print("Spark session created successfully")
        
        # Convert pandas DataFrames to Spark DataFrames
        train_spark = spark.createDataFrame(train_df)
        test_spark = spark.createDataFrame(test_df)
        
        print(f"Spark DataFrames created. Train count: {train_spark.count()}, Test count: {test_spark.count()}")
        
        # Initialize H2O Sparkling
        try:
            # Get the correct H2OConf class from the available module
            if 'h2o_pysparkling_3_5' in sys.modules:
                from h2o_pysparkling_3_5 import H2OConf
            else:
                from pysparkling import H2OConf
            
            # Create H2O configuration with version check disabled
            # Try different H2OConf constructor patterns
            try:
                conf = H2OConf().setInternalClusterMode().setIgnoreVersionCheck(True)
            except Exception as e1:
                print(f"First H2OConf approach failed: {e1}")
                try:
                    # Try with just creating a basic H2OContext
                    print("Trying direct H2OContext creation...")
                    hc = H2OContext.getOrCreate()
                    return
                except Exception as e2:
                    print(f"Direct H2OContext creation failed: {e2}")
                    raise
            hc = H2OContext.getOrCreate(conf)
        except Exception as e:
            print(f"Error initializing H2OContext: {e}")
            # Try to get more debug information
            import traceback
            traceback.print_exc()
            raise
        print("H2O Context created successfully")
        
        # Convert Spark DataFrames to H2O Frames
        train_h2o = hc.asH2OFrame(train_spark)
        test_h2o = hc.asH2OFrame(test_spark)
        
        # Convert target to categorical for classification
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # Set feature names and target
        features = train_df.columns.tolist()
        target = 'target'
        features.remove(target)
        
        # Try Random Forest as a simple model
        try:
            from h2o.estimators import H2ORandomForestEstimator
            start_time = time.time()
            
            model = H2ORandomForestEstimator(
                ntrees=10,
                max_depth=5,
                seed=42
            )
            
            model.train(x=features, y=target, training_frame=train_h2o, validation_frame=test_h2o)
            train_time = time.time() - start_time
            
            # Model performance
            perf = model.model_performance(test_h2o)
            print("\nModel performance metrics:")
            print(f"AUC: {perf.auc()}")
            print(f"Logloss: {perf.logloss()}")
            print(f"Accuracy: {perf.accuracy()[0][1]}")
            print(f"Training completed in {train_time:.2f} seconds")
            
            # Variable importance
            print("\nVariable importance:")
            varimp = model.varimp(use_pandas=True)
            if varimp is not None:
                print(varimp.head())
            
            sparkling_success = True
        except Exception as e:
            print(f"Error in H2O Sparkling Random Forest: {e}")
            sparkling_success = False
            
        # Stop Spark session
        spark.stop()
        return sparkling_success
        
    except Exception as e:
        print(f"Error setting up H2O Sparkling: {e}")
        # Try to get more information about the error
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run tests"""
    print("\n" + "="*80)
    print("H2O AND SPARKLING WATER TEST")
    print("="*80)
    
    # Create synthetic dataset
    train_df, test_df = create_synthetic_dataset()
    
    # Test standalone H2O
    h2o_success = test_standalone_h2o(train_df, test_df)
    
    # Test H2O Sparkling
    sparkling_success = test_h2o_sparkling(train_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Standalone H2O: {'SUCCESS' if h2o_success else 'FAILED'}")
    print(f"H2O Sparkling Water: {'SUCCESS' if sparkling_success else 'FAILED'}")
    
    return 0 if (h2o_success or sparkling_success) else 1

if __name__ == "__main__":
    sys.exit(main())