#!/usr/bin/env python3
"""
Test H2O Sparkling Water with Java 11
This script creates a synthetic dataset and tests H2O Sparkling Water
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Print environment info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Print Java version
try:
    java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
    print(f"Java version:")
    print(java_version)
except Exception as e:
    print(f"Error getting Java version: {e}")

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
    
    # Try to import PySparkling
    try:
        from pysparkling import H2OContext
        import pysparkling
        print("PySparkling imported successfully")
        print(f"PySparkling path: {pysparkling.__file__}")
        sparkling_available = True
    except ImportError as e:
        print(f"PySparkling import error: {e}")
        sparkling_available = False
except ImportError:
    print("PySpark not available")
    sparkling_available = False

def create_synthetic_dataset(n_samples=500, n_features=5, n_classes=2):
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

def test_h2o_sparkling(train_df, test_df):
    """Test H2O Sparkling Water with a synthetic dataset"""
    if not sparkling_available:
        print("PySparkling not available, skipping Sparkling Water test")
        return False
        
    print("\n" + "="*80)
    print("TESTING H2O SPARKLING WATER")
    print("="*80)
    
    try:
        # Create a Spark session with appropriate configuration
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
        
        # Initialize H2O Sparkling with version check disabled
        try:
            from pysparkling import H2OConf
            
            # Create configuration with version check disabled
            conf = H2OConf()
            conf.set_internal_cluster_mode()
            
            # Check if setIgnoreVersionCheck method exists
            if hasattr(conf, 'setIgnoreVersionCheck'):
                conf.setIgnoreVersionCheck(True)
            elif hasattr(conf, 'set_ignore_version_check'):
                conf.set_ignore_version_check(True)
                
            # Initialize H2O Context
            hc = H2OContext.getOrCreate(conf)
            print("H2O Context created successfully!")
        except Exception as e1:
            print(f"Failed to create H2O Context with configuration: {e1}")
            try:
                # Try creating it directly
                hc = H2OContext.getOrCreate()
                print("H2O Context created directly!")
            except Exception as e2:
                print(f"Failed to create H2O Context directly: {e2}")
                raise
        
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
        
        # Skip model training, just check if conversion works
        try:
            # Check H2O Frame
            print("\nH2O Frame info:")
            print(f"Train H2O Frame shape: {train_h2o.shape}")
            print(f"Test H2O Frame shape: {test_h2o.shape}")
            
            # Print the data types
            print("\nColumn types:")
            print(train_h2o.types)
            
            # Success if we get here
            sparkling_success = True
            print("\nH2O Sparkling Water test SUCCESSFUL!")
        except Exception as e:
            print(f"Error in H2O Sparkling: {e}")
            sparkling_success = False
            
        # Stop Spark session
        spark.stop()
        
        # Stop H2O cluster
        h2o.cluster().shutdown()
        
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
    print("H2O SPARKLING WATER TEST WITH JAVA 11")
    print("="*80)
    
    # Create synthetic dataset
    train_df, test_df = create_synthetic_dataset()
    
    # Test H2O Sparkling
    sparkling_success = test_h2o_sparkling(train_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"H2O Sparkling Water with Java 11: {'SUCCESS' if sparkling_success else 'FAILED'}")
    
    return 0 if sparkling_success else 1

if __name__ == "__main__":
    sys.exit(main())