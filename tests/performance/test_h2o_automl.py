#!/usr/bin/env python
"""
Test script for H2O Sparkling Water AutoML.

This script tests:
1. H2O initialization
2. Dataset download capabilities
3. Basic H2O AutoML functionality
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Directory for test data
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
os.makedirs(TEST_DIR, exist_ok=True)

def check_h2o():
    """Check if H2O is installed and working."""
    print("\n=== Testing H2O ===")
    try:
        import h2o
        print(f"H2O version: {h2o.__version__}")
        
        # Try to initialize H2O
        print("Initializing H2O...")
        h2o.init(max_mem_size="2G")  # Use minimal memory for test
        
        print("H2O cluster info:")
        h2o.cluster().show_status()
        
        return h2o, True
    except ImportError:
        print("H2O not installed")
        return None, False
    except Exception as e:
        print(f"Error initializing H2O: {e}")
        return None, False

def check_h2o_extensions():
    """Check H2O extensions like Sparkling Water."""
    print("\n=== Testing H2O Extensions ===")
    try:
        import h2o
        
        # Try to import pysparkling
        try:
            from pysparkling import H2OContext
            print("PySparkling is installed")
            
            # Try to import Spark
            try:
                from pyspark.sql import SparkSession
                print("PySpark is installed")
                
                # Create a minimal Spark session for testing
                print("Creating Spark session...")
                spark = SparkSession.builder \
                    .appName("H2OSparklingTest") \
                    .config("spark.executor.memory", "2g") \
                    .config("spark.driver.memory", "2g") \
                    .getOrCreate()
                
                print(f"Spark version: {spark.version}")
                
                # Try to create H2O context (this connects Spark and H2O)
                try:
                    print("Initializing H2O Sparkling Water context...")
                    h2o_context = H2OContext.getOrCreate(spark)
                    print("H2O Sparkling Water context created successfully")
                    
                    # Return the session and context for later use
                    return spark, h2o_context, True
                except Exception as e:
                    print(f"Error initializing H2O Sparkling Water context: {e}")
                    return spark, None, False
            except ImportError:
                print("PySpark is not installed")
                return None, None, False
            except Exception as e:
                print(f"Error creating Spark session: {e}")
                return None, None, False
                
        except ImportError:
            print("PySparkling is not installed")
            return None, None, False
    except ImportError:
        print("H2O not installed")
        return None, None, False

def download_test_dataset(h2o):
    """Download a small test dataset from H2O."""
    print("\n=== Downloading Test Dataset ===")
    try:
        # Try to download the airlines dataset
        print("Downloading airlines dataset...")
        airlines_path = h2o.api("GET /3/NodePersistentStorage/categories/url/files/airlinesbillion.csv")["dir"]
        
        # Parse dataset
        print("Parsing airlines dataset...")
        start_time = time.time()
        airlines = h2o.import_file(path=airlines_path, destination_frame="airlines_billions")
        parse_time = time.time() - start_time
        print(f"Parse time: {parse_time:.2f} seconds")
        
        # Print dataset info
        print(f"Dataset shape: {airlines.shape}")
        print("Dataset summary:")
        print(airlines.describe())
        
        return airlines
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Try with a smaller built-in dataset as fallback
        try:
            print("\nFalling back to built-in prostate dataset...")
            prostate = h2o.demo("prostate")
            print(f"Dataset shape: {prostate.shape}")
            return prostate
        except Exception as e2:
            print(f"Error with fallback dataset: {e2}")
            return None

def run_automl(h2o, dataset):
    """Run H2O AutoML on the dataset."""
    print("\n=== Running H2O AutoML ===")
    try:
        from h2o.automl import H2OAutoML
        
        # Check if dataset is available
        if dataset is None:
            print("No dataset available for AutoML")
            return False
        
        # Define features and target
        if dataset.names == ['CAPSULE', 'AGE', 'RACE', 'DPROS', 'DCAPS', 'PSA', 'VOL', 'GLEASON']:
            # Prostate dataset
            print("Using prostate dataset for AutoML")
            target = "CAPSULE"
            features = [c for c in dataset.names if c != target]
        else:
            # Airlines dataset
            print("Using airlines dataset for AutoML")
            target = "IsArrDelayed"
            features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'Distance']
        
        # Split dataset
        train, test = dataset.split_frame(ratios=[0.8])
        print(f"Training set size: {train.shape}")
        print(f"Test set size: {test.shape}")
        
        # Run AutoML
        print(f"Running AutoML for target '{target}' with features: {features}")
        print("AutoML will run for max 30 seconds...")
        aml = H2OAutoML(max_runtime_secs=30, seed=1)
        
        start_time = time.time()
        aml.train(x=features, y=target, training_frame=train)
        training_time = time.time() - start_time
        
        # Show results
        print(f"AutoML completed in {training_time:.2f} seconds")
        print(f"Top model: {aml.leader.model_id}")
        print(f"Models trained: {len(aml.leaderboard)}")
        
        # Show leaderboard
        print("Leaderboard:")
        print(aml.leaderboard.head(5))
        
        # Score on test set
        print("Evaluating on test set...")
        perf = aml.leader.model_performance(test)
        if target == "CAPSULE":  # Binary classification
            print(f"AUC: {perf.auc()}")
            print(f"Accuracy: {1 - perf.mean_per_class_error()}")
        else:  # Likely regression/classification for airlines
            try:
                print(f"AUC: {perf.auc()}")
            except:
                print(f"Error: {perf.error()}")
        
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        return False
    except Exception as e:
        print(f"Error during AutoML: {e}")
        return False

def check_sparkling_automl(spark, h2o_context):
    """Check if Sparkling Water AutoML works."""
    print("\n=== Testing Sparkling Water AutoML ===")
    
    if spark is None or h2o_context is None:
        print("Spark or H2O Context not available, skipping Sparkling Water AutoML test")
        return False
    
    try:
        # Try to create a small Spark DataFrame
        print("Creating test Spark DataFrame...")
        import numpy as np
        import pandas as pd
        from pyspark.sql.types import StructType, StructField, DoubleType, StringType
        
        # Create schema
        schema = StructType([
            StructField("id", DoubleType(), False),
            StructField("x1", DoubleType(), False),
            StructField("x2", DoubleType(), False),
            StructField("x3", DoubleType(), False),
            StructField("x4", DoubleType(), False),
            StructField("y", StringType(), False)
        ])
        
        # Generate random data
        np.random.seed(42)
        n_rows = 1000
        data = []
        for i in range(n_rows):
            x1, x2, x3, x4 = np.random.rand(4)
            y = "yes" if x1 + x2 > 1 else "no"
            data.append((float(i), float(x1), float(x2), float(x3), float(x4), y))
        
        # Create Spark DataFrame
        spark_df = spark.createDataFrame(data, schema=schema)
        print(f"Created Spark DataFrame with {spark_df.count()} rows")
        
        # Convert to H2O Frame
        print("Converting to H2O Frame...")
        h2o_df = h2o_context.asH2OFrame(spark_df)
        print(f"H2O Frame shape: {h2o_df.shape}")
        
        # Run AutoML on this data
        print("Running AutoML on Sparkling Water data...")
        from h2o.automl import H2OAutoML
        
        # Split dataset
        train, test = h2o_df.split_frame(ratios=[0.8])
        
        # Run AutoML
        aml = H2OAutoML(max_runtime_secs=30, seed=1)
        features = ["x1", "x2", "x3", "x4"]
        target = "y"
        
        start_time = time.time()
        aml.train(x=features, y=target, training_frame=train)
        training_time = time.time() - start_time
        
        # Show results
        print(f"Sparkling Water AutoML completed in {training_time:.2f} seconds")
        print(f"Top model: {aml.leader.model_id}")
        
        # Score on test set
        print("Evaluating on test set...")
        perf = aml.leader.model_performance(test)
        print(f"AUC: {perf.auc()}")
        
        # Convert predictions back to Spark
        print("Converting predictions back to Spark...")
        preds = aml.leader.predict(test)
        spark_preds = h2o_context.asSparkFrame(preds)
        print(f"Spark predictions shape: ({spark_preds.count()}, {len(spark_preds.columns)})")
        
        return True
    except ImportError as e:
        print(f"Missing required package for Sparkling Water AutoML: {e}")
        return False
    except Exception as e:
        print(f"Error during Sparkling Water AutoML: {e}")
        return False

def test_gpu_enabled_extensions():
    """Test if GPU-enabled extensions are available."""
    print("\n=== Testing GPU-Enabled Extensions ===")
    
    # Check XGBoost with GPU
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        
        # Create a small dataset
        import numpy as np
        X = np.random.rand(1000, 10).astype('float32')
        y = (X[:, 0] + X[:, 1] > 1).astype('float32')
        
        # Try GPU parameters
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        dtrain = xgb.DMatrix(X, label=y)
        
        try:
            # Try to train a small model with GPU
            bst = xgb.train(params, dtrain, num_boost_round=5)
            print("✓ XGBoost GPU support is working")
            return True
        except Exception as e:
            print(f"⨯ XGBoost GPU support not working: {e}")
            return False
    except ImportError:
        print("XGBoost not installed")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("H2O Sparkling Water and AutoML Test")
    print("=" * 60)
    
    # Check GPU-enabled extensions
    test_gpu_enabled_extensions()
    
    # Test H2O
    h2o, h2o_ok = check_h2o()
    
    # If H2O is working, test related functionality
    if h2o_ok:
        # Test H2O extension
        spark, h2o_context, spark_ok = check_h2o_extensions()
        
        # Download test dataset
        dataset = download_test_dataset(h2o)
        
        # Run AutoML
        run_automl(h2o, dataset)
        
        # Test Sparkling Water AutoML if available
        if spark_ok:
            check_sparkling_automl(spark, h2o_context)
        
        # Shutdown H2O and Spark
        print("\n=== Cleaning Up ===")
        h2o.cluster().shutdown()
        if spark_ok:
            spark.stop()
            
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()