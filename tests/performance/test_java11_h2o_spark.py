#!/usr/bin/env python3
"""
Test script to verify Java 11 with H2O Sparkling Water
"""
import sys
import os
import subprocess

# Print Java version
java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
print(f"Java version:")
print(java_version)

# Print Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Import required packages
try:
    import h2o
    print(f"H2O version: {h2o.__version__}")
except ImportError:
    print("H2O not available")

try:
    import pyspark
    print(f"PySpark version: {pyspark.__version__}")
except ImportError:
    print("PySpark not available")

try:
    from pysparkling import H2OContext
    import pysparkling
    print(f"PySparkling imported successfully")
    print(f"PySparkling path: {pysparkling.__file__}")
except ImportError as e:
    print(f"PySparkling import error: {e}")

# Create a minimal Spark session and H2O context
try:
    from pyspark.sql import SparkSession
    
    # Create a simple Spark session
    spark = SparkSession.builder         .appName("H2OSparklingTest")         .config("spark.executor.memory", "1g")         .config("spark.driver.memory", "1g")         .getOrCreate()
    
    print("Spark session created successfully")
    
    # Create a simple DataFrame
    data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    print("Sample DataFrame:")
    df.show()
    
    # Initialize H2O Sparkling
    try:
        hc = H2OContext.getOrCreate()
        print("H2O Context created successfully!")
        
        # Convert Spark DataFrame to H2O Frame
        h2o_df = hc.asH2OFrame(df)
        print("Converted to H2O Frame:")
        print(h2o_df)
        
        # Shutdown
        h2o.cluster().shutdown()
        spark.stop()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error with H2O Context: {e}")
        spark.stop()
except Exception as e:
    print(f"Error with Spark: {e}")
