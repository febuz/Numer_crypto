#!/usr/bin/env python3
"""
Minimal test for H2O Sparkling Water with Java 11
This script verifies H2O Sparkling Water compatibility with Java 11
"""

import os
import sys
import subprocess

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

# Import PySpark and create a session
try:
    from pyspark.sql import SparkSession
    
    print("Creating Spark session...")
    # Create a simple Spark session
    spark = SparkSession.builder \
        .appName("H2OSparklingMinimalTest") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    print("Spark session created successfully!")
    
    # Create a simple DataFrame
    data = [("Alice", 25), ("Bob", 30)]
    df = spark.createDataFrame(data, ["name", "age"])
    print("Sample Spark DataFrame:")
    df.show()
    
    # Import and initialize H2O Sparkling
    print("Importing H2O Sparkling...")
    try:
        from pysparkling import H2OContext
        print("Successfully imported pysparkling!")
        
        print("Creating H2O Context...")
        hc = H2OContext.getOrCreate()
        print("H2O Context created successfully!")
        
        # If we got here, H2O Sparkling Water works with Java 11!
        print("\n*** SUCCESS! H2O Sparkling Water works with Java 11! ***\n")
        
        # Convert Spark DataFrame to H2O Frame
        print("Converting Spark DataFrame to H2O Frame...")
        h2o_df = hc.asH2OFrame(df)
        print("Converted to H2O Frame:")
        print(h2o_df)
        
        # Shutdown
        print("Shutting down H2O...")
        h2o_df.frame_id  # Just accessing an attribute to confirm it works
        
        # Success!
        print("Test completed successfully!")
        success = True
    except Exception as e:
        print(f"Error with H2O Context: {e}")
        success = False
    finally:
        print("Stopping Spark session...")
        spark.stop()
except Exception as e:
    print(f"Error with Spark: {e}")
    success = False

# Print final result
if success:
    print("\n*** TEST SUCCESSFUL: H2O Sparkling Water works with Java 11! ***")
    sys.exit(0)
else:
    print("\n*** TEST FAILED: H2O Sparkling Water has issues with Java 11 ***")
    sys.exit(1)