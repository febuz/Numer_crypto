#!/usr/bin/env python3
"""
Minimal test for H2O Sparkling Water with Java 17
This script verifies H2O Sparkling Water compatibility with Java 17
"""

import os
import sys
import subprocess
import time

print("=" * 80)
print("H2O SPARKLING WATER TEST WITH JAVA 17")
print("=" * 80)

# Print environment info
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Print Java version
try:
    java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
    print(f"Java version:")
    print(java_version)
    
    # Verify we're using Java 17
    if "openjdk version \"17" not in java_version:
        print("WARNING: Not using Java 17. Please run: source setup_java17_env.sh")
except Exception as e:
    print(f"Error getting Java version: {e}")

# Timestamp for performance measurement
start_time = time.time()

# Import PySpark and create a session
try:
    print("\nInitializing Spark...")
    from pyspark.sql import SparkSession
    
    # Create a simple Spark session with minimal resources and Java 17 modules fix
    spark = SparkSession.builder \
        .appName("H2OSparklingJava17Test") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/sun.net=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions", "--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/sun.net=ALL-UNNAMED") \
        .getOrCreate()
    
    print("Spark session created successfully!")
    
    # Create a simple DataFrame
    data = [("Alice", 25), ("Bob", 30)]
    df = spark.createDataFrame(data, ["name", "age"])
    print("\nSample Spark DataFrame:")
    df.show()
    
    # Import and initialize H2O Sparkling
    print("\nImporting H2O Sparkling...")
    try:
        from pysparkling import H2OContext
        print("PySparkling successfully imported!")
        
        print("\nCreating H2O Context (this may take a moment)...")
        # Create H2O Context
        hc = H2OContext.getOrCreate()
        print("H2O Context created successfully!")
        
        # Convert Spark DataFrame to H2O Frame
        print("\nConverting Spark DataFrame to H2O Frame...")
        h2o_df = hc.asH2OFrame(df)
        print("Successfully converted to H2O Frame!")
        print(h2o_df)
        
        # Print H2O status
        import h2o
        h2o_status = h2o.cluster_status()
        if hasattr(h2o_status, "status"):
            print(f"\nH2O cluster status: {h2o_status.status}")
        if hasattr(h2o_status, "version"):
            print(f"H2O version: {h2o_status.version}")
        
        # Success!
        print("\n" + "=" * 80)
        print("SUCCESS: H2O Sparkling Water works with Java 17!")
        print("=" * 80)
        
        # Performance summary
        end_time = time.time()
        print(f"\nTest completed in {end_time - start_time:.2f} seconds")
        
        # Clean up
        print("\nShutting down H2O...")
        h2o.cluster().shutdown()
        print("Stopping Spark session...")
        spark.stop()
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"Error with H2O Context: {e}")
        # Try to get more information about the error
        import traceback
        traceback.print_exc()
        if spark:
            spark.stop()
        sys.exit(1)
        
except Exception as e:
    print(f"Error with Spark: {e}")
    # Try to get more information about the error
    import traceback
    traceback.print_exc()
    sys.exit(1)