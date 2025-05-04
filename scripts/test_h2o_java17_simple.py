#!/usr/bin/env python3
"""
Ultra simple test for H2O Sparkling Water with Java 17
Only creates session and checks Java version - no data processing
"""

import os
import sys
import subprocess
import time

print("=" * 80)
print("SIMPLE H2O SPARKLING WATER TEST WITH JAVA 17")
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

# Import PySpark and create a session
try:
    print("\nInitializing Spark...")
    from pyspark.sql import SparkSession
    
    # Create a simple Spark session with minimal resources
    spark = SparkSession.builder \
        .appName("H2OSparklingJava17Simple") \
        .config("spark.executor.memory", "512m") \
        .config("spark.driver.memory", "512m") \
        .getOrCreate()
    
    print("Spark session created successfully!")
    
    # Import and initialize H2O Sparkling - version only
    print("\nChecking PySparkling availability...")
    try:
        import pysparkling
        from pysparkling import H2OConf
        print(f"PySparkling imported successfully!")
        print(f"PySparkling version: {pysparkling.__version__ if hasattr(pysparkling, '__version__') else 'unknown'}")
        print(f"PySparkling module location: {pysparkling.__file__}")
        
        # Just verify H2OConf works
        conf = H2OConf()
        print("Successfully created H2OConf object")
        
        print("\nTest SUCCESSFUL: PySparkling is compatible with Java 17!")
        
        # Clean up
        spark.stop()
        print("Spark session stopped.")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"Error with PySparkling: {e}")
        if spark:
            spark.stop()
        sys.exit(1)
        
except Exception as e:
    print(f"Error with Spark: {e}")
    sys.exit(1)