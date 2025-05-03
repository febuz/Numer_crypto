"""
Spark utility functions for the Numerai Crypto project.
"""
import os
from pyspark.sql import SparkSession
from pysparkling import H2OContext
import h2o
from numer_crypto.config.settings import SPARK_CONFIG


def create_spark_session():
    """
    Create and return a SparkSession with configured settings.
    
    Returns:
        SparkSession: Configured Spark session
    """
    # Set Java home if needed
    if 'JAVA_HOME' not in os.environ:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName(SPARK_CONFIG['app_name']) \
        .config("spark.executor.memory", SPARK_CONFIG['executor_memory']) \
        .config("spark.driver.memory", SPARK_CONFIG['driver_memory']) \
        .config("spark.executor.cores", SPARK_CONFIG['executor_cores']) \
        .config("spark.driver.extraJavaOptions", SPARK_CONFIG['driver_java_options']) \
        .config("spark.executor.extraJavaOptions", SPARK_CONFIG['executor_java_options']) \
        .config("spark.dynamicAllocation.enabled", "false") \
        .getOrCreate()
        
    return spark


def init_h2o():
    """
    Initialize H2O and return the H2O instance.
    
    Returns:
        h2o_context: Initialized H2O context
    """
    # Initialize H2O
    h2o.init()
    
    return h2o


def init_h2o_sparkling_water(spark):
    """
    Initialize H2O Sparkling Water.
    
    Args:
        spark (SparkSession): The Spark session
        
    Returns:
        H2OContext: Initialized H2O context
    """
    # Initialize H2O Sparkling Water context
    h2o_context = H2OContext.getOrCreate(spark)
    
    return h2o_context


def get_spark_h2o_environment():
    """
    Set up the complete Spark and H2O environment.
    
    Returns:
        tuple: (spark, h2o, h2o_context)
    """
    spark = create_spark_session()
    h2o_instance = init_h2o()
    h2o_context = init_h2o_sparkling_water(spark)
    
    # Print environment information
    print(f"Spark version: {spark.version}")
    print(f"H2O version: {h2o.version()}")
    
    return spark, h2o_instance, h2o_context