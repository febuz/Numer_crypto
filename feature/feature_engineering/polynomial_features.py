#!/usr/bin/env python3
"""
Polynomial Feature Engineering for Numerai Crypto

This module generates polynomial features for crypto datasets
using Spark and H2O to efficiently process large datasets
and handle memory constraints.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Pyspark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler

# H2O imports for distributed processing
try:
    from pysparkling.conf import H2OConf
    from pysparkling import H2OContext
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("H2O not available. Some functionality may be limited.")

# Constants
FSTORE_DIR = "/media/knight2/EDB/fstore"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class PolynomialFeatureGenerator:
    """Generates polynomial features using Spark/H2O for efficient processing"""
    
    def __init__(self, output_dir=FSTORE_DIR, spark_app_name="NumeraiPolyFeatures"):
        """
        Initialize the polynomial feature generator
        
        Args:
            output_dir (str): Directory to store generated features
            spark_app_name (str): Name of the Spark application
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.spark_app_name = spark_app_name
        self.spark = None
        self.h2o_context = None
        
        logger.info(f"Polynomial Feature Generator initialized. Output dir: {self.output_dir}")
    
    def initialize_spark(self, memory="4g", cores=2, enable_gpu=False):
        """
        Initialize Spark session with appropriate configuration
        
        Args:
            memory (str): Memory to allocate for Spark driver/executor
            cores (int): Number of cores to use
            enable_gpu (bool): Whether to enable GPU acceleration
        """
        logger.info("Initializing Spark session")
        
        # Create Spark session builder
        builder = SparkSession.builder \
            .appName(self.spark_app_name) \
            .config("spark.executor.memory", memory) \
            .config("spark.driver.memory", memory) \
            .config("spark.executor.cores", str(cores)) \
            .config("spark.driver.maxResultSize", memory)
        
        # Enable GPU acceleration if requested and available
        if enable_gpu:
            try:
                # Check for RAPIDS acceleration
                import importlib.util
                rapids_available = importlib.util.find_spec("rapids") is not None
                
                if rapids_available:
                    logger.info("Enabling RAPIDS GPU acceleration for Spark")
                    builder = builder \
                        .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
                        .config("spark.rapids.sql.enabled", "true") \
                        .config("spark.rapids.sql.explain", "ALL")
                else:
                    logger.warning("RAPIDS not available for GPU acceleration")
            except ImportError:
                logger.warning("Could not check for RAPIDS availability")
        
        # Create Spark session
        self.spark = builder.getOrCreate()
        
        # Initialize H2O if available
        if H2O_AVAILABLE:
            try:
                logger.info("Initializing H2O")
                h2o.init()
                
                h2o_conf = H2OConf(self.spark)
                self.h2o_context = H2OContext.getOrCreate(h2o_conf)
                
                logger.info(f"H2O cluster initialized. Status:")
                logger.info(f"  - Version: {h2o.cluster().version}")
                logger.info(f"  - Nodes: {h2o.cluster().numberofnodes}")
                logger.info(f"  - Free Memory: {h2o.cluster().free_mem()}")
            except Exception as e:
                logger.error(f"Failed to initialize H2O: {str(e)}")
                self.h2o_context = None
        
        logger.info(f"Spark initialized. Version: {self.spark.version}")
    
    def shutdown(self):
        """Shutdown Spark and H2O sessions"""
        logger.info("Shutting down")
        
        if self.h2o_context:
            try:
                h2o.cluster().shutdown()
                logger.info("H2O shutdown completed")
            except Exception as e:
                logger.warning(f"Error during H2O shutdown: {str(e)}")
        
        if self.spark:
            try:
                self.spark.stop()
                logger.info("Spark shutdown completed")
            except Exception as e:
                logger.warning(f"Error during Spark shutdown: {str(e)}")
    
    def load_data(self, input_path):
        """
        Load data into Spark DataFrame
        
        Args:
            input_path (str): Path to input parquet file
            
        Returns:
            DataFrame: Spark DataFrame with loaded data
        """
        if not self.spark:
            raise ValueError("Spark session not initialized. Call initialize_spark() first.")
        
        logger.info(f"Loading data from {input_path}")
        
        try:
            df = self.spark.read.parquet(input_path)
            logger.info(f"Loaded data with {df.count()} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def generate_polynomial_features(self, df, prefix, degree=2, max_features=5000):
        """
        Generate polynomial features for a specific column prefix
        
        Args:
            df (DataFrame): Spark DataFrame with input data
            prefix (str): Column prefix to select features (e.g., 'pvm' for pvm_*)
            degree (int): Polynomial degree (1, 2, or 3)
            max_features (int): Maximum number of features to generate
            
        Returns:
            DataFrame: Spark DataFrame with polynomial features added
        """
        logger.info(f"Generating polynomial features for prefix '{prefix}' with degree {degree}")
        
        # Select base features with the given prefix
        base_features = [col for col in df.columns if col.startswith(f"{prefix}_")]
        logger.info(f"Found {len(base_features)} base features with prefix '{prefix}'")
        
        if not base_features:
            logger.warning(f"No features found with prefix '{prefix}'")
            return df
        
        # Limit base features if too many
        if len(base_features) > 100:
            logger.info(f"Limiting to 100 base features (from {len(base_features)})")
            base_features = base_features[:100]
        
        # Generate polynomial features
        poly_features = []
        feature_names = []
        feature_count = 0
        
        # First add original features
        for i, col1 in enumerate(base_features):
            feature_name = f"poly_{prefix}_{feature_count}"
            poly_features.append((df[col1].alias(feature_name), feature_name))
            feature_names.append(feature_name)
            feature_count += 1
            
            if feature_count >= max_features:
                break
        
        # Add squared terms (degree 2)
        if degree >= 2 and feature_count < max_features:
            for i, col1 in enumerate(base_features):
                feature_name = f"poly_{prefix}_{feature_count}"
                squared_expr = (df[col1] * df[col1]).alias(feature_name)
                poly_features.append((squared_expr, feature_name))
                feature_names.append(feature_name)
                feature_count += 1
                
                if feature_count >= max_features:
                    break
        
        # Add interaction terms (degree 2)
        if degree >= 2 and feature_count < max_features:
            for i, col1 in enumerate(base_features):
                for j in range(i + 1, len(base_features)):
                    col2 = base_features[j]
                    
                    feature_name = f"poly_{prefix}_{feature_count}"
                    interaction_expr = (df[col1] * df[col2]).alias(feature_name)
                    poly_features.append((interaction_expr, feature_name))
                    feature_names.append(feature_name)
                    feature_count += 1
                    
                    if feature_count >= max_features:
                        break
                
                if feature_count >= max_features:
                    break
        
        # Add cubic terms (degree 3)
        if degree >= 3 and feature_count < max_features:
            for i, col1 in enumerate(base_features):
                feature_name = f"poly_{prefix}_{feature_count}"
                cubed_expr = (df[col1] * df[col1] * df[col1]).alias(feature_name)
                poly_features.append((cubed_expr, feature_name))
                feature_names.append(feature_name)
                feature_count += 1
                
                if feature_count >= max_features:
                    break
        
        # Add more complex interaction terms (degree 3)
        if degree >= 3 and feature_count < max_features:
            for i, col1 in enumerate(base_features):
                for j in range(i + 1, len(base_features)):
                    col2 = base_features[j]
                    
                    # x1^2 * x2
                    feature_name = f"poly_{prefix}_{feature_count}"
                    interaction_expr = (df[col1] * df[col1] * df[col2]).alias(feature_name)
                    poly_features.append((interaction_expr, feature_name))
                    feature_names.append(feature_name)
                    feature_count += 1
                    
                    if feature_count >= max_features:
                        break
                    
                    # x1 * x2^2
                    feature_name = f"poly_{prefix}_{feature_count}"
                    interaction_expr = (df[col1] * df[col2] * df[col2]).alias(feature_name)
                    poly_features.append((interaction_expr, feature_name))
                    feature_names.append(feature_name)
                    feature_count += 1
                    
                    if feature_count >= max_features:
                        break
                
                if feature_count >= max_features:
                    break
        
        logger.info(f"Generated {len(feature_names)} polynomial features")
        
        # Add poly features to the original DataFrame
        result_df = df
        for expr, name in poly_features:
            result_df = result_df.withColumn(name, expr)
        
        return result_df, feature_names
    
    def process_crypto_data(self, input_path, output_path=None, degrees=[1, 2], 
                         prefixes=['pvm', 'sentiment', 'onchain'], max_features_per_prefix=5000):
        """
        Process crypto data to generate polynomial features
        
        Args:
            input_path (str): Path to input parquet file
            output_path (str): Path to save output parquet file (default: auto-generated)
            degrees (list): List of polynomial degrees to generate
            prefixes (list): List of column prefixes to process
            max_features_per_prefix (int): Maximum features per prefix
            
        Returns:
            str: Path to the output parquet file
        """
        if not self.spark:
            raise ValueError("Spark session not initialized. Call initialize_spark() first.")
        
        # Load data
        df = self.load_data(input_path)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"poly_features_{timestamp}.parquet"
            output_path = os.path.join(self.output_dir, output_name)
        
        # Process each prefix and degree
        all_feature_names = []
        
        for prefix in prefixes:
            for degree in degrees:
                try:
                    df, feature_names = self.generate_polynomial_features(
                        df, 
                        prefix=prefix,
                        degree=degree,
                        max_features=max_features_per_prefix
                    )
                    all_feature_names.extend(feature_names)
                except Exception as e:
                    logger.error(f"Error generating features for prefix '{prefix}', degree {degree}: {str(e)}")
        
        # Save feature metadata
        metadata = {
            "input_path": input_path,
            "output_path": output_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prefixes": prefixes,
            "degrees": degrees,
            "feature_count": len(all_feature_names),
            "feature_names": all_feature_names
        }
        
        # Save metadata
        metadata_path = output_path.replace(".parquet", "_metadata.json")
        try:
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved feature metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        df.write.parquet(output_path, mode="overwrite")
        
        logger.info(f"Processing complete. Generated {len(all_feature_names)} polynomial features.")
        return output_path
    
    def add_advanced_statistical_features(self, df, columns=None, windows=[5, 10, 20, 50]):
        """
        Add advanced statistical features using window functions
        
        Args:
            df (DataFrame): Spark DataFrame
            columns (list): Columns to process (default: all numeric)
            windows (list): Window sizes for rolling statistics
            
        Returns:
            DataFrame: Spark DataFrame with added features
        """
        logger.info("Generating advanced statistical features")
        
        # Determine if we have date and symbol columns
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        symbol_col = next((col for col in df.columns 
                           if col.lower() in ['symbol', 'asset', 'crypto']), None)
        
        if not date_col:
            logger.warning("No date column found. Statistical features may be less meaningful.")
            return df
        
        # If no columns specified, use numeric columns
        if not columns:
            numeric_cols = []
            for field in df.schema.fields:
                if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType)):
                    numeric_cols.append(field.name)
            
            # Limit to 20 columns for performance
            if len(numeric_cols) > 20:
                logger.info(f"Limiting to 20 numeric columns from {len(numeric_cols)}")
                numeric_cols = numeric_cols[:20]
            
            columns = numeric_cols
        
        logger.info(f"Generating statistical features for {len(columns)} columns")
        
        # Create window specifications
        if symbol_col:
            # If we have symbol, partition by symbol
            windows_spec = {
                window_size: Window.partitionBy(symbol_col)
                                 .orderBy(date_col)
                                 .rowsBetween(-window_size, 0)
                for window_size in windows
            }
        else:
            # Otherwise, just use date
            windows_spec = {
                window_size: Window.orderBy(date_col)
                                 .rowsBetween(-window_size, 0)
                for window_size in windows
            }
        
        # Generate features
        result_df = df
        feature_count = 0
        
        for col in columns:
            for window_size in windows:
                w = windows_spec[window_size]
                
                try:
                    # Add rolling mean
                    result_df = result_df.withColumn(
                        f"stat_{col}_mean_{window_size}",
                        F.avg(col).over(w)
                    )
                    feature_count += 1
                    
                    # Add rolling std
                    result_df = result_df.withColumn(
                        f"stat_{col}_std_{window_size}",
                        F.stddev(col).over(w)
                    )
                    feature_count += 1
                    
                    # Add rolling min/max
                    result_df = result_df.withColumn(
                        f"stat_{col}_min_{window_size}",
                        F.min(col).over(w)
                    )
                    feature_count += 1
                    
                    result_df = result_df.withColumn(
                        f"stat_{col}_max_{window_size}",
                        F.max(col).over(w)
                    )
                    feature_count += 1
                    
                    # Add lag features
                    lag_window = Window.partitionBy(symbol_col).orderBy(date_col) if symbol_col else Window.orderBy(date_col)
                    result_df = result_df.withColumn(
                        f"stat_{col}_lag_{window_size}",
                        F.lag(col, window_size).over(lag_window)
                    )
                    feature_count += 1
                    
                    # Calculate percent change
                    result_df = result_df.withColumn(
                        f"stat_{col}_pct_change_{window_size}",
                        (F.col(col) - F.lag(col, window_size).over(lag_window)) / F.lag(col, window_size).over(lag_window)
                    )
                    feature_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate feature for {col}, window {window_size}: {str(e)}")
        
        logger.info(f"Generated {feature_count} statistical features")
        return result_df


def main():
    """Main function to generate polynomial features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate polynomial features for Numerai Crypto')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input parquet file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output parquet file')
    parser.add_argument('--degree', '-d', type=int, default=2, choices=[1, 2, 3], help='Maximum polynomial degree')
    parser.add_argument('--prefixes', '-p', type=str, default='pvm,sentiment,onchain', 
                        help='Comma-separated list of column prefixes')
    parser.add_argument('--max-features', '-m', type=int, default=5000, 
                        help='Maximum features per prefix')
    parser.add_argument('--memory', type=str, default='4g', help='Memory for Spark')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration if available')
    
    args = parser.parse_args()
    
    # Parse prefixes
    prefixes = args.prefixes.split(',')
    
    # Create generator
    generator = PolynomialFeatureGenerator()
    
    try:
        # Initialize Spark
        generator.initialize_spark(memory=args.memory, enable_gpu=args.gpu)
        
        # Process data
        output_path = generator.process_crypto_data(
            input_path=args.input,
            output_path=args.output,
            degrees=[args.degree],
            prefixes=prefixes,
            max_features_per_prefix=args.max_features
        )
        
        logger.info(f"Processing complete. Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during feature generation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown
        generator.shutdown()


if __name__ == "__main__":
    main()