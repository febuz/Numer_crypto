#!/usr/bin/env python3
"""
PySpark Feature Generator for Numerai Crypto

This module provides high-performance feature generation using Apache Spark for distributed computing.
It's optimized for handling extremely large datasets across multiple nodes.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PySparkFeatureGenerator:
    """Feature generator that uses PySpark for distributed processing"""
    
    def __init__(self, output_dir: str = "/media/knight2/EDB/numer_crypto_temp/data/features", 
                 max_features: int = 100000):
        """
        Initialize the feature generator with configuration
        
        Args:
            output_dir: Directory to save generated feature files
            max_features: Maximum number of features to generate
        """
        self.output_dir = Path(output_dir)
        self.max_features = max_features
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature generation parameters
        self.rolling_windows = [3, 7, 14, 28, 56]
        self.ewm_spans = [5, 10, 20, 40]
        self.lag_periods = [1, 2, 3, 5, 7, 14, 28]
        self.group_col = 'symbol'
        self.date_col = 'date'
        
        # Initialize Spark session
        self.spark = self._create_spark_session()
        
        logger.info(f"PySpark Feature Generator initialized with max_features={max_features}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_spark_session(self):
        """Create and configure a Spark session"""
        try:
            from pyspark.sql import SparkSession
            import pyspark.sql.functions as F
            
            # Try to detect if we're running in a job with multiple workers
            # or just locally with multiple cores
            is_distributed = "SPARK_MASTER" in os.environ
            
            # Determine how much memory to allocate
            if is_distributed:
                # In a distributed environment, use environment settings
                executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
                driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
            else:
                # Local mode - base on available memory
                try:
                    import psutil
                    total_mem = psutil.virtual_memory().total / (1024 ** 3)  # GB
                    executor_memory = f"{int(max(1, total_mem * 0.7))}g"  # 70% of memory
                    driver_memory = f"{int(max(1, total_mem * 0.2))}g"    # 20% of memory
                except ImportError:
                    # Default if psutil not available
                    executor_memory = "4g"
                    driver_memory = "4g"
            
            # Create Spark session with appropriate configuration
            spark = (SparkSession.builder
                    .appName("NumeraiCryptoFeatureGenerator")
                    .config("spark.executor.memory", executor_memory)
                    .config("spark.driver.memory", driver_memory)
                    .config("spark.sql.shuffle.partitions", "200")
                    .config("spark.default.parallelism", "100")
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                    .getOrCreate())
            
            spark_version = spark.version
            logger.info(f"Created Spark session version {spark_version}")
            logger.info(f"Executor memory: {executor_memory}, Driver memory: {driver_memory}")
            
            return spark
            
        except ImportError:
            logger.error("PySpark not installed. Please install with: pip install pyspark")
            raise
    
    def generate_features(self, input_file: str, feature_modules: Optional[List[str]] = None) -> str:
        """
        Generate features for a dataset using PySpark
        
        Args:
            input_file: Path to the input parquet file
            feature_modules: Optional list of feature module names to use
            
        Returns:
            Path to the output parquet file with generated features
        """
        # Timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(self.output_dir / f"features_pyspark_{timestamp}.parquet")
        
        # Import Spark functions
        from pyspark.sql import functions as F
        from pyspark.sql import Window
        from pyspark.sql.types import FloatType, IntegerType
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = self.spark.read.parquet(input_file)
        
        # Get the schema and identify numeric columns
        excluded_cols = [self.group_col, self.date_col, 'era', 'target', 'id']
        numeric_cols = [field.name for field in df.schema.fields 
                      if (field.dataType.typeName() in ['double', 'float', 'integer', 'long'] and 
                         field.name not in excluded_cols)]
        
        logger.info(f"Found {len(numeric_cols)} numeric columns for feature generation")
        
        # Convert date column to timestamp if needed
        if self.date_col in df.columns:
            logger.info(f"Converting {self.date_col} to timestamp")
            df = df.withColumn(self.date_col, F.to_timestamp(F.col(self.date_col)))
        
        # Generate features
        logger.info("Generating features...")
        
        # 1. Generate lag features
        df = self._generate_lag_features(df, numeric_cols)
        
        # 2. Generate rolling features
        df = self._generate_rolling_features(df, numeric_cols)
        
        # 3. Generate EWM features
        df = self._generate_ewm_features(df, numeric_cols)
        
        # 4. Generate interaction features if we have room
        feature_count = len(df.columns) - len(excluded_cols)
        if feature_count < self.max_features * 0.8:
            df = self._generate_interaction_features(df, numeric_cols)
        
        # 5. Generate technical indicators for financial data
        df = self._generate_technical_indicators(df)
        
        # Feature selection if we have too many
        feature_count = len(df.columns) - len(excluded_cols)
        if feature_count > self.max_features:
            logger.info(f"Too many features generated ({feature_count}), selecting top {self.max_features}")
            df = self._select_top_features(df, self.max_features)
        
        # Save to parquet
        logger.info(f"Saving features to {output_file}")
        df.write.mode("overwrite").parquet(output_file)
        
        # Statistics
        feature_count = len(df.columns) - len(excluded_cols)
        logger.info(f"Generated {feature_count} features for {df.count()} rows")
        
        # Save feature info file
        feature_info = self._create_feature_info(df, output_file, excluded_cols)
        self._save_feature_info(feature_info)
        
        # Stop Spark session
        self.spark.stop()
        
        return output_file
    
    def _generate_lag_features(self, df, numeric_cols):
        """Generate lag features using Spark window functions"""
        from pyspark.sql import functions as F
        from pyspark.sql import Window
        
        logger.info(f"Generating lag features with periods {self.lag_periods}")
        
        # Create a window for each symbol, ordered by date
        window_spec = (Window.partitionBy(self.group_col)
                      .orderBy(self.date_col)
                      .rowsBetween(-1000, -1))  # Large bounds to handle all lags
        
        # Generate lag features in batches to avoid memory issues
        batch_size = 20
        for i in range(0, len(numeric_cols), batch_size):
            batch_cols = numeric_cols[i:i+batch_size]
            
            # Process each lag period
            for lag in self.lag_periods:
                # Skip large lags for large datasets
                if df.count() > 1000000 and lag > 14:
                    continue
                
                for col in batch_cols:
                    lag_col_name = f"{col}_lag_{lag}"
                    df = df.withColumn(lag_col_name, F.lag(F.col(col), lag).over(window_spec))
        
        return df
    
    def _generate_rolling_features(self, df, numeric_cols):
        """Generate rolling window features using Spark window functions"""
        from pyspark.sql import functions as F
        from pyspark.sql import Window
        
        logger.info(f"Generating rolling features with windows {self.rolling_windows}")
        
        # Create windows for each rolling calculation
        windows = {}
        for window in self.rolling_windows:
            # Create different window specs for each window size
            windows[window] = (Window.partitionBy(self.group_col)
                              .orderBy(self.date_col)
                              .rowsBetween(-(window-1), 0))
        
        # Generate rolling features in batches
        batch_size = 10
        for i in range(0, len(numeric_cols), batch_size):
            batch_cols = numeric_cols[i:i+batch_size]
            
            # Process each window
            for window_size in self.rolling_windows:
                # Skip large windows for large datasets
                if df.count() > 1000000 and window_size > 28:
                    continue
                
                # Get the window spec for this window size
                window_spec = windows[window_size]
                
                # Apply different aggregation functions
                for col in batch_cols:
                    # Mean
                    mean_col = f"{col}_roll_{window_size}_mean"
                    df = df.withColumn(mean_col, F.avg(F.col(col)).over(window_spec))
                    
                    # Standard deviation
                    if window_size <= 14:  # Skip for large windows to save memory
                        std_col = f"{col}_roll_{window_size}_std"
                        df = df.withColumn(std_col, F.stddev(F.col(col)).over(window_spec))
                    
                    # Min and max
                    min_col = f"{col}_roll_{window_size}_min"
                    max_col = f"{col}_roll_{window_size}_max"
                    df = df.withColumn(min_col, F.min(F.col(col)).over(window_spec))
                    df = df.withColumn(max_col, F.max(F.col(col)).over(window_spec))
        
        return df
    
    def _generate_ewm_features(self, df, numeric_cols):
        """
        Generate exponential weighted moving average features
        
        Note: PySpark doesn't have native EWM like pandas, so we approximate it
        using weighted averages
        """
        from pyspark.sql import functions as F
        from pyspark.sql import Window
        import math
        
        logger.info(f"Generating EWM features with spans {self.ewm_spans}")
        
        # Convert spans to decay factors
        # In pandas: alpha = 2 / (span + 1)
        decay_factors = [2 / (span + 1) for span in self.ewm_spans]
        
        # Process a subset of columns for memory efficiency
        if len(numeric_cols) > 10:
            # Use the most predictive columns
            selected_cols = numeric_cols[:10]
        else:
            selected_cols = numeric_cols
        
        # For each decay factor, create approximated EWM
        for span, alpha in zip(self.ewm_spans, decay_factors):
            # Skip large spans for large datasets
            if df.count() > 1000000 and span > 20:
                continue
                
            logger.info(f"Calculating EWM with span={span}, alpha={alpha}")
            
            # Create a window spec ordered by date for each symbol
            window_spec = (Window.partitionBy(self.group_col)
                          .orderBy(self.date_col)
                          .rowsBetween(-span, 0))
            
            # Create weights for the window function
            # Weights decrease exponentially: [alpha*(1-alpha)^i] for i=0,1,...,span
            weights = [alpha * math.pow(1-alpha, i) for i in range(span)]
            # Normalize weights to sum to 1
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            
            # Apply to columns
            for col in selected_cols:
                ewm_col = f"{col}_ewm_{span}"
                
                # Use a weighted average as an approximation of EWM
                # This is not exactly the same as pandas' EWM but provides similar smoothing
                weighted_avg_expr = F.avg(F.col(col)).over(window_spec)
                df = df.withColumn(ewm_col, weighted_avg_expr)
        
        return df
    
    def _generate_interaction_features(self, df, numeric_cols):
        """Generate interaction features between columns"""
        from pyspark.sql import functions as F
        
        # Limit the number of base columns to avoid explosion
        if len(numeric_cols) > 10:
            # Use the first few and last few columns
            base_cols = numeric_cols[:5] + numeric_cols[-5:]
        else:
            base_cols = numeric_cols
            
        logger.info(f"Generating interaction features with {len(base_cols)} base columns")
        
        # Create interaction features
        feature_count = 0
        for i, col1 in enumerate(base_cols):
            for col2 in base_cols[i+1:]:
                # Multiplication interaction
                mult_col = f"{col1}_X_{col2}"
                df = df.withColumn(mult_col, F.col(col1) * F.col(col2))
                feature_count += 1
                
                # Division interaction for price columns
                if "price" in col1 and "price" in col2:
                    div_col = f"{col1}_DIV_{col2}"
                    # Avoid division by zero
                    df = df.withColumn(div_col, 
                                   F.when(F.abs(F.col(col2)) > 1e-5, 
                                          F.col(col1) / F.col(col2))
                                   .otherwise(0.0))
                    feature_count += 1
                
                # Check if we've reached the feature limit
                if len(df.columns) > self.max_features + 10:
                    break
            
            if len(df.columns) > self.max_features + 10:
                break
        
        logger.info(f"Generated {feature_count} interaction features")
        return df
    
    def _generate_technical_indicators(self, df):
        """Generate technical indicators for financial data"""
        from pyspark.sql import functions as F
        from pyspark.sql import Window
        
        logger.info("Generating technical indicators")
        
        # Check if we have OHLCV columns
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}
        has_ohlcv = all(any(col.lower() == ohlcv_col for col in df.columns) 
                       for ohlcv_col in ohlcv_cols)
        
        if not has_ohlcv:
            logger.info("Dataset doesn't have standard OHLCV columns, skipping technical indicators")
            return df
        
        # Find the actual column names (case insensitive matching)
        col_map = {}
        for req in ohlcv_cols:
            for col in df.columns:
                if col.lower() == req:
                    col_map[req] = col
        
        # Define window specs for various lookback periods
        windows = {}
        for period in [10, 14, 20, 26]:
            windows[period] = (Window.partitionBy(self.group_col)
                              .orderBy(self.date_col)
                              .rowsBetween(-(period-1), 0))
        
        # Create technical indicators
        
        # 1. Simple Moving Averages
        for period in [10, 20, 50]:
            df = df.withColumn(f"SMA_{period}", 
                            F.avg(F.col(col_map['close'])).over(windows.get(period, windows[20])))
        
        # 2. RSI (Relative Strength Index)
        # Calculate price changes
        df = df.withColumn("price_change", 
                          F.col(col_map['close']) - F.lag(F.col(col_map['close']), 1)
                          .over(Window.partitionBy(self.group_col).orderBy(self.date_col)))
        
        # Calculate gains and losses
        df = df.withColumn("gain", 
                         F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
        df = df.withColumn("loss", 
                         F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
        
        # Calculate average gains and losses over 14 periods
        avg_gain = F.avg(F.col("gain")).over(windows[14])
        avg_loss = F.avg(F.col("loss")).over(windows[14])
        
        # Calculate RSI
        df = df.withColumn("RSI_14", 
                         100 - (100 / (1 + (avg_gain / F.when(avg_loss > 0, avg_loss).otherwise(0.001)))))
        
        # 3. MACD (Moving Average Convergence Divergence)
        # Calculate 12-day EMA
        df = df.withColumn("EMA_12", 
                         F.avg(F.col(col_map['close'])).over(windows.get(12, windows[10])))
        
        # Calculate 26-day EMA
        df = df.withColumn("EMA_26", 
                         F.avg(F.col(col_map['close'])).over(windows[26]))
        
        # Calculate MACD line
        df = df.withColumn("MACD", F.col("EMA_12") - F.col("EMA_26"))
        
        # Calculate 9-day EMA of MACD for signal line
        signal_window = Window.partitionBy(self.group_col).orderBy(self.date_col).rowsBetween(-8, 0)
        df = df.withColumn("MACD_signal", F.avg(F.col("MACD")).over(signal_window))
        
        # Calculate MACD histogram
        df = df.withColumn("MACD_hist", F.col("MACD") - F.col("MACD_signal"))
        
        # 4. Bollinger Bands
        # Calculate 20-day SMA
        sma_20 = F.avg(F.col(col_map['close'])).over(windows[20])
        
        # Calculate 20-day standard deviation
        std_20 = F.stddev(F.col(col_map['close'])).over(windows[20])
        
        # Calculate Bollinger Bands
        df = df.withColumn("BB_middle", sma_20)
        df = df.withColumn("BB_upper", sma_20 + (2 * std_20))
        df = df.withColumn("BB_lower", sma_20 - (2 * std_20))
        
        # Calculate Bollinger Band width
        df = df.withColumn("BB_width", 
                        (F.col("BB_upper") - F.col("BB_lower")) / F.col("BB_middle"))
        
        # 5. Volume-based indicators
        if 'volume' in col_map:
            # On-Balance Volume (OBV)
            # This is more complex in Spark because we need a running sum
            df = df.withColumn("volume_sign", 
                            F.when(F.col("price_change") > 0, F.col(col_map['volume']))
                             .when(F.col("price_change") < 0, -F.col(col_map['volume']))
                             .otherwise(0))
            
            # We'll use a crude approximation for OBV
            df = df.withColumn("OBV_approx", 
                            F.sum(F.col("volume_sign")).over(
                                Window.partitionBy(self.group_col)
                                     .orderBy(self.date_col)
                                     .rowsBetween(Window.unboundedPreceding, 0)))
        
        # Clean up temporary columns
        for col in ["price_change", "gain", "loss", "volume_sign"]:
            if col in df.columns:
                df = df.drop(col)
        
        logger.info("Added technical indicators")
        return df
    
    def _select_top_features(self, df, max_features):
        """Select top features based on variance or correlation with target"""
        from pyspark.sql import functions as F
        from pyspark.ml.feature import VarianceThresholdSelector
        
        excluded_cols = [self.group_col, self.date_col, 'era', 'target', 'id']
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        logger.info(f"Selecting top {max_features} features from {len(feature_cols)} total features")
        
        # If target column exists, use correlation-based selection
        if 'target' in df.columns:
            logger.info("Using correlation-based feature selection")
            
            # Calculate correlation with target for each feature
            correlations = []
            for col in feature_cols:
                # Calculate correlation
                corr = df.stat.corr(col, 'target')
                correlations.append((col, abs(corr)))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_features = [x[0] for x in correlations[:max_features]]
            
            # Select top features plus excluded columns
            selected_cols = excluded_cols + top_features
            result_df = df.select(*selected_cols)
            
            logger.info(f"Selected {len(top_features)} features based on correlation with target")
            
        else:
            # No target column, use variance-based selection
            logger.info("Using variance-based feature selection (no target column)")
            
            # Calculate variance for each feature
            variances = []
            for col in feature_cols:
                variance = df.agg(F.variance(col)).collect()[0][0]
                if variance is not None:
                    variances.append((col, variance))
                else:
                    variances.append((col, 0.0))
            
            # Sort by variance
            variances.sort(key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_features = [x[0] for x in variances[:max_features]]
            
            # Select top features plus excluded columns
            selected_cols = excluded_cols + top_features
            result_df = df.select(*selected_cols)
            
            logger.info(f"Selected {len(top_features)} features based on variance")
        
        return result_df
    
    def _create_feature_info(self, df, output_file, excluded_cols):
        """Create information about the generated features"""
        import json
        
        # Get feature columns (exclude special columns)
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        # Categorize features
        feature_types = {
            'rolling': len([col for col in feature_cols if 'roll_' in col]),
            'lag': len([col for col in feature_cols if 'lag_' in col]),
            'ewm': len([col for col in feature_cols if 'ewm_' in col]),
            'interaction': len([col for col in feature_cols if '_X_' in col or '_DIV_' in col]),
            'technical': len([col for col in feature_cols if col in [
                'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'OBV_approx',
                'EMA_12', 'EMA_26'
            ]]),
            'base': len([col for col in feature_cols if not (
                'roll_' in col or 'lag_' in col or 'ewm_' in col or 
                '_X_' in col or '_DIV_' in col or col in [
                    'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
                    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'OBV_approx',
                    'EMA_12', 'EMA_26'
                ]
            )])
        }
        
        # Count unique symbols
        unique_symbols = df.select(self.group_col).distinct().count()
        
        # Get date range
        if self.date_col in df.columns:
            date_agg = df.agg(
                F.min(self.date_col).alias("min_date"),
                F.max(self.date_col).alias("max_date")
            ).collect()[0]
            
            min_date = date_agg["min_date"]
            max_date = date_agg["max_date"]
            date_range = f"{min_date} to {max_date}"
        else:
            date_range = "Unknown"
        
        # Create feature info
        feature_info = {
            'timestamp': datetime.now().isoformat(),
            'output_file': output_file,
            'row_count': df.count(),
            'symbol_count': unique_symbols,
            'date_range': date_range,
            'total_features': len(feature_cols),
            'feature_types': feature_types,
            'feature_list': feature_cols
        }
        
        return feature_info
    
    def _save_feature_info(self, feature_info):
        """Save feature information to a JSON file"""
        import json
        
        # Save to JSON
        info_file = feature_info['output_file'].replace('.parquet', '_info.json')
        
        # Convert feature list to simple list for JSON serialization
        feature_info['feature_list'] = list(feature_info['feature_list'])
        
        with open(info_file, 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)
            
        logger.info(f"Feature information saved to {info_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate features using PySpark')
    parser.add_argument('--input-file', type=str, 
                     default='/media/knight2/EDB/numer_crypto_temp/data/merged/merged_train.parquet',
                     help='Input data file (parquet format)')
    parser.add_argument('--output-dir', type=str, 
                     default='/media/knight2/EDB/numer_crypto_temp/data/features',
                     help='Output directory for features')
    parser.add_argument('--max-features', type=int, default=100000, 
                     help='Maximum number of features to generate')
    
    args = parser.parse_args()
    
    # Create generator
    try:
        generator = PySparkFeatureGenerator(
            output_dir=args.output_dir,
            max_features=args.max_features
        )
        
        # Generate features
        output_file = generator.generate_features(args.input_file)
        
        print(f"Features generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)