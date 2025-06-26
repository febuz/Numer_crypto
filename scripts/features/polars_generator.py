#!/usr/bin/env python3
"""
Polars Feature Generator for Numerai Crypto

This module provides high-performance feature generation using the Polars DataFrame library.
It's optimized for memory efficiency and speed when working with large datasets.
"""

import os
import sys
import logging
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import relevant utilities
from utils.memory_utils import log_memory_usage, clear_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PolarsFeatureGenerator:
    """Feature generator that uses Polars for high-performance processing"""
    
    def __init__(self, output_dir: str = "/media/knight2/EDB/numer_crypto_temp/data/features", 
                 max_features: int = 100000,
                 enable_gpu: bool = True,
                 enable_mixed_precision: bool = True,
                 feature_reducer=None):
        """
        Initialize the feature generator with configuration
        
        Args:
            output_dir: Directory to save generated feature files
            max_features: Maximum number of features to generate
            enable_gpu: Whether to use GPU acceleration
            enable_mixed_precision: Whether to use mixed precision for GPU
            feature_reducer: Optional FeatureReducer instance for progressive reduction
        """
        self.output_dir = Path(output_dir)
        self.max_features = max_features
        self.enable_gpu = enable_gpu
        self.enable_mixed_precision = enable_mixed_precision
        self.feature_reducer = feature_reducer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature generation parameters
        self.rolling_windows = [3, 7, 14, 28, 56]
        self.ewm_spans = [5, 10, 20, 40]
        self.lag_periods = [1, 2, 3, 5, 7, 14, 28]
        self.group_col = 'symbol'
        self.date_col = 'date'
        
        # Initialize CPU-only feature generation to avoid object array errors
        self.feature_accelerator = None
        try:
            from features.polars_simple_accelerator import PolarsSimpleAccelerator
            self.feature_accelerator = PolarsSimpleAccelerator()
            logger.info("Simple Polars accelerator enabled (CPU-only, no object array errors)")
        except Exception as e:
            logger.warning(f"Simple Polars accelerator failed to initialize: {e}")
            self.feature_accelerator = None
        
        logger.info(f"Polars Feature Generator initialized with max_features={max_features}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Feature accelerator: {'enabled' if self.feature_accelerator else 'disabled'}")
        logger.info(f"Feature reducer: {'enabled' if self.feature_reducer else 'disabled'}")
    
    def _get_numeric_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of numeric columns from a Polars DataFrame, excluding target and metadata"""
        excluded_cols = {'target', 'era', 'data_type', self.group_col, self.date_col, 'id'}
        numeric_cols = []
        
        for col in df.columns:
            if col not in excluded_cols and df[col].dtype.is_numeric():
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _apply_stage_reduction(self, df: pl.DataFrame, stage_name: str) -> Tuple[pl.DataFrame, List[str]]:
        """
        Apply feature reduction after completing a feature generation stage
        
        Args:
            df: Current dataframe with features
            stage_name: Name of the completed stage
            
        Returns:
            Tuple of (reduced dataframe, updated numeric columns list)
        """
        if not self.feature_reducer:
            return df, self._get_numeric_columns(df)
            
        # Mark stage as completed
        self.feature_reducer.mark_stage_completed(stage_name)
        
        # Check if we should reduce after this stage
        if self.feature_reducer.should_reduce_after_stage(stage_name):
            logger.info(f"Applying feature reduction after {stage_name} stage")
            
            try:
                reduction_reason = self.feature_reducer.get_stage_reduction_reason(stage_name)
                reduced_df = self.feature_reducer.reduce_features(
                    df,
                    target_col='target',
                    excluded_cols=[self.group_col, self.date_col, 'era', 'target', 'id'],
                    reduction_reason=reduction_reason
                )
                
                # Check if reduction was successful
                original_count = df.width
                reduced_count = reduced_df.width if hasattr(reduced_df, 'width') else reduced_df.shape[1]
                
                if reduced_count < original_count:
                    logger.info(f"Stage reduction successful: {original_count} -> {reduced_count} features")
                    result_df = pl.from_pandas(reduced_df) if isinstance(reduced_df, pd.DataFrame) else reduced_df
                    updated_numeric_cols = self._get_numeric_columns(result_df)
                    return result_df, updated_numeric_cols
                else:
                    logger.info(f"No reduction applied after {stage_name} stage")
                    return df, self._get_numeric_columns(df)
                    
            except Exception as e:
                logger.error(f"Error during stage reduction for {stage_name}: {e}")
                return df, self._get_numeric_columns(df)
        else:
            return df, self._get_numeric_columns(df)
    
    def _apply_batch_reduction(self, df: pl.DataFrame, stage_context: str = "") -> pl.DataFrame:
        """
        Apply batch-based feature reduction if interval is reached
        
        Args:
            df: Current dataframe with features
            stage_context: Context of current processing stage
            
        Returns:
            Reduced dataframe (or original if no reduction applied)
        """
        if not self.feature_reducer:
            return df
            
        # Increment batch count
        self.feature_reducer.increment_batch_count()
        
        # Check if we should reduce based on batch count
        if self.feature_reducer.should_reduce_features():
            logger.info(f"Applying batch-based feature reduction{' during ' + stage_context if stage_context else ''}")
            
            try:
                reduction_reason = f"Batch interval reached{' during ' + stage_context if stage_context else ''}"
                reduced_df = self.feature_reducer.reduce_features(
                    df,
                    target_col='target',
                    excluded_cols=[self.group_col, self.date_col, 'era', 'target', 'id'],
                    reduction_reason=reduction_reason
                )
                
                # Check if reduction was successful
                original_count = df.width
                reduced_count = reduced_df.width if hasattr(reduced_df, 'width') else reduced_df.shape[1]
                
                if reduced_count < original_count:
                    logger.info(f"Batch reduction successful: {original_count} -> {reduced_count} features")
                    return pl.from_pandas(reduced_df) if isinstance(reduced_df, pd.DataFrame) else reduced_df
                else:
                    logger.info("No batch reduction applied")
                    return df
                    
            except Exception as e:
                logger.error(f"Error during batch reduction: {e}")
                return df
        else:
            return df
        
    def generate_features(self, input_file: str, feature_modules: Optional[List[str]] = None) -> str:
        """
        Generate features for a dataset using Polars
        
        Args:
            input_file: Path to the input parquet file
            feature_modules: Optional list of feature module names to use
            
        Returns:
            Path to the output parquet file with generated features
        """
        # Memory monitoring
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        if available_gb < 2.0:
            logger.warning("Low memory detected - running memory optimized mode")
        
        # Timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(self.output_dir / f"features_polars_{timestamp}.parquet")
        
        # Log before loading data
        log_memory_usage("Before loading data:")
        
        try:
            # Load data
            logger.info(f"Loading data from {input_file}")
            df = self._load_data(input_file)
            
            # Log after loading data
            log_memory_usage("After loading data:")
            
            # Get numeric columns for feature generation (exclude special columns)
            excluded_cols = [self.group_col, self.date_col, 'era', 'target', 'id']
            numeric_cols = [col for col in df.columns 
                        if col not in excluded_cols and 
                        df.get_column(col).dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            
            # Cast numeric columns to float32 for memory efficiency
            logger.info("Converting numeric columns to float32 for memory efficiency")
            cast_exprs = []
            for col in numeric_cols:
                dtype = df.get_column(col).dtype
                if dtype in [pl.Float64]:
                    cast_exprs.append(pl.col(col).cast(pl.Float32).alias(col))
            
            if cast_exprs:
                df = df.with_columns(cast_exprs)
                logger.info(f"Cast {len(cast_exprs)} columns from float64 to float32")
            
            logger.info(f"Found {len(numeric_cols)} numeric columns for feature generation")
            
            # Always convert date column to datetime
            if self.date_col in df.columns:
                logger.info(f"Converting {self.date_col} to datetime")
                # Handle different versions of polars API
                try:
                    # Newer polars versions with strict=False to handle mixed types
                    df = df.with_column(pl.col(self.date_col).cast(pl.Datetime, strict=False))
                except (AttributeError, TypeError) as e:
                    logger.info(f"First attempt at date conversion failed: {e}, trying alternative methods")
                    try:
                        # Try with newer polars, specifying with_columns
                        df = df.with_columns(pl.col(self.date_col).cast(pl.Datetime, strict=False))
                    except Exception as e2:
                        logger.info(f"Second attempt failed: {e2}, trying older polars syntax")
                        # Older polars versions
                        try:
                            df = df.with_columns([(self.date_col, pl.col(self.date_col).cast(pl.Datetime, strict=False))])
                        except Exception as e3:
                            logger.warning(f"All date conversion attempts failed: {e3}")
                            # Last resort: leave as string
                            logger.info("Leaving date column as string type")
            
            # Check dataframe size and dimensions
            logger.info(f"DataFrame dimensions before feature generation: {df.height} rows × {df.width} columns")
            
            # Memory estimates
            row_count = df.height
            mem = self._get_memory_info()
            available_gb = mem.get('available_gb', 4.0)
            total_gb = mem.get('total_gb', 8.0)
            
            logger.info(f"Available memory: {available_gb:.2f} GB out of {total_gb:.2f} GB total")
            
            # Feature acceleration strategy
            estimated_size_gb = (df.height * df.width * 8) / (1024**3)
            
            # Try feature accelerator if available
            if self.feature_accelerator:
                logger.info(f"Attempting accelerated feature generation (data size: {estimated_size_gb:.2f} GB)")
                try:
                    # Create progress tracking
                    unique_symbols = df.select(pl.col(self.group_col)).n_unique()
                    logger.info(f"Processing {unique_symbols} symbols with simple Polars accelerator")
                    
                    # Use simple Polars accelerator for feature generation
                    result_df = self.feature_accelerator.generate_all_features(
                        df=df,
                        group_col=self.group_col,
                        numeric_cols=numeric_cols,
                        rolling_windows=self.rolling_windows,
                        lag_periods=self.lag_periods,
                        ewm_spans=self.ewm_spans,
                        date_col=self.date_col
                    )
                    
                    # Check if feature generation was successful
                    if result_df is not None and result_df.width > df.width:
                        feature_count = result_df.width - df.width
                        logger.info(f"Feature acceleration successful: generated {feature_count} features")
                        df = result_df
                        
                        # Skip CPU processing
                        use_chunks = False
                    else:
                        logger.warning("Feature accelerator returned no features, falling back to CPU")
                        use_chunks = row_count > 100000 or len(numeric_cols) > 100 or available_gb < 4.0
                        
                except Exception as e:
                    logger.error(f"Feature acceleration failed: {e}")
                    logger.info("Falling back to CPU processing")
                    use_chunks = row_count > 100000 or len(numeric_cols) > 100 or available_gb < 4.0
            else:
                # Standard chunked processing decision
                use_chunks = row_count > 100000 or len(numeric_cols) > 100 or available_gb < 4.0
            
            if use_chunks:
                logger.info("Using chunked processing mode for memory efficiency")
                
                # We'll build up the features in stages and save intermediate results
                
                # Generate features - Step 1: Rolling features
                logger.info("Step 1: Generating rolling features")
                df = self._generate_rolling_features_chunked(df, numeric_cols)
                df, numeric_cols = self._apply_stage_reduction(df, 'rolling')
                df = self._apply_batch_reduction(df, 'rolling stage')
                
                # Clear memory after rolling features
                log_memory_usage("After rolling features:")
                clear_memory()
                
                # Save intermediate result
                tmp_file_1 = str(self.output_dir / f"tmp_rolling_{timestamp}.parquet")
                logger.info(f"Saving intermediate result after rolling features to {tmp_file_1}")
                df.write_parquet(tmp_file_1)
                
                # Reload to ensure clean memory state
                logger.info("Reloading from intermediate file to ensure clean memory state")
                df = pl.read_parquet(tmp_file_1)
                clear_memory()
                
                # Step 2: Lag features
                logger.info("Step 2: Generating lag features")
                df = self._generate_lag_features(df, numeric_cols)
                df, numeric_cols = self._apply_stage_reduction(df, 'lag')
                df = self._apply_batch_reduction(df, 'lag stage')
                
                # Clear memory after lag features
                log_memory_usage("After lag features:")
                clear_memory()
                
                # Save intermediate result
                tmp_file_2 = str(self.output_dir / f"tmp_lag_{timestamp}.parquet")
                logger.info(f"Saving intermediate result after lag features to {tmp_file_2}")
                df.write_parquet(tmp_file_2)
                
                # Reload to ensure clean memory state
                logger.info("Reloading from intermediate file to ensure clean memory state")
                df = pl.read_parquet(tmp_file_2)
                clear_memory()
                
                # Step 3: EWM features (consider skipping if memory is tight)
                if available_gb > 3.0:
                    logger.info("Step 3: Generating EWM features")
                    df = self._generate_ewm_features(df, numeric_cols[:min(20, len(numeric_cols))])  # Limit column count
                    df, numeric_cols = self._apply_stage_reduction(df, 'ewm')
                    df = self._apply_batch_reduction(df, 'ewm stage')
                    
                    # Clear memory after EWM features
                    log_memory_usage("After EWM features:")
                    clear_memory()
                    
                    # Save intermediate result
                    tmp_file_3 = str(self.output_dir / f"tmp_ewm_{timestamp}.parquet")
                    logger.info(f"Saving intermediate result after EWM features to {tmp_file_3}")
                    df.write_parquet(tmp_file_3)
                    
                    # Reload to ensure clean memory state
                    logger.info("Reloading from intermediate file to ensure clean memory state")
                    df = pl.read_parquet(tmp_file_3)
                    clear_memory()
                else:
                    logger.warning("Skipping EWM features due to memory constraints")
                
                # Step 4: Generate interaction features if we have room
                current_feature_count = len(df.columns) - len(excluded_cols)
                if current_feature_count < self.max_features * 0.8 and available_gb > 2.0:
                    logger.info("Step 4: Generating interaction features")
                    df = self._generate_interaction_features(df, numeric_cols[:min(10, len(numeric_cols))])  # Limit column count
                    df, numeric_cols = self._apply_stage_reduction(df, 'interaction')
                    df = self._apply_batch_reduction(df, 'interaction stage')
                    
                    # Clear memory after interaction features
                    log_memory_usage("After interaction features:")
                    clear_memory()
                else:
                    logger.warning("Skipping interaction features due to memory constraints or feature count")
                
                # Step 5: Generate technical indicators for financial data
                logger.info("Step 5: Generating technical indicators")
                df = self._generate_technical_indicators(df)
                df, numeric_cols = self._apply_stage_reduction(df, 'technical')
                df = self._apply_batch_reduction(df, 'technical stage')
                
                # Clear memory after technical indicators
                log_memory_usage("After technical indicators:")
                clear_memory()
                
            else:
                # Standard non-chunked processing for smaller datasets
                logger.info("Using standard processing mode")
                
                # Generate features
                logger.info("Generating features...")
                
                # 1. Generate rolling features
                df = self._generate_rolling_features(df, numeric_cols)
                df, numeric_cols = self._apply_stage_reduction(df, 'rolling')
                df = self._apply_batch_reduction(df, 'rolling stage')
                
                # 2. Generate lag features
                df = self._generate_lag_features(df, numeric_cols)
                df, numeric_cols = self._apply_stage_reduction(df, 'lag')
                df = self._apply_batch_reduction(df, 'lag stage')
                
                # 3. Generate EWM features
                df = self._generate_ewm_features(df, numeric_cols)
                df, numeric_cols = self._apply_stage_reduction(df, 'ewm')
                df = self._apply_batch_reduction(df, 'ewm stage')
                
                # 4. Generate interaction features if we have room
                current_feature_count = len(df.columns) - len(excluded_cols)
                if current_feature_count < self.max_features * 0.8:
                    df = self._generate_interaction_features(df, numeric_cols)
                    df, numeric_cols = self._apply_stage_reduction(df, 'interaction')
                    df = self._apply_batch_reduction(df, 'interaction stage')
                
                # 5. Generate technical indicators for financial data
                df = self._generate_technical_indicators(df)
                df, numeric_cols = self._apply_stage_reduction(df, 'technical')
                df = self._apply_batch_reduction(df, 'technical stage')
            
            # Feature selection if we have too many
            if len(df.columns) > self.max_features + len(excluded_cols):
                logger.info(f"Too many features generated ({len(df.columns) - len(excluded_cols)}), selecting top {self.max_features}")
                df = self._select_top_features(df, self.max_features)
            
            # Save to parquet
            logger.info(f"Saving features to {output_file}")
            df.write_parquet(output_file)
            
            # Statistics
            feature_count = len(df.columns) - len(excluded_cols)
            logger.info(f"Generated {feature_count} features for {df.height} rows")
            
            # Clean up temp files if they exist
            for tmp_file in [f"tmp_rolling_{timestamp}.parquet", f"tmp_lag_{timestamp}.parquet", f"tmp_ewm_{timestamp}.parquet"]:
                full_path = str(self.output_dir / tmp_file)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        logger.info(f"Removed temporary file: {full_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {full_path}: {e}")
            
            # Save feature info file
            self._save_feature_info(df, output_file, excluded_cols)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error in feature generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self, input_file: str) -> pl.DataFrame:
        """Load data from parquet file with Polars"""
        try:
            # Try to load with polars first - adding type detection and validation
            try:
                # Use more robust loading with schema detection and validation
                df = pl.read_parquet(input_file, use_pyarrow=True)
                logger.info(f"Loaded data with {df.height} rows and {df.width} columns")
                
                # Verify column count
                if df.width < 3000:
                    logger.warning(f"WARNING: Input file has only {df.width} columns. For proper feature generation in Numerai Crypto, the input should have at least 3000 columns.")
                    logger.warning("This indicates the merged dataset wasn't correctly created.")
                    
                    # Get file size for additional context
                    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
                    logger.warning(f"File size: {file_size_mb:.2f} MB")
                    
                    # Check for files in the same directory and list alternatives
                    dir_path = os.path.dirname(input_file)
                    other_files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
                    if other_files:
                        logger.warning(f"Other parquet files in the same directory: {other_files}")
                        
                        # Check column counts in other files to find potential alternatives
                        for other_file in other_files[:3]:  # Check up to 3 alternatives
                            if other_file == os.path.basename(input_file):
                                continue
                                
                            try:
                                other_path = os.path.join(dir_path, other_file)
                                other_df = pl.read_parquet(other_path)
                                if other_df.width >= 3000:
                                    logger.warning(f"Alternative file with sufficient columns found: {other_file} ({other_df.width} columns)")
                                    logger.warning(f"Consider using this file instead: {other_path}")
                            except Exception as e_alt:
                                logger.debug(f"Could not check alternative file {other_file}: {e_alt}")
                
                # Handle date column immediately after loading
                if self.date_col in df.columns:
                    logger.info(f"Handling date column {self.date_col}")
                    # Check current data type
                    date_type = df.get_column(self.date_col).dtype
                    logger.info(f"Date column current type: {date_type}")
                    
                    # If not already datetime, try to convert
                    if not str(date_type).startswith(('Date', 'Datetime')):
                        try:
                            # Try with strict=False first
                            df = df.with_column(pl.col(self.date_col).cast(pl.Datetime, strict=False))
                            logger.info(f"Successfully converted {self.date_col} to datetime with strict=False")
                        except Exception as e_dt1:
                            logger.warning(f"First attempt to convert date column failed: {e_dt1}")
                            try:
                                # Try with_columns instead
                                df = df.with_columns(pl.col(self.date_col).cast(pl.Datetime, strict=False))
                                logger.info(f"Successfully converted {self.date_col} to datetime with with_columns")
                            except Exception as e_dt2:
                                logger.warning(f"Second attempt to convert date column failed: {e_dt2}")
                                try:
                                    # Try older polars syntax
                                    df = df.with_columns([(self.date_col, pl.col(self.date_col).cast(pl.Datetime, strict=False))])
                                    logger.info(f"Successfully converted {self.date_col} to datetime with older syntax")
                                except Exception as e_dt3:
                                    logger.warning(f"All attempts to convert date column in polars failed: {e_dt3}")
                                    logger.info("Will try pandas fallback if needed later")
                
                return df
            except Exception as e:
                logger.error(f"Error loading data with Polars: {e}")
                
                # Try pandas fallback
                logger.info("Trying pandas fallback for loading...")
                pandas_df = pd.read_parquet(input_file)
                logger.info(f"Loaded with pandas: {pandas_df.shape[0]} rows, {pandas_df.shape[1]} columns")
                
                # Check column count
                if pandas_df.shape[1] < 3000:
                    logger.warning(f"WARNING: Input file has only {pandas_df.shape[1]} columns. For proper feature generation in Numerai Crypto, the input should have at least 3000 columns.")
                    logger.warning("This indicates the merged dataset wasn't correctly created.")
                    
                    # Check processing_dir for files with more columns
                    processed_dir = "/media/knight2/EDB/numer_crypto_temp/data/processed"
                    if os.path.exists(processed_dir):
                        logger.info(f"Checking processed directory for better alternatives: {processed_dir}")
                        try:
                            import glob
                            # Look for large parquet files in the processed directory
                            parquet_files = glob.glob(os.path.join(processed_dir, "*.parquet"))
                            parquet_files += glob.glob(os.path.join(processed_dir, "*/*.parquet"))
                            
                            # Get files with size > 100MB (likely to have many columns)
                            large_files = []
                            for pf in parquet_files:
                                try:
                                    file_size_mb = os.path.getsize(pf) / (1024 * 1024)
                                    if file_size_mb > 100:
                                        large_files.append((pf, file_size_mb))
                                except:
                                    pass
                            
                            # Sort by size, descending
                            large_files.sort(key=lambda x: x[1], reverse=True)
                            
                            if large_files:
                                logger.info(f"Found {len(large_files)} large parquet files. Top files:")
                                for i, (file_path, size_mb) in enumerate(large_files[:3]):
                                    logger.info(f"  {i+1}. {file_path} ({size_mb:.1f} MB)")
                                
                                # Try the largest file
                                if large_files:
                                    largest_file = large_files[0][0]
                                    logger.info(f"Checking column count in largest file: {largest_file}")
                                    try:
                                        largest_df = pd.read_parquet(largest_file, engine='pyarrow')
                                        if largest_df.shape[1] >= 3000:
                                            logger.warning(f"Found better alternative with {largest_df.shape[1]} columns: {largest_file}")
                                            logger.warning(f"You may want to use this file instead.")
                                    except Exception as e:
                                        logger.debug(f"Error checking largest file: {e}")
                        except Exception as e_glob:
                            logger.debug(f"Error searching for alternatives: {e_glob}")
                
                # Handle date conversion before converting to polars
                if self.date_col in pandas_df.columns:
                    try:
                        # Try to convert date column to datetime format in pandas first
                        if pandas_df[self.date_col].dtype == 'object':
                            logger.info(f"Converting {self.date_col} to datetime in pandas")
                            pandas_df[self.date_col] = pd.to_datetime(pandas_df[self.date_col], errors='coerce')
                            # Fill NA values that couldn't be converted
                            if pandas_df[self.date_col].isna().any():
                                logger.warning(f"Found {pandas_df[self.date_col].isna().sum()} NA values after date conversion")
                                # Fill with a default date value
                                pandas_df[self.date_col] = pandas_df[self.date_col].fillna(pd.Timestamp('2000-01-01'))
                    except Exception as e_dt:
                        logger.warning(f"Error converting date column in pandas: {e_dt}")
                
                # Convert to polars
                df = pl.from_pandas(pandas_df)
                logger.info(f"Converted to Polars: {df.height} rows, {df.width} columns")
                return df
        except Exception as e_pd:
            logger.error(f"All data loading methods failed: {e_pd}")
            raise
    
    def _generate_rolling_features(self, df: pl.DataFrame, numeric_cols: List[str]) -> pl.DataFrame:
        """Generate rolling window features"""
        logger.info(f"Generating rolling features with windows {self.rolling_windows}")
        
        # Check if date column exists
        has_date_col = self.date_col in df.columns
        if not has_date_col:
            logger.warning(f"Date column '{self.date_col}' not found in dataframe. Using index for time ordering.")
            # Create a dummy date column - use row index
            try:
                # Try using index_level if it exists
                if "__index_level_0__" in df.columns:
                    # Create a new date column based on the index
                    try:
                        df = df.with_columns(
                            pl.col("__index_level_0__").alias(self.date_col)
                        )
                    except Exception as e:
                        logger.debug(f"First attempt to create date column failed: {e}")
                        try:
                            df = df.with_column(
                                pl.col("__index_level_0__").alias(self.date_col)
                            )
                        except Exception as e2:
                            logger.debug(f"Second attempt to create date column failed: {e2}")
                            df = df.with_columns([(self.date_col, pl.col("__index_level_0__"))])
                    
                    has_date_col = True
                    logger.info(f"Created date column from __index_level_0__")
                else:
                    # Create a sequence
                    try:
                        df = df.with_columns(
                            pl.arange(0, df.height).alias(self.date_col)
                        )
                    except Exception as e:
                        logger.debug(f"First attempt to create date sequence failed: {e}")
                        try:
                            df = df.with_column(
                                pl.arange(0, df.height).alias(self.date_col)
                            )
                        except Exception as e2:
                            logger.debug(f"Second attempt to create date sequence failed: {e2}")
                            df = df.with_columns([(self.date_col, pl.arange(0, df.height))])
                    
                    has_date_col = True
                    logger.info(f"Created date column as integer sequence")
            except Exception as e:
                logger.error(f"Failed to create date column: {e}")
                # Continue without date column, using row order
        
        # Count the number of unique symbols to better understand memory requirements
        unique_symbols = df.select(pl.col(self.group_col)).n_unique()
        logger.info(f"Dataset has {unique_symbols} unique symbols")
        
        # Estimate memory needed per window/function/batch
        approx_mem_per_col_gb = (df.height * 8) / (1024**3)  # 8 bytes per float64 value
        logger.info(f"Approx memory per column: {approx_mem_per_col_gb:.4f} GB")
        logger.info(f"Dataframe dimensions: {df.height} rows × {df.width} columns")
        
        # Calculate a safe batch size based on available memory
        mem = self._get_memory_info()
        available_gb = mem.get('available_gb', 4.0)  # Default to 4GB if can't detect
        
        # Adjust batch size based on available memory (target using max 25% of available)
        target_mem_usage = available_gb * 0.25
        safe_batch_size = max(1, min(10, int(target_mem_usage / (approx_mem_per_col_gb * 4))))
        
        logger.info(f"Calculated safe batch size: {safe_batch_size} columns (available memory: {available_gb:.2f} GB)")
        
        # For memory efficiency, process in batches
        all_features = []
        
        # Keep the original DataFrame
        all_features.append(df)
        
        # Generate features for each window
        for window_idx, window in enumerate(self.rolling_windows):
            logger.info(f"Processing window {window} ({window_idx+1}/{len(self.rolling_windows)})")
            
            # Skip large windows if we have lots of data
            if df.height > 1000000 and window > 28:
                logger.info(f"Skipping large window {window} for large dataset with {df.height} rows")
                continue
                
            # Create a suffix for column names
            suffix = f"_roll_{window}"
            
            # Start with a skeleton DataFrame with required columns
            select_cols = [self.group_col]
            if has_date_col:
                select_cols.append(self.date_col)
            
            window_df = df.select(select_cols)
            
            # Process each function
            for agg_fn in ["mean", "std", "max", "min"]:
                # Skip some combinations for memory efficiency
                if window > 14 and agg_fn in ["std", "min"]:
                    logger.info(f"Skipping {agg_fn} for large window {window}")
                    continue
                
                fn_suffix = f"{suffix}_{agg_fn}"
                logger.info(f"Processing {agg_fn} function for window {window}")
                
                # Use our dynamically calculated batch size
                batch_size = safe_batch_size
                
                # Process columns in smaller batches
                for i in range(0, len(numeric_cols), batch_size):
                    batch_cols = numeric_cols[i:i+batch_size]
                    
                    # Only log occasionally to reduce verbosity
                    if i % (batch_size * 20) == 0 or i == 0 or i >= len(numeric_cols) - batch_size:
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(numeric_cols) + batch_size - 1)//batch_size}")
                    
                    # Build expressions for rolling aggregations
                    exprs = []
                    for col in batch_cols:
                        # Handle both older and newer Polars versions
                        try:
                            # Newer versions use min_samples with float32 casting
                            if agg_fn == "mean":
                                expr = pl.col(col).rolling_mean(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "std":
                                expr = pl.col(col).rolling_std(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "max":
                                expr = pl.col(col).rolling_max(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "min":
                                expr = pl.col(col).rolling_min(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                        except TypeError:
                            # Older versions use min_periods with float32 casting
                            if agg_fn == "mean":
                                expr = pl.col(col).rolling_mean(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "std":
                                expr = pl.col(col).rolling_std(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "max":
                                expr = pl.col(col).rolling_max(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                            elif agg_fn == "min":
                                expr = pl.col(col).rolling_min(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                        exprs.append(expr)
                    
                    # Add aggregated columns to window DataFrame
                    select_cols = [self.group_col] + batch_cols
                    if has_date_col:
                        select_cols.append(self.date_col)
                    
                    # Reduce verbosity - only log at key points
                    try:
                        # Group by symbol without detailed logging
                        batch_df = df.select(select_cols).group_by(self.group_col).agg(exprs)
                        
                        # Merge with window_df - join only on symbol
                        window_df = window_df.join(batch_df, on=[self.group_col], how="left")
                        
                        # Force clear memory intermediates
                        del batch_df
                        from utils.memory_utils import clear_memory
                        clear_memory()
                        
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        # If we get an error, try with a smaller batch size
                        if len(batch_cols) > 1:
                            logger.info("Trying with smaller batch_size to recover")
                            # Try with half the batch size for the next iteration
                            safe_batch_size = max(1, safe_batch_size // 2)
                            logger.info(f"Reduced batch size to {safe_batch_size}")
                            
                            # Skip this batch and continue with smaller batches
                            continue
                        else:
                            # If we're already at batch size 1, we have a more fundamental problem
                            logger.error("Cannot process even a single column, skipping this aggregation function")
                            break
            
            # Add the new features to our collection, but only keep the actual feature columns
            try:
                feature_cols = [c for c in window_df.columns if c not in [self.group_col, self.date_col]]
                logger.info(f"Adding {len(feature_cols)} new feature columns from window {window}")
                
                if feature_cols:  # Only append if we have features
                    all_features.append(window_df.select(feature_cols))
                    
                # Clear window_df to free memory
                del window_df
                from utils.memory_utils import clear_memory
                clear_memory()
                
            except Exception as e:
                logger.error(f"Error adding features for window {window}: {e}")
        
        # Combine all features with the original DataFrame - minimal logging
        try:
            logger.info(f"Combining {len(all_features)} feature sets")
            result_df = df
            
            # Add feature sets one at a time without excessive logging
            for idx, feature_df in enumerate(all_features[1:]):  # Skip the first one which is the original df
                # Only log at major milestones
                if idx == 0 or idx == len(all_features)-2 or idx % max(5, len(all_features) // 4) == 0:
                    logger.info(f"Adding feature set {idx+1}/{len(all_features)-1}")
                
                new_cols = [c for c in feature_df.columns if c not in result_df.columns]
                if new_cols:
                    result_df = pl.concat([result_df, feature_df.select(new_cols)], how="horizontal")
                
                # Clean up to free memory
                del feature_df
                from utils.memory_utils import clear_memory
                clear_memory()
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            # Return original DataFrame as fallback
            return df
        
        logger.info(f"Generated {result_df.width - df.width} rolling features")
        return result_df
        
    def _get_memory_info(self) -> dict:
        """Get memory information using psutil if available, otherwise estimate"""
        try:
            import psutil
            
            # Current process
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024 ** 3)  # GB
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_total = system_memory.total / (1024 ** 3)  # GB
            system_available = system_memory.available / (1024 ** 3)  # GB
            system_used = system_memory.used / (1024 ** 3)  # GB
            
            return {
                'process_gb': process_memory,
                'total_gb': system_total,
                'used_gb': system_used,
                'available_gb': system_available
            }
        except ImportError:
            # Fallback if psutil is not available - make conservative estimate
            return {
                'process_gb': 2.0,
                'total_gb': 8.0,
                'used_gb': 4.0,
                'available_gb': 4.0
            }
            
    def _generate_rolling_features_chunked(self, df: pl.DataFrame, numeric_cols: List[str]) -> pl.DataFrame:
        """Generate rolling window features using a memory-efficient chunked approach"""
        from utils.memory_utils import clear_memory
        
        logger.info(f"Generating rolling features (chunked) with windows {self.rolling_windows}")
        
        # Check if date column exists
        has_date_col = self.date_col in df.columns
        if not has_date_col:
            logger.warning(f"Date column '{self.date_col}' not found in dataframe. Using index for time ordering.")
            # Create a sequence
            try:
                df = df.with_columns(
                    pl.arange(0, df.height).alias(self.date_col)
                )
            except Exception as e:
                logger.debug(f"Failed to create date column: {e}")
                
                # Try alternate API
                try:
                    df = df.with_column(
                        pl.arange(0, df.height).alias(self.date_col)
                    )
                except Exception as e2:
                    logger.error(f"All attempts to create date column failed: {e2}")
                    # Continue without date column
        
        # Count unique symbols
        unique_symbols = df.select(pl.col(self.group_col)).n_unique()
        logger.info(f"Dataset has {unique_symbols} unique symbols")
        
        # Approximate memory needed per column
        approx_mem_per_col_gb = (df.height * 8) / (1024**3)  # 8 bytes per float64 value
        logger.info(f"Approx memory per column: {approx_mem_per_col_gb:.4f} GB")
        
        # Get memory info
        mem = self._get_memory_info()
        available_gb = mem.get('available_gb', 4.0)
        
        # Calculate a safe number of columns per symbol chunk based on available memory
        target_mem_gb = available_gb * 0.25  # Target using 25% of available memory
        cols_per_batch = max(1, min(5, int(target_mem_gb / (approx_mem_per_col_gb * 2))))
        
        # Calculate how many symbols to process at once
        symbols_per_chunk = max(1, min(10, int(unique_symbols / 10)))
        logger.info(f"Processing {symbols_per_chunk} symbols at a time, with {cols_per_batch} columns per batch")
        
        # Get all unique symbols
        all_symbols = df.select(pl.col(self.group_col)).unique().to_pandas()[self.group_col].tolist()
        
        # Create a dictionary to store all generated features
        all_features = {}
        
        # Process in chunks by symbol groups
        for symbol_chunk_idx in range(0, len(all_symbols), symbols_per_chunk):
            symbol_chunk = all_symbols[symbol_chunk_idx:symbol_chunk_idx + symbols_per_chunk]
            logger.info(f"Processing symbol chunk {symbol_chunk_idx//symbols_per_chunk + 1}/{(len(all_symbols) + symbols_per_chunk - 1)//symbols_per_chunk}")
            logger.info(f"Symbols in this chunk: {symbol_chunk}")
            
            # Filter dataframe for just this set of symbols
            chunk_filter = pl.col(self.group_col).is_in(symbol_chunk)
            symbol_df = df.filter(chunk_filter)
            
            logger.info(f"Symbol chunk dataframe has shape: {symbol_df.height} rows x {symbol_df.width} columns")
            
            # Process each window
            for window_idx, window in enumerate(self.rolling_windows):
                logger.info(f"Processing window {window} ({window_idx+1}/{len(self.rolling_windows)}) for symbol chunk")
                
                # Skip large windows if we have lots of data
                if df.height > 1000000 and window > 28:
                    logger.info(f"Skipping large window {window} for large dataset")
                    continue
                
                # Create a suffix for column names
                suffix = f"_roll_{window}"
                
                # Process functions
                for agg_fn in ["mean", "std", "max", "min"]:
                    # Skip some combinations for memory efficiency
                    if window > 14 and agg_fn in ["std", "min"]:
                        logger.info(f"Skipping {agg_fn} for large window {window}")
                        continue
                    
                    fn_suffix = f"{suffix}_{agg_fn}"
                    logger.info(f"Processing {agg_fn} function for window {window}")
                    
                    # Process columns in smaller batches
                    for col_idx in range(0, len(numeric_cols), cols_per_batch):
                        batch_cols = numeric_cols[col_idx:col_idx + cols_per_batch]
                        if not batch_cols:
                            continue
                            
                        # Only log very occasionally for extreme reduction in verbosity
                        batch_num = col_idx//cols_per_batch + 1
                        total_batches = (len(numeric_cols) + cols_per_batch - 1)//cols_per_batch
                        
                        if batch_num % max(20, total_batches // 5) == 0 or batch_num == 1 or batch_num == total_batches:
                            logger.info(f"Processing column batch {batch_num}/{total_batches}")
                        
                        try:
                            # Build expressions for rolling aggregations
                            exprs = []
                            for col in batch_cols:
                                if agg_fn == "mean":
                                    try:
                                        expr = pl.col(col).rolling_mean(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                    except TypeError:
                                        expr = pl.col(col).rolling_mean(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                elif agg_fn == "std":
                                    try:
                                        expr = pl.col(col).rolling_std(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                    except TypeError:
                                        expr = pl.col(col).rolling_std(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                elif agg_fn == "max":
                                    try:
                                        expr = pl.col(col).rolling_max(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                    except TypeError:
                                        expr = pl.col(col).rolling_max(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                elif agg_fn == "min":
                                    try:
                                        expr = pl.col(col).rolling_min(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                    except TypeError:
                                        expr = pl.col(col).rolling_min(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                exprs.append(expr)
                            
                            # Select required columns from the symbol dataframe
                            select_cols = [self.group_col] + batch_cols
                            if has_date_col:
                                select_cols.append(self.date_col)
                            
                            # Group by symbol and calculate rolling features
                            # Only log operation details on very rare occasions - just when we hit major milestones
                            if batch_num % max(10, total_batches // 5) == 0 or batch_num == 1 or batch_num == total_batches:
                                logger.info(f"Processing window {window} {agg_fn} (batch {batch_num})")
                            
                            try:
                                # Perform group_by without any memory logging except at very major milestones
                                feature_df = symbol_df.select(select_cols).group_by(self.group_col).agg(exprs)
                                
                                # Store feature columns in our dictionary
                                for col in batch_cols:
                                    feature_col = f"{col}{fn_suffix}"
                                    if feature_col not in all_features:
                                        # Extract just this column and the group column
                                        col_df = feature_df.select([self.group_col, feature_col])
                                        all_features[feature_col] = col_df
                                    
                                # Clean up to save memory
                                del feature_df
                                # Only do full GC periodically
                                should_full_gc = (batch_num % max(10, total_batches // 5) == 0)
                                # Only be verbose every 50 batches
                                verbose_log = should_full_gc and (batch_num % 50 == 0)
                                clear_memory()
                                
                            except Exception as e:
                                # Only log the first few errors in detail
                                if batch_num < 10 or batch_num % 50 == 0:
                                    logger.error(f"Error calculating features for batch {batch_num}: {e}")
                                # If batch size > 1, try with even smaller batches
                                if len(batch_cols) > 1:
                                    if batch_num < 10 or batch_num % 50 == 0:
                                        logger.info("Trying with individual columns")
                                    
                                    # Try each column individually but don't log each one
                                    individual_success = 0
                                    for col in batch_cols:
                                        try:
                                            # Build expression for single column with float32 precision
                                            if agg_fn == "mean":
                                                try:
                                                    expr = pl.col(col).rolling_mean(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                                except TypeError:
                                                    expr = pl.col(col).rolling_mean(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                            elif agg_fn == "std":
                                                try:
                                                    expr = pl.col(col).rolling_std(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                                except TypeError:
                                                    expr = pl.col(col).rolling_std(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                            elif agg_fn == "max":
                                                try:
                                                    expr = pl.col(col).rolling_max(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                                except TypeError:
                                                    expr = pl.col(col).rolling_max(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                            elif agg_fn == "min":
                                                try:
                                                    expr = pl.col(col).rolling_min(window, min_samples=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                                except TypeError:
                                                    expr = pl.col(col).rolling_min(window, min_periods=1).cast(pl.Float32).alias(f"{col}{fn_suffix}")
                                            
                                            # Select required columns from the symbol dataframe
                                            select_cols = [self.group_col, col]
                                            if has_date_col:
                                                select_cols.append(self.date_col)
                                            
                                            # Process single column without verbose logging
                                            col_df = symbol_df.select(select_cols).group_by(self.group_col).agg([expr])
                                            
                                            # Store in our features dictionary
                                            feature_col = f"{col}{fn_suffix}"
                                            if feature_col not in all_features:
                                                all_features[feature_col] = col_df.select([self.group_col, feature_col])
                                            
                                            # Clean up
                                            del col_df
                                            # Only clear memory every few columns
                                            if individual_success % 10 == 0:
                                                clear_memory()
                                                
                                            individual_success += 1
                                            
                                        except Exception:
                                            # Skip this column without detailed logging
                                            continue
                                    
                                    # Log summary of individual column processing
                                    if individual_success > 0 and (batch_num < 10 or batch_num % 50 == 0):
                                        logger.info(f"Successfully processed {individual_success}/{len(batch_cols)} individual columns")
                        except Exception as e_batch:
                            if batch_num < 10 or batch_num % 50 == 0:
                                logger.error(f"Error processing batch {batch_num}: {e_batch}")
                            # Continue with next batch
                            continue
            
            # Clear memory after processing a symbol chunk
            chunk_num = symbol_chunk_idx//symbols_per_chunk + 1
            total_chunks = (len(all_symbols) + symbols_per_chunk - 1)//symbols_per_chunk
            logger.info(f"Completed symbol chunk {chunk_num}/{total_chunks}")
            del symbol_df
            clear_memory()
        
        # Now combine all feature columns with the original dataframe - minimal logging
        logger.info(f"Generated {len(all_features)} feature columns, merging with original dataframe")
        
        # Start with original dataframe
        result_df = df
        
        # Keep track of skipped features due to errors
        skipped_features = []
        
        # Merge each feature column one by one
        total_features = len(all_features)
        for feature_idx, (feature_name, feature_df) in enumerate(all_features.items()):
            try:
                # Only log at major milestones to greatly reduce verbosity
                if feature_idx % max(100, total_features // 4) == 0 or feature_idx == 0 or feature_idx == total_features - 1:
                    logger.info(f"Merging feature {feature_idx + 1}/{total_features}")
                
                # Join with original dataframe on group column
                result_df = result_df.join(
                    feature_df,
                    on=self.group_col,
                    how="left"
                )
                
                # Clean up to save memory
                del feature_df
                
                # Periodically force GC to keep memory usage low
                if feature_idx % max(100, total_features // 4) == 0:
                    clear_memory()
                
            except Exception as e:
                # Only log errors at very rare intervals
                if feature_idx < 5 or feature_idx % 100 == 0:
                    logger.error(f"Error merging feature {feature_name}: {e}")
                skipped_features.append(feature_name)
                continue
        
        # Report on any skipped features
        if skipped_features:
            logger.warning(f"Skipped {len(skipped_features)} features due to errors")
        
        # Final statistics - minimal logging
        feature_count = result_df.width - df.width
        logger.info(f"Successfully added {feature_count} rolling features")
        
        return result_df
    
    def _generate_lag_features(self, df: pl.DataFrame, numeric_cols: List[str]) -> pl.DataFrame:
        """Generate lag features"""
        logger.info(f"Generating lag features with periods {self.lag_periods}")
        
        # For memory efficiency, process in batches
        all_features = []
        
        # Check if date column exists
        has_date_col = self.date_col in df.columns
        
        # Start with a skeleton DataFrame with required columns
        select_cols = [self.group_col]
        if has_date_col:
            select_cols.append(self.date_col)
        
        lag_df = df.select(select_cols)
        
        # Process columns in smaller batches
        batch_size = 20
        for i in range(0, len(numeric_cols), batch_size):
            batch_cols = numeric_cols[i:i+batch_size]
            
            # Process each lag period
            for lag in self.lag_periods:
                # Skip large lags if we have lots of data
                if df.height > 1000000 and lag > 14:
                    continue
                
                # Create lag features for this batch with float32 precision
                batch_exprs = []
                for col in batch_cols:
                    batch_exprs.append(pl.col(col).shift(lag).cast(pl.Float32).alias(f"{col}_lag_{lag}"))
                
                # Add to the lag DataFrame
                select_cols = [self.group_col] + batch_cols
                if has_date_col:
                    select_cols.append(self.date_col)
                
                # Group by symbol only (not date)
                batch_df = df.select(select_cols).group_by(self.group_col).agg(batch_exprs)
                
                # Log columns for debugging
                logger.debug(f"Lag DataFrame columns: {lag_df.columns}")
                logger.debug(f"Batch DataFrame columns: {batch_df.columns}")
                
                # Merge with lag_df - join only on symbol
                lag_df = lag_df.join(batch_df, on=[self.group_col], how="left")
        
        # Add new columns to result
        new_cols = [c for c in lag_df.columns if c not in df.columns]
        if new_cols:
            result_df = pl.concat([df, lag_df.select(new_cols)], how="horizontal")
        else:
            result_df = df
        
        logger.info(f"Generated {result_df.width - df.width} lag features")
        return result_df
    
    def _generate_ewm_features(self, df: pl.DataFrame, numeric_cols: List[str]) -> pl.DataFrame:
        """Generate exponential weighted moving average features using Polars"""
        from utils.memory_utils import clear_memory
        
        logger.info(f"Generating EWM features with spans {self.ewm_spans}")
        
        # Check if date column exists (required for proper ordering)
        has_date_col = self.date_col in df.columns
        
        # Polars doesn't have a direct EWM function, but we can approximate it
        # using a weighted moving average calculation
        
        # Start with the original DataFrame
        result_df = df.clone()
        
        # Total number of EWM features to generate
        total_ewm_features = len(self.ewm_spans) * min(20, len(numeric_cols))
        features_added = 0
        
        # For memory efficiency, process in batches
        batch_size = 5  # Process 5 columns at a time
        
        # Limit the number of columns to process to avoid memory issues
        max_cols = min(20, len(numeric_cols))
        limited_cols = numeric_cols[:max_cols]
        
        for span_idx, span in enumerate(self.ewm_spans):
            # Skip large spans if we have lots of data
            if df.height > 1000000 and span > 20:
                logger.info(f"Skipping large span {span} for large dataset")
                continue
                
            logger.info(f"Calculating EWM-like features with span={span} ({span_idx+1}/{len(self.ewm_spans)})")
            
            # Calculate alpha (same as pandas ewm uses)
            alpha = 2.0 / (span + 1.0)
            logger.info(f"Alpha value: {alpha}")
            
            # Process columns in batches
            for i in range(0, len(limited_cols), batch_size):
                batch_cols = limited_cols[i:i+batch_size]
                logger.info(f"Processing columns {i+1}-{i+len(batch_cols)} of {len(limited_cols)}")
                
                # Track column progress
                col_idx = 0
                for col in batch_cols:
                    col_idx += 1
                    # Determine if we should log based on batch progress
                    should_log = (i == 0 and col_idx == 1) or (i + col_idx == len(limited_cols)) or (col_idx == 1 and i % 50 == 0)
                    
                    try:
                        new_col_name = f"{col}_ewm_{span}"
                        if should_log:
                            logger.info(f"Creating EWM features batch {i//batch_size + 1}/{(len(limited_cols) + batch_size - 1)//batch_size}")
                        
                        # We'll implement EWM using Polars' window functions as an approximation
                        # This won't be exactly the same as pandas' EWM but close enough for feature generation
                        
                        # First, make sure we're sorted by group and date (if available)
                        if has_date_col:
                            # For sorting, create a new df with just the columns we need
                            # This approach is more memory efficient
                            select_cols = [self.group_col, col]
                            if has_date_col:
                                select_cols.append(self.date_col)
                                
                            temp_df = df.select(select_cols)
                            
                            # Calculate EWM-like feature using weighted average with exponential decay
                            # We'll approximate this using window functions
                            try:
                                # Check if pl.repeat is available (newer Polars versions)
                                if hasattr(pl, 'repeat'):
                                    weights = pl.repeat(alpha, span).cumprod().reverse()
                                    
                                    # Try with weights parameter (newer Polars)
                                    ewm_col = (
                                        temp_df
                                        .sort([self.group_col, self.date_col])
                                        .group_by(self.group_col)
                                        .agg(
                                            # This creates a weighted moving average using rolling window
                                            pl.col(col).rolling_mean(
                                                window_size=span,
                                                weights=weights,
                                                min_periods=1
                                            ).cast(pl.Float32).alias(new_col_name)
                                        )
                                    )
                                else:
                                    # For older Polars versions without pl.repeat
                                    if should_log:
                                        logger.info("pl.repeat not available, using standard rolling_mean as fallback")
                                    raise AttributeError("pl.repeat not available")
                                    
                            except (TypeError, AttributeError, ValueError):
                                # Try older Polars version syntax without weights
                                try:
                                    if should_log:
                                        logger.info("Using standard rolling_mean as fallback")
                                    # Approximate using simple rolling average (fallback)
                                    ewm_col = (
                                        temp_df
                                        .sort([self.group_col, self.date_col])
                                        .group_by(self.group_col)
                                        .agg(
                                            pl.col(col).rolling_mean(
                                                window_size=span,
                                                min_periods=1
                                            ).cast(pl.Float32).alias(new_col_name)
                                        )
                                    )
                                except Exception as e:
                                    # Last fallback: use even simpler method
                                    if should_log:
                                        logger.error(f"Standard rolling_mean failed: {e}")
                                    try:
                                        # Try with different API style
                                        ewm_col = (
                                            temp_df
                                            .sort([self.group_col, self.date_col])
                                            .group_by(self.group_col)
                                            .agg([
                                                pl.col(col).rolling_mean(
                                                    window_size=span,
                                                    min_periods=1
                                                ).cast(pl.Float32).alias(new_col_name)
                                            ])
                                        )
                                    except Exception:
                                        # Just continue without detailed logging
                                        continue
                        else:
                            # Without date column, use a simpler approach with just rolling mean
                            if should_log:
                                logger.info("Using simple rolling mean as EWM approximation (no date column)")
                            select_cols = [self.group_col, col]
                            temp_df = df.select(select_cols)
                            
                            ewm_col = (
                                temp_df
                                .group_by(self.group_col)
                                .agg(
                                    pl.col(col).rolling_mean(
                                        window_size=span,
                                        min_periods=1
                                    ).cast(pl.Float32).alias(new_col_name)
                                )
                            )
                        
                        # Add to result DataFrame
                        if new_col_name not in result_df.columns:
                            result_df = result_df.join(
                                ewm_col.select([self.group_col, new_col_name]),
                                on=self.group_col,
                                how="left"
                            )
                            features_added += 1
                            
                            # Log progress periodically but minimally
                            if features_added % 50 == 0 or features_added == total_ewm_features:
                                logger.info(f"Added {features_added}/{total_ewm_features} EWM features")
                        
                        # Clean up temp DataFrames to save memory
                        del temp_df
                        del ewm_col
                        # Only do full GC periodically
                        clear_memory()
                        
                    except Exception as e:
                        if col_idx == 1 or col_idx % 20 == 0:
                            logger.error(f"Error creating EWM feature: {e}")
                        continue
            
            # Clean memory after each span
            clear_memory()
        
        # Final log - minimal output
        logger.info(f"Generated {features_added} EWM features")
        
        return result_df
    
    def _generate_interaction_features(self, df: pl.DataFrame, numeric_cols: List[str]) -> pl.DataFrame:
        """Generate interaction features between columns"""
        # Limit the number of base columns to avoid explosion
        if len(numeric_cols) > 10:
            # Use the first few and last few columns
            base_cols = numeric_cols[:5] + numeric_cols[-5:]
        else:
            base_cols = numeric_cols
            
        logger.info(f"Generating interaction features with {len(base_cols)} base columns")
        
        # Start with a skeleton DataFrame 
        result_df = df.clone()
        
        # Create only a few key interactions to avoid feature explosion
        for i, col1 in enumerate(base_cols):
            for col2 in base_cols[i+1:]:
                # Only create multiply interactions (others can be derived)
                try:
                    # Try the newest polars API first
                    result_df = result_df.with_columns(
                        (pl.col(col1) * pl.col(col2)).cast(pl.Float32).alias(f"{col1}_X_{col2}")
                    )
                except Exception as e1:
                    logger.debug(f"First API style failed: {e1}")
                    try:
                        # Try another newer polars API style
                        result_df = result_df.with_column(
                            (pl.col(col1) * pl.col(col2)).cast(pl.Float32).alias(f"{col1}_X_{col2}")
                        )
                    except (AttributeError, TypeError) as e2:
                        logger.debug(f"Second API style failed: {e2}")
                        # Fall back to most compatible version
                        result_df = result_df.with_columns([(f"{col1}_X_{col2}", (pl.col(col1) * pl.col(col2)).cast(pl.Float32))])
                
                # Add division if it won't create infinities
                if "price" in col1 and "price" in col2:
                    try:
                        # Try the newest polars API first
                        result_df = result_df.with_columns(
                            (pl.col(col1) / pl.col(col2).clip_min(0.0001)).cast(pl.Float32).alias(f"{col1}_DIV_{col2}")
                        )
                    except Exception as e1:
                        logger.debug(f"First division API style failed: {e1}")
                        try:
                            # Try another newer polars API style
                            result_df = result_df.with_column(
                                (pl.col(col1) / pl.col(col2).clip_min(0.0001)).cast(pl.Float32).alias(f"{col1}_DIV_{col2}")
                            )
                        except (AttributeError, TypeError) as e2:
                            logger.debug(f"Second division API style failed: {e2}")
                            # Fall back to most compatible version
                            result_df = result_df.with_columns([(f"{col1}_DIV_{col2}", (pl.col(col1) / pl.col(col2).clip_min(0.0001)).cast(pl.Float32))])
                
                # Check if we're exceeding max features
                if result_df.width > self.max_features + 10:  # +10 for reserved cols
                    break
            
            if result_df.width > self.max_features + 10:
                break
        
        logger.info(f"Generated {result_df.width - df.width} interaction features")
        return result_df
    
    def _generate_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate financial technical indicators"""
        logger.info("Generating technical indicators")
        
        # Check if we have the typical OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        has_ohlcv = all(col.lower() in [c.lower() for c in df.columns] for col in required_cols)
        
        if not has_ohlcv:
            logger.info("Dataset doesn't have standard OHLCV columns, skipping technical indicators")
            return df
        
        # Map to our actual column names (case insensitive)
        col_map = {}
        for req in required_cols:
            for col in df.columns:
                if col.lower() == req:
                    col_map[req] = col
        
        # Use pandas-ta if available, otherwise calculate basic indicators
        try:
            import pandas_ta as ta
            
            logger.info("Using pandas-ta for technical indicators")
            
            # Convert to pandas
            pandas_df = df.to_pandas()
            
            # Set up ta datafame
            for symbol in pandas_df[self.group_col].unique():
                symbol_df = pandas_df[pandas_df[self.group_col] == symbol].copy()
                
                # Basic momentum
                symbol_df[f'rsi_14'] = ta.rsi(symbol_df[col_map['close']], length=14)
                
                # Volume indicators
                symbol_df[f'volume_delta'] = ta.volume.volume_delta(
                    close=symbol_df[col_map['close']], 
                    volume=symbol_df[col_map['volume']]
                )
                
                # Trend
                symbol_df[f'ema_12'] = ta.ema(symbol_df[col_map['close']], length=12)
                symbol_df[f'ema_26'] = ta.ema(symbol_df[col_map['close']], length=26)
                symbol_df[f'macd'] = ta.macd(symbol_df[col_map['close']])['MACD_12_26_9']
                
                # Update original
                pandas_df.loc[pandas_df[self.group_col] == symbol] = symbol_df
            
            # Convert back to polars
            result_df = pl.from_pandas(pandas_df)
            
        except ImportError:
            logger.info("pandas-ta not available, calculating basic indicators")
            
            # Start with original DataFrame
            result_df = df.clone()
            
            # Calculate some basic indicators
            
            # 1. Price change
            try:
                # Try the newest polars API first
                result_df = result_df.with_columns(
                    (pl.col(col_map['close']) - pl.col(col_map['open'])).cast(pl.Float32).alias('price_change')
                )
            except Exception as e1:
                logger.debug(f"First price change API style failed: {e1}")
                try:
                    # Try another newer polars API style
                    result_df = result_df.with_column(
                        (pl.col(col_map['close']) - pl.col(col_map['open'])).cast(pl.Float32).alias('price_change')
                    )
                except (AttributeError, TypeError) as e2:
                    logger.debug(f"Second price change API style failed: {e2}")
                    # Fall back to most compatible version
                    result_df = result_df.with_columns([('price_change', (pl.col(col_map['close']) - pl.col(col_map['open'])).cast(pl.Float32))])
            
            # 2. Price range
            try:
                # Try the newest polars API first
                result_df = result_df.with_columns(
                    (pl.col(col_map['high']) - pl.col(col_map['low'])).cast(pl.Float32).alias('price_range')
                )
            except Exception as e1:
                logger.debug(f"First price range API style failed: {e1}")
                try:
                    # Try another newer polars API style
                    result_df = result_df.with_column(
                        (pl.col(col_map['high']) - pl.col(col_map['low'])).cast(pl.Float32).alias('price_range')
                    )
                except (AttributeError, TypeError) as e2:
                    logger.debug(f"Second price range API style failed: {e2}")
                    # Fall back to most compatible version
                    result_df = result_df.with_columns([('price_range', (pl.col(col_map['high']) - pl.col(col_map['low'])).cast(pl.Float32))])
            
            # 3. Volume * price change (approximation of volume delta)
            try:
                # Try the newest polars API first
                result_df = result_df.with_columns(
                    (pl.col(col_map['volume']) * (pl.col(col_map['close']) - pl.col(col_map['open']))).cast(pl.Float32).alias('volume_price_change')
                )
            except Exception as e1:
                logger.debug(f"First volume API style failed: {e1}")
                try:
                    # Try another newer polars API style
                    result_df = result_df.with_column(
                        (pl.col(col_map['volume']) * (pl.col(col_map['close']) - pl.col(col_map['open']))).cast(pl.Float32).alias('volume_price_change')
                    )
                except (AttributeError, TypeError) as e2:
                    logger.debug(f"Second volume API style failed: {e2}")
                    # Fall back to most compatible version
                    result_df = result_df.with_columns([('volume_price_change', (pl.col(col_map['volume']) * (pl.col(col_map['close']) - pl.col(col_map['open']))).cast(pl.Float32))])
        
        logger.info(f"Added {result_df.width - df.width} technical indicators")
        return result_df
    
    def _select_top_features(self, df: pl.DataFrame, max_features: int) -> pl.DataFrame:
        """Select the top most important features if we have too many"""
        # Exclude special columns
        excluded_cols = [self.group_col, self.date_col, 'era', 'target', 'id']
        
        # If target column exists, use correlation-based selection
        if 'target' in df.columns:
            logger.info("Using correlation-based feature selection")
            
            # Get feature columns
            feature_cols = [col for col in df.columns if col not in excluded_cols]
            
            # Calculate correlation with target
            correlations = {}
            
            for col in feature_cols:
                # Convert to pandas for correlation computation
                try:
                    col_data = df.select([col, 'target']).to_pandas()
                    correlation = abs(col_data[col].corr(col_data['target']))
                    if not np.isnan(correlation):
                        correlations[col] = correlation
                except:
                    pass
            
            # Sort features by correlation
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_feature_names = [feature for feature, corr in sorted_features[:max_features]]
            
            # Select top features plus excluded columns
            selected_cols = excluded_cols + top_feature_names
            result_df = df.select(selected_cols)
            
            logger.info(f"Selected {len(top_feature_names)} features based on correlation with target")
            
        else:
            # No target column, use variance-based selection
            logger.info("Using variance-based feature selection (no target column)")
            
            # Get feature columns
            feature_cols = [col for col in df.columns if col not in excluded_cols]
            
            # Calculate variance for each feature
            variances = {}
            
            for col in feature_cols:
                try:
                    variance = df.select(pl.col(col)).to_pandas()[col].var()
                    if not np.isnan(variance):
                        variances[col] = variance
                except:
                    pass
            
            # Sort features by variance
            sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_feature_names = [feature for feature, var in sorted_features[:max_features]]
            
            # Select top features plus excluded columns
            selected_cols = excluded_cols + top_feature_names
            result_df = df.select(selected_cols)
            
            logger.info(f"Selected {len(top_feature_names)} features based on variance")
        
        return result_df
    
    def _save_feature_info(self, df: pl.DataFrame, output_file: str, excluded_cols: List[str]) -> None:
        """Save information about the generated features"""
        try:
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
                    'rsi_14', 'volume_delta', 'ema_12', 'ema_26', 'macd', 
                    'price_change', 'price_range', 'volume_price_change'
                ]]),
                'base': len([col for col in feature_cols if not (
                    'roll_' in col or 'lag_' in col or 'ewm_' in col or 
                    '_X_' in col or '_DIV_' in col or col in [
                        'rsi_14', 'volume_delta', 'ema_12', 'ema_26', 'macd',
                        'price_change', 'price_range', 'volume_price_change'
                    ]
                )])
            }
            
            # Count unique symbols
            unique_symbols = df.select(pl.col(self.group_col)).n_unique()
            
            # Get dates range
            if self.date_col in df.columns:
                min_date = df.select(pl.min(self.date_col)).item()
                max_date = df.select(pl.max(self.date_col)).item()
                date_range = f"{min_date} to {max_date}"
            else:
                date_range = "Unknown"
            
            # Create feature info
            feature_info = {
                'timestamp': datetime.now().isoformat(),
                'output_file': output_file,
                'row_count': df.height,
                'symbol_count': unique_symbols,
                'date_range': date_range,
                'total_features': len(feature_cols),
                'feature_types': feature_types,
                'feature_list': feature_cols
            }
            
            # Save to JSON
            info_file = output_file.replace('.parquet', '_info.json')
            with open(info_file, 'w') as f:
                json.dump(feature_info, f, indent=2, default=str)
                
            logger.info(f"Feature information saved to {info_file}")
            
        except Exception as e:
            logger.warning(f"Error saving feature info: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate features using Polars')
    parser.add_argument('--input-file', type=str, 
                      default='/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet',
                      help='Input data file (parquet format)')
    parser.add_argument('--output-dir', type=str, 
                      default='/media/knight2/EDB/numer_crypto_temp/data/features',
                      help='Output directory for features')
    parser.add_argument('--max-features', type=int, default=100000, 
                      help='Maximum number of features to generate')
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        # Try to find an alternative input file
        possible_files = [
            "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet",
            "/media/knight2/EDB/numer_crypto_temp/data/processed/train/train_data.parquet",
            "/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_latest.parquet"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                logger.info(f"Found alternative input file: {file_path}")
                args.input_file = file_path
                break
        else:
            logger.error("No suitable input files found. Exiting.")
            sys.exit(1)
    
    try:
        # Create generator
        generator = PolarsFeatureGenerator(
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