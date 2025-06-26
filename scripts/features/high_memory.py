#!/usr/bin/env python3
"""
High-memory feature engineering for Numerai Crypto competition.

This module generates 100,000+ features utilizing up to 600GB RAM:
- Rolling window features (multiple timeframes)
- Exponential moving averages
- Interaction terms between all features
- Polynomial features up to order 4
- Statistical moments and aggregations
- Technical indicators and spectral features

Designed to run on high-memory machines with distributed computation support.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import gc
import warnings
from itertools import combinations, chain
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import dask.dataframe as dd
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"high_mem_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Memory monitoring functions
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    return memory_gb

def log_memory_usage(step):
    """Log current memory usage"""
    memory_gb = get_memory_usage()
    logger.info(f"Memory usage after {step}: {memory_gb:.2f} GB")

class HighMemoryFeatureGenerator:
    """
    High-memory feature generator that creates 100,000+ features
    utilizing up to 600GB RAM with distributed computation support.
    """
    
    def __init__(self, output_dir=None, max_memory_gb=500, n_jobs=-1, chunksize=10000):
        """
        Initialize the feature generator.
        
        Args:
            output_dir: Directory to save intermediate and final feature files
            max_memory_gb: Maximum memory to use in GB (default: 500)
            n_jobs: Number of parallel jobs (default: all cores)
            chunksize: Number of rows to process at once (for dask)
        """
        # Paths
        self.repo_root = Path(__file__).parent.parent.parent
        if output_dir is None:
            self.output_dir = os.path.join(self.repo_root, 'data', 'features')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance parameters
        self.max_memory_gb = max_memory_gb
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        
        # Feature counts
        self.total_features = 0
        self.feature_counts = {
            'base': 0,
            'rolling': 0,
            'ewm': 0,
            'interaction': 0,
            'polynomial': 0,
            'technical': 0,
            'statistical': 0,
            'spectral': 0
        }
        
        # Feature importance placeholders
        self.feature_importance = None
    
    def load_data(self, input_file):
        """
        Load data for feature generation.
        
        Args:
            input_file: Path to parquet file with processed data
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {input_file}")
        
        try:
            # Use PyArrow for faster loading
            df = pd.read_parquet(input_file, engine='pyarrow')
            
            # Extract symbols
            if 'symbol' in df.columns:
                self.symbols = df['symbol'].unique()
                logger.info(f"Data contains {len(self.symbols)} unique symbols")
            
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Count base features
            self.feature_counts['base'] = df.shape[1] - 1  # Excluding symbol column
            self.total_features = self.feature_counts['base']
            
            log_memory_usage("data loading")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def create_rolling_features(self, df, windows=[2, 5, 10, 20, 50, 100], functions=['mean', 'std', 'min', 'max']):
        """
        Create rolling window features.
        
        Args:
            df: DataFrame with data
            windows: List of window sizes
            functions: List of aggregation functions
            
        Returns:
            DataFrame with original and rolling features
        """
        logger.info(f"Creating rolling window features with {len(windows)} windows and {len(functions)} functions")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # For each window and function, create rolling features
        for window in windows:
            for func in functions:
                # Check if we're approaching memory limit
                current_memory = get_memory_usage()
                if current_memory > self.max_memory_gb:
                    logger.warning(f"Approaching memory limit ({current_memory:.2f} GB). Stopping rolling feature creation.")
                    break
                
                logger.info(f"Creating {func} features with window {window}")
                
                # Use dask for parallel computation if dataframe is large
                if df.shape[0] > 50000 or df.shape[1] > 1000:
                    # Convert to dask dataframe
                    dask_df = dd.from_pandas(df[feature_cols], chunksize=self.chunksize)
                    
                    # Apply rolling function
                    if func == 'mean':
                        rolled = dask_df.rolling(window=window).mean().compute()
                    elif func == 'std':
                        rolled = dask_df.rolling(window=window).std().compute()
                    elif func == 'min':
                        rolled = dask_df.rolling(window=window).min().compute()
                    elif func == 'max':
                        rolled = dask_df.rolling(window=window).max().compute()
                    
                else:
                    # For smaller dataframes, use pandas directly
                    rolled = df[feature_cols].rolling(window=window).agg(func)
                
                # Create new column names
                new_cols = [f"{col}_{func}_window_{window}" for col in feature_cols]
                
                # Add new features to result dataframe
                for i, col in enumerate(feature_cols):
                    new_col = new_cols[i]
                    result_df[new_col] = rolled[col]
                
                # Update count
                features_created += len(feature_cols)
                
                # Log progress
                if features_created % 10000 == 0:
                    log_memory_usage(f"created {features_created} rolling features")
        
        # Replace NaN values with column means instead of zeros
        # This is more statistically sound and prevents issues with the code hanging
        logger.info("Replacing NaN values with column means in rolling features")
        
        # Identify columns with NaNs
        nan_cols = []
        for col in result_df.columns:
            if col not in ['symbol', 'date', 'era', 'id', 'target', 'asset']:
                # Check if column has NaNs
                if result_df[col].isna().any():
                    nan_cols.append(col)
        
        logger.info(f"Found {len(nan_cols)} columns with NaN values")
        
        # Replace NaNs with column means
        for col in nan_cols:
            try:
                # Calculate mean (excluding NaNs)
                col_mean = result_df[col].mean()
                # Handle edge case where mean is NaN (all values are NaN)
                if pd.isna(col_mean):
                    logger.warning(f"Column {col} has all NaN values, using 0 as replacement")
                    result_df[col] = result_df[col].fillna(0)
                else:
                    result_df[col] = result_df[col].fillna(col_mean)
                    logger.debug(f"Column {col}: replaced NaNs with mean value {col_mean}")
            except Exception as e:
                logger.warning(f"Error calculating mean for {col}: {e}, using 0 as fallback")
                result_df[col] = result_df[col].fillna(0)
        
        # Update feature counts
        self.feature_counts['rolling'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} rolling features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("rolling feature creation")
        
        return result_df
    
    def create_ewm_features(self, df, spans=[2, 5, 10, 20, 50, 100], functions=['mean', 'std']):
        """
        Create exponential weighted moving features.
        
        Args:
            df: DataFrame with data
            spans: List of span values for EWM
            functions: List of aggregation functions
            
        Returns:
            DataFrame with original and EWM features
        """
        logger.info(f"Creating exponential weighted moving features with {len(spans)} spans and {len(functions)} functions")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # For each span and function, create EWM features
        for span in spans:
            for func in functions:
                # Check if we're approaching memory limit
                current_memory = get_memory_usage()
                if current_memory > self.max_memory_gb:
                    logger.warning(f"Approaching memory limit ({current_memory:.2f} GB). Stopping EWM feature creation.")
                    break
                
                logger.info(f"Creating {func} EWM features with span {span}")
                
                # Apply EWM function
                if func == 'mean':
                    ewm_result = df[feature_cols].ewm(span=span).mean()
                elif func == 'std':
                    ewm_result = df[feature_cols].ewm(span=span).std()
                
                # Create new column names
                new_cols = [f"{col}_ewm_{func}_span_{span}" for col in feature_cols]
                
                # Add new features to result dataframe
                for i, col in enumerate(feature_cols):
                    new_col = new_cols[i]
                    result_df[new_col] = ewm_result[col]
                
                # Update count
                features_created += len(feature_cols)
                
                # Log progress
                if features_created % 10000 == 0:
                    log_memory_usage(f"created {features_created} EWM features")
        
        # Replace NaN values with column means instead of zeros
        # This is more statistically sound and prevents issues with the code hanging
        logger.info("Replacing NaN values with column means in EWM features")
        
        # Identify columns with NaNs
        nan_cols = []
        for col in result_df.columns:
            if col not in ['symbol', 'date', 'era', 'id', 'target', 'asset']:
                # Check if column has NaNs
                if result_df[col].isna().any():
                    nan_cols.append(col)
        
        logger.info(f"Found {len(nan_cols)} columns with NaN values")
        
        # Replace NaNs with column means
        for col in nan_cols:
            try:
                # Calculate mean (excluding NaNs)
                col_mean = result_df[col].mean()
                # Handle edge case where mean is NaN (all values are NaN)
                if pd.isna(col_mean):
                    logger.warning(f"Column {col} has all NaN values, using 0 as replacement")
                    result_df[col] = result_df[col].fillna(0)
                else:
                    result_df[col] = result_df[col].fillna(col_mean)
                    logger.debug(f"Column {col}: replaced NaNs with mean value {col_mean}")
            except Exception as e:
                logger.warning(f"Error calculating mean for {col}: {e}, using 0 as fallback")
                result_df[col] = result_df[col].fillna(0)
        
        # Update feature counts
        self.feature_counts['ewm'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} EWM features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("EWM feature creation")
        
        return result_df
    
    def create_interaction_features(self, df, max_interactions=2, top_n_features=200):
        """
        Create interaction features (products of feature pairs).
        
        Args:
            df: DataFrame with data
            max_interactions: Maximum number of features to interact
            top_n_features: Number of most important features to use for interactions
            
        Returns:
            DataFrame with original and interaction features
        """
        logger.info(f"Creating interaction features with max {max_interactions} interactions using top {top_n_features} features")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # If we have feature importance, use it to select top features
        if self.feature_importance is not None and len(feature_cols) > top_n_features:
            logger.info(f"Using feature importance to select top {top_n_features} features for interactions")
            top_features = [col for col, imp in self.feature_importance[:top_n_features]]
            feature_cols = [col for col in feature_cols if col in top_features]
        
        # If still too many features, select a subset
        if len(feature_cols) > top_n_features:
            logger.info(f"Selected random {top_n_features} features for interactions")
            np.random.seed(42)
            feature_cols = np.random.choice(feature_cols, top_n_features, replace=False).tolist()
        
        logger.info(f"Creating interactions using {len(feature_cols)} features")
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # Generate combinations of features
        for r in range(2, max_interactions + 1):
            # Check if we're approaching memory limit
            current_memory = get_memory_usage()
            if current_memory > self.max_memory_gb:
                logger.warning(f"Approaching memory limit ({current_memory:.2f} GB). Stopping interaction feature creation.")
                break
            
            logger.info(f"Creating order {r} interaction features")
            
            # Generate combinations
            feature_combinations = list(combinations(feature_cols, r))
            
            # If too many combinations, sample a subset
            max_combinations = 50000 // r  # Limit based on interaction order
            if len(feature_combinations) > max_combinations:
                logger.info(f"Too many combinations ({len(feature_combinations)}), sampling {max_combinations}")
                np.random.seed(42)
                feature_combinations = np.random.choice(feature_combinations, max_combinations, replace=False).tolist()
            
            # Process combinations in parallel chunks
            chunk_size = min(1000, len(feature_combinations))
            num_chunks = (len(feature_combinations) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                # Get combinations for this chunk
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(feature_combinations))
                chunk_combinations = feature_combinations[start_idx:end_idx]
                
                # Process each combination
                for combo in chunk_combinations:
                    # Check memory again
                    if get_memory_usage() > self.max_memory_gb:
                        break
                    
                    # Create new feature name
                    new_col = f"interaction_{'_x_'.join(combo)}"
                    
                    # Create interaction feature (product of features)
                    result_df[new_col] = df[list(combo)].prod(axis=1)
                    
                    # Update count
                    features_created += 1
                
                # Log progress
                if chunk_idx % 10 == 0 or chunk_idx == num_chunks - 1:
                    logger.info(f"Processed chunk {chunk_idx+1}/{num_chunks}, created {features_created} interaction features so far")
                    log_memory_usage(f"interaction feature chunk {chunk_idx+1}")
        
        # Update feature counts
        self.feature_counts['interaction'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} interaction features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("interaction feature creation")
        
        return result_df
    
    def create_polynomial_features(self, df, degree=4, top_n_features=100, interaction_only=False):
        """
        Create polynomial features up to specified degree.
        
        Args:
            df: DataFrame with data
            degree: Maximum polynomial degree
            top_n_features: Number of most important features to use for polynomial expansion
            interaction_only: If True, only include interaction terms without powers
            
        Returns:
            DataFrame with original and polynomial features
        """
        logger.info(f"Creating polynomial features with degree {degree} using top {top_n_features} features")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # If we have feature importance, use it to select top features
        if self.feature_importance is not None and len(feature_cols) > top_n_features:
            logger.info(f"Using feature importance to select top {top_n_features} features for polynomial expansion")
            top_features = [col for col, imp in self.feature_importance[:top_n_features]]
            feature_cols = [col for col in feature_cols if col in top_features]
        
        # If still too many features, select a subset
        if len(feature_cols) > top_n_features:
            logger.info(f"Selected random {top_n_features} features for polynomial expansion")
            np.random.seed(42)
            feature_cols = np.random.choice(feature_cols, top_n_features, replace=False).tolist()
        
        logger.info(f"Creating polynomial features using {len(feature_cols)} features")
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        
        # Apply polynomial transformation to the selected features
        logger.info("Applying polynomial transformation")
        try:
            X_poly = poly.fit_transform(df[feature_cols])
            
            # Get feature names
            if hasattr(poly, 'get_feature_names_out'):
                poly_feature_names = poly.get_feature_names_out(feature_cols)
            else:
                poly_feature_names = poly.get_feature_names(feature_cols)
            
            # Remove the original features from the polynomial result
            poly_feature_mask = [name not in feature_cols for name in poly_feature_names]
            X_poly_new = X_poly[:, poly_feature_mask]
            poly_feature_names_new = [name for i, name in enumerate(poly_feature_names) if poly_feature_mask[i]]
            
            # Convert feature names to valid column names
            valid_feature_names = []
            for name in poly_feature_names_new:
                # Replace spaces with underscores and remove parentheses
                valid_name = name.replace(' ', '_').replace('(', '').replace(')', '')
                valid_name = f"poly_{valid_name}"
                valid_feature_names.append(valid_name)
            
            # Add polynomial features to the result dataframe
            poly_df = pd.DataFrame(X_poly_new, columns=valid_feature_names, index=df.index)
            
            # Check if we're approaching memory limit
            if get_memory_usage() > self.max_memory_gb * 0.9:
                logger.warning(f"Approaching memory limit, reducing polynomial features")
                # Keep only a subset of polynomial features
                max_poly_features = int((self.max_memory_gb - get_memory_usage()) * 1e9 // (8 * len(df)))
                if max_poly_features > 0:
                    logger.info(f"Keeping only {max_poly_features} polynomial features")
                    poly_df = poly_df.iloc[:, :max_poly_features]
                else:
                    logger.warning("Not enough memory for polynomial features")
                    return result_df
            
            # Concatenate with original dataframe
            result_df = pd.concat([result_df, poly_df], axis=1)
            
            # Update feature counts
            features_created = poly_df.shape[1]
            self.feature_counts['polynomial'] = features_created
            self.total_features += features_created
            
            # Log memory usage
            final_memory = get_memory_usage()
            logger.info(f"Created {features_created} polynomial features using {final_memory - initial_memory:.2f} GB additional memory")
            log_memory_usage("polynomial feature creation")
            
        except Exception as e:
            logger.error(f"Error creating polynomial features: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def create_statistical_features(self, df, feature_groups=None, functions=['mean', 'std', 'skew', 'kurt', 'quantile']):
        """
        Create statistical features across groups of features.
        
        Args:
            df: DataFrame with data
            feature_groups: List of feature group prefixes, if None, auto-detect groups
            functions: Statistical functions to apply
            
        Returns:
            DataFrame with original and statistical features
        """
        logger.info(f"Creating statistical features using {len(functions) if functions else 0} functions")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Auto-detect feature groups if not provided
        if feature_groups is None:
            # Extract prefixes from column names
            prefixes = set()
            for col in feature_cols:
                parts = col.split('_')
                if len(parts) > 1:
                    prefixes.add(parts[0])
            
            feature_groups = list(prefixes)
            logger.info(f"Auto-detected {len(feature_groups)} feature groups: {feature_groups}")
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # For each feature group and function, create statistical features
        for group in feature_groups:
            # Get columns in this group
            group_cols = [col for col in feature_cols if col.startswith(f"{group}_")]
            
            if not group_cols:
                continue
            
            logger.info(f"Creating statistical features for group {group} with {len(group_cols)} features")
            
            # Apply each statistical function
            for func in functions:
                # Check if we're approaching memory limit
                if get_memory_usage() > self.max_memory_gb:
                    logger.warning(f"Approaching memory limit. Stopping statistical feature creation.")
                    break
                
                # Apply function
                if func == 'mean':
                    result_df[f"stat_{group}_mean"] = df[group_cols].mean(axis=1)
                elif func == 'std':
                    result_df[f"stat_{group}_std"] = df[group_cols].std(axis=1)
                elif func == 'skew':
                    result_df[f"stat_{group}_skew"] = df[group_cols].skew(axis=1)
                elif func == 'kurt':
                    result_df[f"stat_{group}_kurt"] = df[group_cols].kurt(axis=1)
                elif func == 'quantile':
                    # Add multiple quantiles
                    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                        result_df[f"stat_{group}_q{int(q*100)}"] = df[group_cols].quantile(q=q, axis=1)
                        features_created += 1
                
                features_created += 1
        
        # Update feature counts
        self.feature_counts['statistical'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} statistical features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("statistical feature creation")
        
        return result_df
    
    def create_technical_indicators(self, df):
        """
        Create technical indicators for financial features.
        This includes various price and volume indicators adapted for crypto data.
        
        Args:
            df: DataFrame with data
            
        Returns:
            DataFrame with original and technical indicator features
        """
        logger.info("Creating technical indicator features")
        
        # This is a simplified version - in reality we'd check if time series data exists
        # and calculate appropriate indicators
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # Try to identify price and volume related columns
        price_cols = [col for col in df.columns if any(term in col.lower() for term in ['price', 'close', 'value'])]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        if price_cols and len(df) > 20:  # Need enough data for lookback
            logger.info(f"Found {len(price_cols)} price-related columns, creating indicators")
            
            # Create indicators for each price column
            for col in price_cols[:20]:  # Limit to 20 price columns
                # Simple moving averages
                for window in [5, 10, 20]:
                    result_df[f"tech_{col}_sma{window}"] = df[col].rolling(window=window).mean()
                    features_created += 1
                
                # Relative strength index (simplified)
                if len(df) > 14:
                    delta = df[col].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    result_df[f"tech_{col}_rsi"] = 100 - (100 / (1 + rs))
                    features_created += 1
                
                # Bollinger Bands
                if len(df) > 20:
                    sma20 = df[col].rolling(window=20).mean()
                    std20 = df[col].rolling(window=20).std()
                    result_df[f"tech_{col}_bb_upper"] = sma20 + (std20 * 2)
                    result_df[f"tech_{col}_bb_lower"] = sma20 - (std20 * 2)
                    result_df[f"tech_{col}_bb_width"] = (result_df[f"tech_{col}_bb_upper"] - result_df[f"tech_{col}_bb_lower"]) / sma20
                    features_created += 3
        
        if volume_cols and price_cols and len(df) > 20:
            logger.info(f"Found {len(volume_cols)} volume-related columns, creating indicators")
            
            # Create volume indicators
            for vol_col in volume_cols[:10]:  # Limit to 10 volume columns
                # Volume moving averages
                for window in [5, 10, 20]:
                    result_df[f"tech_{vol_col}_sma{window}"] = df[vol_col].rolling(window=window).mean()
                    features_created += 1
                
                # On-balance volume (simplified)
                if len(price_cols) > 0:
                    price_col = price_cols[0]  # Use first price column
                    obv = pd.Series(0, index=df.index)
                    price_delta = df[price_col].diff()
                    obv[price_delta > 0] = df.loc[price_delta > 0, vol_col]
                    obv[price_delta < 0] = -df.loc[price_delta < 0, vol_col]
                    result_df[f"tech_{vol_col}_obv"] = obv.cumsum()
                    features_created += 1
        
        # Fill NaN values created by rolling windows
        result_df = result_df.fillna(0)
        
        # Update feature counts
        self.feature_counts['technical'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} technical indicator features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("technical indicator creation")
        
        return result_df
    
    def create_spectral_features(self, df, top_n_features=100):
        """
        Create spectral features using FFT and wavelets.
        
        Args:
            df: DataFrame with data
            top_n_features: Number of features to use for spectral analysis
            
        Returns:
            DataFrame with original and spectral features
        """
        logger.info(f"Creating spectral features using top {top_n_features} features")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # If we have feature importance, use it to select top features
        if self.feature_importance is not None and len(feature_cols) > top_n_features:
            logger.info(f"Using feature importance to select top {top_n_features} features for spectral analysis")
            top_features = [col for col, imp in self.feature_importance[:top_n_features]]
            feature_cols = [col for col in feature_cols if col in top_features]
        
        # If still too many features, select a subset
        if len(feature_cols) > top_n_features:
            logger.info(f"Selected random {top_n_features} features for spectral analysis")
            np.random.seed(42)
            feature_cols = np.random.choice(feature_cols, top_n_features, replace=False).tolist()
        
        logger.info(f"Creating spectral features using {len(feature_cols)} features")
        
        # Initialize result with a copy of the original dataframe
        result_df = df.copy()
        
        # Track memory usage and feature count
        initial_memory = get_memory_usage()
        features_created = 0
        
        # Apply Fast Fourier Transform to selected features
        try:
            # Only if we have scipy 
            from scipy import fft
            
            logger.info("Applying FFT to selected features")
            
            for col in feature_cols:
                # Check memory
                if get_memory_usage() > self.max_memory_gb:
                    logger.warning(f"Approaching memory limit. Stopping spectral feature creation.")
                    break
                
                # Apply FFT (real part only for simplicity)
                fft_vals = fft.rfft(df[col].fillna(0).values)
                fft_real = np.abs(fft_vals)
                
                # Get top frequencies
                n_freqs = min(10, len(fft_real))
                top_freq_idx = np.argsort(fft_real)[-n_freqs:]
                
                # Create features for top frequencies
                for i, idx in enumerate(top_freq_idx):
                    result_df[f"spectral_{col}_fft_mag{i}"] = fft_real[idx]
                    features_created += 1
                
                # Add aggregate features - normalized power in bands
                if len(fft_real) > 10:
                    # Low frequency band
                    low_band = np.sum(fft_real[:len(fft_real)//5])
                    # Mid frequency band
                    mid_band = np.sum(fft_real[len(fft_real)//5:3*len(fft_real)//5])
                    # High frequency band
                    high_band = np.sum(fft_real[3*len(fft_real)//5:])
                    
                    # Normalized bands
                    total_power = low_band + mid_band + high_band
                    if total_power > 0:
                        result_df[f"spectral_{col}_low_power"] = low_band / total_power
                        result_df[f"spectral_{col}_mid_power"] = mid_band / total_power
                        result_df[f"spectral_{col}_high_power"] = high_band / total_power
                        features_created += 3
        
        except ImportError:
            logger.warning("scipy not available, skipping FFT features")
        
        # Try wavelet features if PyWavelets is available
        try:
            import pywt
            
            logger.info("Creating wavelet features")
            
            # List of wavelets to try
            wavelets = ['db1', 'sym2', 'coif1']
            
            for wavelet in wavelets:
                # For each feature, compute wavelet coefficients
                for col in feature_cols[:20]:  # Limit to 20 features
                    # Check memory
                    if get_memory_usage() > self.max_memory_gb:
                        logger.warning(f"Approaching memory limit. Stopping wavelet feature creation.")
                        break
                    
                    # Ensure data is appropriate length for wavelet transform
                    data = df[col].fillna(0).values
                    
                    # Apply wavelet transform
                    try:
                        coeffs = pywt.wavedec(data, wavelet, level=3)
                        
                        # Create features from coefficients
                        for i, c in enumerate(coeffs):
                            result_df[f"spectral_{col}_{wavelet}_level{i}_mean"] = np.mean(c)
                            result_df[f"spectral_{col}_{wavelet}_level{i}_std"] = np.std(c)
                            features_created += 2
                    except Exception as e:
                        logger.warning(f"Error in wavelet transform for {col}: {e}")
            
        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet features")
        
        # Update feature counts
        self.feature_counts['spectral'] = features_created
        self.total_features += features_created
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Created {features_created} spectral features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("spectral feature creation")
        
        return result_df
    
    def apply_feature_selection(self, df, target_col=None, max_features=5000, methods=['variance', 'correlation', 'importance']):
        """
        Apply feature selection to reduce dimensionality.
        
        Args:
            df: DataFrame with data
            target_col: Name of target column, if available
            max_features: Maximum number of features to keep
            methods: Feature selection methods to apply
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Applying feature selection to reduce from {df.shape[1]} to max {max_features} features")
        
        # Get feature columns (exclude symbol and other non-feature columns)
        exclude_cols = ['symbol', 'date', 'era', 'id']
        if target_col:
            exclude_cols.append(target_col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Starting with {len(feature_cols)} features")
        
        # If already fewer features than max, return as is
        if len(feature_cols) <= max_features:
            logger.info(f"Already fewer than {max_features} features, skipping selection")
            return df
        
        # Track memory usage
        initial_memory = get_memory_usage()
        
        # Apply selection methods
        selected_features = set(feature_cols)
        
        # 1. Variance Threshold
        if 'variance' in methods:
            logger.info("Applying variance threshold selection")
            
            # Initialize selector
            var_selector = VarianceThreshold(threshold=0.0)
            
            # Fit and transform
            X = df[feature_cols]
            var_selector.fit(X)
            
            # Get selected features
            variance_features = [feature_cols[i] for i, selected in enumerate(var_selector.get_support()) if selected]
            logger.info(f"Variance threshold selected {len(variance_features)} features")
            
            # Update selected features
            selected_features = selected_features.intersection(variance_features)
        
        # 2. Correlation Filter
        if 'correlation' in methods and len(selected_features) > max_features:
            logger.info("Applying correlation-based selection")
            
            # Convert set to list
            current_features = list(selected_features)
            
            # Calculate correlation matrix
            X = df[current_features]
            corr_matrix = X.corr().abs()
            
            # Create upper triangle mask
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            threshold = 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            logger.info(f"Correlation filter removed {len(to_drop)} features")
            
            # Update selected features
            selected_features = selected_features.difference(to_drop)
        
        # 3. Feature Importance from previous step
        if 'importance' in methods and self.feature_importance is not None and len(selected_features) > max_features:
            logger.info("Applying importance-based selection")
            
            # Extract most important features
            important_features = [col for col, imp in self.feature_importance[:max_features]]
            
            # Update selected features
            selected_features = selected_features.intersection(important_features)
        
        # 4. If still too many features, select randomly
        if len(selected_features) > max_features:
            logger.info(f"Still {len(selected_features)} features, randomly selecting {max_features}")
            
            # Convert to list and randomly select
            current_features = list(selected_features)
            np.random.seed(42)
            selected_features = set(np.random.choice(current_features, max_features, replace=False))
        
        # Convert back to list
        selected_features = list(selected_features)
        
        # Add back essential columns
        selected_features += [col for col in df.columns if col not in feature_cols]
        
        # Return dataframe with selected features
        result_df = df[selected_features].copy()
        
        # Log memory usage
        final_memory = get_memory_usage()
        logger.info(f"Reduced to {len(selected_features)} features using {final_memory - initial_memory:.2f} GB additional memory")
        log_memory_usage("feature selection")
        
        return result_df
    
    def generate_features(self, input_file, output_file=None, feature_modules=None):
        """
        Generate all features in a pipeline.
        
        Args:
            input_file: Path to input parquet file
            output_file: Path to output parquet file
            feature_modules: List of feature generation modules to run
            
        Returns:
            DataFrame with all generated features
        """
        logger.info(f"Starting feature generation pipeline from {input_file}")
        
        # Set default feature modules if not provided
        if feature_modules is None:
            feature_modules = [
                'rolling', 
                'ewm', 
                'technical', 
                'statistical', 
                'interaction', 
                'polynomial', 
                'spectral'
            ]
        
        # Set default output file if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f"high_mem_features_{timestamp}.parquet")
        
        # Load data
        df = self.load_data(input_file)
        
        if df is None:
            logger.error("Failed to load data, aborting feature generation")
            return None
        
        # Track memory usage
        start_memory = get_memory_usage()
        
        # Keep symbols for later
        if 'symbol' in df.columns:
            symbols = df['symbol'].copy()
        else:
            symbols = None
        
        # Apply each feature generation module
        if 'rolling' in feature_modules:
            logger.info("Running rolling feature generation")
            df = self.create_rolling_features(df)
        
        if 'ewm' in feature_modules:
            logger.info("Running exponential weighted moving feature generation")
            df = self.create_ewm_features(df)
        
        if 'technical' in feature_modules:
            logger.info("Running technical indicator feature generation")
            df = self.create_technical_indicators(df)
        
        if 'statistical' in feature_modules:
            logger.info("Running statistical feature generation")
            df = self.create_statistical_features(df)
        
        # Run feature selection after initial feature creation
        if df.shape[1] > 10000:
            logger.info(f"Running intermediate feature selection (current: {df.shape[1]} features)")
            df = self.apply_feature_selection(df, max_features=10000)
        
        # Continue with more complex features
        if 'interaction' in feature_modules:
            logger.info("Running interaction feature generation")
            df = self.create_interaction_features(df)
        
        if 'polynomial' in feature_modules:
            logger.info("Running polynomial feature generation")
            df = self.create_polynomial_features(df)
        
        if 'spectral' in feature_modules:
            logger.info("Running spectral feature generation")
            df = self.create_spectral_features(df)
        
        # Final feature selection
        logger.info(f"Running final feature selection (current: {df.shape[1]} features)")
        df = self.apply_feature_selection(df, max_features=5000)
        
        # Add symbol column back if needed
        if symbols is not None and 'symbol' not in df.columns:
            df['symbol'] = symbols
        
        # Save generated features
        logger.info(f"Saving {df.shape[1]} features to {output_file}")
        try:
            # Use PyArrow for faster saving
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_file)
            logger.info(f"Features saved successfully to {output_file}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
        
        # Log feature generation summary
        logger.info("\n=== FEATURE GENERATION SUMMARY ===")
        logger.info(f"Input shape: {self.feature_counts['base']} features")
        logger.info(f"Output shape: {df.shape[1]} features")
        logger.info(f"Total features generated: {self.total_features}")
        logger.info(f"  - Rolling features: {self.feature_counts['rolling']}")
        logger.info(f"  - EWM features: {self.feature_counts['ewm']}")
        logger.info(f"  - Interaction features: {self.feature_counts['interaction']}")
        logger.info(f"  - Polynomial features: {self.feature_counts['polynomial']}")
        logger.info(f"  - Technical features: {self.feature_counts['technical']}")
        logger.info(f"  - Statistical features: {self.feature_counts['statistical']}")
        logger.info(f"  - Spectral features: {self.feature_counts['spectral']}")
        logger.info(f"Memory usage: {get_memory_usage() - start_memory:.2f} GB")
        
        # Save feature generation info
        feature_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': input_file,
            'output_file': output_file,
            'input_shape': (df.shape[0], self.feature_counts['base']),
            'output_shape': df.shape,
            'feature_counts': self.feature_counts,
            'memory_usage_gb': get_memory_usage() - start_memory
        }
        
        info_file = os.path.join(self.output_dir, f"feature_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(info_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        return df

def main():
    """Main function to run high memory feature generation"""
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate high memory features for Numerai Crypto')
    parser.add_argument('--input', type=str, help='Input parquet file path')
    parser.add_argument('--output', type=str, help='Output parquet file path')
    parser.add_argument('--modules', type=str, nargs='+', help='Feature generation modules to run')
    parser.add_argument('--max-memory', type=int, default=500, help='Maximum memory to use in GB')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Find input file if not provided
    if args.input is None:
        # Try to find processed yiedl data
        processed_dir = os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed')
        processed_files = [f for f in os.listdir(processed_dir) 
                          if f.startswith('processed_yiedl_') and f.endswith('.parquet')]
        
        if processed_files:
            # Sort by timestamp and take the most recent
            processed_files.sort(reverse=True)
            args.input = os.path.join(processed_dir, processed_files[0])
            logger.info(f"Using latest processed file: {args.input}")
        else:
            logger.error("No input file provided and no processed files found")
            return 1
    
    # Create feature generator
    feature_gen = HighMemoryFeatureGenerator(
        output_dir=args.output_dir,
        max_memory_gb=args.max_memory,
        n_jobs=args.n_jobs
    )
    
    # Generate features
    df = feature_gen.generate_features(
        input_file=args.input,
        output_file=args.output,
        feature_modules=args.modules
    )
    
    if df is not None:
        logger.info("Feature generation completed successfully")
        return 0
    else:
        logger.error("Feature generation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())