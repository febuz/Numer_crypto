#!/usr/bin/env python3
"""
Feature Extractor Module for Numer_crypto
Handles feature extraction and preprocessing for predictions.
"""

import os
import sys
import logging
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.log_utils import setup_logging

logger = setup_logging(__name__)


class FeatureExtractor:
    """
    Handles feature extraction and preprocessing for the Numerai crypto pipeline.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.excluded_cols = ['symbol', 'date', 'era', 'id', 'target']
        logger.info("FeatureExtractor initialized")
    
    def extract_features(self, df: pl.DataFrame, target_symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Extract features from the DataFrame.
        
        Args:
            df: Input Polars DataFrame
            target_symbols: Optional list of symbols to filter for
            
        Returns:
            Polars DataFrame with extracted features
        """
        try:
            logger.info(f"Extracting features from DataFrame with shape: {df.shape}")
            
            # Filter for target symbols if provided
            if target_symbols and "symbol" in df.columns:
                df = df.filter(pl.col("symbol").is_in(target_symbols))
                logger.info(f"Filtered to {df.height} rows for {len(target_symbols)} symbols")
            
            # Get feature columns (exclude metadata columns)
            feature_cols = [col for col in df.columns if col not in self.excluded_cols]
            logger.info(f"Found {len(feature_cols)} feature columns")
            
            # Select feature columns plus necessary metadata
            metadata_cols = [col for col in ['symbol', 'date', 'era', 'id'] if col in df.columns]
            selected_cols = metadata_cols + feature_cols
            
            result_df = df.select(selected_cols)
            logger.info(f"Extracted features with shape: {result_df.shape}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return df
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Get the list of feature columns from a DataFrame.
        
        Args:
            df: Input Polars DataFrame
            
        Returns:
            List of feature column names
        """
        try:
            feature_cols = [col for col in df.columns if col not in self.excluded_cols]
            logger.info(f"Identified {len(feature_cols)} feature columns")
            return feature_cols
            
        except Exception as e:
            logger.error(f"Error getting feature columns: {e}")
            return []
    
    def prepare_for_prediction(self, df: pl.DataFrame, model_features: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Prepare features for model prediction.
        
        Args:
            df: Input Polars DataFrame with features
            model_features: Optional list of required model features
            
        Returns:
            Polars DataFrame ready for prediction
        """
        try:
            logger.info(f"Preparing features for prediction from DataFrame with shape: {df.shape}")
            
            # Get available feature columns
            available_features = self.get_feature_columns(df)
            
            if model_features:
                # Check which model features are available
                available_model_features = [col for col in model_features if col in available_features]
                missing_features = [col for col in model_features if col not in available_features]
                
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} required features: {missing_features[:10]}...")
                
                # Use available model features
                use_features = available_model_features
                logger.info(f"Using {len(use_features)} model features out of {len(model_features)} required")
            else:
                # Use all available features
                use_features = available_features
                logger.info(f"Using all {len(use_features)} available features")
            
            # Select metadata columns plus features
            metadata_cols = [col for col in ['symbol', 'date', 'era', 'id'] if col in df.columns]
            selected_cols = metadata_cols + use_features
            
            result_df = df.select(selected_cols)
            
            # Handle missing values
            result_df = self._handle_missing_values(result_df, use_features)
            
            logger.info(f"Prepared features with shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing features for prediction: {e}")
            return df
    
    def _handle_missing_values(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """
        Handle missing values in feature columns.
        
        Args:
            df: Input Polars DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Polars DataFrame with missing values handled
        """
        try:
            # Fill NaN values with 0 for numeric columns
            fill_exprs = []
            for col in df.columns:
                if col in feature_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    fill_exprs.append(pl.col(col).fill_null(0.0).fill_nan(0.0))
                else:
                    fill_exprs.append(pl.col(col))
            
            if fill_exprs:
                df = df.select(fill_exprs)
                logger.info("Filled missing values with 0.0")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df
    
    def get_latest_data_for_symbols(self, symbols: List[str], df: pl.DataFrame) -> pl.DataFrame:
        """
        Get the latest data for specified symbols.
        
        Args:
            symbols: List of symbol names
            df: Input DataFrame
            
        Returns:
            DataFrame with latest data for each symbol
        """
        try:
            if "symbol" not in df.columns:
                logger.warning("No 'symbol' column found")
                return df
            
            # Filter for target symbols
            filtered_df = df.filter(pl.col("symbol").is_in(symbols))
            
            if "date" in filtered_df.columns:
                # Get latest date for each symbol
                latest_df = (
                    filtered_df
                    .group_by("symbol")
                    .agg(pl.col("date").max().alias("max_date"))
                    .join(filtered_df, left_on=["symbol", "max_date"], right_on=["symbol", "date"])
                    .drop("max_date")
                )
                logger.info(f"Got latest data for {latest_df.height} symbol entries")
            else:
                # If no date column, just return filtered data
                latest_df = filtered_df
                logger.info(f"Got data for {latest_df.height} symbol entries (no date filtering)")
            
            return latest_df
            
        except Exception as e:
            logger.error(f"Error getting latest data for symbols: {e}")
            return df.filter(pl.col("symbol").is_in(symbols)) if "symbol" in df.columns else df
    
    def filter_features(self, df: pl.DataFrame, required_features: List[str]) -> pl.DataFrame:
        """
        Filter DataFrame to include only required features plus metadata columns.
        
        Args:
            df: Input Polars DataFrame
            required_features: List of required feature column names
            
        Returns:
            Polars DataFrame with only required features and metadata
        """
        try:
            logger.info(f"Filtering features to {len(required_features)} required features")
            
            # Get metadata columns that exist in the DataFrame
            metadata_cols = [col for col in ['symbol', 'date', 'era', 'id'] if col in df.columns]
            
            # Get available required features
            available_features = [col for col in required_features if col in df.columns]
            missing_features = [col for col in required_features if col not in df.columns]
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} required features: {missing_features[:10]}...")
            
            # Select metadata columns plus available required features
            selected_cols = metadata_cols + available_features
            
            if not selected_cols:
                logger.error("No valid columns to select")
                return df
            
            result_df = df.select(selected_cols)
            logger.info(f"Filtered features from {len(df.columns)} to {len(result_df.columns)} columns")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error filtering features: {e}")
            return df
    
    def validate_features(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate the feature DataFrame.
        
        Args:
            df: Input Polars DataFrame
            
        Returns:
            Dictionary with validation results
        """
        try:
            feature_cols = self.get_feature_columns(df)
            
            validation = {
                "total_rows": df.height,
                "total_columns": len(df.columns),
                "feature_columns": len(feature_cols),
                "has_symbol_col": "symbol" in df.columns,
                "has_date_col": "date" in df.columns,
                "missing_values": {},
                "dtype_issues": []
            }
            
            # Check for missing values
            for col in feature_cols:
                null_count = df[col].null_count()
                if null_count > 0:
                    validation["missing_values"][col] = null_count
            
            # Check data types
            for col in feature_cols:
                if df[col].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    validation["dtype_issues"].append(f"{col}: {df[col].dtype}")
            
            # Symbol information
            if "symbol" in df.columns:
                symbols = df["symbol"].unique().to_list()
                validation["symbols"] = len(symbols)
                validation["sample_symbols"] = symbols[:10]
            
            logger.info(f"Feature validation completed: {validation}")
            return validation
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    # Create sample data for testing
    import polars as pl
    
    sample_data = pl.DataFrame({
        "symbol": ["BTC", "ETH", "BTC", "ETH"],
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "feature_2": [0.5, 1.5, 2.5, 3.5],
        "target": [0.1, 0.2, 0.3, 0.4]
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test feature extraction
    features = extractor.extract_features(sample_data, target_symbols=["BTC"])
    print(f"Extracted features shape: {features.shape}")
    
    # Test feature column identification
    feature_cols = extractor.get_feature_columns(sample_data)
    print(f"Feature columns: {feature_cols}")
    
    # Test validation
    validation = extractor.validate_features(sample_data)
    print(f"Validation results: {validation}")