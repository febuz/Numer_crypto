#!/usr/bin/env python3
"""
Feature Store Module for Numer_crypto
Handles storage and retrieval of feature data for predictions.
"""

import os
import sys
import logging
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.log_utils import setup_logging

logger = setup_logging(__name__)


class FeatureStore:
    """
    Manages feature storage and retrieval for the Numerai crypto pipeline.
    """
    
    def __init__(self, base_dir: str = "/media/knight2/EDB/numer_crypto_temp/data"):
        """
        Initialize the feature store.
        
        Args:
            base_dir: Base directory for feature data storage
        """
        self.base_dir = Path(base_dir)
        self.features_dir = self.base_dir / "features"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories if they don't exist
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FeatureStore initialized with base_dir: {self.base_dir}")
    
    def get_latest_features(self, feature_type: str = "gpu_features") -> Optional[pl.DataFrame]:
        """
        Get the latest feature data.
        
        Args:
            feature_type: Type of features to retrieve
            
        Returns:
            Polars DataFrame with features or None if not found
        """
        try:
            # Look for feature files
            feature_files = [
                self.features_dir / f"{feature_type}.parquet",
                self.features_dir / "polars_features.parquet", 
                self.features_dir / "features.parquet",
                self.processed_dir / "crypto_train.parquet",
                self.processed_dir / "crypto_train_features.parquet"
            ]
            
            for feature_file in feature_files:
                if feature_file.exists():
                    logger.info(f"Loading features from: {feature_file}")
                    df = pl.read_parquet(str(feature_file))
                    logger.info(f"Loaded features with shape: {df.shape}")
                    return df
            
            logger.warning("No feature files found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None
    
    def get_features_for_symbols(self, symbols: List[str], feature_type: str = "gpu_features") -> Optional[pl.DataFrame]:
        """
        Get features for specific symbols.
        
        Args:
            symbols: List of crypto symbols
            feature_type: Type of features to retrieve
            
        Returns:
            Polars DataFrame with features for specified symbols
        """
        try:
            df = self.get_latest_features(feature_type)
            if df is None:
                return None
            
            # Filter for specified symbols
            if "symbol" in df.columns:
                filtered_df = df.filter(pl.col("symbol").is_in(symbols))
                logger.info(f"Filtered features to {filtered_df.shape[0]} rows for {len(symbols)} symbols")
                return filtered_df
            else:
                logger.warning("No 'symbol' column found in features")
                return df
                
        except Exception as e:
            logger.error(f"Error filtering features for symbols: {e}")
            return None
    
    def get_live_features(self) -> Optional[pl.DataFrame]:
        """
        Get the most recent features for live prediction.
        
        Returns:
            Polars DataFrame with latest features
        """
        try:
            # Look for live feature files first
            live_files = [
                self.features_dir / "live_features.parquet",
                self.features_dir / "latest_features.parquet"
            ]
            
            for live_file in live_files:
                if live_file.exists():
                    logger.info(f"Loading live features from: {live_file}")
                    return pl.read_parquet(str(live_file))
            
            # Fall back to latest training features
            logger.info("No live features found, using latest training features")
            return self.get_latest_features()
            
        except Exception as e:
            logger.error(f"Error loading live features: {e}")
            return None
    
    def save_features(self, df: pl.DataFrame, feature_type: str = "features") -> bool:
        """
        Save features to the feature store.
        
        Args:
            df: Polars DataFrame with features
            feature_type: Type identifier for the features
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = self.features_dir / f"{feature_type}.parquet"
            df.write_parquet(str(output_file))
            logger.info(f"Saved features to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False
    
    def list_available_features(self) -> List[str]:
        """
        List all available feature files.
        
        Returns:
            List of feature file names
        """
        try:
            feature_files = []
            
            # Check features directory
            if self.features_dir.exists():
                for file_path in self.features_dir.glob("*.parquet"):
                    feature_files.append(file_path.name)
            
            # Check processed directory
            if self.processed_dir.exists():
                for file_path in self.processed_dir.glob("*features*.parquet"):
                    feature_files.append(f"processed/{file_path.name}")
            
            logger.info(f"Found {len(feature_files)} feature files")
            return feature_files
            
        except Exception as e:
            logger.error(f"Error listing feature files: {e}")
            return []
    
    def list_feature_sets(self) -> List[str]:
        """
        List all available feature sets.
        
        Returns:
            List of feature set names (without .parquet extension)
        """
        try:
            feature_sets = []
            
            # Check features directory
            if self.features_dir.exists():
                for file_path in self.features_dir.glob("*.parquet"):
                    feature_sets.append(file_path.stem)  # Remove .parquet extension
            
            # Check processed directory
            if self.processed_dir.exists():
                for file_path in self.processed_dir.glob("*features*.parquet"):
                    feature_sets.append(f"processed/{file_path.stem}")
            
            logger.info(f"Found {len(feature_sets)} feature sets")
            return feature_sets
            
        except Exception as e:
            logger.error(f"Error listing feature sets: {e}")
            return []
    
    def get_feature_set(self, feature_set_id: str) -> Optional[pl.DataFrame]:
        """
        Get a specific feature set by ID.
        
        Args:
            feature_set_id: Feature set identifier
            
        Returns:
            Polars DataFrame with the feature set or None if not found
        """
        try:
            # Try to load the feature set by name
            feature_files = [
                self.features_dir / f"{feature_set_id}.parquet",
                self.features_dir / f"{feature_set_id}_features.parquet", 
                self.processed_dir / f"{feature_set_id}.parquet",
                self.processed_dir / f"{feature_set_id}_features.parquet"
            ]
            
            for feature_file in feature_files:
                if feature_file.exists():
                    logger.info(f"Loading feature set {feature_set_id} from: {feature_file}")
                    df = pl.read_parquet(str(feature_file))
                    logger.info(f"Loaded feature set with shape: {df.shape}")
                    return df
            
            logger.warning(f"Feature set {feature_set_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading feature set {feature_set_id}: {e}")
            return None
    
    def get_feature_info(self, feature_type: str = "gpu_features") -> Dict[str, Any]:
        """
        Get information about stored features.
        
        Args:
            feature_type: Type of features to analyze
            
        Returns:
            Dictionary with feature information
        """
        try:
            df = self.get_latest_features(feature_type)
            if df is None:
                return {"error": "No features found"}
            
            info = {
                "shape": df.shape,
                "columns": len(df.columns),
                "rows": df.height,
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
            }
            
            # Add symbol information if available
            if "symbol" in df.columns:
                symbols = df["symbol"].unique().to_list()
                info["symbols"] = len(symbols)
                info["sample_symbols"] = symbols[:10]
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting feature info: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the feature store
    store = FeatureStore()
    
    # List available features
    available = store.list_available_features()
    print(f"Available features: {available}")
    
    # Get feature info
    info = store.get_feature_info()
    print(f"Feature info: {info}")
    
    # Try to load latest features
    df = store.get_latest_features()
    if df is not None:
        print(f"Loaded features with shape: {df.shape}")
    else:
        print("No features could be loaded")