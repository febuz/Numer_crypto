#!/usr/bin/env python3
"""
Feature Store for Numerai Crypto

This module provides a feature store for persistent storage of computed features:
- Store and retrieve feature sets
- Track feature metadata and statistics
- Cache computed features for faster reuse
- Version and timestamp feature data
- Store feature importance data from trained models

Usage:
    # Import and initialize
    from feature_store import FeatureStore
    store = FeatureStore('/path/to/store')
    
    # Store features
    store.store_features(feature_df, 'polynomial_features_v1', metadata={'degree': 2})
    
    # Retrieve features
    features_df = store.get_features('polynomial_features_v1')
"""

import os
import json
import shutil
import logging
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Persistent store for features and their metadata
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the feature store
        
        Args:
            base_dir: Base directory for the feature store
        """
        self.base_dir = Path(base_dir)
        self.features_dir = self.base_dir / "features"
        self.metadata_dir = self.base_dir / "metadata"
        self.tmp_dir = self.base_dir / "tmp"
        
        # Create directories if they don't exist
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Load feature registry
        self.registry_path = self.base_dir / "feature_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"Feature store initialized at {self.base_dir}")
        logger.info(f"Found {len(self.registry)} feature sets in registry")
    
    def _load_registry(self) -> Dict:
        """Load the feature registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading registry: {e}. Creating new registry.")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the feature registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _get_feature_path(self, feature_set_name: str) -> Path:
        """Get the path for feature data"""
        return self.features_dir / f"{feature_set_name}.parquet"
    
    def _get_metadata_path(self, feature_set_name: str) -> Path:
        """Get the path for feature metadata"""
        return self.metadata_dir / f"{feature_set_name}.json"
    
    def _compute_features_hash(self, df: pd.DataFrame) -> str:
        """
        Compute a hash of the dataframe to use as a unique identifier
        Uses a sample of the data for large dataframes
        """
        # For large dataframes, sample data to compute hash
        if len(df) > 10000:
            sample = df.sample(n=10000, random_state=42)
        else:
            sample = df
        
        # Convert to string and hash
        sample_str = pd.util.hash_pandas_object(sample).sum()
        return hashlib.md5(str(sample_str).encode()).hexdigest()
    
    def _compute_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Compute basic statistics for features"""
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Compute statistics
        stats = {}
        for col in numeric_cols:
            col_stats = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isnull().sum())
            }
            stats[col] = col_stats
        
        return stats
    
    def store_features(self, 
                       df: pd.DataFrame, 
                       feature_set_name: str, 
                       metadata: Optional[Dict] = None,
                       overwrite: bool = False) -> bool:
        """
        Store features in the feature store
        
        Args:
            df: DataFrame containing features
            feature_set_name: Name for this feature set
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing feature set
            
        Returns:
            bool: Success or failure
        """
        feature_path = self._get_feature_path(feature_set_name)
        metadata_path = self._get_metadata_path(feature_set_name)
        
        # Check if feature set already exists
        if feature_path.exists() and not overwrite:
            logger.warning(f"Feature set '{feature_set_name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Compute feature hash
        features_hash = self._compute_features_hash(df)
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            'name': feature_set_name,
            'created_at': datetime.now().isoformat(),
            'num_samples': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'hash': features_hash
        })
        
        # Add computed statistics
        try:
            meta['statistics'] = self._compute_feature_stats(df)
        except Exception as e:
            logger.warning(f"Could not compute statistics: {e}")
        
        # Save to temporary locations first
        tmp_feature_path = self.tmp_dir / f"{feature_set_name}_{int(datetime.now().timestamp())}.parquet"
        tmp_metadata_path = self.tmp_dir / f"{feature_set_name}_{int(datetime.now().timestamp())}.json"
        
        try:
            # Save feature data
            df.to_parquet(tmp_feature_path)
            
            # Save metadata
            with open(tmp_metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Move to final locations
            shutil.move(str(tmp_feature_path), str(feature_path))
            shutil.move(str(tmp_metadata_path), str(metadata_path))
            
            # Update registry
            self.registry[feature_set_name] = {
                'path': str(feature_path),
                'metadata_path': str(metadata_path),
                'created_at': meta['created_at'],
                'hash': features_hash
            }
            self._save_registry()
            
            logger.info(f"Stored feature set '{feature_set_name}' with {len(df)} samples and {len(df.columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features '{feature_set_name}': {e}")
            # Clean up temporary files
            if tmp_feature_path.exists():
                tmp_feature_path.unlink()
            if tmp_metadata_path.exists():
                tmp_metadata_path.unlink()
            return False
    
    def get_features(self, feature_set_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve features from the store
        
        Args:
            feature_set_name: Name of the feature set to retrieve
            
        Returns:
            DataFrame containing the features or None if not found
        """
        feature_path = self._get_feature_path(feature_set_name)
        
        if not feature_path.exists():
            logger.warning(f"Feature set '{feature_set_name}' not found")
            return None
        
        try:
            df = pd.read_parquet(feature_path)
            logger.info(f"Retrieved feature set '{feature_set_name}' with {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error retrieving feature set '{feature_set_name}': {e}")
            return None
    
    def get_metadata(self, feature_set_name: str) -> Optional[Dict]:
        """
        Retrieve metadata for a feature set
        
        Args:
            feature_set_name: Name of the feature set
            
        Returns:
            Dictionary with metadata or None if not found
        """
        metadata_path = self._get_metadata_path(feature_set_name)
        
        if not metadata_path.exists():
            logger.warning(f"Metadata for feature set '{feature_set_name}' not found")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving metadata for feature set '{feature_set_name}': {e}")
            return None
    
    def list_feature_sets(self) -> List[Dict]:
        """
        List all available feature sets with basic metadata
        
        Returns:
            List of dictionaries with feature set information
        """
        result = []
        for name, info in self.registry.items():
            result.append({
                'name': name,
                'created_at': info.get('created_at', 'Unknown'),
                'hash': info.get('hash', 'Unknown')
            })
        return result
    
    def feature_set_exists(self, feature_set_name: str) -> bool:
        """
        Check if a feature set exists
        
        Args:
            feature_set_name: Name of the feature set
            
        Returns:
            True if the feature set exists, False otherwise
        """
        return feature_set_name in self.registry
    
    def delete_feature_set(self, feature_set_name: str) -> bool:
        """
        Delete a feature set and its metadata
        
        Args:
            feature_set_name: Name of the feature set to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        feature_path = self._get_feature_path(feature_set_name)
        metadata_path = self._get_metadata_path(feature_set_name)
        
        try:
            if feature_path.exists():
                feature_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            if feature_set_name in self.registry:
                del self.registry[feature_set_name]
                self._save_registry()
            
            logger.info(f"Deleted feature set '{feature_set_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting feature set '{feature_set_name}': {e}")
            return False
    
    def store_feature_importance(self, 
                                feature_set_name: str, 
                                importance_dict: Dict[str, float],
                                model_name: str) -> bool:
        """
        Store feature importance for a feature set
        
        Args:
            feature_set_name: Name of the feature set
            importance_dict: Dictionary mapping feature names to importance values
            model_name: Name of the model that generated the importance values
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.feature_set_exists(feature_set_name):
            logger.warning(f"Feature set '{feature_set_name}' does not exist")
            return False
        
        metadata = self.get_metadata(feature_set_name)
        if not metadata:
            return False
        
        # Initialize importance field if it doesn't exist
        if 'feature_importance' not in metadata:
            metadata['feature_importance'] = {}
        
        # Add model importance
        metadata['feature_importance'][model_name] = {
            'timestamp': datetime.now().isoformat(),
            'importance': importance_dict
        }
        
        # Save updated metadata
        metadata_path = self._get_metadata_path(feature_set_name)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Stored feature importance for '{feature_set_name}' from model '{model_name}'")
            return True
        except Exception as e:
            logger.error(f"Error storing feature importance: {e}")
            return False
    
    def get_top_features(self, 
                         feature_set_name: str, 
                         model_name: str = None, 
                         top_n: int = 100) -> List[str]:
        """
        Get the top N most important features
        
        Args:
            feature_set_name: Name of the feature set
            model_name: Name of the model (if None, average across all models)
            top_n: Number of top features to return
            
        Returns:
            List of feature names sorted by importance
        """
        metadata = self.get_metadata(feature_set_name)
        if not metadata or 'feature_importance' not in metadata:
            logger.warning(f"No feature importance data for '{feature_set_name}'")
            return []
        
        if model_name:
            # Get importance from specific model
            if model_name not in metadata['feature_importance']:
                logger.warning(f"No feature importance data for model '{model_name}'")
                return []
            
            importance = metadata['feature_importance'][model_name]['importance']
        else:
            # Average across all models
            all_models = metadata['feature_importance'].keys()
            if not all_models:
                return []
            
            # Initialize with zeros
            first_model = next(iter(all_models))
            features = metadata['feature_importance'][first_model]['importance'].keys()
            importance = {feature: 0.0 for feature in features}
            
            # Sum importance across models
            for model in all_models:
                model_imp = metadata['feature_importance'][model]['importance']
                for feature, value in model_imp.items():
                    if feature in importance:
                        importance[feature] += value
            
            # Average
            for feature in importance:
                importance[feature] /= len(all_models)
        
        # Sort by importance and get top N
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:top_n]]


if __name__ == "__main__":
    """Test the feature store functionality"""
    import tempfile
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize feature store
        store = FeatureStore(temp_dir)
        
        # Create a test dataframe
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        
        # Store features
        store.store_features(df, 'test_features', metadata={'source': 'test'})
        
        # List feature sets
        feature_sets = store.list_feature_sets()
        print("Feature sets:", feature_sets)
        
        # Get features
        retrieved_df = store.get_features('test_features')
        print("Retrieved features shape:", retrieved_df.shape)
        
        # Get metadata
        metadata = store.get_metadata('test_features')
        print("Metadata:", metadata)
        
        # Store feature importance
        importance = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        store.store_feature_importance('test_features', importance, 'test_model')
        
        # Get top features
        top_features = store.get_top_features('test_features', 'test_model', 2)
        print("Top features:", top_features)
        
        print("Feature store test completed successfully")