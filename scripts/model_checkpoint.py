#!/usr/bin/env python3
"""
Model Checkpoint System for Numerai Crypto

This module provides a checkpoint system for saving and loading models:
- Store models with metadata
- Track training history and performance metrics
- Version models with validation results
- Support for multiple model types including LightGBM, H2O XGBoost, etc.
- Compare model performance across checkpoints

Usage:
    # Import and initialize
    from model_checkpoint import ModelCheckpoint
    checkpoint = ModelCheckpoint('/path/to/checkpoints')
    
    # Save a model checkpoint
    checkpoint.save_model(model, 'lightgbm_v1', model_type='lightgbm', 
                          metrics={'validation_auc': 0.78})
    
    # Load a model checkpoint
    model = checkpoint.load_model('lightgbm_v1')
"""

import os
import json
import pickle
import logging
import shutil
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """
    Checkpoint system for saving and loading models with metadata
    """
    
    # Supported model types and their save/load handlers
    MODEL_HANDLERS = {
        'lightgbm': {
            'save': lambda model, path: model.model.save_model(path),
            'load': lambda path: None  # Loaded differently
        },
        'h2o_xgboost': {
            'save': lambda model, path: pickle.dump(model, open(path, 'wb')),
            'load': lambda path: pickle.load(open(path, 'rb'))
        },
        'h2o_automl': {
            'save': lambda model, path: pickle.dump(model, open(path, 'wb')),
            'load': lambda path: pickle.load(open(path, 'rb'))
        },
        'generic': {
            'save': lambda model, path: pickle.dump(model, open(path, 'wb')),
            'load': lambda path: pickle.load(open(path, 'rb'))
        }
    }
    
    def __init__(self, base_dir: str):
        """
        Initialize the model checkpoint system
        
        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.tmp_dir = self.base_dir / "tmp"
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Load model registry
        self.registry_path = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"Model checkpoint system initialized at {self.base_dir}")
        logger.info(f"Found {len(self.registry)} models in registry")
    
    def _load_registry(self) -> Dict:
        """Load the model registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading registry: {e}. Creating new registry.")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the model registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path for the model data"""
        return self.models_dir / f"{model_name}.model"
    
    def _get_metadata_path(self, model_name: str) -> Path:
        """Get the path for the model metadata"""
        return self.metadata_dir / f"{model_name}.json"
    
    @contextmanager
    def _temp_paths(self, model_name: str):
        """Create temporary paths for atomic writes"""
        timestamp = int(datetime.now().timestamp())
        tmp_model_path = self.tmp_dir / f"{model_name}_{timestamp}.model"
        tmp_metadata_path = self.tmp_dir / f"{model_name}_{timestamp}.json"
        
        try:
            yield tmp_model_path, tmp_metadata_path
        finally:
            # Clean up any leftover temporary files
            if tmp_model_path.exists():
                tmp_model_path.unlink()
            if tmp_metadata_path.exists():
                tmp_metadata_path.unlink()
    
    def save_model(self, 
                   model: Any, 
                   model_name: str, 
                   model_type: str = 'generic',
                   metrics: Optional[Dict[str, float]] = None,
                   params: Optional[Dict] = None,
                   feature_names: Optional[List[str]] = None,
                   feature_importance: Optional[Dict[str, float]] = None,
                   overwrite: bool = False) -> bool:
        """
        Save a model checkpoint
        
        Args:
            model: The model object to save
            model_name: Unique name for this model
            model_type: Type of model (lightgbm, h2o_xgboost, h2o_automl, generic)
            metrics: Dictionary of performance metrics
            params: Model parameters/hyperparameters
            feature_names: List of feature names used by the model
            feature_importance: Dictionary mapping feature names to importance values
            overwrite: Whether to overwrite an existing model with the same name
            
        Returns:
            bool: Success or failure
        """
        model_path = self._get_model_path(model_name)
        metadata_path = self._get_metadata_path(model_name)
        
        # Check if model already exists
        if model_path.exists() and not overwrite:
            logger.warning(f"Model '{model_name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Check if model type is supported
        if model_type not in self.MODEL_HANDLERS:
            logger.warning(f"Unsupported model type: {model_type}. Falling back to generic.")
            model_type = 'generic'
        
        # Prepare metadata
        meta = {
            'name': model_name,
            'created_at': datetime.now().isoformat(),
            'model_type': model_type,
            'metrics': metrics or {},
            'params': params or {},
            'feature_names': feature_names or [],
            'feature_importance': feature_importance or {}
        }
        
        # Use temporary paths for atomic write
        with self._temp_paths(model_name) as (tmp_model_path, tmp_metadata_path):
            try:
                # Save model using the appropriate handler
                if model_type == 'lightgbm':
                    # Special case for LightGBM models
                    if hasattr(model, 'model') and model.model is not None:
                        self.MODEL_HANDLERS[model_type]['save'](model, str(tmp_model_path))
                    else:
                        raise ValueError("LightGBM model does not have a valid internal model")
                else:
                    # Generic handler for other model types
                    self.MODEL_HANDLERS[model_type]['save'](model, str(tmp_model_path))
                
                # Save metadata
                with open(tmp_metadata_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                # Move to final locations
                shutil.move(str(tmp_model_path), str(model_path))
                shutil.move(str(tmp_metadata_path), str(metadata_path))
                
                # Update registry
                self.registry[model_name] = {
                    'path': str(model_path),
                    'metadata_path': str(metadata_path),
                    'model_type': model_type,
                    'created_at': meta['created_at']
                }
                self._save_registry()
                
                logger.info(f"Saved model '{model_name}' of type {model_type}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving model '{model_name}': {e}")
                return False
    
    def load_model(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        Load a model checkpoint
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, metadata) or (None, None) if not found
        """
        if model_name not in self.registry:
            logger.warning(f"Model '{model_name}' not found in registry")
            return None, None
        
        model_info = self.registry[model_name]
        model_path = Path(model_info['path'])
        metadata_path = Path(model_info['metadata_path'])
        model_type = model_info['model_type']
        
        if not model_path.exists() or not metadata_path.exists():
            logger.warning(f"Model or metadata file for '{model_name}' not found")
            return None, None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Special handling for LightGBM models
            if model_type == 'lightgbm':
                try:
                    # Try to dynamically import LightGBM
                    import importlib
                    if importlib.util.find_spec('lightgbm') is not None:
                        import lightgbm as lgb
                        from numer_crypto.models.lightgbm_model import LightGBMModel
                        
                        # Create a new model instance
                        params = metadata.get('params', {})
                        model = LightGBMModel(params=params)
                        
                        # Load the saved model
                        model.model = lgb.Booster(model_file=str(model_path))
                        
                        # Set feature names if available
                        if 'feature_names' in metadata:
                            model.feature_names = metadata['feature_names']
                        
                        return model, metadata
                    else:
                        logger.warning("LightGBM not available to load model. Returning metadata only.")
                        return None, metadata
                except Exception as e:
                    logger.error(f"Error loading LightGBM model: {e}")
                    return None, metadata
            
            # Handle H2O models
            elif model_type in ['h2o_xgboost', 'h2o_automl']:
                try:
                    # Load the model using pickle
                    model = self.MODEL_HANDLERS[model_type]['load'](str(model_path))
                    return model, metadata
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {e}")
                    return None, metadata
            
            # Generic model loader
            else:
                model = self.MODEL_HANDLERS['generic']['load'](str(model_path))
                return model, metadata
                
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            return None, None
    
    def get_metadata(self, model_name: str) -> Optional[Dict]:
        """
        Get metadata for a model without loading the model itself
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with metadata or None if not found
        """
        if model_name not in self.registry:
            logger.warning(f"Model '{model_name}' not found in registry")
            return None
        
        metadata_path = Path(self.registry[model_name]['metadata_path'])
        
        if not metadata_path.exists():
            logger.warning(f"Metadata for model '{model_name}' not found")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for model '{model_name}': {e}")
            return None
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model checkpoint
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            bool: Success or failure
        """
        if model_name not in self.registry:
            logger.warning(f"Model '{model_name}' not found in registry")
            return False
        
        model_path = Path(self.registry[model_name]['path'])
        metadata_path = Path(self.registry[model_name]['metadata_path'])
        
        try:
            # Delete files
            if model_path.exists():
                model_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update registry
            del self.registry[model_name]
            self._save_registry()
            
            logger.info(f"Deleted model '{model_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting model '{model_name}': {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """
        List all available models with basic metadata
        
        Returns:
            List of dictionaries with model information
        """
        result = []
        for name, info in self.registry.items():
            # Try to load metadata for metrics
            try:
                metadata = self.get_metadata(name)
                metrics = metadata.get('metrics', {}) if metadata else {}
            except:
                metrics = {}
            
            result.append({
                'name': name,
                'model_type': info.get('model_type', 'unknown'),
                'created_at': info.get('created_at', 'Unknown'),
                'metrics': metrics
            })
        
        # Sort by creation time (newest first)
        result.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return result
    
    def get_best_model(self, metric: str = 'validation_auc', higher_is_better: bool = True) -> str:
        """
        Get the name of the best model according to a specific metric
        
        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Name of the best model or None if no models found
        """
        models = self.list_models()
        if not models:
            return None
        
        valid_models = []
        for model in models:
            # Check if the model has the specified metric
            if metric in model.get('metrics', {}):
                valid_models.append(model)
        
        if not valid_models:
            logger.warning(f"No models found with metric '{metric}'")
            return None
        
        # Sort by metric
        if higher_is_better:
            best_model = max(valid_models, key=lambda x: x['metrics'].get(metric, float('-inf')))
        else:
            best_model = min(valid_models, key=lambda x: x['metrics'].get(metric, float('inf')))
        
        return best_model['name']
    
    def compare_models(self, model_names: List[str], metrics: List[str] = None) -> Dict:
        """
        Compare models across specified metrics
        
        Args:
            model_names: List of model names to compare
            metrics: List of metrics to compare (if None, compare all available metrics)
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            'models': {},
            'metrics': {}
        }
        
        all_metrics = set()
        
        # Gather metrics for each model
        for name in model_names:
            metadata = self.get_metadata(name)
            if not metadata:
                continue
            
            model_metrics = metadata.get('metrics', {})
            results['models'][name] = {
                'model_type': metadata.get('model_type', 'unknown'),
                'created_at': metadata.get('created_at', 'Unknown'),
                'metrics': model_metrics
            }
            
            # Track all available metrics
            all_metrics.update(model_metrics.keys())
        
        # Use specified metrics or all available ones
        compare_metrics = metrics if metrics else list(all_metrics)
        
        # Organize by metric for easier comparison
        for metric in compare_metrics:
            results['metrics'][metric] = {}
            for name in model_names:
                if name in results['models']:
                    model_metrics = results['models'][name]['metrics']
                    if metric in model_metrics:
                        results['metrics'][metric][name] = model_metrics[metric]
        
        return results
    
    def update_metrics(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """
        Update metrics for an existing model
        
        Args:
            model_name: Name of the model
            metrics: New metrics to add or update
            
        Returns:
            bool: Success or failure
        """
        metadata = self.get_metadata(model_name)
        if not metadata:
            return False
        
        # Update metrics
        if 'metrics' not in metadata:
            metadata['metrics'] = {}
        
        metadata['metrics'].update(metrics)
        
        # Save updated metadata
        metadata_path = self._get_metadata_path(model_name)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated metrics for model '{model_name}'")
            return True
        except Exception as e:
            logger.error(f"Error updating metrics for model '{model_name}': {e}")
            return False


if __name__ == "__main__":
    """Test the model checkpoint system"""
    import tempfile
    import numpy as np
    
    # Create a mock LightGBM model class
    class MockLGBModel:
        def __init__(self):
            self.model = None
        
        def save_model(self, path):
            # Just write a dummy file
            with open(path, 'w') as f:
                f.write("MOCK MODEL")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize checkpoint system
        checkpoint = ModelCheckpoint(temp_dir)
        
        # Create a mock model
        mock_model = MockLGBModel()
        mock_model.model = MockLGBModel()
        
        # Mock metrics
        metrics = {
            'validation_auc': 0.75,
            'training_time': 10.5
        }
        
        # Mock params
        params = {
            'learning_rate': 0.1,
            'max_depth': 5
        }
        
        # Save the model
        checkpoint.save_model(
            mock_model,
            'test_model',
            model_type='lightgbm',
            metrics=metrics,
            params=params,
            feature_names=['f1', 'f2', 'f3'],
            feature_importance={'f1': 0.5, 'f2': 0.3, 'f3': 0.2}
        )
        
        # List models
        models = checkpoint.list_models()
        print("Models:", models)
        
        # Get metadata
        metadata = checkpoint.get_metadata('test_model')
        print("Metadata:", metadata)
        
        # Update metrics
        checkpoint.update_metrics('test_model', {'test_metric': 0.85})
        
        # Compare models
        comparison = checkpoint.compare_models(['test_model'])
        print("Comparison:", comparison)
        
        print("Model checkpoint test completed successfully")