"""
Model storage and registry utilities.

This module provides functionality for:
- Registering and tracking trained models
- Loading models for prediction
- Managing model metadata
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ModelStore:
    """
    Model store for managing and registering trained models.
    
    Tracks all trained models with their metadata and provides
    utilities for finding and loading models by type and version.
    """
    
    def __init__(self, base_dir="/media/knight2/EDB/numer_crypto_temp/models"):
        """
        Initialize the model store.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.registry_file = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load model registry from file"""
        if not self.registry_file.exists():
            return {"models": []}
        
        with open(self.registry_file, 'r') as f:
            return json.load(f)
    
    def _save_registry(self):
        """Save model registry to file"""
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, model_type, feature_set_id=None, 
                     version="1", metrics=None, description=None):
        """
        Register a trained model.
        
        Args:
            model_path: Path to saved model file
            model_type: Type of model (lightgbm, xgboost, etc.)
            feature_set_id: ID of feature set used for training
            version: Model version
            metrics: Dictionary of evaluation metrics
            description: Description of the model
            
        Returns:
            str: Model ID
        """
        model_id = f"{model_type}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_entry = {
            "id": model_id,
            "model_path": str(model_path),
            "model_type": model_type,
            "feature_set_id": feature_set_id,
            "version": version,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics or {},
            "description": description or f"{model_type} model version {version}"
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        logger.info(f"Registered model {model_id}")
        return model_id
    
    def get_model_by_id(self, model_id):
        """
        Get model entry by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            dict: Model entry or None if not found
        """
        for model in self.registry["models"]:
            if model["id"] == model_id:
                return model
        return None
    
    def get_latest_model(self, model_type=None):
        """
        Get the latest model of a given type.
        
        Args:
            model_type: Type of model (if None, return the latest of any type)
            
        Returns:
            dict: Model entry or None if not found
        """
        models = self.registry["models"]
        
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        if not models:
            return None
        
        # Sort by creation date (latest first)
        models.sort(key=lambda m: m["created_at"], reverse=True)
        return models[0]
    
    def list_models(self, model_type=None, feature_set_id=None, limit=10):
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            feature_set_id: Filter by feature set ID
            limit: Maximum number of models to return
            
        Returns:
            dict: Dictionary mapping model_id to model metadata
        """
        models = self.registry["models"]
        
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        if feature_set_id:
            models = [m for m in models if m["feature_set_id"] == feature_set_id]
        
        # Sort by creation date (latest first)
        models.sort(key=lambda m: m["created_at"], reverse=True)
        
        # Convert to dictionary with id as key
        result = {}
        for model in models[:limit]:
            result[model["id"]] = model
            
        return result
    
    def get_model(self, model_id=None, model_type=None):
        """
        Load a model by ID or type.
        
        Args:
            model_id: Model ID (if None, latest model of model_type is used)
            model_type: Model type to load (used if model_id is None)
            
        Returns:
            tuple: (model, metadata) or (None, None) if not found
            
        Raises:
            FileNotFoundError: If model file does not exist
        """
        # Get model entry
        if model_id:
            model_entry = self.get_model_by_id(model_id)
        else:
            model_entry = self.get_latest_model(model_type)
        
        if not model_entry:
            return None, None
        
        # Check if model file exists
        model_path = model_entry["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with error handling
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, model_entry
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            # If model file is corrupt or otherwise unloadable, return None
            return None, model_entry
        
    # Alias for backward compatibility
    load_model = get_model
    
    def update_model_metrics(self, model_id, metrics):
        """
        Update metrics for a model.
        
        Args:
            model_id: Model ID
            metrics: Dictionary of metrics to update
            
        Returns:
            bool: Whether update was successful
        """
        model_entry = self.get_model_by_id(model_id)
        
        if not model_entry:
            return False
        
        # Update metrics
        model_entry["metrics"].update(metrics)
        
        # Save registry
        self._save_registry()
        
        return True
    
    def create_model_metadata(self, model, model_path, model_type, 
                            feature_set_id=None, params=None, metrics=None,
                            version="1", description=None):
        """
        Create and save model metadata file.
        
        Args:
            model: Trained model object
            model_path: Path to saved model file
            model_type: Type of model
            feature_set_id: ID of feature set used for training
            params: Model parameters
            metrics: Dictionary of evaluation metrics
            version: Model version
            description: Description of the model
            
        Returns:
            str: Path to metadata file
        """
        metadata = {
            "model_type": model_type,
            "version": version,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(model_path),
            "feature_set_id": feature_set_id,
            "params": params or {},
            "metrics": metrics or {},
            "description": description or f"{model_type} model version {version}"
        }
        
        # Try to get model parameters
        if hasattr(model, "get_params"):
            try:
                metadata["params"] = model.get_params()
            except:
                pass
        
        # Create metadata file path
        metadata_path = str(model_path).replace('.pkl', '_metadata.json')
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created model metadata at {metadata_path}")
        return metadata_path