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
import sys
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime
import torch

# Configure for better PyTorch model loading
torch.multiprocessing.set_sharing_strategy('file_system')

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
        # Import os at function level to ensure availability
        import os
        
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
        # Import os at function level to ensure availability
        import os
        
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
        # Import os at function level to ensure availability
        import os
        
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
        
        # Load model with error handling - use appropriate method based on model type
        try:
            model_type = model_entry.get("model_type", "").lower()
            
            if "h2o" in model_type:
                # H2O models need special loading
                try:
                    import h2o
                    # Initialize H2O if not running
                    cluster = h2o.cluster()
                    if cluster is None or not cluster.is_running():
                        # Set environment variables to force Java to use NVMe disk
                        import os
                        os.environ['TMPDIR'] = '/media/knight2/EDB/tmp/h2o'
                        os.environ['TMP'] = '/media/knight2/EDB/tmp/h2o'
                        os.environ['TEMP'] = '/media/knight2/EDB/tmp/h2o'
                        os.environ['JAVA_OPTS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
                        os.environ['_JAVA_OPTIONS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
                        
                        # Ensure temp directory exists
                        os.makedirs('/media/knight2/EDB/tmp/h2o', mode=0o755, exist_ok=True)
                        
                        h2o.init(
                            nthreads=1, 
                            max_mem_size="2G", 
                            verbose=False, 
                            ice_root="/media/knight2/EDB/tmp/h2o",
                            jvm_custom_args=["-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o"]
                        )
                    
                    # Load H2O model
                    model = h2o.load_model(model_path)
                    logger.info(f"Successfully loaded H2O model from {model_path}")
                    return model, model_entry
                except Exception as h2o_error:
                    logger.error(f"Error loading H2O model from {model_path}: {h2o_error}")
                    return None, model_entry
            elif "pytorch" in model_type:
                # Load PyTorch model using special handler
                try:
                    # Import PyTorch utilities
                    from utils.model.pytorch_utils import load_pytorch_model
                    
                    # Load model with the specialized function
                    model, metadata = load_pytorch_model(model_path)
                    if model is not None:
                        logger.info(f"Successfully loaded PyTorch model from {model_path}")
                        # Merge metadata from model_entry and metadata from loader
                        model_entry.update(metadata)
                        return model, model_entry
                    else:
                        logger.error(f"Failed to load PyTorch model from {model_path}")
                        return None, model_entry
                except ImportError:
                    logger.error("PyTorch utilities not available, falling back to standard loading")
                    # Fall back to standard loading
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    return model, model_entry
                except Exception as torch_error:
                    logger.error(f"Error loading PyTorch model from {model_path}: {torch_error}")
                    return None, model_entry
            elif "pytorch" in model_type.lower() or model_path.endswith('.pt') or "neural" in model_type.lower():
                # PyTorch model loading with error handling
                try:
                    # First, try to import and define the model class if needed
                    # This is critical for pickle-based loading to work
                    try:
                        # Define the model class in the main module for pickle to find
                        if 'SimpleNeuralNet' not in sys.modules.get('__main__', {}).__dict__:
                            class SimpleNeuralNet(torch.nn.Module):
                                """Simple feedforward neural network for regression"""
                                def __init__(self, input_size=500, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
                                    super(SimpleNeuralNet, self).__init__()
                                    
                                    layers = []
                                    prev_size = input_size
                                    
                                    for hidden_size in hidden_sizes:
                                        layers.extend([
                                            torch.nn.Linear(prev_size, hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.BatchNorm1d(hidden_size),
                                            torch.nn.Dropout(dropout_rate)
                                        ])
                                        prev_size = hidden_size
                                    
                                    # Output layer
                                    layers.append(torch.nn.Linear(prev_size, 1))
                                    
                                    self.network = torch.nn.Sequential(*layers)
                                
                                def forward(self, x):
                                    return self.network(x).squeeze()
                            
                            # Set the class in the main module
                            sys.modules['__main__'].SimpleNeuralNet = SimpleNeuralNet
                            logger.info(f"Registered SimpleNeuralNet class for PyTorch model loading")
                    except Exception as class_error:
                        logger.warning(f"Could not register model class: {class_error}")
                    
                    # Try loading the model with torch.load first
                    try:
                        model_data = torch.load(model_path, map_location='cpu')
                        
                        # Handle different saved formats
                        if isinstance(model_data, dict) and 'model' in model_data:
                            model = model_data['model']
                            # Update model entry with additional metadata
                            if 'scaler' in model_data:
                                model_entry['scaler'] = model_data['scaler']
                        elif isinstance(model_data, dict) and 'model_state_dict' in model_data:
                            # Need to reconstruct model from state dict
                            if 'SimpleNeuralNet' in sys.modules['__main__'].__dict__:
                                input_size = model_data.get('input_size', 500)  # Default if not specified
                                model = SimpleNeuralNet(input_size)
                                model.load_state_dict(model_data['model_state_dict'])
                                if 'scaler' in model_data:
                                    model_entry['scaler'] = model_data['scaler']
                            else:
                                raise ValueError("Model class not available to load state dict")
                        else:
                            # Assume model_data is the model itself
                            model = model_data
                        
                        logger.info(f"Successfully loaded PyTorch model from {model_path}")
                        return model, model_entry
                        
                    except Exception as torch_error:
                        logger.warning(f"Torch load failed, trying pickle: {torch_error}")
                        # Fall back to pickle loading
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        return model, model_entry
                        
                except Exception as e:
                    logger.error(f"All PyTorch load attempts failed: {e}")
                    return None, model_entry
            else:
                # Standard pickle loading for other models
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
        # Import os at function level to ensure availability
        import os
        
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
        # Import os at function level to ensure availability
        import os
        
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