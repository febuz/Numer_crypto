"""
Model ensemble methods for Numerai Crypto.
Combines predictions from multiple models to improve performance.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

try:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.settings import CHECKPOINTS_DIR
except ImportError:
    CHECKPOINTS_DIR = os.path.expanduser('~/numer_crypto_checkpoints')

def ensemble_predictions(predictions: List[np.ndarray], 
                         weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Ensemble multiple model predictions.
    
    Args:
        predictions: List of prediction arrays (each with same shape)
        weights: Optional weights for weighted average. If None, equal weights used.
        
    Returns:
        Ensembled predictions as numpy array
    """
    if not predictions:
        raise ValueError("No predictions provided for ensembling")
    
    # Verify all predictions have same shape
    pred_shape = predictions[0].shape
    for i, pred in enumerate(predictions):
        if pred.shape != pred_shape:
            raise ValueError(f"Prediction {i} has shape {pred.shape}, expected {pred_shape}")
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Compute weighted average
    result = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        result += pred * weight
    
    return result

class ModelEnsemble:
    """
    Ensemble multiple models with flexible weighting strategies.
    """
    
    def __init__(self, name: str = "ensemble", weights_strategy: str = "equal"):
        """
        Initialize model ensemble.
        
        Args:
            name: Name for this ensemble
            weights_strategy: Strategy for weight determination
                - 'equal': Equal weights for all models
                - 'performance': Weights based on validation performance
                - 'custom': Use custom weights provided during predict
        """
        self.name = name
        self.weights_strategy = weights_strategy
        self.models = []
        self.weights = []
        self.performance_metrics = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def add_model(self, model: Any, weight: Optional[float] = None, 
                  performance_metric: Optional[float] = None):
        """
        Add a model to the ensemble.
        
        Args:
            model: Model object with predict method
            weight: Optional custom weight for this model
            performance_metric: Validation metric (lower is better)
        """
        self.models.append(model)
        
        if weight is not None:
            self.weights.append(weight)
        else:
            # Default to equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        if performance_metric is not None:
            self.performance_metrics[len(self.models) - 1] = performance_metric
            # Update weights if using performance strategy
            if self.weights_strategy == "performance" and self.performance_metrics:
                self._update_performance_weights()
    
    def _update_performance_weights(self):
        """Update weights based on performance metrics."""
        if not self.performance_metrics:
            return
        
        # Lower metrics (like RMSE) are better, so use inverse
        inv_metrics = [1.0 / self.performance_metrics[i] for i in range(len(self.models))]
        total = sum(inv_metrics)
        self.weights = [m / total for m in inv_metrics]
    
    def predict(self, X: pd.DataFrame, custom_weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Features to predict on
            custom_weights: Optional custom weights (overrides weights_strategy)
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Generate predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Determine weights to use
        weights_to_use = None
        if custom_weights is not None:
            weights_to_use = custom_weights
        elif self.weights_strategy == "equal":
            weights_to_use = [1.0 / len(self.models)] * len(self.models)
        elif self.weights_strategy == "performance":
            weights_to_use = self.weights
        else:
            weights_to_use = self.weights
        
        # Ensemble predictions
        return ensemble_predictions(predictions, weights_to_use)
    
    def save(self, directory: str = CHECKPOINTS_DIR):
        """
        Save ensemble metadata (does not save individual models).
        
        Args:
            directory: Directory to save metadata
        """
        os.makedirs(directory, exist_ok=True)
        
        # Create metadata dict
        metadata = {
            "name": self.name,
            "weights_strategy": self.weights_strategy,
            "model_count": len(self.models),
            "weights": self.weights,
            "performance_metrics": self.performance_metrics,
            "timestamp": self.timestamp
        }
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path