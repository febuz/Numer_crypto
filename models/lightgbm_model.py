"""
LightGBM model implementation for Numerai predictions with GPU support.
This module provides a GPU-accelerated LightGBM model wrapper with the same
interface as the XGBoost model.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import project settings
from config.settings import HARDWARE_CONFIG

class LightGBMModel:
    """LightGBM model implementation with GPU support"""
    
    def __init__(self, 
                 params: Optional[Dict[str, Any]] = None,
                 use_gpu: bool = True,
                 gpu_device_id: int = 0,
                 seed: int = 42,
                 name: str = "lightgbm_model"):
        """
        Initialize LightGBM model
        
        Args:
            params: LightGBM parameters dictionary
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device ID to use
            seed: Random seed
            name: Model name for saving/loading
        """
        self.name = name
        self.seed = seed
        self.model = None
        self.feature_names = None
        
        # Check if LightGBM is available
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install it first.")
        
        # Check if GPU is available and requested
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        
        # Get hardware configuration
        self.gpu_count = HARDWARE_CONFIG.get('gpu_count', 0) if 'HARDWARE_CONFIG' in globals() else 0
        
        # If no GPUs available or not requested, fallback to CPU
        if self.gpu_count == 0 or not self.use_gpu:
            self.use_gpu = False
            print("Using CPU for LightGBM.")
        else:
            if self.gpu_device_id >= self.gpu_count:
                print(f"Warning: Requested GPU ID {self.gpu_device_id} exceeds available GPUs ({self.gpu_count}). Using GPU 0.")
                self.gpu_device_id = 0
            print(f"Using GPU {self.gpu_device_id} for LightGBM.")
        
        # Set up default parameters
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': self.seed
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': self.gpu_device_id
            })
    
    def _prepare_data(self, 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> lgb.Dataset:
        """
        Prepare data for LightGBM
        
        Args:
            X: Features
            y: Target values (optional)
            
        Returns:
            LightGBM Dataset
        """
        # Save feature names for prediction
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_data = X.values
        else:
            self.feature_names = [f'f{i}' for i in range(X.shape[1])]
            X_data = X
        
        # Create dataset
        if y is not None:
            y_data = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
            return lgb.Dataset(X_data, label=y_data, feature_name=self.feature_names)
        else:
            return X_data
    
    def train(self, 
              X_train: Union[pd.DataFrame, np.ndarray],
              y_train: Union[pd.Series, np.ndarray],
              X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
              y_val: Optional[Union[pd.Series, np.ndarray]] = None,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Train the LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Dictionary with training history and results
        """
        # Prepare training data
        train_data = self._prepare_data(X_train, y_train)
        
        # Prepare validation data if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = self._prepare_data(X_val, y_val)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train the model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Return evaluation results
        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.get_feature_importance()
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Features
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        X_data = self._prepare_data(X)
        
        # Generate predictions
        return self.model.predict(X_data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        
        # Create dictionary of feature importance
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, directory: str) -> str:
        """
        Save model to disk
        
        Args:
            directory: Directory to save model
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(directory, f"{self.name}.txt")
        self.model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'feature_names': self.feature_names,
            'params': self.params,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, directory: str, model_name: Optional[str] = None) -> None:
        """
        Load model from disk
        
        Args:
            directory: Directory containing the model
            model_name: Model name (without extension)
        """
        name = model_name or self.name
        
        # Load model file
        model_path = os.path.join(directory, f"{name}.txt")
        self.model = lgb.Booster(model_file=model_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{name}_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.name = metadata.get('name', name)
                self.feature_names = metadata.get('feature_names')
                self.params = metadata.get('params', self.params)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Metadata file not found or invalid. Using defaults.")