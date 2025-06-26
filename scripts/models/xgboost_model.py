"""
XGBoost model implementation for the Numerai Crypto project.
"""
import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost is not installed. Using fallback implementation.")

# Import project settings
try:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.settings import HARDWARE_CONFIG
except ImportError:
    HARDWARE_CONFIG = {'gpu_count': 0}
    logger.warning("Could not import hardware config. Using defaults.")

class XGBoostModel:
    """XGBoost model implementation with GPU support"""
    
    def __init__(self, 
                 params: Optional[Dict[str, Any]] = None,
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 seed: int = 42,
                 model_id: str = "xgboost_model"):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameters dictionary
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID to use
            seed: Random seed
            model_id: Model ID for saving/loading
        """
        self.model_id = model_id
        self.seed = seed
        self.model = None
        self.feature_names = None
        
        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not installed. Using fallback model.")
            self.xgboost_available = False
            return
        else:
            self.xgboost_available = True
        
        # Check if GPU is available and requested
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Get hardware configuration
        self.gpu_count = HARDWARE_CONFIG.get('gpu_count', 0)
        
        # If no GPUs available or not requested, fallback to CPU
        if self.gpu_count == 0 or not self.use_gpu:
            self.use_gpu = False
            logger.info("Using CPU for XGBoost.")
        else:
            if self.gpu_id >= self.gpu_count:
                logger.warning(f"Requested GPU ID {self.gpu_id} exceeds available GPUs ({self.gpu_count}). Using GPU 0.")
                self.gpu_id = 0
            logger.info(f"Using GPU {self.gpu_id} for XGBoost.")
        
        # Set up default parameters
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0,
            'seed': self.seed
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': self.gpu_id
            })
    
    def train(self, 
              X_train: Union[pd.DataFrame, np.ndarray],
              y_train: Union[pd.Series, np.ndarray],
              X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
              y_val: Optional[Union[pd.Series, np.ndarray]] = None,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Train the XGBoost model
        
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
        if not self.xgboost_available:
            logger.warning("XGBoost not available. Using fallback training.")
            return self._fallback_train(X_train, y_train, X_val, y_val)
        
        # Save feature names for prediction
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        # Prepare training data
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Prepare validation data if provided
        watchlist = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            watchlist.append((dval, 'val'))
        
        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Return evaluation results
        if hasattr(self.model, 'best_iteration'):
            best_iteration = self.model.best_iteration
        else:
            best_iteration = num_boost_round
            
        if hasattr(self.model, 'best_score'):
            best_score = self.model.best_score
        else:
            best_score = None
            
        return {
            'best_iteration': best_iteration,
            'best_score': best_score,
            'feature_importance': self.get_feature_importance()
        }
    
    def _fallback_train(self, X_train, y_train, X_val=None, y_val=None):
        """Simple fallback training when XGBoost is not available"""
        # For fallback, just create a simple model that returns the mean
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
            
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            self.mean_value = y_train.mean()
        else:
            self.mean_value = np.mean(y_train)
            
        # Create a simple model object
        self.model = {
            'type': 'fallback',
            'mean': self.mean_value,
            'feature_names': self.feature_names,
            'shape': X_train.shape
        }
        
        logger.warning(f"Trained fallback model with mean value: {self.mean_value}")
        
        return {
            'best_iteration': 1,
            'best_score': None,
            'feature_importance': {name: 1.0 for name in self.feature_names}
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
        
        # Use fallback prediction if needed
        if not self.xgboost_available or isinstance(self.model, dict) and self.model.get('type') == 'fallback':
            return self._fallback_predict(X)
        
        # Prepare data
        if self.feature_names:
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        else:
            dtest = xgb.DMatrix(X)
        
        # Generate predictions
        return self.model.predict(dtest)
    
    def _fallback_predict(self, X):
        """Fallback prediction method when XGBoost is not available"""
        # Just return the mean value for all samples
        n_samples = X.shape[0]
        return np.full(n_samples, self.model['mean'])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle fallback model
        if not self.xgboost_available or isinstance(self.model, dict) and self.model.get('type') == 'fallback':
            # Return equal importance for all features
            return {name: 1.0 for name in self.feature_names}
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        
        # Add missing features with zero importance
        for feature in self.feature_names:
            if feature not in importance:
                importance[feature] = 0.0
        
        return importance
    
    def save(self, directory: str) -> str:
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
        
        # Handle fallback model
        if not self.xgboost_available or isinstance(self.model, dict) and self.model.get('type') == 'fallback':
            model_path = os.path.join(directory, f"{self.model_id}.json")
            with open(model_path, 'w') as f:
                json.dump(self.model, f, indent=2)
        else:
            # Save model file
            model_path = os.path.join(directory, f"{self.model_id}.model")
            self.model.save_model(model_path)
        
        # Save metadata
        metadata = {
            'model_id': self.model_id,
            'feature_names': self.feature_names,
            'params': self.params,
            'fallback': not self.xgboost_available
        }
        
        if hasattr(self.model, 'best_iteration'):
            metadata['best_iteration'] = self.model.best_iteration
            
        if hasattr(self.model, 'best_score'):
            metadata['best_score'] = self.model.best_score
        
        metadata_path = os.path.join(directory, f"{self.model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load(self, model_path: str) -> None:
        """
        Load model from disk
        
        Args:
            model_path: Path to saved model or directory
        """
        # If directory provided, construct model path
        if os.path.isdir(model_path):
            directory = model_path
            model_path = os.path.join(directory, f"{self.model_id}.model")
            json_path = os.path.join(directory, f"{self.model_id}.json")
            
            # Check if we have a fallback model
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.model = json.load(f)
                self.feature_names = self.model.get('feature_names', [])
                self.mean_value = self.model.get('mean', 0.0)
                return
        
        # Load metadata if available
        metadata_path = model_path.replace('.model', '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.model_id = metadata.get('model_id', self.model_id)
                    self.feature_names = metadata.get('feature_names', [])
                    self.params = metadata.get('params', self.params)
                    
                    # Check if it's a fallback model
                    if metadata.get('fallback', False):
                        with open(model_path.replace('.model', '.json'), 'r') as f:
                            self.model = json.load(f)
                        self.mean_value = self.model.get('mean', 0.0)
                        return
            except:
                logger.warning("Could not load metadata")
        
        # Load model file
        if not self.xgboost_available:
            logger.error("Cannot load XGBoost model: XGBoost not available")
            return
        
        if os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            logger.error(f"Model file not found: {model_path}")