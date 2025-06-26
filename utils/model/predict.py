"""
Prediction utilities for different model types.
Provides consistent prediction interfaces for various model types.
"""

import logging
import numpy as np
import os

# Configure logging
logger = logging.getLogger(__name__)

def predict_with_model(model, X, model_type=None, metadata=None):
    """
    Make predictions using a model with appropriate handling for different model types.
    
    Args:
        model: The loaded model
        X: Features for prediction (numpy array, pandas DataFrame, or polars DataFrame)
        model_type: Type of model (optional, will be inferred if not provided)
        metadata: Model metadata (optional)
    
    Returns:
        numpy.ndarray: Predictions
    """
    # Convert X to numpy if it's not already
    if hasattr(X, 'to_numpy'):
        X_numpy = X.to_numpy()
    else:
        X_numpy = np.asarray(X)
    
    # Infer model type if not provided
    if model_type is None:
        model_type = infer_model_type(model)
    
    model_type = model_type.lower() if isinstance(model_type, str) else ''
    
    try:
        # Route to appropriate prediction method based on model type
        if 'xgboost' in model_type:
            return predict_xgboost(model, X_numpy)
        elif 'lightgbm' in model_type:
            return predict_lightgbm(model, X_numpy)
        elif 'catboost' in model_type:
            return predict_catboost(model, X_numpy)
        elif 'h2o' in model_type:
            return predict_h2o(model, X_numpy)
        elif 'pytorch' in model_type or hasattr(model, 'forward'):
            return predict_pytorch(model, X_numpy, metadata)
        else:
            # Default scikit-learn compatible prediction
            return model.predict(X_numpy)
    
    except Exception as e:
        logger.error(f"Error in predict_with_model: {e}")
        # Try to use a fallback method
        return fallback_prediction(model, X_numpy)

def infer_model_type(model):
    """Infer model type from model object"""
    model_str = str(type(model)).lower()
    
    if 'xgboost' in model_str:
        return 'xgboost'
    elif 'lightgbm' in model_str:
        return 'lightgbm'
    elif 'catboost' in model_str:
        return 'catboost'
    elif 'h2o' in model_str:
        return 'h2o'
    elif 'module' in model_str or hasattr(model, 'forward'):
        return 'pytorch'
    else:
        return 'unknown'

def predict_xgboost(model, X):
    """Make predictions with XGBoost model"""
    try:
        # Try sklearn-style prediction
        if hasattr(model, 'predict') and hasattr(model, 'get_params'):
            return model.predict(X)
        
        # Native XGBoost model
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)
    
    except Exception as e:
        logger.error(f"XGBoost prediction error: {e}")
        raise

def predict_lightgbm(model, X):
    """Make predictions with LightGBM model"""
    try:
        # LightGBM models sometimes need shape check disabled
        return model.predict(X, predict_disable_shape_check=True)
    except Exception as e:
        logger.error(f"LightGBM prediction error: {e}")
        # Try without the flag
        return model.predict(X)

def predict_catboost(model, X):
    """Make predictions with CatBoost model"""
    try:
        return model.predict(X)
    except Exception as e:
        logger.error(f"CatBoost prediction error: {e}")
        raise

def predict_h2o(model, X):
    """Make predictions with H2O model"""
    try:
        import h2o
        # Create H2O frame with generic feature names
        h2o_frame = h2o.H2OFrame(X)
        h2o_frame.columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Get predictions
        preds = model.predict(h2o_frame).as_data_frame()
        
        # Return the predicted column as a flat array
        return preds.values.flatten()
    
    except Exception as e:
        logger.error(f"H2O prediction error: {e}")
        raise

def predict_pytorch(model, X, metadata=None):
    """Make predictions with PyTorch model"""
    try:
        import torch
        
        # Apply feature scaling if a scaler is provided
        if metadata and 'scaler' in metadata:
            X = metadata['scaler'].transform(X)
        
        # Convert to float32 for GPU compatibility
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = X_tensor.to(device)
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make predictions without gradient tracking
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        
        # Ensure predictions are 1D
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    except Exception as e:
        logger.error(f"PyTorch prediction error: {e}")
        raise

def fallback_prediction(model, X):
    """Attempt to make predictions using various methods when standard approaches fail"""
    logger.warning("Using fallback prediction method")
    
    try:
        # Try various approaches
        
        # 1. Try direct prediction
        if hasattr(model, 'predict'):
            return model.predict(X)
        
        # 2. Try XGBoost with DMatrix
        if 'Booster' in str(type(model)):
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                return model.predict(dmatrix)
            except:
                pass
        
        # 3. Try PyTorch forward method
        if hasattr(model, 'forward') or hasattr(model, '__call__'):
            try:
                import torch
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    preds = model(X_tensor).numpy()
                return preds.flatten() if len(preds.shape) > 1 else preds
            except:
                pass
        
        # 4. As a last resort, create dummy predictions
        logger.error("All prediction methods failed, returning neutral predictions")
        return np.full(X.shape[0], 0.5)
        
    except Exception as e:
        logger.error(f"Fallback prediction error: {e}")
        # Return neutral predictions
        return np.full(X.shape[0], 0.5)