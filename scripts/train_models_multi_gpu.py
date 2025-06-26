#!/usr/bin/env python3
"""
Multi-GPU Training Worker Functions for Numerai Crypto

This module provides worker functions for training models on multiple GPUs in parallel.
Supports LightGBM, XGBoost, CatBoost, and PyTorch models with GPU acceleration.
"""

import os
import sys
import time
import logging
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Configure GPU settings for better performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "4"

def train_lightgbm_worker(gpu_id: int, X_train: np.ndarray, y_train: np.ndarray, 
                        model_name: str, output_dir: str) -> Dict[str, Any]:
    """
    Train LightGBM model on a specific GPU
    
    Args:
        gpu_id: GPU device ID to use
        X_train: Training features
        y_train: Training targets
        model_name: Base name for the model
        output_dir: Directory to save model
        
    Returns:
        Dictionary with training results and model path
    """
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure subprocess inherits correct Python environment
    import sys
    venv_path = '/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env/lib/python3.12/site-packages'
    if venv_path not in sys.path:
        sys.path.insert(0, venv_path)
    
    # Get memory limit from environment or use default
    memory_limit_gb = int(os.environ.get("GPU_MEMORY_LIMIT", "24"))
    
    # Enable Azure Synapse LightGBM mode
    os.environ["USE_AZURE_SYNAPSE_LIGHTGBM"] = "1"
    os.environ["LIGHTGBM_SYNAPSE_MODE"] = "1"
    os.environ["LIGHTGBM_USE_SYNAPSE"] = "1"
    
    # Set CUDA device for this process
    logger.info(f"Worker {gpu_id} training LightGBM model: {model_name}")
    
    try:
        # Ensure we can import LightGBM
        try:
            import lightgbm as lgb
        except ImportError as e:
            logger.error(f"GPU {gpu_id} cannot import LightGBM: {e}")
            return {
                'gpu_id': gpu_id,
                'model_name': model_name,
                'status': 'failed',
                'error': f'LightGBM import failed: {e}'
            }
        
        from sklearn.model_selection import train_test_split
        
        # Check for existing model with today's date
        today_str = datetime.today().strftime("%Y%m%d")
        model_file_pattern = f"{model_name}_gpu{gpu_id}_{today_str}"
        existing_models = [f for f in os.listdir(output_dir) if model_file_pattern in f] if os.path.exists(output_dir) else []
        
        if existing_models and not os.environ.get("FORCE_RETRAIN", "0") == "1":
            logger.info(f"GPU {gpu_id} found existing model {existing_models[0]}, skipping training")
            return {
                'gpu_id': gpu_id,
                'model_name': model_name,
                'model_path': os.path.join(output_dir, existing_models[0]),
                'status': 'skipped'
            }
            
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # GPU-optimized parameters with variation per GPU for best results
        # Each GPU gets a different hyperparameter configuration for ensemble diversity
        base_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'first_metric_only': True,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,  # Set to 0 since CUDA_VISIBLE_DEVICES controls the actual GPU
            'verbose': -1,
            'deterministic': True,
            'seed': 42 + gpu_id,
            'num_iterations': 2000,
            'early_stopping_rounds': 100,
            'bagging_seed': 42 + gpu_id,
            'feature_fraction_seed': 42 + gpu_id
        }
        
        # Enable Azure Synapse mode
        if os.environ.get("USE_AZURE_SYNAPSE_LIGHTGBM", "1") == "1":
            base_params['azure_synapse_mode'] = True
            base_params['synapse_mode'] = True
            base_params['use_synapse'] = True
            logger.info(f"GPU {gpu_id} using Azure Synapse mode for LightGBM")
        
        # Vary parameters by GPU for best prediction quality
        if gpu_id == 0:
            # First GPU: Deeper model with lower learning rate
            base_params.update({
                'learning_rate': 0.01,
                'max_depth': 12,
                'num_leaves': 128,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0
            })
        elif gpu_id == 1:
            # Second GPU: More aggressive learning with more regularization
            base_params.update({
                'learning_rate': 0.05,
                'max_depth': 10,
                'num_leaves': 64,
                'min_data_in_leaf': 30,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.9,
                'bagging_freq': 3,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5
            })
        else:
            # Third GPU and beyond: Faster learning with more aggressive regularization
            base_params.update({
                'learning_rate': 0.1,
                'max_depth': 8,
                'num_leaves': 32,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'lambda_l1': 1.0,
                'lambda_l2': 0.1
            })
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with learning_rate={base_params['learning_rate']}, max_depth={base_params['max_depth']}")
        
        # Create datasets
        train_dataset = lgb.Dataset(X_train_split, label=y_train_split)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        start_time = time.time()
        
        # Train model
        model = lgb.train(
            params=base_params,
            train_set=train_dataset,
            valid_sets=[val_dataset],
            callbacks=[lgb.log_evaluation(period=100, show_stdv=False)]
        )
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed LightGBM training in {training_time:.2f} seconds")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}_{timestamp}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        # Get best score
        best_score = min(model.best_score['valid_0']['rmse'])
        best_iteration = model.best_iteration
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'model_path': model_path,
            'training_time': training_time,
            'best_score': best_score,
            'best_iteration': best_iteration,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} LightGBM training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def train_xgboost_worker(gpu_id: int, X_train: np.ndarray, y_train: np.ndarray, 
                        model_name: str, output_dir: str) -> Dict[str, Any]:
    """
    Train XGBoost model on a specific GPU
    
    Args:
        gpu_id: GPU device ID to use
        X_train: Training features
        y_train: Training targets
        model_name: Base name for the model
        output_dir: Directory to save model
        
    Returns:
        Dictionary with training results and model path
    """
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure subprocess inherits correct Python environment
    import sys
    venv_path = '/media/knight2/EDB/repos/Numer_crypto/numer_crypto_env/lib/python3.12/site-packages'
    if venv_path not in sys.path:
        sys.path.insert(0, venv_path)
    
    # Get memory limit from environment or use default
    memory_limit_gb = int(os.environ.get("GPU_MEMORY_LIMIT", "24"))
    
    logger.info(f"Worker {gpu_id} training XGBoost model: {model_name}")
    
    try:
        # Ensure we can import XGBoost
        try:
            import xgboost as xgb
        except ImportError as e:
            logger.error(f"GPU {gpu_id} cannot import XGBoost: {e}")
            return {
                'gpu_id': gpu_id,
                'model_name': model_name,
                'status': 'failed',
                'error': f'XGBoost import failed: {e}'
            }
        
        from sklearn.model_selection import train_test_split
        
        # Check for existing model with today's date
        today_str = datetime.today().strftime("%Y%m%d")
        model_file_pattern = f"{model_name}_gpu{gpu_id}_{today_str}"
        existing_models = [f for f in os.listdir(output_dir) if model_file_pattern in f] if os.path.exists(output_dir) else []
        
        if existing_models and not os.environ.get("FORCE_RETRAIN", "0") == "1":
            logger.info(f"GPU {gpu_id} found existing model {existing_models[0]}, skipping training")
            return {
                'gpu_id': gpu_id,
                'model_name': model_name,
                'model_path': os.path.join(output_dir, existing_models[0]),
                'status': 'skipped'
            }
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # GPU-optimized parameters with variation per GPU for best results
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'gpu_id': 0,  # Use first visible GPU (controlled by CUDA_VISIBLE_DEVICES)
            'verbosity': 0,
            'random_state': 42 + gpu_id,
        }
        
        # Vary parameters by GPU for best prediction quality
        if gpu_id == 0:
            # First GPU: Higher learning rate with more depth
            base_params.update({
                'max_depth': 12,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.9,
                'reg_alpha': 0.01,
                'reg_lambda': 1,
                'scale_pos_weight': 1
            })
        elif gpu_id == 1:
            # Second GPU: More conservative learning with more regularization
            base_params.update({
                'max_depth': 8,
                'learning_rate': 0.05,
                'min_child_weight': 3,
                'gamma': 0.2,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'reg_alpha': 0.1,
                'reg_lambda': 10,
                'scale_pos_weight': 1
            })
        else:
            # Third GPU and beyond: Even more conservative
            base_params.update({
                'max_depth': 6,
                'learning_rate': 0.01,
                'min_child_weight': 5,
                'gamma': 0.5,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'colsample_bylevel': 0.6,
                'reg_alpha': 1,
                'reg_lambda': 5,
                'scale_pos_weight': 1
            })
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with learning_rate={base_params['learning_rate']}, max_depth={base_params['max_depth']}")
        
        # Create datasets
        dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        start_time = time.time()
        
        # Train model
        model = xgb.train(
            params=base_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed XGBoost training in {training_time:.2f} seconds")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}_{timestamp}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        # Get best score
        best_score = model.best_score
        best_iteration = model.best_iteration
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'model_path': model_path,
            'training_time': training_time,
            'best_score': best_score,
            'best_iteration': best_iteration,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} XGBoost training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

if __name__ == "__main__":
    logger.info("This module provides worker functions for multi-GPU training and shouldn't be run directly.")
    logger.info("Import the worker functions in your training script instead.")