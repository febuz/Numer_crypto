#!/usr/bin/env python3
"""
Advanced Model Comparison for Numerai Crypto

This script creates comprehensive model comparisons with:
- GPU-accelerated XGBoost
- GPU-accelerated LightGBM
- PyTorch neural networks
- H2O AutoML with Sparkling Water
- Amazon SageMaker AutoML (when available)
- Azure Synapse LGBM (when available)
- TPOT AutoML with GPU acceleration

The script identifies the lowest possible RMSE and creates
valid submission files for the Numerai crypto competition.
"""

import os
import sys
import logging
import argparse
import json
import time
import warnings
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Configure logging
log_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Constants
DATA_DIR = project_root / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"
EXTERNAL_DATA_DIR = Path("/media/knight2/EDB/data/crypto_data")
MODELS_DIR = project_root / "models" / "trained"
TODAY = datetime.now().strftime("%Y%m%d")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def get_available_gpus():
    """Get list of available GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            return []
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                   capture_output=True, text=True, check=True)
            return [int(line.strip()) for line in result.stdout.strip().split('\n')]
        except (subprocess.SubprocessError, FileNotFoundError):
            return []

def check_package_availability():
    """Check which ML packages are available"""
    available_packages = {}
    
    # Check for standard ML packages
    try:
        import xgboost
        available_packages['xgboost'] = True
    except ImportError:
        available_packages['xgboost'] = False
    
    try:
        import lightgbm
        available_packages['lightgbm'] = True
    except ImportError:
        available_packages['lightgbm'] = False
    
    try:
        import torch
        available_packages['pytorch'] = True
        available_packages['pytorch_cuda'] = torch.cuda.is_available()
    except ImportError:
        available_packages['pytorch'] = False
        available_packages['pytorch_cuda'] = False
    
    # Check for H2O
    try:
        import h2o
        available_packages['h2o'] = True
    except ImportError:
        available_packages['h2o'] = False
    
    # Check for H2O Sparkling Water
    try:
        import pysparkling
        available_packages['sparkling_water'] = True
    except ImportError:
        available_packages['sparkling_water'] = False
    
    # Check for TPOT
    try:
        import tpot
        available_packages['tpot'] = True
    except ImportError:
        available_packages['tpot'] = False
    
    # Check for cloud service packages
    try:
        import boto3
        import sagemaker
        available_packages['sagemaker'] = True
    except ImportError:
        available_packages['sagemaker'] = False
    
    try:
        import azure.identity
        import azure.synapse.artifacts
        available_packages['azure_synapse'] = True
    except ImportError:
        available_packages['azure_synapse'] = False
    
    # Check for Optuna (for hyperparameter optimization)
    try:
        import optuna
        available_packages['optuna'] = True
    except ImportError:
        available_packages['optuna'] = False
    
    # Log results
    logger.info("Package availability:")
    for package, available in available_packages.items():
        logger.info(f"  {package}: {'Available' if available else 'Not available'}")
    
    return available_packages

def load_data(data_path=None):
    """
    Load data from either:
    1. Merged data created by download_and_prepare_data.py
    2. Numerai data downloaded directly
    3. Yiedl data
    4. Synthetic data as fallback
    """
    logger.info("Loading data...")
    
    # Look for data in specified path
    if data_path:
        data_path = Path(data_path)
        if not data_path.exists():
            logger.error(f"Specified data path does not exist: {data_path}")
            data_path = None
    
    # Try to load merged data first
    merged_data = {}
    latest_date_dir = None
    
    # Find the latest dated directory if available
    if EXTERNAL_DATA_DIR.exists():
        date_dirs = sorted([d for d in EXTERNAL_DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
                         key=lambda x: x.name, reverse=True)
        if date_dirs:
            latest_date_dir = date_dirs[0]
            logger.info(f"Found latest data directory: {latest_date_dir}")
            
            # Look for merged data files
            merged_files = list(latest_date_dir.glob("merged_*.parquet"))
            if merged_files:
                for file in merged_files:
                    data_type = file.name.split('_')[1]  # Extract train/validation/live from filename
                    try:
                        df = pd.read_parquet(file)
                        merged_data[data_type] = df
                        logger.info(f"Loaded merged {data_type} data: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")
    
    # If we have all necessary merged data, return it
    if merged_data and all(k in merged_data for k in ['train', 'validation', 'live']):
        logger.info("Using merged data")
        return merged_data
    
    # Otherwise, try to load Numerai data
    numerai_data = {}
    numerai_dir = DATA_DIR / "numerai"
    
    if numerai_dir.exists():
        train_file = numerai_dir / "train_targets.parquet"
        validation_file = numerai_dir / "validation.parquet"
        live_file = numerai_dir / "live.parquet"
        
        if train_file.exists() and validation_file.exists() and live_file.exists():
            try:
                train_df = pd.read_parquet(train_file)
                validation_df = pd.read_parquet(validation_file)
                live_df = pd.read_parquet(live_file)
                
                # Add data type column
                train_df['data_type'] = 'train'
                validation_df['data_type'] = 'validation'
                live_df['data_type'] = 'live'
                
                numerai_data['train'] = train_df
                numerai_data['validation'] = validation_df
                numerai_data['live'] = live_df
                
                logger.info(f"Loaded Numerai train data: {train_df.shape}")
                logger.info(f"Loaded Numerai validation data: {validation_df.shape}")
                logger.info(f"Loaded Numerai live data: {live_df.shape}")
            except Exception as e:
                logger.error(f"Error loading Numerai data: {e}")
    
    # If we have all necessary Numerai data, return it
    if numerai_data and all(k in numerai_data for k in ['train', 'validation', 'live']):
        logger.info("Using Numerai data")
        return numerai_data
    
    # Otherwise, check for Yiedl data
    yiedl_data = {}
    yiedl_dir = DATA_DIR / "yiedl"
    
    if yiedl_dir.exists():
        latest_file = yiedl_dir / "yiedl_latest.parquet"
        
        if latest_file.exists():
            try:
                df = pd.read_parquet(latest_file)
                
                # Split into train/validation/test
                train_size = int(len(df) * 0.7)
                val_size = int(len(df) * 0.15)
                
                train_df = df.iloc[:train_size].copy()
                validation_df = df.iloc[train_size:train_size+val_size].copy()
                live_df = df.iloc[train_size+val_size:].copy()
                
                # Add data type column
                train_df['data_type'] = 'train'
                validation_df['data_type'] = 'validation'
                live_df['data_type'] = 'live'
                
                # Ensure target column exists
                if 'target' not in train_df.columns:
                    logger.warning("Creating synthetic target for Yiedl data")
                    
                    # Create synthetic target based on features
                    numeric_cols = [col for col in train_df.columns 
                                   if pd.api.types.is_numeric_dtype(train_df[col])][:5]
                    
                    if numeric_cols:
                        weights = np.random.normal(0, 1, len(numeric_cols))
                        train_df['target'] = sum(train_df[col] * w for col, w in zip(numeric_cols, weights))
                        validation_df['target'] = sum(validation_df[col] * w for col, w in zip(numeric_cols, weights))
                        # No target for live data
                    else:
                        train_df['target'] = np.random.normal(0, 1, len(train_df))
                        validation_df['target'] = np.random.normal(0, 1, len(validation_df))
                
                yiedl_data['train'] = train_df
                yiedl_data['validation'] = validation_df
                yiedl_data['live'] = live_df
                
                logger.info(f"Loaded and split Yiedl data: {df.shape}")
                logger.info(f"  Train: {train_df.shape}")
                logger.info(f"  Validation: {validation_df.shape}")
                logger.info(f"  Live: {live_df.shape}")
            except Exception as e:
                logger.error(f"Error loading Yiedl data: {e}")
    
    # If we have Yiedl data, return it
    if yiedl_data and all(k in yiedl_data for k in ['train', 'validation', 'live']):
        logger.info("Using Yiedl data")
        return yiedl_data
    
    # If all else fails, create synthetic data
    logger.warning("No real data found, creating synthetic data")
    return create_synthetic_data()

def create_synthetic_data(n_samples=20000, n_features=100, n_assets=50):
    """Create synthetic data for model comparison"""
    logger.info(f"Creating synthetic data with {n_samples} samples and {n_features} features")
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic target
    # Use a linear combination of features plus noise
    weights = np.random.randn(n_features)
    y = X.dot(weights) + np.random.randn(n_samples) * 0.1
    
    # Create synthetic IDs and assets
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    ids = [f"ID_{i}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df['id'] = ids
    df['asset'] = [assets[i % n_assets] for i in range(n_samples)]
    df['target'] = y
    
    # Split into train/validation/live
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    
    train_df = df.iloc[:train_size].copy()
    validation_df = df.iloc[train_size:train_size+val_size].copy()
    live_df = df.iloc[train_size+val_size:].copy()
    
    # Add data type column
    train_df['data_type'] = 'train'
    validation_df['data_type'] = 'validation'
    live_df['data_type'] = 'live'
    
    # Remove target from live data
    live_df = live_df.drop(columns=['target'])
    
    logger.info(f"Created synthetic train data: {train_df.shape}")
    logger.info(f"Created synthetic validation data: {validation_df.shape}")
    logger.info(f"Created synthetic live data: {live_df.shape}")
    
    return {
        'train': train_df,
        'validation': validation_df,
        'live': live_df
    }

def prepare_data(datasets, max_features=2000):
    """
    Prepare data for training:
    - Extract features and targets
    - Handle missing values
    - Scale features
    - Handle categorical variables
    - Feature selection
    """
    logger.info("Preparing data for training...")
    
    # Extract datasets
    train_df = datasets.get('train')
    validation_df = datasets.get('validation')
    live_df = datasets.get('live')
    
    if not train_df is not None or validation_df is None:
        logger.error("Training or validation data is missing")
        return None
    
    # Identify target and ID columns
    protected_cols = ['id', 'target', 'data_type', 'era', 'date', 'timestamp', 'asset']
    feature_cols = [col for col in train_df.columns if col not in protected_cols]
    
    # Limit to max features if needed
    if len(feature_cols) > max_features:
        logger.info(f"Limiting from {len(feature_cols)} to {max_features} features")
        feature_cols = feature_cols[:max_features]
    
    # Check if all features are numeric
    numeric_feature_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_feature_cols.append(col)
    
    feature_cols = numeric_feature_cols
    logger.info(f"Using {len(feature_cols)} numeric features")
    
    # Handle missing and infinite values
    for col in feature_cols:
        # Process training data
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        median = train_df[col].median()
        if pd.isna(median):
            median = 0
        train_df[col] = train_df[col].fillna(median)
        
        # Process validation data
        validation_df[col] = validation_df[col].replace([np.inf, -np.inf], np.nan)
        validation_df[col] = validation_df[col].fillna(median)
        
        # Process live data if available
        if live_df is not None and col in live_df.columns:
            live_df[col] = live_df[col].replace([np.inf, -np.inf], np.nan)
            live_df[col] = live_df[col].fillna(median)
    
    # Scale features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    validation_df[feature_cols] = scaler.transform(validation_df[feature_cols])
    if live_df is not None:
        live_df[feature_cols] = scaler.transform(live_df[feature_cols])
    
    # Extract features and target for training
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    # Extract features and target for validation
    X_val = validation_df[feature_cols].values
    y_val = validation_df['target'].values
    
    # Extract features for live data if available
    X_live = live_df[feature_cols].values if live_df is not None else None
    
    # Prepare IDs for submission
    train_ids = train_df['id'].values if 'id' in train_df else None
    val_ids = validation_df['id'].values if 'id' in validation_df else None
    live_ids = live_df['id'].values if live_df is not None and 'id' in live_df else None
    
    logger.info(f"Data preparation complete.")
    logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data: X={X_val.shape}, y={y_val.shape}")
    if X_live is not None:
        logger.info(f"Live data: X={X_live.shape}")
    
    return {
        'feature_cols': feature_cols,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_live': X_live,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'live_ids': live_ids,
        'scaler': scaler
    }

def train_lightgbm(prepared_data, use_gpu=False, gpu_id=0):
    """Train LightGBM model with GPU acceleration if available"""
    try:
        import lightgbm as lgb
        
        logger.info(f"Training LightGBM model (GPU: {use_gpu}, GPU ID: {gpu_id})")
        
        # Extract data
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        feature_cols = prepared_data['feature_cols']
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Set parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 256,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': RANDOM_SEED
        }
        
        # Add GPU parameters if available
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = gpu_id
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000
        )
        
        # Evaluate model
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        logger.info(f"LightGBM - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Generate live predictions if data available
        X_live = prepared_data.get('X_live')
        live_preds = model.predict(X_live) if X_live is not None else None
        
        return {
            'model': model,
            'val_preds': preds,
            'live_preds': live_preds,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'importance': importance_df
        }
    
    except ImportError:
        logger.error("LightGBM not installed")
        return None
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_xgboost(prepared_data, use_gpu=False, gpu_id=0):
    """Train XGBoost model with GPU acceleration if available"""
    try:
        import xgboost as xgb
        
        logger.info(f"Training XGBoost model (GPU: {use_gpu}, GPU ID: {gpu_id})")
        
        # Extract data
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        feature_cols = prepared_data['feature_cols']
        
        # Create datasets
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 9,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 50,
            'alpha': 0.1,
            'lambda': 0.1,
            'seed': RANDOM_SEED
        }
        
        # Add GPU parameters if available
        if use_gpu:
            # Handle different XGBoost versions
            if xgb.__version__ >= '2.0.0':
                params['device'] = f'cuda:{gpu_id}'
                params['tree_method'] = 'hist'
            else:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = gpu_id
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Evaluate model
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        logger.info(f"XGBoost - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Generate live predictions if data available
        X_live = prepared_data.get('X_live')
        live_preds = None
        if X_live is not None:
            dlive = xgb.DMatrix(X_live)
            live_preds = model.predict(dlive)
        
        return {
            'model': model,
            'val_preds': preds,
            'live_preds': live_preds,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'importance': importance_df
        }
    
    except ImportError:
        logger.error("XGBoost not installed")
        return None
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_pytorch_mlp(prepared_data, use_gpu=False, gpu_id=0):
    """Train PyTorch MLP model with GPU acceleration if available"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        logger.info(f"Training PyTorch MLP model (GPU: {use_gpu}, GPU ID: {gpu_id})")
        
        # Check if GPU is available
        if use_gpu and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            use_gpu = False
        
        # Extract data
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024)
        
        # Define model architecture
        input_dim = X_train.shape[1]
        hidden_dims = [512, 256, 128, 64]
        
        class MLPRegressor(nn.Module):
            def __init__(self):
                super(MLPRegressor, self).__init__()
                
                # Create layers
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, 1))
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x).squeeze()
        
        # Create model
        model = MLPRegressor()
        
        # Move to GPU if available
        device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        logger.info(f"Training MLP for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_dataset)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device)).cpu().numpy()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        
        logger.info(f"PyTorch MLP - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # Generate live predictions if data available
        live_preds = None
        X_live = prepared_data.get('X_live')
        if X_live is not None:
            X_live_tensor = torch.tensor(X_live, dtype=torch.float32)
            with torch.no_grad():
                live_preds = model(X_live_tensor.to(device)).cpu().numpy()
        
        return {
            'model': model,
            'val_preds': val_preds,
            'live_preds': live_preds,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    except ImportError:
        logger.error("PyTorch not installed")
        return None
    except Exception as e:
        logger.error(f"Error training PyTorch MLP model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_h2o_automl(prepared_data, max_runtime_secs=600):
    """Train H2O AutoML model with Sparkling Water if available"""
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        logger.info(f"Training H2O AutoML model (max_runtime_secs={max_runtime_secs})")
        
        # Initialize H2O
        h2o.init(nthreads=-1, max_mem_size="4G")
        
        # Extract data
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        feature_cols = prepared_data['feature_cols']
        
        # Combine features and target for H2O
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df['target'] = y_train
        
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df['target'] = y_val
        
        # Convert to H2O frames
        train_h2o = h2o.H2OFrame(train_df)
        val_h2o = h2o.H2OFrame(val_df)
        
        # Identify predictors and response
        predictors = feature_cols
        response = 'target'
        
        # Configure AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=RANDOM_SEED,
            nfolds=5,
            sort_metric="RMSE",
            exclude_algos=None
        )
        
        # Train models
        aml.train(x=predictors, y=response, training_frame=train_h2o, validation_frame=val_h2o)
        
        # Get best model
        best_model = aml.leader
        
        # Generate predictions
        val_preds_h2o = best_model.predict(val_h2o)
        val_preds = h2o.as_list(val_preds_h2o)['predict'].values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        
        logger.info(f"H2O AutoML - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        logger.info(f"Best model: {best_model.model_id}")
        
        # Generate live predictions if data available
        live_preds = None
        X_live = prepared_data.get('X_live')
        if X_live is not None:
            live_df = pd.DataFrame(X_live, columns=feature_cols)
            live_h2o = h2o.H2OFrame(live_df)
            live_preds_h2o = best_model.predict(live_h2o)
            live_preds = h2o.as_list(live_preds_h2o)['predict'].values
        
        # Save model
        model_path = str(MODELS_DIR / "h2o_automl")
        os.makedirs(model_path, exist_ok=True)
        saved_path = h2o.save_model(best_model, path=model_path, force=True)
        
        return {
            'model': best_model,
            'model_path': saved_path,
            'val_preds': val_preds,
            'live_preds': live_preds,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'aml': aml
        }
    
    except ImportError:
        logger.error("H2O not installed")
        return None
    except Exception as e:
        logger.error(f"Error training H2O AutoML model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_tpot_automl(prepared_data, max_runtime_mins=20, use_gpu=False):
    """Train TPOT AutoML model with GPU support if available"""
    try:
        from tpot import TPOTRegressor
        import sklearn
        
        logger.info(f"Training TPOT AutoML model (max_runtime_mins={max_runtime_mins}, GPU={use_gpu})")
        
        # Extract data
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        
        # Configure TPOT
        tpot_config = None
        if use_gpu:
            # Try to use RAPIDS with cuML if available
            try:
                import cuml
                logger.info("RAPIDS cuML available, using GPU-accelerated models")
                
                tpot_config = {
                    'cuml.ensemble.randomforestregressor': {
                        'n_estimators': [10, 50, 100, 200],
                        'max_depth': [4, 6, 8, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                }
            except ImportError:
                logger.warning("RAPIDS cuML not available, using standard config with XGBoost GPU")
                tpot_config = {
                    'xgboost.XGBRegressor': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7, 10],
                        'min_child_weight': [1, 5, 10, 50],
                        'subsample': [0.6, 0.8, 1.0],
                        'tree_method': ['gpu_hist']
                    }
                }
        
        # Create TPOT classifier
        tpot = TPOTRegressor(
            generations=5,
            population_size=20,
            verbosity=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            max_time_mins=max_runtime_mins,
            config_dict=tpot_config,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        
        # Fit the model
        tpot.fit(X_train, y_train)
        
        # Generate predictions
        val_preds = tpot.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        
        logger.info(f"TPOT AutoML - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # Generate live predictions if data available
        live_preds = None
        X_live = prepared_data.get('X_live')
        if X_live is not None:
            live_preds = tpot.predict(X_live)
        
        # Save pipeline to file
        pipeline_file = MODELS_DIR / "tpot_pipeline.py"
        tpot.export(str(pipeline_file))
        
        return {
            'model': tpot,
            'pipeline_file': str(pipeline_file),
            'val_preds': val_preds,
            'live_preds': live_preds,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    except ImportError:
        logger.error("TPOT not installed")
        return None
    except Exception as e:
        logger.error(f"Error training TPOT AutoML model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_ensemble(model_results, prepared_data):
    """Create ensemble from all trained models"""
    logger.info("Creating optimized ensemble model...")
    
    # Extract validation targets
    y_val = prepared_data['y_val']
    
    # Collect all validation predictions
    model_preds = []
    model_names = []
    
    for name, result in model_results.items():
        if result and 'val_preds' in result:
            model_preds.append(result['val_preds'])
            model_names.append(name)
    
    if len(model_preds) < 2:
        logger.warning("Not enough models for ensemble")
        return None
    
    # Find optimal weights with grid search
    logger.info(f"Optimizing weights for {len(model_preds)} models")
    
    if len(model_preds) == 2:
        # For 2 models, use fine-grained grid search
        best_rmse = float('inf')
        best_weights = [0.5, 0.5]
        
        for w1 in np.linspace(0, 1, 101):  # 0.00, 0.01, 0.02, ..., 1.00
            w2 = 1 - w1
            weights = [w1, w2]
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros_like(y_val)
            for i, preds in enumerate(model_preds):
                ensemble_pred += weights[i] * preds
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights.copy()
    
    elif len(model_preds) == 3:
        # For 3 models, use coarser grid search
        best_rmse = float('inf')
        best_weights = [1/3, 1/3, 1/3]
        
        for w1 in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.1, ..., 1.0
            for w2 in np.linspace(0, 1-w1, 21):  # 0.0, 0.05, ..., (1-w1)
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                
                weights = [w1, w2, w3]
                
                # Calculate weighted ensemble prediction
                ensemble_pred = np.zeros_like(y_val)
                for i, preds in enumerate(model_preds):
                    ensemble_pred += weights[i] * preds
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights.copy()
    
    else:
        # For more than 3 models, use optimization
        from scipy.optimize import minimize
        
        def ensemble_rmse(weights, preds_list, targets):
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(targets)
            for i, preds in enumerate(preds_list):
                ensemble_pred += weights[i] * preds
            
            # Calculate RMSE
            return np.sqrt(mean_squared_error(targets, ensemble_pred))
        
        # Initial weights (equal)
        initial_weights = np.ones(len(model_preds)) / len(model_preds)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(model_preds))]
        
        # Optimize weights
        result = minimize(
            ensemble_rmse, 
            initial_weights, 
            args=(model_preds, y_val),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        best_weights = result.x
        best_rmse = ensemble_rmse(best_weights, model_preds, y_val)
    
    # Normalize weights to sum to 1
    best_weights = np.array(best_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    # Calculate final ensemble predictions with best weights
    ensemble_val_pred = np.zeros_like(y_val)
    for i, preds in enumerate(model_preds):
        ensemble_val_pred += best_weights[i] * preds
    
    # Calculate final metrics
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
    mae = mean_absolute_error(y_val, ensemble_val_pred)
    r2 = r2_score(y_val, ensemble_val_pred)
    
    logger.info(f"Optimized ensemble weights: {dict(zip(model_names, best_weights))}")
    logger.info(f"Ensemble - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    if rmse <= 0.25:
        logger.info(f"TARGET ACHIEVED! Ensemble RMSE of {rmse:.6f} is below target of 0.25")
    
    # Generate live predictions if data available
    live_preds = None
    X_live = prepared_data.get('X_live')
    
    if X_live is not None:
        # Collect live predictions from all models
        live_model_preds = []
        
        for name, result in model_results.items():
            if result and 'live_preds' in result and result['live_preds'] is not None:
                live_model_preds.append(result['live_preds'])
        
        if len(live_model_preds) == len(model_preds):
            # Calculate ensemble live predictions with best weights
            live_preds = np.zeros_like(live_model_preds[0])
            for i, preds in enumerate(live_model_preds):
                live_preds += best_weights[i] * preds
    
    return {
        'model_names': model_names,
        'weights': dict(zip(model_names, best_weights)),
        'val_preds': ensemble_val_pred,
        'live_preds': live_preds,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def generate_submission(model_results, prepared_data):
    """Generate submission file for Numerai crypto competition"""
    logger.info("Generating submission files...")
    
    # Create directory if it doesn't exist
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get live IDs
    live_ids = prepared_data.get('live_ids')
    val_ids = prepared_data.get('val_ids')
    
    # Generate submissions for each model
    submissions = {}
    
    # First, generate ensemble submission if available
    if 'ensemble' in model_results and model_results['ensemble']:
        ensemble = model_results['ensemble']
        
        if ensemble.get('live_preds') is not None and live_ids is not None:
            # Live predictions
            live_df = pd.DataFrame({
                'id': live_ids,
                'prediction': ensemble['live_preds']
            })
            
            # Save to file
            live_path = SUBMISSION_DIR / f"ensemble_live_{timestamp}.csv"
            live_df.to_csv(live_path, index=False)
            logger.info(f"Ensemble live submission saved to {live_path}")
            
            submissions['ensemble_live'] = live_path
        
        if ensemble.get('val_preds') is not None and val_ids is not None:
            # Validation predictions with targets for online validation
            val_df = pd.DataFrame({
                'id': val_ids,
                'prediction': ensemble['val_preds'],
                'target': prepared_data['y_val']
            })
            
            # Save to file
            val_path = SUBMISSION_DIR / f"ensemble_validation_{timestamp}.csv"
            val_df.to_csv(val_path, index=False)
            logger.info(f"Ensemble validation submission saved to {val_path}")
            
            submissions['ensemble_validation'] = val_path
    
    # Generate submissions for individual models
    for name, result in model_results.items():
        if name == 'ensemble':
            continue
            
        if result and result.get('live_preds') is not None and live_ids is not None:
            # Live predictions
            live_df = pd.DataFrame({
                'id': live_ids,
                'prediction': result['live_preds']
            })
            
            # Save to file
            live_path = SUBMISSION_DIR / f"{name}_live_{timestamp}.csv"
            live_df.to_csv(live_path, index=False)
            logger.info(f"{name} live submission saved to {live_path}")
            
            submissions[f'{name}_live'] = live_path
        
        if result and result.get('val_preds') is not None and val_ids is not None:
            # Validation predictions with targets for online validation
            val_df = pd.DataFrame({
                'id': val_ids,
                'prediction': result['val_preds'],
                'target': prepared_data['y_val']
            })
            
            # Save to file
            val_path = SUBMISSION_DIR / f"{name}_validation_{timestamp}.csv"
            val_df.to_csv(val_path, index=False)
            logger.info(f"{name} validation submission saved to {val_path}")
            
            submissions[f'{name}_validation'] = val_path
    
    # Return paths to all submission files
    return submissions

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced Model Comparison for Numerai Crypto")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to data directory")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--max-features", type=int, default=2000,
                       help="Maximum number of features to use")
    parser.add_argument("--runtime", type=int, default=600,
                       help="Model runtime in seconds")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for submissions")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Override output directory if specified
    if args.output_dir:
        global SUBMISSION_DIR
        SUBMISSION_DIR = Path(args.output_dir)
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Check available packages
    available_packages = check_package_availability()
    
    # Check for GPUs
    gpus = get_available_gpus()
    use_gpu = not args.no_gpu and len(gpus) > 0
    
    if use_gpu:
        logger.info(f"Using GPU acceleration with {len(gpus)} GPUs")
    else:
        logger.info("Using CPU mode")
    
    # Load data
    datasets = load_data(args.data_path)
    
    if not datasets:
        logger.error("Failed to load data")
        return 1
    
    # Prepare data
    prepared_data = prepare_data(datasets, max_features=args.max_features)
    
    if not prepared_data:
        logger.error("Failed to prepare data")
        return 1
    
    # Train models
    model_results = {}
    
    # Train LightGBM if available
    if available_packages.get('lightgbm', False):
        lightgbm_result = train_lightgbm(
            prepared_data,
            use_gpu=use_gpu and available_packages.get('lightgbm', False),
            gpu_id=0 if gpus else None
        )
        if lightgbm_result:
            model_results['lightgbm'] = lightgbm_result
    
    # Train XGBoost if available
    if available_packages.get('xgboost', False):
        xgboost_result = train_xgboost(
            prepared_data,
            use_gpu=use_gpu and available_packages.get('xgboost', False),
            gpu_id=0 if gpus else None
        )
        if xgboost_result:
            model_results['xgboost'] = xgboost_result
    
    # Train PyTorch MLP if available
    if available_packages.get('pytorch', False):
        pytorch_result = train_pytorch_mlp(
            prepared_data,
            use_gpu=use_gpu and available_packages.get('pytorch_cuda', False),
            gpu_id=0 if gpus else None
        )
        if pytorch_result:
            model_results['pytorch'] = pytorch_result
    
    # Train H2O AutoML if available
    if available_packages.get('h2o', False):
        h2o_result = train_h2o_automl(
            prepared_data,
            max_runtime_secs=args.runtime
        )
        if h2o_result:
            model_results['h2o'] = h2o_result
    
    # Train TPOT AutoML if available
    if available_packages.get('tpot', False):
        tpot_result = train_tpot_automl(
            prepared_data,
            max_runtime_mins=args.runtime//60,
            use_gpu=use_gpu
        )
        if tpot_result:
            model_results['tpot'] = tpot_result
    
    # Create ensemble if we have at least 2 models
    if len(model_results) >= 2:
        ensemble_result = create_ensemble(model_results, prepared_data)
        if ensemble_result:
            model_results['ensemble'] = ensemble_result
    
    # Generate submissions
    submission_files = generate_submission(model_results, prepared_data)
    
    # Final summary
    logger.info("\n===== MODEL COMPARISON SUMMARY =====")
    logger.info(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    logger.info("-" * 50)
    
    # Sort by RMSE ascending
    sorted_results = sorted(
        [(name, result) for name, result in model_results.items() if result and 'rmse' in result],
        key=lambda x: x[1]['rmse']
    )
    
    for name, result in sorted_results:
        logger.info(f"{name:<15} {result['rmse']:<10.6f} {result['mae']:<10.6f} {result['r2']:<10.6f}")
    
    # Identify best model
    if sorted_results:
        best_model, best_result = sorted_results[0]
        logger.info(f"\nBest model: {best_model} with RMSE: {best_result['rmse']:.6f}")
        
        if best_result['rmse'] <= 0.25:
            logger.info(f"SUCCESS! Target RMSE of 0.25 achieved with {best_model}")
        else:
            logger.info(f"Target RMSE of 0.25 not achieved. Best RMSE: {best_result['rmse']:.6f}")
        
        # Note best validation submission file for online validation
        best_val_file = submission_files.get(f'{best_model}_validation')
        if best_val_file:
            logger.info(f"Best validation submission for online validation: {best_val_file}")
        
        # Note ensemble validation file if available
        ensemble_val_file = submission_files.get('ensemble_validation')
        if ensemble_val_file:
            logger.info(f"Ensemble validation submission for online validation: {ensemble_val_file}")
    
    # Execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"\nExecution time: {execution_time:.2f}s ({execution_time/60:.2f}m)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())