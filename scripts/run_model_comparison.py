#!/usr/bin/env python3
"""
Model Comparison for Numerai Crypto Competition

This script compares multiple machine learning models on the Numerai Crypto dataset:
- LightGBM (GPU-accelerated)
- XGBoost (GPU-accelerated)
- H2O AutoML (CPU-based but highly parallel)
- PyTorch LSTM (GPU-accelerated)
- PyTorch CNN (GPU-accelerated)
- CatBoost (GPU-accelerated)
- Ensemble of all models

With extensive feature engineering to utilize 600GB RAM.
Each model is timeboxed to 12 minutes maximum runtime.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import multiprocessing

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

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# GPU utilities
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

# Generate synthetic data with many features
def create_synthetic_data(n_samples=10000, n_features=1000, n_assets=100, forecast_days=20):
    """Create synthetic crypto data with many features to utilize RAM"""
    logger.info(f"Creating synthetic data with {n_samples} samples, {n_features} features, {n_assets} assets")
    
    asset_ids = [f"CRYPTO_{i}" for i in range(n_assets)]
    dates = pd.date_range(start='2020-01-01', periods=n_samples//n_assets + forecast_days, freq='D')
    
    data = []
    
    for asset_id in asset_ids:
        # Base price series with random walk
        base_series = np.random.randn(len(dates)).cumsum()
        
        # Add seasonality and trend
        seasonality = 0.2 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))
        trend = np.linspace(0, 5, len(dates))
        price_series = base_series + seasonality + trend
        
        # Generate samples with target and features
        for i in range(len(dates) - forecast_days):
            sample = {
                'id': f"{asset_id}_{i}",
                'asset': asset_id,
                'date': dates[i],
                'era': i
            }
            
            # Generate raw features (time series lags, moving averages, volatility)
            for j in range(min(50, n_features)):
                if j < 20:  # Price lags
                    lag = j + 1
                    if i - lag >= 0:
                        sample[f'price_lag_{lag}'] = price_series[i - lag]
                    else:
                        sample[f'price_lag_{lag}'] = np.nan
                elif j < 35:  # Moving averages
                    window = (j - 20) * 2 + 2
                    if i - window >= 0:
                        sample[f'ma_{window}'] = np.mean(price_series[i-window:i])
                    else:
                        sample[f'ma_{window}'] = np.nan
                elif j < 50:  # Volatility
                    window = (j - 35) * 2 + 2
                    if i - window >= 0:
                        sample[f'vol_{window}'] = np.std(price_series[i-window:i])
                    else:
                        sample[f'vol_{window}'] = np.nan
            
            # Add target (n-day ahead return)
            if i + forecast_days < len(dates):
                forward_return = price_series[i + forecast_days] - price_series[i]
                # Binary target
                sample['target_binary'] = 1 if forward_return > 0 else 0
                # Regression target (normalized)
                sample['target'] = np.tanh(forward_return)  # Squash to [-1, 1]
            
            data.append(sample)
    
    df = pd.DataFrame(data)
    df = df.fillna(method='bfill').fillna(0)
    
    # Generate additional polynomial features to use more RAM
    logger.info("Generating polynomial features to utilize RAM...")
    base_features = [col for col in df.columns if col.startswith(('price_lag_', 'ma_', 'vol_'))]
    
    # Calculate how many polynomial features we need based on available RAM
    ram_gb = int(os.environ.get("MODEL_RAM_GB", "500"))
    approx_bytes_per_float = 8  # Each float is roughly 8 bytes
    approx_bytes_per_feature = approx_bytes_per_float * len(df)
    approx_gb_per_feature = approx_bytes_per_feature / (1024**3)
    
    target_features = min(n_features, int(ram_gb * 0.7 / approx_gb_per_feature))
    features_to_add = max(0, target_features - len(df.columns))
    
    logger.info(f"Adding {features_to_add} polynomial features to utilize {ram_gb}GB RAM")
    
    # Start with pairwise interactions for most important features
    top_features = base_features[:min(20, len(base_features))]
    n_added = 0
    
    # Add polynomial features in batches to avoid memory spikes
    batch_size = 100
    
    # 1. Pairwise interactions
    for i, feat1 in enumerate(top_features):
        batch_features = []
        for j, feat2 in enumerate(top_features[i+1:]):
            if n_added >= features_to_add:
                break
                
            # Create interaction feature
            col_name = f"{feat1}_x_{feat2}"
            df[col_name] = df[feat1] * df[feat2]
            n_added += 1
            
            if n_added % batch_size == 0:
                logger.info(f"Added {n_added} features so far")
                
        if n_added >= features_to_add:
            break
    
    # 2. Polynomial expansions if we still need more features
    if n_added < features_to_add:
        for power in range(2, 4):  # Squares and cubes
            batch_features = []
            for feat in base_features:
                if n_added >= features_to_add:
                    break
                    
                col_name = f"{feat}_power_{power}"
                df[col_name] = df[feat] ** power
                n_added += 1
                
                if n_added % batch_size == 0:
                    logger.info(f"Added {n_added} features so far")
                    
            if n_added >= features_to_add:
                break
    
    # 3. Random features if we still need more features
    if n_added < features_to_add:
        batch_size = 1000  # Larger batches for random features
        remaining = features_to_add - n_added
        batches = (remaining + batch_size - 1) // batch_size
        
        for batch in range(batches):
            batch_features = min(batch_size, remaining - batch * batch_size)
            if batch_features <= 0:
                break
                
            logger.info(f"Adding batch of {batch_features} random features")
            for i in range(batch_features):
                col_name = f"random_feature_{n_added + i}"
                df[col_name] = np.random.randn(len(df))
            
            n_added += batch_features
    
    logger.info(f"Final dataframe shape: {df.shape}")
    
    return df

# Feature engineering and selection
def engineer_features(df, max_features=5000):
    """
    Engineer additional features from the base features
    using parallel processing for large dataframes
    """
    logger.info(f"Starting feature engineering with dataframe of shape {df.shape}")
    
    # Identify features and non-features
    non_feature_cols = ['id', 'target', 'target_binary', 'era', 'date', 'asset', 'data_type']
    base_features = [col for col in df.columns if col not in non_feature_cols]
    
    logger.info(f"Starting with {len(base_features)} base features")
    
    # Only keep up to max_features
    if len(base_features) > max_features:
        logger.info(f"Reducing feature count from {len(base_features)} to {max_features}")
        # Keep most important features based on naming pattern (domain knowledge)
        priority_patterns = [
            'price_lag_', 'ma_', 'vol_',  # Original features first
            '_x_', '_power_'  # Then interactions and polynomials
        ]
        
        selected_features = []
        for pattern in priority_patterns:
            pattern_features = [f for f in base_features if pattern in f]
            selected_features.extend(pattern_features)
            if len(selected_features) >= max_features:
                selected_features = selected_features[:max_features]
                break
        
        # If still not enough, add random features
        if len(selected_features) < max_features:
            random_features = [f for f in base_features if f not in selected_features]
            random.shuffle(random_features)
            selected_features.extend(random_features[:max_features-len(selected_features)])
        
        # Keep only selected features
        base_features = selected_features[:max_features]
    
    feature_cols = base_features
    logger.info(f"Using {len(feature_cols)} features for modeling")
    
    return df, feature_cols

# Model classes
class LightGBMModel:
    """LightGBM model with GPU acceleration"""
    
    def __init__(self, use_gpu=True, gpu_id=0, params=None, time_limit=720):
        self.name = "lightgbm"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.time_limit = time_limit  # seconds
        self.start_time = None
        
        # Default parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': RANDOM_SEED
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': self.gpu_id
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model"""
        import lightgbm as lgb
        
        self.start_time = time.time()
        logger.info(f"Training LightGBM model with {X_train.shape[1]} features")
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        callbacks = [lgb.callback.early_stopping(50), lgb.callback.log_evaluation(100)]
        
        # Add time limit callback
        def time_limit_callback():
            def callback(env):
                elapsed = time.time() - self.start_time
                if elapsed > self.time_limit:
                    logger.info(f"Time limit reached at {elapsed:.2f} seconds. Stopping training.")
                    raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
                return False
            return callback
        
        callbacks.append(time_limit_callback())
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=10000,  # high number, will be stopped by callbacks
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            verbose_eval=False
        )
        
        training_time = time.time() - self.start_time
        logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importance'):
            self.feature_importance = dict(zip(
                feature_names if feature_names else [f'f{i}' for i in range(X_train.shape[1])],
                self.model.feature_importance(importance_type='gain')
            ))
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance if hasattr(self, 'feature_importance') else {}

class XGBoostModel:
    """XGBoost model with GPU acceleration"""
    
    def __init__(self, use_gpu=True, gpu_id=0, params=None, time_limit=720):
        self.name = "xgboost"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.time_limit = time_limit  # seconds
        self.start_time = None
        
        # Default parameters
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': RANDOM_SEED,
            'silent': 1
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model"""
        import xgboost as xgb
        
        self.start_time = time.time()
        logger.info(f"Training XGBoost model with {X_train.shape[1]} features")
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            watchlist.append((dval, 'valid'))
        
        # Define callback for time limit
        class TimeoutCallback(xgb.callback.TrainingCallback):
            def __init__(self, start_time, time_limit):
                self.start_time = start_time
                self.time_limit = time_limit
            
            def after_iteration(self, model, epoch, evals_log):
                elapsed = time.time() - self.start_time
                if elapsed > self.time_limit:
                    logger.info(f"Time limit reached at {elapsed:.2f} seconds. Stopping training.")
                    return True
                return False
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=10000,  # high number, will be stopped by callback
            evals=watchlist,
            early_stopping_rounds=50,
            verbose_eval=100,
            callbacks=[TimeoutCallback(self.start_time, self.time_limit)]
        )
        
        training_time = time.time() - self.start_time
        logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        # Get feature importance
        if feature_names:
            self.feature_importance = self.model.get_score(importance_type='gain')
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance if hasattr(self, 'feature_importance') else {}

class CatBoostModel:
    """CatBoost model with GPU acceleration"""
    
    def __init__(self, use_gpu=True, gpu_id=0, params=None, time_limit=720):
        self.name = "catboost"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.time_limit = time_limit  # seconds
        self.start_time = None
        
        # Default parameters
        self.params = {
            'loss_function': 'RMSE',
            'iterations': 10000,
            'learning_rate': 0.05,
            'depth': 8,
            'random_seed': RANDOM_SEED,
            'verbose': 100
        }
        
        # Update with provided parameters
        if params:
            self.params.update(params)
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'task_type': 'GPU',
                'devices': str(self.gpu_id)
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model"""
        from catboost import CatBoost, Pool
        
        self.start_time = time.time()
        logger.info(f"Training CatBoost model with {X_train.shape[1]} features")
        
        # Prepare data
        train_pool = Pool(X_train, y_train, feature_names=feature_names)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val, feature_names=feature_names)
        
        # Set time limit
        self.params['early_stopping_rounds'] = 50
        if self.time_limit:
            self.params['time_limit'] = self.time_limit
        
        # Train model
        self.model = CatBoost(self.params)
        self.model.fit(train_pool, eval_set=eval_set, verbose=False)
        
        training_time = time.time() - self.start_time
        logger.info(f"CatBoost training completed in {training_time:.2f} seconds")
        
        # Get feature importance
        self.feature_importance = dict(zip(
            feature_names if feature_names else [f'f{i}' for i in range(X_train.shape[1])],
            self.model.get_feature_importance()
        ))
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance if hasattr(self, 'feature_importance') else {}

class PyTorchLSTMModel:
    """PyTorch LSTM model with GPU acceleration"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3, use_gpu=True, gpu_id=0, time_limit=720):
        self.name = "pytorch_lstm"
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.time_limit = time_limit  # seconds
        self.start_time = None
        
        import torch
        self.device = torch.device(f"cuda:{gpu_id}" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch LSTM using device: {self.device}")
    
    def _create_model(self):
        """Create the model architecture"""
        import torch.nn as nn
        
        class LSTMRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super(LSTMRegressor, self).__init__()
                
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dims[0],
                    num_layers=len(hidden_dims),
                    batch_first=True,
                    dropout=dropout if len(hidden_dims) > 1 else 0
                )
                
                layers = []
                for i in range(len(hidden_dims)-1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                
                layers.append(nn.Linear(hidden_dims[-1], 1))
                
                self.fc_layers = nn.Sequential(*layers)
            
            def forward(self, x):
                # x shape: (batch_size, seq_len=1, input_dim)
                lstm_out, _ = self.lstm(x)
                
                # Use only the last timestep
                lstm_out = lstm_out[:, -1, :]
                
                # Pass through fully connected layers
                output = self.fc_layers(lstm_out)
                
                return output.squeeze()
        
        return LSTMRegressor(self.input_dim, self.hidden_dims, self.dropout).to(self.device)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        self.start_time = time.time()
        logger.info(f"Training PyTorch LSTM model with {X_train.shape[1]} features")
        
        # Create model
        self.model = self._create_model()
        
        # Reshape input for LSTM [batch, sequence_length=1, features]
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=256)
        else:
            val_loader = None
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        epochs = 1000  # high number, will be stopped by time limit
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Check time limit
            if time.time() - self.start_time > self.time_limit:
                logger.info(f"Time limit reached after {epoch} epochs. Stopping training.")
                break
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_dataset)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_dataset)
                self.history['val_loss'].append(val_loss)
                
                # Learning rate scheduler
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}"
                if val_loader:
                    msg += f", Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        # Restore best model if validation was used
        if val_loader and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - self.start_time
        logger.info(f"PyTorch LSTM training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import torch
        
        # Reshape input for LSTM
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def get_feature_importance(self):
        """
        Generate approximate feature importance by measuring
        the change in loss when each feature is zeroed out
        """
        return {}  # Not implemented for PyTorch

class PyTorchCNNModel:
    """PyTorch CNN model with GPU acceleration"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3, use_gpu=True, gpu_id=0, time_limit=720):
        self.name = "pytorch_cnn"
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.time_limit = time_limit  # seconds
        self.start_time = None
        
        import torch
        self.device = torch.device(f"cuda:{gpu_id}" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch CNN using device: {self.device}")
    
    def _create_model(self):
        """Create the model architecture"""
        import torch.nn as nn
        
        class CNNRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super(CNNRegressor, self).__init__()
                
                # Reshape input for 1D CNN (batch, channels=1, sequence=input_dim)
                self.reshape = lambda x: x.view(x.size(0), 1, x.size(1))
                
                # 1D CNN layers
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout)
                )
                
                # Calculate the size of the flattened CNN output
                self.fc_input_dim = 128 * (input_dim // 4)  # Due to two maxpool layers with stride 2
                
                # Fully connected layers
                fc_layers = []
                fc_layers.append(nn.Linear(self.fc_input_dim, hidden_dims[0]))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
                
                for i in range(len(hidden_dims)-1):
                    fc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    fc_layers.append(nn.ReLU())
                    fc_layers.append(nn.Dropout(dropout))
                
                fc_layers.append(nn.Linear(hidden_dims[-1], 1))
                
                self.fc_layers = nn.Sequential(*fc_layers)
            
            def forward(self, x):
                # Reshape for CNN
                x = self.reshape(x)
                
                # Pass through CNN layers
                x = self.conv_layers(x)
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Pass through fully connected layers
                output = self.fc_layers(x)
                
                return output.squeeze()
        
        return CNNRegressor(self.input_dim, self.hidden_dims, self.dropout).to(self.device)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        self.start_time = time.time()
        logger.info(f"Training PyTorch CNN model with {X_train.shape[1]} features")
        
        # Create model
        self.model = self._create_model()
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=256)
        else:
            val_loader = None
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        epochs = 1000  # high number, will be stopped by time limit
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Check time limit
            if time.time() - self.start_time > self.time_limit:
                logger.info(f"Time limit reached after {epoch} epochs. Stopping training.")
                break
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_dataset)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_dataset)
                self.history['val_loss'].append(val_loss)
                
                # Learning rate scheduler
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}"
                if val_loader:
                    msg += f", Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        # Restore best model if validation was used
        if val_loader and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - self.start_time
        logger.info(f"PyTorch CNN training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import torch
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def get_feature_importance(self):
        """Feature importance not directly available for CNN"""
        return {}

class H2OAutoMLModel:
    """H2O AutoML model"""
    
    def __init__(self, time_limit=720, nfolds=5, seed=RANDOM_SEED):
        self.name = "h2o_automl"
        self.time_limit = time_limit
        self.nfolds = nfolds
        self.seed = seed
        self.start_time = None
        
        # Initialize H2O
        self._init_h2o()
    
    def _init_h2o(self):
        """Initialize H2O if not already running"""
        import h2o
        
        try:
            h2o.init(nthreads=-1, max_mem_size="200G")
            self.h2o = h2o
            logger.info(f"H2O version: {h2o.__version__}")
        except Exception as e:
            logger.error(f"Error initializing H2O: {e}")
            raise
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the model using H2O AutoML"""
        from h2o.automl import H2OAutoML
        
        self.start_time = time.time()
        logger.info(f"Training H2O AutoML model with {X_train.shape[1]} features")
        
        # Combine X and y into a single DataFrame
        train_df = pd.DataFrame(X_train, columns=feature_names if feature_names else [f'f{i}' for i in range(X_train.shape[1])])
        train_df['target'] = y_train
        
        # Convert to H2O Frame
        train_h2o = self.h2o.H2OFrame(train_df)
        
        # Define features and target
        x = train_h2o.columns
        x.remove('target')
        y = 'target'
        
        # Configure AutoML
        aml = H2OAutoML(
            max_runtime_secs=self.time_limit,
            nfolds=self.nfolds,
            seed=self.seed,
            sort_metric="RMSE"
        )
        
        # Train AutoML
        aml.train(x=x, y=y, training_frame=train_h2o)
        
        # Store the AutoML object
        self.model = aml
        
        training_time = time.time() - self.start_time
        logger.info(f"H2O AutoML training completed in {training_time:.2f} seconds")
        logger.info(f"Best model: {aml.leader.model_id}")
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        # Convert to H2O Frame
        test_df = pd.DataFrame(X, columns=self.model.leader.names[:-1])
        test_h2o = self.h2o.H2OFrame(test_df)
        
        # Generate predictions
        preds_h2o = self.model.leader.predict(test_h2o)
        preds = preds_h2o.as_data_frame().values.flatten()
        
        return preds
    
    def get_feature_importance(self):
        """Get feature importance from the leader model"""
        try:
            varimp = self.model.leader.varimp(use_pandas=True)
            if varimp is not None:
                return dict(zip(varimp['variable'], varimp['relative_importance']))
        except:
            pass
        return {}

class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self, models, weights=None):
        self.name = "ensemble"
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        """Generate ensemble predictions"""
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                model_preds = model.predict(X)
                predictions.append(model_preds)
            except Exception as e:
                logger.error(f"Error generating predictions from {model.name}: {e}")
                # Use zero weights for failed models
                self.weights[i] = 0
        
        # Normalize weights
        if sum(self.weights) > 0:
            normalized_weights = [w/sum(self.weights) for w in self.weights]
        else:
            normalized_weights = [1/len(predictions)] * len(predictions)
        
        # Weighted average
        ensemble_preds = np.zeros(X.shape[0])
        for i, preds in enumerate(predictions):
            if i < len(normalized_weights):
                ensemble_preds += preds * normalized_weights[i]
        
        return ensemble_preds
    
    def get_feature_importance(self):
        """Aggregate feature importance from all models"""
        # Collect feature importance from all models
        all_importances = {}
        
        for model in self.models:
            model_importance = model.get_feature_importance()
            for feature, importance in model_importance.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        # Average importance across models
        avg_importance = {}
        for feature, importances in all_importances.items():
            avg_importance[feature] = sum(importances) / len(importances)
        
        return avg_importance

def evaluate_model(model, X, y, metric='rmse'):
    """Evaluate model performance using specified metric"""
    # Generate predictions
    preds = model.predict(X)
    
    # Calculate metrics
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y, preds))
    elif metric == 'mae':
        return mean_absolute_error(y, preds)
    elif metric == 'r2':
        return r2_score(y, preds)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def visualize_feature_importance(models, top_n=20, output_dir='reports/plots'):
    """Visualize feature importance from models"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model in models:
        importances = model.get_feature_importance()
        if not importances:
            continue
        
        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        feature_names = [f[:20] + '...' if len(f) > 20 else f for f, _ in top_features]
        values = [v for _, v in top_features]
        
        plt.barh(feature_names, values)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features by Importance - {model.name}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'feature_importance_{model.name}.png'))
        plt.close()

def save_model_predictions(models, X, data_id, output_dir='data/submissions'):
    """Save predictions from multiple models to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in models:
        try:
            # Generate predictions
            preds = model.predict(X)
            
            # Create DataFrame with IDs and predictions
            if isinstance(data_id, np.ndarray) or isinstance(data_id, list):
                df = pd.DataFrame({'id': data_id, 'prediction': preds})
            else:
                df = pd.DataFrame({'prediction': preds})
                df['id'] = [f"pred_{i}" for i in range(len(df))]
            
            # Save to CSV
            filename = os.path.join(output_dir, f"predictions_{model.name}_{timestamp}.csv")
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            
            logger.info(f"Saved predictions from {model.name} to {filename}")
        except Exception as e:
            logger.error(f"Error saving predictions from {model.name}: {e}")
    
    return saved_files

def main():
    parser = argparse.ArgumentParser(description='Run model comparison for Numerai Crypto')
    parser.add_argument('--ram', type=int, default=500, help='RAM to use in GB (default: 500)')
    parser.add_argument('--gpus', type=str, default='0,1,2', help='GPU IDs to use (comma-separated)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (default)')
    parser.add_argument('--features', type=int, default=5000, help='Max number of features to use')
    parser.add_argument('--time-limit', type=int, default=720, help='Time limit per model in seconds (default: 720)')
    parser.add_argument('--output-dir', type=str, default='data/submissions', help='Output directory for submissions')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_RAM_GB"] = str(args.ram)
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip()]
    if not gpu_ids:
        gpu_ids = [0]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create synthetic data
    if args.synthetic:
        # Calculate number of samples based on RAM
        n_features = min(args.features, 10000)  # Cap features at 10000
        approx_samples = args.ram * 1024**3 / (n_features * 8) / 2  # Rough estimation
        n_samples = min(int(approx_samples), 100000)  # Cap samples
        
        logger.info(f"Creating synthetic data with {n_samples} samples and {n_features} features")
        df = create_synthetic_data(n_samples=n_samples, n_features=n_features)
    else:
        # TODO: Add support for loading real data
        # For now, default to synthetic
        logger.info("Real data loading not implemented, using synthetic data")
        df = create_synthetic_data(n_samples=100000, n_features=args.features)
    
    # Feature engineering
    df, feature_cols = engineer_features(df, max_features=args.features)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    
    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # List of models to train
    models_to_train = []
    
    # Add gradient boosting models
    if len(gpu_ids) >= 3:
        # Use 3 separate GPUs
        models_to_train.append(('lightgbm', LightGBMModel(use_gpu=True, gpu_id=gpu_ids[0], time_limit=args.time_limit)))
        models_to_train.append(('xgboost', XGBoostModel(use_gpu=True, gpu_id=gpu_ids[1], time_limit=args.time_limit)))
        try:
            import catboost
            models_to_train.append(('catboost', CatBoostModel(use_gpu=True, gpu_id=gpu_ids[2], time_limit=args.time_limit)))
        except ImportError:
            logger.warning("CatBoost not available, skipping")
    else:
        # Use primary GPU for all models
        models_to_train.append(('lightgbm', LightGBMModel(use_gpu=True, gpu_id=gpu_ids[0], time_limit=args.time_limit)))
        models_to_train.append(('xgboost', XGBoostModel(use_gpu=True, gpu_id=gpu_ids[0], time_limit=args.time_limit)))
    
    # Add deep learning models if PyTorch is available
    try:
        import torch
        if torch.cuda.is_available():
            if len(gpu_ids) >= 2:
                # Use separate GPUs for the models
                models_to_train.append(('pytorch_lstm', PyTorchLSTMModel(input_dim=X_train.shape[1], use_gpu=True, gpu_id=gpu_ids[0], time_limit=args.time_limit)))
                models_to_train.append(('pytorch_cnn', PyTorchCNNModel(input_dim=X_train.shape[1], use_gpu=True, gpu_id=gpu_ids[1], time_limit=args.time_limit)))
            else:
                # Use primary GPU for both models
                models_to_train.append(('pytorch_lstm', PyTorchLSTMModel(input_dim=X_train.shape[1], use_gpu=True, gpu_id=gpu_ids[0], time_limit=args.time_limit)))
    except ImportError:
        logger.warning("PyTorch not available, skipping neural network models")
    
    # Add H2O AutoML
    try:
        import h2o
        models_to_train.append(('h2o_automl', H2OAutoMLModel(time_limit=args.time_limit)))
    except ImportError:
        logger.warning("H2O not available, skipping AutoML model")
    
    # Train models and collect results
    trained_models = []
    results = {}
    
    for name, model in models_to_train:
        logger.info(f"Training {name} model")
        start_time = time.time()
        
        try:
            model.train(X_train, y_train, X_val=X_test, y_val=y_test, feature_names=feature_cols)
            trained_models.append(model)
            
            # Evaluate on test set
            rmse = evaluate_model(model, X_test, y_test, metric='rmse')
            results[name] = {
                'rmse': rmse,
                'training_time': time.time() - start_time
            }
            
            logger.info(f"{name} model - RMSE: {rmse:.6f}, Time: {results[name]['training_time']:.2f}s")
            
            # Clean up to free memory
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error training {name} model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Create ensemble model if we have at least 2 trained models
    if len(trained_models) >= 2:
        logger.info("Creating ensemble model")
        ensemble = EnsembleModel(trained_models)
        trained_models.append(ensemble)
        
        # Evaluate ensemble
        rmse = evaluate_model(ensemble, X_test, y_test, metric='rmse')
        results['ensemble'] = {
            'rmse': rmse,
            'training_time': sum(results[m.name]['training_time'] for m in trained_models[:-1])
        }
        
        logger.info(f"Ensemble model - RMSE: {rmse:.6f}")
    
    # Visualize feature importance
    visualize_feature_importance(trained_models)
    
    # Save model predictions
    prediction_files = save_model_predictions(trained_models, X_test, test_df['id'] if 'id' in test_df.columns else None)
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, f"model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        # Convert any numpy types to native Python types
        clean_results = {}
        for model_name, metrics in results.items():
            clean_results[model_name] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                          for k, v in metrics.items()}
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print final summary
    logger.info("\n===== MODEL COMPARISON SUMMARY =====")
    logger.info(f"{'Model':<15} {'RMSE':<10} {'Training Time':<15}")
    logger.info("-" * 40)
    
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['rmse']):
        logger.info(f"{model_name:<15} {metrics['rmse']:<10.6f} {metrics['training_time']:<15.2f}s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())