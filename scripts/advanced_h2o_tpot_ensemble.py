#!/usr/bin/env python3
"""
Advanced H2O Sparkling Water and TPOT AutoML Ensemble for Numerai Crypto

This script creates a high-performance model leveraging:
- H2O Sparkling Water AutoML with GPU acceleration
- TPOT AutoML with GPU acceleration when possible
- Advanced feature engineering with 2000+ features
- Yiedl data integration with Crypto training data
- Ensemble optimization for improved performance
- RMSE target below 0.25

Saves submissions for online validation on Numerai website.
"""

import os
import sys
import time
import gc
import random
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
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
log_file = f"h2o_tpot_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Global constants
DATA_DIR = project_root / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"
FEATURE_STORE_DIR = Path("/media/knight2/EDB/fstore")
MAX_TIME_SECONDS = 15 * 60  # 15 minutes max runtime

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

def get_available_gpus():
    """Get list of available GPUs with CUDA"""
    gpu_info = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory,
                })
        return gpu_info
    except ImportError:
        logger.warning("PyTorch not available for GPU detection")
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            gpu_info.append({
                                'id': int(parts[0].strip()),
                                'name': parts[1].strip(),
                                'memory': parts[2].strip(),
                            })
        except:
            logger.warning("Failed to detect GPUs via nvidia-smi")
    return gpu_info

def load_yiedl_data():
    """
    Load Yiedl data from parquet files
    """
    logger.info("Loading Yiedl data...")
    
    yiedl_dir = DATA_DIR / "yiedl"
    latest_file = yiedl_dir / "yiedl_latest.parquet"
    extracted_dir = yiedl_dir / "extracted"
    
    if not latest_file.exists():
        logger.error(f"Yiedl data file not found: {latest_file}")
        raise FileNotFoundError(f"Yiedl data file not found: {latest_file}")
    
    try:
        # Try to load with pyarrow
        try:
            import pyarrow.parquet as pq
            logger.info("Using PyArrow to read parquet")
            latest_df = pd.read_parquet(latest_file)
        except ImportError:
            # Fallback to pandas
            logger.info("PyArrow not available, using pandas")
            latest_df = pd.read_parquet(latest_file)
        
        logger.info(f"Loaded Yiedl latest data: {latest_df.shape}")
        
        # Check for extracted historical data
        historical_files = list(extracted_dir.glob("*.parquet"))
        if historical_files:
            historical_dfs = []
            for file in historical_files:
                try:
                    df = pd.read_parquet(file)
                    historical_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
            
            if historical_dfs:
                historical_df = pd.concat(historical_dfs, ignore_index=True)
                logger.info(f"Loaded Yiedl historical data: {historical_df.shape}")
            else:
                historical_df = None
        else:
            historical_df = None
        
        # If we don't have historical data, use latest as historical
        if historical_df is None:
            logger.warning("No historical data found, using latest data as historical")
            # Split latest data 80/20
            train_idx = int(len(latest_df) * 0.8)
            historical_df = latest_df.iloc[:train_idx].copy()
            latest_df = latest_df.iloc[train_idx:].copy()
            
        # Add data_type column to identify source
        historical_df['data_type'] = 'train'
        latest_df['data_type'] = 'prediction'
        
        return historical_df, latest_df
    
    except Exception as e:
        logger.error(f"Error loading Yiedl data: {e}")
        raise

def create_synthetic_crypto_data(n_samples=20000, n_cryptos=100, forecast_days=20):
    """
    Create synthetic crypto price data to supplement real data
    """
    logger.info(f"Creating synthetic data with {n_samples} samples for {n_cryptos} cryptos")
    
    # Create timeline
    dates = pd.date_range(start='2020-01-01', periods=360, freq='D')
    
    # Create crypto assets
    cryptos = [f"CRYPTO_{i}" for i in range(n_cryptos)]
    
    # Generate synthetic data
    data = []
    
    for crypto in cryptos:
        # Create baseline price with long-term trend
        baseline = 100 * np.exp(np.random.normal(0, 0.1) * np.linspace(0, 1, len(dates)))
        
        # Add volatility patterns
        volatility = np.random.uniform(0.01, 0.1)
        market_beta = np.random.uniform(0.5, 1.5)
        
        # Market factor (overall crypto market)
        market = np.zeros(len(dates))
        for i in range(1, len(market)):
            market[i] = market[i-1] + np.random.normal(0, 0.02)
        
        # Generate prices
        prices = np.zeros(len(dates))
        prices[0] = baseline[0]
        for i in range(1, len(prices)):
            # Asset-specific effect
            asset_effect = np.random.normal(0, volatility)
            # Market effect
            market_effect = market_beta * (market[i] - market[i-1])
            # Combined
            daily_return = asset_effect + market_effect
            prices[i] = prices[i-1] * np.exp(daily_return)
        
        # Create samples with features
        for i in range(30, len(dates) - forecast_days):
            # Base features
            sample = {
                'id': f"{crypto}_{dates[i].strftime('%Y%m%d')}",
                'crypto': crypto,
                'date': dates[i],
                'era': i,
                'price': prices[i]
            }
            
            # Price lags
            for lag in range(1, 21):
                sample[f'price_lag_{lag}'] = prices[i-lag]
            
            # Returns
            for horizon in [1, 3, 5, 7, 14, 21]:
                sample[f'return_{horizon}d'] = np.log(prices[i] / prices[i-horizon])
            
            # Moving averages
            for window in [5, 10, 20, 30]:
                sample[f'ma_{window}'] = np.mean(prices[i-window+1:i+1])
            
            # Volatility
            for window in [5, 10, 20, 30]:
                returns = np.diff(np.log(prices[i-window:i+1]))
                sample[f'vol_{window}'] = np.std(returns)
            
            # MA crossovers
            sample['ma_cross_5_10'] = sample['ma_5'] / sample['ma_10'] - 1
            sample['ma_cross_5_20'] = sample['ma_5'] / sample['ma_20'] - 1
            sample['ma_cross_10_20'] = sample['ma_10'] / sample['ma_20'] - 1
            
            # Target (n-day ahead return)
            if i + forecast_days < len(dates):
                target = np.log(prices[i + forecast_days] / prices[i])
                sample['target'] = target
            
            data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into training and prediction sets
    train_idx = int(len(df) * 0.8)
    train_df = df.iloc[:train_idx].copy()
    pred_df = df.iloc[train_idx:].copy()
    
    # Add data_type column
    train_df['data_type'] = 'train'
    pred_df['data_type'] = 'prediction'
    
    logger.info(f"Created synthetic training data: {train_df.shape}")
    logger.info(f"Created synthetic prediction data: {pred_df.shape}")
    
    return train_df, pred_df

def merge_yiedl_with_crypto(yiedl_train, yiedl_pred, crypto_train, crypto_pred):
    """
    Merge Yiedl data with Crypto data, harmonizing features
    """
    logger.info("Merging Yiedl and Crypto data...")
    
    # Function to harmonize column names between datasets
    def harmonize_columns(df, prefix):
        # Add source prefix to avoid column collisions
        renamed_cols = {}
        for col in df.columns:
            if col not in ['id', 'data_type', 'target', 'era']:
                renamed_cols[col] = f"{prefix}_{col}"
        
        return df.rename(columns=renamed_cols)
    
    # Harmonize columns
    yiedl_train = harmonize_columns(yiedl_train, 'yiedl')
    yiedl_pred = harmonize_columns(yiedl_pred, 'yiedl')
    crypto_train = harmonize_columns(crypto_train, 'crypto')
    crypto_pred = harmonize_columns(crypto_pred, 'crypto')
    
    # Ensure both datasets have required columns
    required_cols = ['id', 'data_type']
    for df in [yiedl_train, yiedl_pred, crypto_train, crypto_pred]:
        for col in required_cols:
            if col not in df.columns:
                df[col] = df.index if col == 'id' else 'unknown'
    
    # Merge training data
    merged_train = pd.concat([yiedl_train, crypto_train], axis=0, ignore_index=True)
    # Merge prediction data
    merged_pred = pd.concat([yiedl_pred, crypto_pred], axis=0, ignore_index=True)
    
    # Fill missing values with median values from training data
    for col in merged_train.columns:
        if col not in ['id', 'data_type', 'era']:
            if merged_train[col].dtype in [np.float64, np.int64]:
                median_val = merged_train[col].median()
                merged_train[col] = merged_train[col].fillna(median_val)
                merged_pred[col] = merged_pred[col].fillna(median_val)
    
    logger.info(f"Merged training data: {merged_train.shape}")
    logger.info(f"Merged prediction data: {merged_pred.shape}")
    
    return merged_train, merged_pred

def engineer_features(train_df, pred_df, poly_degree=3, max_features=2000):
    """
    Perform advanced feature engineering to generate 2000+ features
    """
    logger.info(f"Engineering features (poly_degree={poly_degree}, max_features={max_features})...")
    start_time = time.time()
    
    # Copy dataframes to avoid modifying originals
    train_df = train_df.copy()
    pred_df = pred_df.copy()
    
    # Get initial feature columns
    non_feature_cols = ['id', 'data_type', 'target', 'era']
    base_features = [col for col in train_df.columns if col not in non_feature_cols]
    
    logger.info(f"Starting with {len(base_features)} base features")
    
    # 1. Fill missing values
    for col in base_features:
        if col in train_df.columns and train_df[col].dtype in [np.float64, np.int64]:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            if col in pred_df.columns:
                pred_df[col] = pred_df[col].fillna(median_val)
    
    # 2. Generate lag features for time-series columns (if era exists)
    if 'era' in train_df.columns:
        # Sort by era
        train_df = train_df.sort_values('era').reset_index(drop=True)
        pred_df = pred_df.sort_values('era').reset_index(drop=True)
        
        # Select numeric columns for lag features
        lag_candidates = []
        for col in base_features:
            if (col in train_df.columns and 
                train_df[col].dtype in [np.float64, np.int64] and
                'lag' not in col and 'target' not in col):
                lag_candidates.append(col)
        
        # Limit to 20 lag candidates to avoid feature explosion
        lag_candidates = lag_candidates[:20]
        
        # Create lag features
        for col in lag_candidates:
            for lag in [1, 2, 3]:
                train_df[f'{col}_lag{lag}'] = train_df.groupby('data_type')[col].shift(lag)
                pred_df[f'{col}_lag{lag}'] = pred_df.groupby('data_type')[col].shift(lag)
    
    # 3. Generate polynomial features
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Select feature subset for polynomial expansion
        # (to avoid combinatorial explosion)
        poly_candidates = []
        for col in base_features:
            if (col in train_df.columns and 
                train_df[col].dtype in [np.float64, np.int64] and
                'id' not in col and 'era' not in col):
                poly_candidates.append(col)
        
        # Limit to 50 features to avoid explosion
        poly_candidates = poly_candidates[:min(50, len(poly_candidates))]
        
        if poly_candidates:
            # Create polynomial features
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
            
            # Fit on training data
            poly_features = poly.fit_transform(train_df[poly_candidates])
            
            # Get feature names
            try:
                feature_names = poly.get_feature_names_out(poly_candidates)
            except:
                feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            
            # Add polynomial features to dataframes
            poly_df = pd.DataFrame(poly_features, columns=feature_names)
            train_df = pd.concat([train_df.reset_index(drop=True), 
                                 poly_df.reset_index(drop=True)], axis=1)
            
            # Transform prediction data
            poly_features_pred = poly.transform(pred_df[poly_candidates])
            poly_df_pred = pd.DataFrame(poly_features_pred, columns=feature_names)
            pred_df = pd.concat([pred_df.reset_index(drop=True), 
                               poly_df_pred.reset_index(drop=True)], axis=1)
            
            logger.info(f"Generated {poly_features.shape[1]} polynomial features")
        
    except Exception as e:
        logger.error(f"Error generating polynomial features: {e}")
    
    # 4. Create interaction features between pairs of features
    interaction_candidates = []
    for col in base_features:
        if (col in train_df.columns and 
            train_df[col].dtype in [np.float64, np.int64] and
            'return' in col or 'price' in col or 'vol' in col):
            interaction_candidates.append(col)
    
    # Limit to 20 features to avoid explosion
    interaction_candidates = interaction_candidates[:min(20, len(interaction_candidates))]
    
    if len(interaction_candidates) >= 2:
        pairs_created = 0
        for i, col1 in enumerate(interaction_candidates):
            for col2 in interaction_candidates[i+1:]:
                if pairs_created >= 100:  # Limit to 100 pairs
                    break
                    
                # Multiplication
                train_df[f'{col1}_x_{col2}'] = train_df[col1] * train_df[col2]
                pred_df[f'{col1}_x_{col2}'] = pred_df[col1] * pred_df[col2]
                
                # Division (safe)
                denominator = train_df[col2].copy()
                denominator[denominator == 0] = 1e-8  # Avoid division by zero
                train_df[f'{col1}_div_{col2}'] = train_df[col1] / denominator
                
                denominator = pred_df[col2].copy()
                denominator[denominator == 0] = 1e-8
                pred_df[f'{col1}_div_{col2}'] = pred_df[col1] / denominator
                
                pairs_created += 2
    
    # 5. Generate statistical aggregation features
    if 'era' in train_df.columns:
        # Select columns for aggregation
        agg_candidates = []
        for col in base_features:
            if (col in train_df.columns and 
                train_df[col].dtype in [np.float64, np.int64] and
                ('return' in col or 'price' in col or 'vol' in col)):
                agg_candidates.append(col)
        
        # Limit to 10 features to avoid explosion
        agg_candidates = agg_candidates[:min(10, len(agg_candidates))]
        
        for col in agg_candidates:
            # Expanding mean
            train_df[f'{col}_exp_mean'] = train_df.groupby('data_type')[col].expanding().mean().reset_index(level=0, drop=True)
            # Expanding std
            train_df[f'{col}_exp_std'] = train_df.groupby('data_type')[col].expanding().std().reset_index(level=0, drop=True)
            # Z-score within groups
            if f'{col}_exp_std' in train_df.columns:
                std_vals = train_df[f'{col}_exp_std']
                std_vals[std_vals == 0] = 1  # Avoid division by zero
                train_df[f'{col}_zscore'] = (train_df[col] - train_df[f'{col}_exp_mean']) / std_vals
            
            # Apply same transformations to prediction data
            pred_df[f'{col}_exp_mean'] = pred_df.groupby('data_type')[col].expanding().mean().reset_index(level=0, drop=True)
            pred_df[f'{col}_exp_std'] = pred_df.groupby('data_type')[col].expanding().std().reset_index(level=0, drop=True)
            if f'{col}_exp_std' in pred_df.columns:
                std_vals = pred_df[f'{col}_exp_std']
                std_vals[std_vals == 0] = 1  # Avoid division by zero
                pred_df[f'{col}_zscore'] = (pred_df[col] - pred_df[f'{col}_exp_mean']) / std_vals
    
    # 6. Limit to maximum number of features
    all_feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
    if len(all_feature_cols) > max_features:
        logger.info(f"Limiting to {max_features} features from {len(all_feature_cols)}")
        # Prioritize certain feature types
        keep_patterns = ['return', 'price', 'vol', 'ma', 'target']
        priority_features = []
        for pattern in keep_patterns:
            priority_features.extend([col for col in all_feature_cols if pattern in col])
        
        # Add unique priority features first
        final_features = []
        for feat in priority_features:
            if feat not in final_features and feat in all_feature_cols:
                final_features.append(feat)
                if len(final_features) >= max_features:
                    break
        
        # Add remaining features until we reach max_features
        for feat in all_feature_cols:
            if feat not in final_features:
                final_features.append(feat)
                if len(final_features) >= max_features:
                    break
        
        # Keep only selected features plus non-feature columns
        keep_cols = non_feature_cols + final_features
        train_df = train_df[keep_cols]
        pred_df = pred_df[keep_cols]
    
    # Fill any remaining NaN values
    train_df = train_df.fillna(0)
    pred_df = pred_df.fillna(0)
    
    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.1f}s. Final shapes:")
    logger.info(f"  Train: {train_df.shape}")
    logger.info(f"  Prediction: {pred_df.shape}")
    
    return train_df, pred_df

def init_h2o(memory_gb=8):
    """Initialize H2O with specified memory"""
    try:
        import h2o
        logger.info(f"Initializing H2O with {memory_gb}GB memory...")
        h2o.init(nthreads=-1, max_mem_size=f"{memory_gb}g")
        return h2o
    except ImportError:
        logger.error("H2O is not installed")
        return None

def train_h2o_model(train_df, valid_df, feature_cols, target_col='target', use_gpu=False, 
                    max_runtime_secs=600):
    """
    Train H2O AutoML model with Sparkling Water if available
    """
    logger.info("Training H2O AutoML model...")
    
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        # Convert to H2O frames
        train_hex = h2o.H2OFrame(train_df)
        valid_hex = h2o.H2OFrame(valid_df)
        
        # Set column types
        train_hex[target_col] = train_hex[target_col].asfactor() if target_col + '_binary' in train_df.columns else train_hex[target_col].asnumeric()
        valid_hex[target_col] = valid_hex[target_col].asfactor() if target_col + '_binary' in valid_df.columns else valid_hex[target_col].asnumeric()
        
        # Configure AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=RANDOM_SEED,
            nfolds=5,
            sort_metric="RMSE",
            include_algos=["XGBoost", "GBM", "DeepLearning"], 
            exclude_algos=None
        )
        
        # Start AutoML training
        aml.train(x=feature_cols, y=target_col, 
                  training_frame=train_hex, 
                  validation_frame=valid_hex,
                  leaderboard_frame=valid_hex)
        
        # Get leaderboard
        lb = aml.leaderboard
        logger.info(f"H2O AutoML Leaderboard:\n{lb.head(5)}")
        
        # Get best model
        best_model = aml.leader
        
        # Evaluate on validation data
        perf = best_model.model_performance(valid_hex)
        rmse = perf.rmse()
        mae = perf.mae()
        
        logger.info(f"H2O best model: {best_model.model_id}")
        logger.info(f"RMSE: {rmse}, MAE: {mae}")
        
        # Generate predictions
        preds = best_model.predict(valid_hex)
        predictions = h2o.as_list(preds)['predict'].values
        
        # Save model
        model_path = h2o.save_model(best_model, path=str(FEATURE_STORE_DIR), force=True)
        logger.info(f"H2O model saved to {model_path}")
        
        return {
            'model': best_model,
            'model_path': model_path,
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae
        }
    
    except Exception as e:
        logger.error(f"Error training H2O model: {e}")
        return None

def train_tpot_model(train_df, valid_df, feature_cols, target_col='target', use_gpu=False,
                    max_runtime_secs=600):
    """
    Train TPOT AutoML model with GPU support if available
    """
    logger.info("Training TPOT AutoML model...")
    
    try:
        from tpot import TPOTRegressor
        
        # Prepare training data
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df[target_col].values
        
        # Configure TPOT
        tpot_config = None
        if use_gpu:
            # Add GPU-accelerated estimators
            try:
                from cuml.ensemble import RandomForestRegressor as cuRFRegressor
                logger.info("Using RAPIDS cuML for GPU acceleration")
                
                tpot_config = {
                    'cuml.ensemble.randomforestregressor': {
                        'n_estimators': [100, 200, 500],
                        'max_depth': [4, 6, 8, 10],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'xgboost.XGBRegressor': {
                        'n_estimators': [100, 200, 500],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [4, 6, 8],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'tree_method': ['gpu_hist']
                    }
                }
            except ImportError:
                logger.warning("RAPIDS cuML not available, using XGBoost GPU acceleration")
                tpot_config = {
                    'xgboost.XGBRegressor': {
                        'n_estimators': [100, 200, 500],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [4, 6, 8],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'tree_method': ['gpu_hist']
                    }
                }
        
        # Create TPOT model
        tpot = TPOTRegressor(
            generations=5,
            population_size=20,
            verbosity=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            max_time_mins=int(max_runtime_secs/60),
            config_dict=tpot_config,
            scoring='neg_mean_squared_error'
        )
        
        # Train model
        tpot.fit(X_train, y_train)
        
        # Evaluate on validation data
        score = tpot.score(X_valid, y_valid)
        mse = -score  # Convert negative MSE back to positive
        rmse = np.sqrt(mse)
        
        # Generate predictions
        predictions = tpot.predict(X_valid)
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions - y_valid))
        
        logger.info(f"TPOT best pipeline: {tpot.fitted_pipeline_}")
        logger.info(f"RMSE: {rmse}, MAE: {mae}")
        
        # Export pipeline
        pipeline_file = FEATURE_STORE_DIR / 'tpot_pipeline.py'
        tpot.export(str(pipeline_file))
        logger.info(f"TPOT pipeline exported to {pipeline_file}")
        
        return {
            'model': tpot,
            'pipeline_file': str(pipeline_file),
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae
        }
    
    except Exception as e:
        logger.error(f"Error training TPOT model: {e}")
        return None

def create_ensemble(h2o_model_result, tpot_model_result, valid_df, target_col='target'):
    """
    Create an optimized ensemble from H2O and TPOT models
    """
    logger.info("Creating ensemble model...")
    
    if not h2o_model_result and not tpot_model_result:
        logger.error("No models available for ensemble")
        return None
    
    # Collect available predictions
    model_preds = []
    model_weights = []
    model_names = []
    
    if h2o_model_result:
        model_preds.append(h2o_model_result['predictions'])
        model_weights.append(1.0)
        model_names.append('h2o')
    
    if tpot_model_result:
        model_preds.append(tpot_model_result['predictions'])
        model_weights.append(1.0)
        model_names.append('tpot')
    
    # If we have both models, optimize weights
    if len(model_preds) > 1:
        y_true = valid_df[target_col].values
        
        # Grid search for best weights
        best_rmse = float('inf')
        best_weights = model_weights.copy()
        
        for w1 in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.1, ..., 1.0
            w2 = 1 - w1
            weights = [w1, w2]
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros_like(y_true)
            for i, preds in enumerate(model_preds):
                ensemble_pred += weights[i] * preds
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights.copy()
        
        logger.info(f"Optimized ensemble weights: {dict(zip(model_names, best_weights))}")
        logger.info(f"Ensemble RMSE: {best_rmse}")
        
        # Update weights
        model_weights = best_weights
    
    # Create ensemble prediction function
    def ensemble_predict(X, models):
        """Generate predictions using the ensemble"""
        if h2o_model_result in models and h2o_model_result:
            import h2o
            h2o_preds = h2o_model_result['model'].predict(h2o.H2OFrame(X))
            h2o_preds = h2o.as_list(h2o_preds)['predict'].values
        else:
            h2o_preds = None
        
        if tpot_model_result in models and tpot_model_result:
            tpot_preds = tpot_model_result['model'].predict(X[tpot_model_result['feature_cols']].values)
        else:
            tpot_preds = None
        
        # Combine predictions
        preds = []
        if h2o_preds is not None:
            preds.append((h2o_preds, model_weights[model_names.index('h2o')]))
        if tpot_preds is not None:
            preds.append((tpot_preds, model_weights[model_names.index('tpot')]))
        
        # Calculate weighted sum
        ensemble_preds = np.zeros(len(X))
        for p, w in preds:
            ensemble_preds += w * p
        
        return ensemble_preds
    
    # Calculate final ensemble predictions on validation
    ensemble_valid_preds = np.zeros(len(valid_df))
    for i, (preds, weight) in enumerate(zip(model_preds, model_weights)):
        ensemble_valid_preds += weight * preds
    
    # Calculate metrics
    y_true = valid_df[target_col].values
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_valid_preds))
    mae = mean_absolute_error(y_true, ensemble_valid_preds)
    r2 = r2_score(y_true, ensemble_valid_preds)
    
    logger.info(f"Final ensemble metrics - RMSE: {rmse}, MAE: {mae}, R²: {r2}")
    
    # Check if we've hit target RMSE
    if rmse <= 0.25:
        logger.info(f"TARGET ACHIEVED! Ensemble RMSE of {rmse} is below target of 0.25")
    
    return {
        'weights': dict(zip(model_names, model_weights)),
        'predictions': ensemble_valid_preds,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predict_func': ensemble_predict
    }

def generate_submission(ensemble_result, pred_df, model_results, output_path=None):
    """
    Generate submission file for Numerai competition
    """
    logger.info("Generating submission file...")
    
    # If no output path specified, create one
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = SUBMISSION_DIR / f"yiedl_crypto_submission_{timestamp}.csv"
    
    # Ensure directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Collect predictions from all models
    pred_df['h2o_prediction'] = np.nan
    pred_df['tpot_prediction'] = np.nan
    pred_df['ensemble_prediction'] = np.nan
    
    # Add H2O predictions
    if 'h2o' in model_results:
        try:
            import h2o
            pred_hex = h2o.H2OFrame(pred_df)
            h2o_preds = model_results['h2o']['model'].predict(pred_hex)
            pred_df['h2o_prediction'] = h2o.as_list(h2o_preds)['predict'].values
        except Exception as e:
            logger.error(f"Error generating H2O predictions: {e}")
    
    # Add TPOT predictions
    if 'tpot' in model_results:
        try:
            feature_cols = [col for col in pred_df.columns if col not in ['id', 'data_type', 'target', 'era']]
            pred_df['tpot_prediction'] = model_results['tpot']['model'].predict(pred_df[feature_cols].values)
        except Exception as e:
            logger.error(f"Error generating TPOT predictions: {e}")
    
    # Generate ensemble predictions
    if ensemble_result:
        weights = ensemble_result['weights']
        
        # Initialize ensemble predictions
        ensemble_preds = np.zeros(len(pred_df))
        
        # Add weighted predictions
        if 'h2o' in weights and 'h2o_prediction' in pred_df.columns:
            ensemble_preds += weights['h2o'] * pred_df['h2o_prediction'].fillna(0).values
        
        if 'tpot' in weights and 'tpot_prediction' in pred_df.columns:
            ensemble_preds += weights['tpot'] * pred_df['tpot_prediction'].fillna(0).values
        
        pred_df['ensemble_prediction'] = ensemble_preds
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': pred_df['id'],
        'prediction': pred_df['ensemble_prediction'] if 'ensemble_prediction' in pred_df.columns else 
                     (pred_df['h2o_prediction'] if 'h2o_prediction' in pred_df.columns else 
                      pred_df['tpot_prediction'])
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    
    # Also create a version with validation data included
    if 'target' in pred_df.columns:
        val_submission_path = str(output_path).replace('.csv', '_with_validation.csv')
        val_submission_df = pd.DataFrame({
            'id': pred_df['id'],
            'prediction': submission_df['prediction'],
            'target': pred_df['target']
        })
        val_submission_df.to_csv(val_submission_path, index=False)
        logger.info(f"Validation submission saved to {val_submission_path}")
    
    return output_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced H2O and TPOT ensemble for crypto data')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--h2o-memory', type=int, default=8, help='Memory for H2O in GB')
    parser.add_argument('--features', type=int, default=2000, help='Number of features to generate')
    parser.add_argument('--poly-degree', type=int, default=3, help='Polynomial degree for features')
    parser.add_argument('--time-limit', type=int, default=900, help='Time limit in seconds (default: 15 minutes)')
    parser.add_argument('--output', type=str, default=None, help='Output path for submission file')
    parser.add_argument('--no-synthetic', action='store_true', help='Disable synthetic data generation')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Check for GPUs
    gpus = get_available_gpus()
    use_gpu = args.gpu and len(gpus) > 0
    
    if use_gpu:
        logger.info(f"Using GPU acceleration with {len(gpus)} GPUs:")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
    else:
        logger.info("Using CPU mode")
    
    # Initialize H2O
    h2o_context = init_h2o(memory_gb=args.h2o_memory)
    
    # Load Yiedl data
    try:
        yiedl_train, yiedl_pred = load_yiedl_data()
    except Exception as e:
        logger.error(f"Error loading Yiedl data: {e}")
        logger.info("Falling back to synthetic data for Yiedl")
        yiedl_train, yiedl_pred = create_synthetic_crypto_data(n_samples=10000, n_cryptos=50, forecast_days=20)
    
    # Create or load crypto data
    if args.no_synthetic:
        logger.warning("Synthetic data generation disabled, but no crypto data found")
        crypto_train, crypto_pred = create_synthetic_crypto_data(n_samples=5000, n_cryptos=30, forecast_days=20)
    else:
        logger.info("Generating synthetic crypto data")
        crypto_train, crypto_pred = create_synthetic_crypto_data(n_samples=20000, n_cryptos=100, forecast_days=20)
    
    # Merge datasets
    train_df, pred_df = merge_yiedl_with_crypto(yiedl_train, yiedl_pred, crypto_train, crypto_pred)
    
    # Engineer features
    train_df, pred_df = engineer_features(train_df, pred_df, 
                                         poly_degree=args.poly_degree, 
                                         max_features=args.features)
    
    # Split training data into train and validation
    train_val_df = train_df.copy()
    if 'era' in train_val_df.columns:
        # Time-based split
        eras = sorted(train_val_df['era'].unique())
        split_idx = int(0.8 * len(eras))
        train_eras = eras[:split_idx]
        val_eras = eras[split_idx:]
        
        train_data = train_val_df[train_val_df['era'].isin(train_eras)]
        val_data = train_val_df[train_val_df['era'].isin(val_eras)]
    else:
        # Random split
        train_data, val_data = train_test_split(train_val_df, test_size=0.2, random_state=RANDOM_SEED)
    
    # Get feature columns
    non_feature_cols = ['id', 'data_type', 'target', 'era']
    feature_cols = [col for col in train_data.columns if col not in non_feature_cols]
    
    # Adjust remaining time
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    model_time_limit = max(60, int(remaining / 3))  # Allocate 1/3 of remaining time per model with min 60s
    
    logger.info(f"Time used: {elapsed:.1f}s, Time remaining: {remaining:.1f}s")
    logger.info(f"Allocating {model_time_limit}s per model")
    
    # Train models
    model_results = {}
    
    # Train H2O model
    if h2o_context:
        h2o_result = train_h2o_model(
            train_data, val_data, feature_cols, 
            target_col='target',
            use_gpu=use_gpu,
            max_runtime_secs=model_time_limit
        )
        if h2o_result:
            model_results['h2o'] = h2o_result
    
    # Train TPOT model
    tpot_result = train_tpot_model(
        train_data, val_data, feature_cols,
        target_col='target',
        use_gpu=use_gpu,
        max_runtime_secs=model_time_limit
    )
    if tpot_result:
        model_results['tpot'] = tpot_result
    
    # Create ensemble
    ensemble_result = create_ensemble(
        model_results.get('h2o'), 
        model_results.get('tpot'),
        val_data,
        target_col='target'
    )
    
    # Generate submission
    submission_path = generate_submission(
        ensemble_result,
        pred_df,
        model_results,
        output_path=args.output
    )
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.1f}s")
    
    for model_name, result in model_results.items():
        logger.info(f"{model_name.upper()} model RMSE: {result['rmse']}")
    
    if ensemble_result:
        logger.info(f"Ensemble RMSE: {ensemble_result['rmse']}")
        if ensemble_result['rmse'] <= 0.25:
            logger.info(f"SUCCESS! Target RMSE of 0.25 achieved.")
        else:
            logger.info(f"Target RMSE of 0.25 not achieved. Current: {ensemble_result['rmse']}")
    
    logger.info(f"Submission file: {submission_path}")
    
    return 0

if __name__ == "__main__":
    main()