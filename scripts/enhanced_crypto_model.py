#!/usr/bin/env python3
"""
Enhanced Crypto Model with Advanced Feature Engineering

This script implements a high-performance model for crypto prediction with:
- Advanced feature engineering including polynomial features
- Price return aggregation and statistical features
- Cross-asset correlation features
- Time-series momentum features
- Ensemble of gradient boosting and neural networks
- Target RMSE: 0.25 or lower

Utilizes full hardware capabilities (3x 24GB GPUs and 600GB RAM)
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Configure logging
log_file = f"enhanced_crypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def load_yiedl_data(data_dir=None):
    """
    Load data from Yiedl parquet and zip files.
    This function now handles proper loading of parquet files.
    """
    if data_dir is None:
        data_dir = Path(project_root) / "data" / "yiedl"
    else:
        data_dir = Path(data_dir)
    
    logger.info(f"Loading data from {data_dir}")
    
    latest_file = data_dir / "yiedl_latest.parquet"
    historical_zip = data_dir / "yiedl_historical.zip"
    extracted_dir = data_dir / "extracted"
    
    os.makedirs(extracted_dir, exist_ok=True)
    
    latest_df = None
    historical_df = None
    
    # Try to load latest data from parquet
    if latest_file.exists():
        try:
            import pyarrow.parquet as pq
            latest_df = pd.read_parquet(latest_file)
            logger.info(f"Successfully loaded latest data: {latest_df.shape}")
        except Exception as e:
            logger.error(f"Error loading latest parquet file: {e}")
    
    # Try to extract and load historical data from zip
    if historical_zip.exists():
        try:
            import zipfile
            with zipfile.ZipFile(historical_zip, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            # Look for parquet files in extracted directory
            parquet_files = list(extracted_dir.glob("**/*.parquet"))
            if parquet_files:
                # Load and concatenate all parquet files if multiple exist
                dfs = []
                for file in parquet_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                
                if dfs:
                    historical_df = pd.concat(dfs, ignore_index=True)
                    logger.info(f"Successfully loaded historical data: {historical_df.shape}")
        except Exception as e:
            logger.error(f"Error extracting/loading historical data: {e}")
    
    # Generate synthetic data if real data couldn't be loaded
    if latest_df is None or historical_df is None:
        logger.warning("Using synthetic data because real data couldn't be loaded")
        historical_df, latest_df = create_synthetic_data()
    
    return historical_df, latest_df

def create_synthetic_data(n_samples=100000, n_assets=100, lookback_days=60, forecast_days=20):
    """
    Create synthetic crypto data with realistic price patterns.
    Enhanced to create more realistic crypto market behaviors.
    """
    logger.info(f"Creating synthetic data with {n_samples} samples, {n_assets} assets")
    
    # Create timeline
    all_dates = pd.date_range(start='2020-01-01', 
                              periods=lookback_days + n_samples//n_assets + forecast_days, 
                              freq='D')
    
    # Asset properties - each crypto has different volatility and trend
    asset_properties = {
        f"CRYPTO_{i}": {
            'volatility': np.random.uniform(0.01, 0.1),  # Daily volatility
            'trend': np.random.uniform(-0.0005, 0.001),  # Daily trend
            'market_beta': np.random.uniform(0.5, 1.5),  # Correlation to market
            'seasonality_amplitude': np.random.uniform(0, 0.05)  # Seasonality effect
        } 
        for i in range(n_assets)
    }
    
    # Create market factor - overall crypto market
    market_factor = np.zeros(len(all_dates))
    market_volatility = 0.015  # Market volatility
    
    # Market follows random walk with momentum and reversion
    for i in range(1, len(market_factor)):
        # Random shock
        shock = np.random.normal(0, market_volatility)
        # Momentum (continuation of previous movement)
        momentum = 0.2 * (market_factor[i-1] - (0 if i < 2 else market_factor[i-2]))
        # Mean reversion (pull toward zero)
        reversion = -0.1 * market_factor[i-1]
        # Combined effect
        market_factor[i] = market_factor[i-1] + shock + momentum + reversion
    
    # Create crypto prices
    all_prices = {}
    for asset_id, properties in asset_properties.items():
        # Starting price around 100 with some randomness
        start_price = np.random.uniform(10, 1000)
        
        # Create price series
        prices = np.zeros(len(all_dates))
        prices[0] = start_price
        
        # Add price movement based on properties
        for i in range(1, len(prices)):
            # Random component - asset-specific
            asset_shock = np.random.normal(0, properties['volatility'])
            
            # Market component - correlation with overall market
            market_component = properties['market_beta'] * (market_factor[i] - market_factor[i-1])
            
            # Trend component
            trend_component = properties['trend']
            
            # Seasonality - weekly cycle (some cryptos have weekend patterns)
            day_of_week = i % 7
            seasonality = properties['seasonality_amplitude'] * np.sin(day_of_week * np.pi / 3.5)
            
            # Volume-related component (higher volatility during high volume periods)
            volume_impact = np.random.normal(0, properties['volatility'] * (1 + abs(market_factor[i])))
            
            # Calculate daily return
            daily_return = asset_shock + market_component + trend_component + seasonality + 0.2 * volume_impact
            
            # Apply return to price with log-normal distribution (prevents negative prices)
            prices[i] = prices[i-1] * np.exp(daily_return)
        
        all_prices[asset_id] = prices
    
    # Generate dataset with features
    data = []
    
    for asset_id, prices in all_prices.items():
        # Get asset properties
        props = asset_properties[asset_id]
        
        # Process each day with sufficient history
        for i in range(lookback_days, len(all_dates) - forecast_days):
            sample = {
                'id': f"{asset_id}_{i-lookback_days}",
                'asset': asset_id,
                'date': all_dates[i],
                'era': i-lookback_days,
                'price': prices[i],  # Current price
                'market': market_factor[i]  # Market factor
            }
            
            # === FEATURE ENGINEERING ===
            
            # 1. Price lags
            for lag in range(1, min(31, lookback_days+1)):
                sample[f'price_lag_{lag}'] = prices[i-lag]
            
            # 2. Returns at various horizons
            for horizon in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
                if i >= horizon:
                    sample[f'return_{horizon}d'] = np.log(prices[i] / prices[i-horizon])
            
            # 3. Moving averages
            for window in [5, 10, 20, 30, 50]:
                if i >= window:
                    sample[f'ma_{window}'] = np.mean(prices[i-window+1:i+1])
                    # MA crossovers
                    if window > 5 and 'ma_5' in sample:
                        sample[f'ma_cross_5_{window}'] = sample['ma_5'] / sample[f'ma_{window}'] - 1
            
            # 4. Volatility (standard deviation of returns)
            for window in [5, 10, 20, 30]:
                if i >= window:
                    returns = np.diff(np.log(prices[i-window:i+1]))
                    sample[f'vol_{window}'] = np.std(returns)
            
            # 5. Price momentum indicators
            for window in [10, 20, 30]:
                if i >= window:
                    sample[f'momentum_{window}'] = prices[i] / prices[i-window] - 1
            
            # 6. RSI (Relative Strength Index)
            for window in [7, 14, 21]:
                if i >= window+1:
                    diff = np.diff(prices[i-window:i+1])
                    gains = np.sum(np.maximum(diff, 0))
                    losses = np.sum(np.abs(np.minimum(diff, 0)))
                    if losses == 0:
                        sample[f'rsi_{window}'] = 100
                    else:
                        rs = gains / losses
                        sample[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # 7. Distance from moving averages
            for window in [10, 20, 50]:
                if i >= window:
                    ma = np.mean(prices[i-window+1:i+1])
                    sample[f'dist_ma_{window}'] = prices[i] / ma - 1
            
            # 8. Market correlation
            for window in [20, 30]:
                if i >= window:
                    price_returns = np.diff(np.log(prices[i-window:i+1]))
                    market_returns = np.diff(market_factor[i-window:i+1])
                    if np.std(market_returns) > 0 and np.std(price_returns) > 0:
                        corr = np.corrcoef(price_returns, market_returns)[0, 1]
                        sample[f'market_corr_{window}'] = corr if not np.isnan(corr) else 0
                    else:
                        sample[f'market_corr_{window}'] = 0
            
            # 9. Bollinger Bands
            for window in [20]:
                if i >= window:
                    rolling_mean = np.mean(prices[i-window+1:i+1])
                    rolling_std = np.std(prices[i-window+1:i+1])
                    sample[f'bb_upper_{window}'] = rolling_mean + 2 * rolling_std
                    sample[f'bb_lower_{window}'] = rolling_mean - 2 * rolling_std
                    sample[f'bb_width_{window}'] = (sample[f'bb_upper_{window}'] - sample[f'bb_lower_{window}']) / rolling_mean
                    sample[f'bb_position_{window}'] = (prices[i] - sample[f'bb_lower_{window}']) / (sample[f'bb_upper_{window}'] - sample[f'bb_lower_{window}'])
            
            # 10. Price acceleration (change in returns)
            for horizon in [3, 5, 10]:
                if i >= horizon*2:
                    return_t = np.log(prices[i] / prices[i-horizon])
                    return_t_prev = np.log(prices[i-horizon] / prices[i-horizon*2])
                    sample[f'return_accel_{horizon}'] = return_t - return_t_prev
            
            # Add target (n-day ahead return)
            if i + forecast_days < len(all_dates):
                forward_return = np.log(prices[i + forecast_days] / prices[i])
                # Regression target
                sample['target'] = forward_return
                # Classification target
                sample['target_binary'] = 1 if forward_return > 0 else 0
            
            data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Handle missing values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Split data into historical and latest
    train_idx = int(len(df) * 0.8)
    historical_df = df.iloc[:train_idx].copy()
    latest_df = df.iloc[train_idx:].copy()
    
    logger.info(f"Created synthetic historical data: {historical_df.shape}")
    logger.info(f"Created synthetic latest data: {latest_df.shape}")
    
    return historical_df, latest_df

def engineer_features(df, is_training=True, polynomial_degree=3):
    """
    Advanced feature engineering to improve model performance
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data
        polynomial_degree: Degree for polynomial features
        
    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Starting advanced feature engineering on dataframe of shape {df.shape}")
    
    # Copy dataframe to avoid modifying the original
    engineered_df = df.copy()
    
    # Step 1: Identify feature columns
    non_feature_cols = ['id', 'target', 'target_binary', 'era', 'date', 'asset', 'data_type']
    base_features = [col for col in engineered_df.columns if col not in non_feature_cols]
    
    logger.info(f"Starting with {len(base_features)} base features")
    
    # Step 2: Price-based features (if not already created in synthetic data)
    price_cols = [col for col in base_features if 'price' in col and 'lag' in col]
    if price_cols and 'price_lag_1' in engineered_df.columns:
        logger.info("Creating price-based features")
        
        # Create return features if they don't exist
        if not any('return_' in col for col in engineered_df.columns):
            for lag in range(1, min(10, len(price_cols))):
                col = f'price_lag_{lag}'
                prev_col = f'price_lag_{lag+1}'
                if col in engineered_df.columns and prev_col in engineered_df.columns:
                    engineered_df[f'return_{lag}d'] = np.log(engineered_df[col] / engineered_df[prev_col])
        
        # Create moving average features if they don't exist
        if not any('ma_' in col for col in engineered_df.columns):
            for window in [5, 10, 20]:
                if len(price_cols) >= window:
                    cols = [f'price_lag_{i}' for i in range(1, window+1) if f'price_lag_{i}' in engineered_df.columns]
                    if len(cols) == window:
                        engineered_df[f'ma_{window}'] = engineered_df[cols].mean(axis=1)
    
    # Step 3: Asset-specific aggregations
    if 'asset' in engineered_df.columns:
        logger.info("Creating asset-specific aggregation features")
        assets = engineered_df['asset'].unique()
        
        # Only process if we have manageable number of assets
        if len(assets) > 1 and len(assets) <= 200:
            # Features to aggregate
            agg_features = [col for col in engineered_df.columns 
                           if any(x in col for x in ['return', 'vol', 'momentum', 'rsi'])]
            
            if len(agg_features) > 0:
                # Create asset stats dictionary
                asset_stats = {}
                
                # Calculate asset-level statistics
                for asset in assets:
                    asset_data = engineered_df[engineered_df['asset'] == asset]
                    
                    for feat in agg_features:
                        if feat in asset_data.columns:
                            # Get statistics
                            mean = asset_data[feat].mean()
                            std = asset_data[feat].std()
                            if asset not in asset_stats:
                                asset_stats[asset] = {}
                            asset_stats[asset][f'{feat}_mean'] = mean
                            asset_stats[asset][f'{feat}_std'] = std
                
                # Apply asset-level normalization
                for asset in assets:
                    mask = engineered_df['asset'] == asset
                    
                    for feat in agg_features:
                        if feat in engineered_df.columns and asset in asset_stats and f'{feat}_mean' in asset_stats[asset]:
                            # Z-score normalization within each asset
                            mean = asset_stats[asset][f'{feat}_mean']
                            std = asset_stats[asset][f'{feat}_std']
                            if std > 0:
                                engineered_df.loc[mask, f'{feat}_zscore'] = (engineered_df.loc[mask, feat] - mean) / std
                
                # Market-level features
                for feat in agg_features[:10]:  # Limit to avoid too many features
                    if feat in engineered_df.columns:
                        # Calculate distance from market mean
                        market_mean = engineered_df[feat].mean()
                        engineered_df[f'{feat}_mkt_diff'] = engineered_df[feat] - market_mean
                
                # Calculate cross-asset correlations
                if 'return_1d' in engineered_df.columns:
                    # Group by era to calculate correlations
                    eras = engineered_df['era'].unique()
                    selected_eras = sorted(eras)[-min(20, len(eras)):]  # Use recent eras
                    
                    # Matrix for storing correlations
                    era_correlations = {}
                    
                    for era in selected_eras:
                        era_data = engineered_df[engineered_df['era'] == era]
                        
                        # Pivot to get asset returns
                        pivot_df = era_data.pivot(index='era', columns='asset', values='return_1d')
                        
                        # Calculate correlation matrix
                        if pivot_df.shape[1] > 1:  # Need at least 2 assets
                            corr_matrix = pivot_df.corr()
                            era_correlations[era] = corr_matrix
                    
                    # Calculate median correlation for each asset
                    median_correlations = {}
                    
                    for asset in assets:
                        asset_corrs = []
                        for era, corr_matrix in era_correlations.items():
                            if asset in corr_matrix.index:
                                # Get correlations excluding self-correlation
                                corrs = corr_matrix[asset].drop(asset, errors='ignore')
                                if not corrs.empty:
                                    asset_corrs.append(corrs.median())
                        
                        if asset_corrs:
                            median_correlations[asset] = np.median(asset_corrs)
                    
                    # Add correlation feature
                    for asset in assets:
                        if asset in median_correlations:
                            mask = engineered_df['asset'] == asset
                            engineered_df.loc[mask, 'median_market_corr'] = median_correlations[asset]
    
    # Step 4: Polynomial features for important predictors
    logger.info(f"Creating polynomial features of degree {polynomial_degree}")
    
    # Identify most important feature types for polynomials
    important_patterns = ['return', 'momentum', 'vol', 'ma_cross', 'rsi', 'bb_position']
    important_features = []
    
    for pattern in important_patterns:
        pattern_cols = [col for col in engineered_df.columns if pattern in col]
        important_features.extend(pattern_cols[:min(5, len(pattern_cols))])
    
    # Limit to reasonable number
    important_features = important_features[:20]
    
    # Create polynomial features
    if important_features:
        for feat in important_features:
            if feat in engineered_df.columns:
                for degree in range(2, polynomial_degree + 1):
                    engineered_df[f'{feat}_pow{degree}'] = engineered_df[feat] ** degree
    
    # Step 5: Interaction features between key predictors
    logger.info("Creating interaction features between predictors")
    
    # Select types of features for interactions
    return_features = [col for col in engineered_df.columns if 'return_' in col][:3]
    momentum_features = [col for col in engineered_df.columns if 'momentum_' in col][:2]
    vol_features = [col for col in engineered_df.columns if 'vol_' in col][:2]
    
    # Technical indicator interactions
    interaction_groups = [
        return_features,
        momentum_features,
        vol_features
    ]
    
    # Create interaction features between different types
    for i, group1 in enumerate(interaction_groups):
        for group2 in interaction_groups[i+1:]:
            for feat1 in group1:
                for feat2 in group2:
                    if feat1 in engineered_df.columns and feat2 in engineered_df.columns:
                        # Multiplication interaction
                        engineered_df[f'{feat1}_x_{feat2}'] = engineered_df[feat1] * engineered_df[feat2]
                        
                        # Ratio interaction (safely)
                        engineered_df[f'{feat1}_div_{feat2}'] = engineered_df[feat1] / (engineered_df[feat2] + 1e-8)
    
    # Step 6: Time-series features
    if 'era' in engineered_df.columns:
        logger.info("Creating time series features")
        
        # Sort by asset and era
        engineered_df = engineered_df.sort_values(['asset', 'era']).reset_index(drop=True)
        
        # Group by asset
        for asset in engineered_df['asset'].unique():
            asset_mask = engineered_df['asset'] == asset
            asset_data = engineered_df[asset_mask].copy()
            
            # Skip if too few data points
            if len(asset_data) <= 5:
                continue
            
            # Create lagged returns
            for lag in range(1, min(5, len(asset_data))):
                for col in ['return_1d', 'return_5d', 'return_10d']:
                    if col in asset_data.columns:
                        lagged_values = asset_data[col].shift(lag)
                        
                        if not lagged_values.isnull().all():
                            engineered_df.loc[asset_mask, f'{col}_lag{lag}'] = lagged_values
            
            # Calculate expanding means for key metrics
            for col in ['return_1d', 'vol_10', 'momentum_10']:
                if col in asset_data.columns:
                    expanding_mean = asset_data[col].expanding().mean()
                    engineered_df.loc[asset_mask, f'{col}_expand_mean'] = expanding_mean
                    
                    # Add distance from expanding mean
                    if not expanding_mean.isnull().all():
                        engineered_df.loc[asset_mask, f'{col}_dev_mean'] = asset_data[col] - expanding_mean
    
    # Step 7: Clean up and finalize
    logger.info("Finalizing feature engineering")
    
    # Fill missing values
    engineered_df = engineered_df.fillna(0)
    
    # Identify all feature columns
    feature_cols = [col for col in engineered_df.columns if col not in non_feature_cols]
    
    logger.info(f"Feature engineering complete. Final shape: {engineered_df.shape} with {len(feature_cols)} features")
    
    return engineered_df, feature_cols

def select_features(X, y, feature_names, max_features=300):
    """
    Select best features based on correlation with target
    """
    logger.info(f"Selecting up to {max_features} best features from {X.shape[1]} total features")
    
    if X.shape[1] <= max_features:
        logger.info(f"Already have {X.shape[1]} features which is <= {max_features}, skipping selection")
        return list(feature_names)
    
    # Use SelectKBest with f_regression for feature selection
    selector = SelectKBest(f_regression, k=max_features)
    selector.fit(X, y)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    # Get selected feature names
    selected_features = [feature_names[i] for i in selected_indices]
    
    logger.info(f"Selected {len(selected_features)} features")
    
    return selected_features

class LightGBMModel:
    """LightGBM model with GPU acceleration and enhanced parameters"""
    
    def __init__(self, use_gpu=True, gpu_id=0, params=None, categorical_features=None):
        self.name = "lightgbm"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.categorical_features = categorical_features
        
        # Default parameters optimized for RMSE
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'max_depth': 12,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, early_stopping_rounds=50):
        """Train the model"""
        import lightgbm as lgb
        
        logger.info(f"Training LightGBM model with {X_train.shape[1]} features")
        
        # Convert to native numpy arrays
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train_np, label=y_train_np, feature_name=feature_names,
                                categorical_feature=self.categorical_features)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            valid_data = lgb.Dataset(X_val_np, label=y_val_np, feature_name=feature_names,
                                    categorical_feature=self.categorical_features)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=10000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Get feature importance
        self.feature_importance = dict(zip(
            feature_names if feature_names else [f'f{i}' for i in range(X_train.shape[1])],
            self.model.feature_importance(importance_type='gain')
        ))
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance if hasattr(self, 'feature_importance') else {}

class XGBoostModel:
    """XGBoost model with GPU acceleration and enhanced parameters"""
    
    def __init__(self, use_gpu=True, gpu_id=0, params=None):
        self.name = "xgboost"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Default parameters optimized for RMSE
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 10,
            'eta': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'min_child_weight': 20,
            'alpha': 0.1,
            'lambda': 0.3,
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, early_stopping_rounds=50):
        """Train the model"""
        import xgboost as xgb
        
        logger.info(f"Training XGBoost model with {X_train.shape[1]} features")
        
        # Convert to numpy if needed
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train_np, label=y_train_np, feature_names=feature_names)
        
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            dval = xgb.DMatrix(X_val_np, label=y_val_np, feature_names=feature_names)
            watchlist.append((dval, 'valid'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=10000,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Get feature importance
        if feature_names:
            self.feature_importance = self.model.get_score(importance_type='gain')
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import xgboost as xgb
        
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        
        dtest = xgb.DMatrix(X_np)
        return self.model.predict(dtest)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance if hasattr(self, 'feature_importance') else {}

class PyTorchMLPModel:
    """PyTorch MLP model with advanced architecture and GPU acceleration"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, use_gpu=True, gpu_id=0, learning_rate=0.001):
        self.name = "pytorch_mlp"
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        
        import torch
        self.device = torch.device(f"cuda:{gpu_id}" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch MLP using device: {self.device}")
    
    def _create_model(self):
        """Create the model architecture"""
        import torch.nn as nn
        
        class MLPRegressor(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super(MLPRegressor, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                # Hidden layers
                for i, hidden_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, 1))
                
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x).squeeze()
        
        return MLPRegressor(self.input_dim, self.hidden_dims, self.dropout).to(self.device)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             feature_names=None, batch_size=1024, epochs=500, patience=20):
        """Train the model"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        logger.info(f"Training PyTorch MLP model with {X_train.shape[1]} features")
        
        # Convert data to numpy if needed
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Create model
        self.model = self._create_model()
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
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
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import torch
        
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions

class EnsembleModel:
    """Ensemble of multiple models with weighted blending"""
    
    def __init__(self, models, weights=None):
        self.name = "ensemble"
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        """
        Generate ensemble predictions using weighted average
        """
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
    
    def optimize_weights(self, X, y, method='grid_search'):
        """
        Optimize ensemble weights using grid search or optimization
        """
        logger.info(f"Optimizing ensemble weights using {method}")
        
        # Get predictions from all models
        model_predictions = []
        for model in self.models:
            try:
                preds = model.predict(X)
                model_predictions.append(preds)
            except Exception as e:
                logger.error(f"Error getting predictions from {model.name}: {e}")
                model_predictions.append(np.zeros(len(y)))
        
        if method == 'grid_search':
            best_weights = self._grid_search_weights(model_predictions, y)
        else:
            # Default to equal weights
            best_weights = [1/len(self.models)] * len(self.models)
        
        self.weights = best_weights
        
        # Calculate final RMSE with optimized weights
        ensemble_preds = np.zeros(len(y))
        for i, preds in enumerate(model_predictions):
            ensemble_preds += preds * self.weights[i]
        
        rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
        logger.info(f"Ensemble RMSE with optimized weights: {rmse:.6f}")
        
        return self
    
    def _grid_search_weights(self, predictions, y_true, steps=10):
        """
        Find optimal weights using grid search
        """
        n_models = len(predictions)
        if n_models <= 1:
            return [1.0]
        
        # Only optimize weights for 2-3 models with grid search
        # For more models, use default equal weights
        if n_models > 3:
            return [1/n_models] * n_models
        
        # Generate weight combinations
        if n_models == 2:
            # For 2 models, we only need to search one parameter
            # since weights sum to 1
            best_rmse = float('inf')
            best_weights = [0.5, 0.5]
            
            for w1 in np.linspace(0, 1, steps):
                w2 = 1 - w1
                weights = [w1, w2]
                
                # Calculate weighted predictions
                ensemble_preds = np.zeros(len(y_true))
                for i, preds in enumerate(predictions):
                    ensemble_preds += preds * weights[i]
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights
            
            return best_weights
        
        elif n_models == 3:
            best_rmse = float('inf')
            best_weights = [1/3, 1/3, 1/3]
            
            for w1 in np.linspace(0, 1, steps):
                remaining = 1 - w1
                for w2_ratio in np.linspace(0, 1, steps):
                    w2 = remaining * w2_ratio
                    w3 = remaining * (1 - w2_ratio)
                    
                    weights = [w1, w2, w3]
                    
                    # Calculate weighted predictions
                    ensemble_preds = np.zeros(len(y_true))
                    for i, preds in enumerate(predictions):
                        ensemble_preds += preds * weights[i]
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = weights
            
            return best_weights
    
    def get_feature_importance(self):
        """Aggregate feature importance from all models"""
        all_importances = {}
        
        for model in self.models:
            model_importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}
            for feature, importance in model_importance.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        # Average importance across models
        avg_importance = {}
        for feature, importances in all_importances.items():
            avg_importance[feature] = sum(importances) / len(importances)
        
        return avg_importance

def optimize_hyperparameters(X_train, y_train, X_val, y_val, feature_names, use_gpu=True, gpu_id=0):
    """
    Find optimal hyperparameters for LightGBM and XGBoost models
    """
    logger.info("Starting hyperparameter optimization")
    
    import optuna
    from optuna.samplers import TPESampler
    
    # Define objective function for LightGBM
    def objective_lgb(trial):
        # Define hyperparameters to optimize
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'seed': RANDOM_SEED,
            'verbose': -1
        }
        
        # Add GPU params if available
        if use_gpu:
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': gpu_id
            })
        
        # Train model
        model = LightGBMModel(use_gpu=use_gpu, gpu_id=gpu_id, params=params)
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names, early_stopping_rounds=30)
        
        # Predict and calculate validation RMSE
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        return rmse
    
    # Create and run the study
    logger.info("Optimizing LightGBM hyperparameters")
    lgb_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
    lgb_study.optimize(objective_lgb, n_trials=15)
    
    logger.info(f"Best LightGBM RMSE: {lgb_study.best_value:.6f}")
    logger.info(f"Best LightGBM params: {lgb_study.best_params}")
    
    # Define objective function for XGBoost
    def objective_xgb(trial):
        # Define hyperparameters to optimize
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'eta': trial.suggest_float('eta', 0.005, 0.05),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'lambda': trial.suggest_float('lambda', 0.0, 1.0),
            'seed': RANDOM_SEED,
            'silent': 1
        }
        
        # Add GPU params if available
        if use_gpu:
            params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': gpu_id
            })
        
        # Train model
        model = XGBoostModel(use_gpu=use_gpu, gpu_id=gpu_id, params=params)
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names, early_stopping_rounds=30)
        
        # Predict and calculate validation RMSE
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        return rmse
    
    # Create and run the study
    logger.info("Optimizing XGBoost hyperparameters")
    xgb_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_SEED))
    xgb_study.optimize(objective_xgb, n_trials=15)
    
    logger.info(f"Best XGBoost RMSE: {xgb_study.best_value:.6f}")
    logger.info(f"Best XGBoost params: {xgb_study.best_params}")
    
    # Return best parameters
    return {
        'lightgbm': lgb_study.best_params,
        'xgboost': xgb_study.best_params
    }

def evaluate_model(model, X, y, model_name=None):
    """
    Evaluate model performance
    """
    model_name = model_name or model.name
    
    # Generate predictions
    preds = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    logger.info(f"{model_name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': preds}

def plot_predictions(true_values, predictions, model_name, output_dir):
    """
    Create scatter plot of predictions vs actual values
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Add RMSE to plot
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    plt.annotate(f'RMSE: {rmse:.6f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_predictions.png'))
    plt.close()

def plot_feature_importance(model, feature_names, n_top=30, output_dir=None):
    """
    Plot feature importance for a model
    """
    if not hasattr(model, 'get_feature_importance'):
        return
    
    importances = model.get_feature_importance()
    if not importances:
        return
    
    # Sort features by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:n_top]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    y_pos = range(len(top_features))
    
    feature_names = [f[:30] + '...' if len(f) > 30 else f for f, _ in top_features]
    importances = [imp for _, imp in top_features]
    
    plt.barh(y_pos, importances)
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Importance')
    plt.title(f'Top {n_top} Feature Importance - {model.name}')
    plt.tight_layout()
    
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model.name}_feature_importance.png'))
        plt.close()
    else:
        plt.show()

def save_submission(predictions, ids, filename):
    """
    Save submission file for Numerai
    """
    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    submission_df.to_csv(filename, index=False)
    logger.info(f"Saved submission to {filename}")
    
    return filename

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Crypto Model with Advanced Feature Engineering')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Directory containing Yiedl data files')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory for saving outputs')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                      help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--max-features', type=int, default=300,
                      help='Maximum number of features to use')
    parser.add_argument('--synthetic', action='store_true',
                      help='Use synthetic data even if real data is available')
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize hyperparameters')
    parser.add_argument('--forecast-days', type=int, default=20,
                      help='Number of days to forecast')
    parser.add_argument('--polynomial-degree', type=int, default=3,
                      help='Degree for polynomial features')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'data', 'submissions')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse GPU IDs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip()]
        if not gpu_ids:
            gpu_ids = [0]
    else:
        gpu_ids = []
    
    # Determine if GPU is available
    use_gpu = len(gpu_ids) > 0
    primary_gpu = gpu_ids[0] if gpu_ids else 0
    
    logger.info(f"Using GPUs: {gpu_ids if use_gpu else 'None (CPU mode)'}")
    
    # Load or create data
    if args.synthetic:
        logger.info("Using synthetic data as requested")
        historical_df, latest_df = create_synthetic_data(forecast_days=args.forecast_days)
    else:
        logger.info("Attempting to load real data")
        historical_df, latest_df = load_yiedl_data(args.data_dir)
    
    # Apply advanced feature engineering
    logger.info("Applying advanced feature engineering to historical data")
    historical_df, feature_names = engineer_features(
        historical_df, 
        is_training=True, 
        polynomial_degree=args.polynomial_degree
    )
    
    logger.info("Applying advanced feature engineering to latest data")
    latest_df, _ = engineer_features(
        latest_df, 
        is_training=False, 
        polynomial_degree=args.polynomial_degree
    )
    
    # Split historical data into train and validation
    logger.info("Splitting historical data into train and validation sets")
    
    if 'era' in historical_df.columns:
        # Time-based split (preferred for time series)
        eras = sorted(historical_df['era'].unique())
        split_point = int(len(eras) * 0.8)
        train_eras = eras[:split_point]
        val_eras = eras[split_point:]
        
        train_df = historical_df[historical_df['era'].isin(train_eras)]
        val_df = historical_df[historical_df['era'].isin(val_eras)]
    else:
        # Random split if eras not available
        train_df, val_df = train_test_split(historical_df, test_size=0.2, random_state=RANDOM_SEED)
    
    logger.info(f"Train data: {train_df.shape}, Validation data: {val_df.shape}")
    
    # Prepare feature matrices and target variables
    X_train = train_df[feature_names]
    y_train = train_df['target']
    
    X_val = val_df[feature_names]
    y_val = val_df['target']
    
    X_test = latest_df[feature_names]
    test_ids = latest_df['id'] if 'id' in latest_df.columns else [f"id_{i}" for i in range(len(latest_df))]
    
    # Select important features
    selected_features = select_features(X_train, y_train, feature_names, max_features=args.max_features)
    
    # Update feature matrices
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # Optimize hyperparameters if requested
    if args.optimize:
        best_params = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, selected_features,
            use_gpu=use_gpu, gpu_id=primary_gpu
        )
        
        lgb_params = best_params['lightgbm']
        xgb_params = best_params['xgboost']
    else:
        lgb_params = None
        xgb_params = None
    
    # Initialize models
    models = []
    
    # LightGBM model on first GPU
    if use_gpu and len(gpu_ids) >= 1:
        models.append(LightGBMModel(use_gpu=True, gpu_id=gpu_ids[0], params=lgb_params))
    else:
        models.append(LightGBMModel(use_gpu=False, params=lgb_params))
    
    # XGBoost model on second GPU if available
    if use_gpu and len(gpu_ids) >= 2:
        models.append(XGBoostModel(use_gpu=True, gpu_id=gpu_ids[1], params=xgb_params))
    else:
        models.append(XGBoostModel(use_gpu=use_gpu, gpu_id=primary_gpu, params=xgb_params))
    
    # PyTorch MLP model on third GPU if available
    if use_gpu and len(gpu_ids) >= 3:
        models.append(PyTorchMLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=[256, 128, 64],
            dropout=0.3,
            use_gpu=True,
            gpu_id=gpu_ids[2]
        ))
    elif use_gpu:
        models.append(PyTorchMLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=[256, 128, 64],
            dropout=0.3,
            use_gpu=True,
            gpu_id=primary_gpu
        ))
    
    # Train models and collect results
    model_results = {}
    trained_models = []
    
    for i, model in enumerate(models):
        logger.info(f"Training {model.name} model")
        
        try:
            # Train the model
            model.train(X_train, y_train, X_val, y_val, feature_names=selected_features)
            trained_models.append(model)
            
            # Evaluate on validation set
            results = evaluate_model(model, X_val, y_val)
            model_results[model.name] = results
            
            # Plot feature importance
            plot_feature_importance(model, selected_features, 
                                  output_dir=os.path.join(args.output_dir, 'plots'))
            
            # Plot predictions
            plot_predictions(y_val, results['predictions'], model.name, 
                           output_dir=os.path.join(args.output_dir, 'plots'))
            
            # Generate and save submission
            test_preds = model.predict(X_test)
            save_submission(
                test_preds,
                test_ids,
                os.path.join(args.output_dir, f"{model.name}_submission.csv")
            )
        
        except Exception as e:
            logger.error(f"Error training {model.name} model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Create ensemble if we have at least 2 trained models
    if len(trained_models) >= 2:
        logger.info("Training ensemble model")
        
        ensemble = EnsembleModel(trained_models)
        
        # Optimize ensemble weights
        ensemble.optimize_weights(X_val, y_val)
        
        # Evaluate ensemble
        ensemble_results = evaluate_model(ensemble, X_val, y_val)
        model_results['ensemble'] = ensemble_results
        
        # Plot ensemble predictions
        plot_predictions(y_val, ensemble_results['predictions'], 'ensemble', 
                       output_dir=os.path.join(args.output_dir, 'plots'))
        
        # Generate and save ensemble submission
        ensemble_preds = ensemble.predict(X_test)
        save_submission(
            ensemble_preds,
            test_ids,
            os.path.join(args.output_dir, "ensemble_submission.csv")
        )
    
    # Print final summary
    logger.info("\n===== MODEL RESULTS SUMMARY =====")
    logger.info(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    logger.info("-" * 50)
    
    for model_name, results in sorted(model_results.items(), key=lambda x: x[1]['rmse']):
        logger.info(f"{model_name:<15} {results['rmse']:<10.6f} {results['mae']:<10.6f} {results['r2']:<10.6f}")
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        # Remove predictions from results to make JSON serializable
        serializable_results = {}
        for model_name, results in model_results.items():
            serializable_results[model_name] = {
                'rmse': float(results['rmse']),
                'mae': float(results['mae']),
                'r2': float(results['r2'])
            }
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Return best model RMSE
    best_model = min(model_results.items(), key=lambda x: x[1]['rmse'])
    logger.info(f"Best model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.6f}")
    
    return 0

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced Crypto Model with Advanced Feature Engineering')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Directory containing Yiedl data files')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory for saving outputs')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                      help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--max-features', type=int, default=300,
                      help='Maximum number of features to use')
    parser.add_argument('--synthetic', action='store_true',
                      help='Use synthetic data even if real data is available')
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize hyperparameters')
    parser.add_argument('--forecast-days', type=int, default=20,
                      help='Number of days to forecast')
    parser.add_argument('--polynomial-degree', type=int, default=3,
                      help='Degree for polynomial features')
    
    args = parser.parse_args()
    
    sys.exit(main())