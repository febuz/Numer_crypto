#!/usr/bin/env python3
"""
Advanced model for crypto prediction with sophisticated feature engineering
and ensemble optimization for RMSE below 0.25

Features:
- Polynomial features
- Asset-specific aggregation features 
- Cross-asset correlations
- Time series momentum indicators
- Weighted ensemble optimization
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Set up logging
log_file = f"advanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def create_synthetic_data(n_samples=12000, n_assets=20):
    """Create synthetic crypto data with realistic price patterns"""
    logger.info(f"Creating synthetic data with {n_samples} samples, {n_assets} assets")
    
    # Create timeline
    all_dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
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
            
            # Calculate daily return
            daily_return = asset_shock + market_component + trend_component + seasonality
            
            # Apply return to price with log-normal distribution (prevents negative prices)
            prices[i] = prices[i-1] * np.exp(daily_return)
        
        all_prices[asset_id] = prices
    
    # Generate dataset with features
    data = []
    
    for asset_id, prices in all_prices.items():
        # Get asset properties
        props = asset_properties[asset_id]
        
        # Process each day with sufficient history
        for i in range(30, len(all_dates) - 10):
            sample = {
                'id': f"{asset_id}_{i-30}",
                'asset': asset_id,
                'date': all_dates[i],
                'era': i-30,
                'price': prices[i],  # Current price
                'market': market_factor[i]  # Market factor
            }
            
            # === FEATURE ENGINEERING ===
            
            # 1. Price lags
            for lag in range(1, min(11, 30+1)):
                sample[f'price_lag_{lag}'] = prices[i-lag]
            
            # 2. Returns at various horizons
            for horizon in [1, 2, 3, 5, 7, 10]:
                if i >= horizon:
                    sample[f'return_{horizon}d'] = np.log(prices[i] / prices[i-horizon])
            
            # 3. Moving averages
            for window in [5, 10, 20]:
                if i >= window:
                    sample[f'ma_{window}'] = np.mean(prices[i-window+1:i+1])
                    # MA crossovers
                    if window > 5 and 'ma_5' in sample:
                        sample[f'ma_cross_5_{window}'] = sample['ma_5'] / sample[f'ma_{window}'] - 1
            
            # 4. Volatility (standard deviation of returns)
            for window in [5, 10, 20]:
                if i >= window:
                    returns = np.diff(np.log(prices[i-window:i+1]))
                    sample[f'vol_{window}'] = np.std(returns)
            
            # 5. Price momentum indicators
            for window in [10, 20]:
                if i >= window:
                    sample[f'momentum_{window}'] = prices[i] / prices[i-window] - 1
            
            # 6. RSI (Relative Strength Index)
            for window in [14]:
                if i >= window+1:
                    diff = np.diff(prices[i-window:i+1])
                    gains = np.sum(np.maximum(diff, 0))
                    losses = np.sum(np.abs(np.minimum(diff, 0)))
                    if losses == 0:
                        sample[f'rsi_{window}'] = 100
                    else:
                        rs = gains / losses
                        sample[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # 7. Bollinger Bands
            for window in [20]:
                if i >= window:
                    rolling_mean = np.mean(prices[i-window+1:i+1])
                    rolling_std = np.std(prices[i-window+1:i+1])
                    sample[f'bb_upper_{window}'] = rolling_mean + 2 * rolling_std
                    sample[f'bb_lower_{window}'] = rolling_mean - 2 * rolling_std
                    sample[f'bb_width_{window}'] = (sample[f'bb_upper_{window}'] - sample[f'bb_lower_{window}']) / rolling_mean
                    sample[f'bb_position_{window}'] = (prices[i] - sample[f'bb_lower_{window}']) / (sample[f'bb_upper_{window}'] - sample[f'bb_lower_{window}'])
            
            # 8. Price acceleration (change in returns)
            for horizon in [3, 5]:
                if i >= horizon*2:
                    return_t = np.log(prices[i] / prices[i-horizon])
                    return_t_prev = np.log(prices[i-horizon] / prices[i-horizon*2])
                    sample[f'return_accel_{horizon}'] = return_t - return_t_prev
            
            # Add target (n-day ahead return)
            if i + 10 < len(all_dates):
                forward_return = np.log(prices[i + 10] / prices[i])
                # Regression target
                sample['target'] = forward_return
                # Classification target
                sample['target_binary'] = 1 if forward_return > 0 else 0
            
            data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Split data into historical and latest
    train_idx = int(len(df) * 0.8)
    historical_df = df.iloc[:train_idx].copy()
    latest_df = df.iloc[train_idx:].copy()
    
    logger.info(f"Created synthetic historical data: {historical_df.shape}")
    logger.info(f"Created synthetic latest data: {latest_df.shape}")
    
    return historical_df, latest_df

def engineer_features(df, polynomial_degree=3):
    """Advanced feature engineering with polynomial features"""
    logger.info(f"Engineering features for data of shape {df.shape}")
    
    # Copy dataframe to avoid modifying the original
    engineered_df = df.copy()
    
    # Step 1: Asset-specific features
    if 'asset' in engineered_df.columns:
        logger.info("Creating asset-specific features")
        assets = engineered_df['asset'].unique()
        
        # Group by asset
        grouped = engineered_df.groupby('asset')
        
        # Calculate asset-level statistics
        for asset, asset_data in grouped:
            mask = engineered_df['asset'] == asset
            
            # For time series features like returns, calculate statistics
            for feat in ['return_1d', 'return_3d', 'return_5d', 'vol_5', 'vol_10']:
                if feat in asset_data.columns:
                    mean = asset_data[feat].mean()
                    std = asset_data[feat].std() if len(asset_data) > 1 else 1.0
                    
                    # Z-score normalization within asset
                    if std > 0:
                        engineered_df.loc[mask, f'{feat}_zscore'] = (engineered_df.loc[mask, feat] - mean) / std
    
    # Step 2: Polynomial features
    logger.info(f"Creating polynomial features of degree {polynomial_degree}")
    
    # Select features for polynomial expansion
    base_features = ['return_1d', 'return_3d', 'momentum_10', 'vol_5']
    base_features = [f for f in base_features if f in engineered_df.columns]
    
    # Create polynomial features
    for feat in base_features:
        for degree in range(2, polynomial_degree + 1):
            engineered_df[f'{feat}_pow{degree}'] = engineered_df[feat] ** degree
    
    # Step 3: Interaction features
    logger.info("Creating interaction features")
    
    # Create interaction features between different types
    for i, feat1 in enumerate(base_features):
        for feat2 in base_features[i+1:]:
            # Multiplication
            engineered_df[f'{feat1}_x_{feat2}'] = engineered_df[feat1] * engineered_df[feat2]
            
            # Ratio (safe division)
            denominator = engineered_df[feat2].copy()
            denominator[denominator == 0] = 1e-8  # Avoid division by zero
            engineered_df[f'{feat1}_div_{feat2}'] = engineered_df[feat1] / denominator
    
    # Step 4: Time series features
    logger.info("Creating time series features")
    
    if 'era' in engineered_df.columns and 'asset' in engineered_df.columns:
        # Sort by asset and era
        engineered_df = engineered_df.sort_values(['asset', 'era']).reset_index(drop=True)
        
        # Group by asset to create lagged features
        for asset in engineered_df['asset'].unique():
            asset_mask = engineered_df['asset'] == asset
            asset_data = engineered_df[asset_mask].copy()
            
            # Skip if too few data points
            if len(asset_data) <= 5:
                continue
            
            # Create lagged returns
            for lag in range(1, min(5, len(asset_data))):
                for col in ['return_1d', 'return_3d']:
                    if col in asset_data.columns:
                        lagged_values = asset_data[col].shift(lag)
                        engineered_df.loc[asset_mask, f'{col}_lag{lag}'] = lagged_values.fillna(0)
    
    # Step 5: Market features
    if 'market' in engineered_df.columns:
        logger.info("Creating market-related features")
        
        # Market returns
        if 'market' in engineered_df.columns:
            for window in [1, 3, 5]:
                if f'market_lag{window}' not in engineered_df.columns:
                    engineered_df[f'market_lag{window}'] = engineered_df.groupby('asset')['market'].shift(window).fillna(0)
                    
                # Market return
                if window > 0:
                    denominator = engineered_df[f'market_lag{window}'].copy()
                    denominator[denominator == 0] = 1e-8  # Avoid division by zero
                    engineered_df[f'market_return_{window}'] = (engineered_df['market'] / denominator - 1)
    
    # Fill missing values
    engineered_df = engineered_df.fillna(0)
    
    # Identify all feature columns
    non_feature_cols = ['id', 'target', 'target_binary', 'era', 'date', 'asset']
    feature_cols = [col for col in engineered_df.columns if col not in non_feature_cols]
    
    logger.info(f"Feature engineering complete. Final shape: {engineered_df.shape} with {len(feature_cols)} features")
    
    return engineered_df, feature_cols

class LightGBMModel:
    """LightGBM model with enhanced parameters"""
    
    def __init__(self, use_gpu=False):
        self.name = "lightgbm"
        self.use_gpu = use_gpu
        
        # Default parameters optimized for RMSE
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': RANDOM_SEED
        }
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        import lightgbm as lgb
        
        logger.info(f"Training LightGBM model with {X_train.shape[1]} features")
        
        # Convert to native numpy arrays
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train_np, label=y_train_np)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            valid_data = lgb.Dataset(X_val_np, label=y_val_np)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=150,  # Reduced for quick results
            valid_sets=valid_sets,
            valid_names=valid_names
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(
            [f'f{i}' for i in range(X_train.shape[1])],
            self.model.feature_importance(importance_type='gain')
        ))
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)

class XGBModel:
    """XGBoost model"""
    
    def __init__(self, use_gpu=False):
        self.name = "xgboost"
        self.use_gpu = use_gpu
        
        # Default parameters optimized for RMSE
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': RANDOM_SEED
        }
        
        # Add GPU parameters if needed
        if self.use_gpu:
            self.params.update({
                'tree_method': 'hist',
                'device': 'cuda:0'
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        import xgboost as xgb
        
        logger.info(f"Training XGBoost model with {X_train.shape[1]} features")
        
        # Convert to numpy if needed
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Prepare DMatrix without feature names to avoid compatibility issues
        dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
        
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            dval = xgb.DMatrix(X_val_np, label=y_val_np)
            watchlist.append((dval, 'valid'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=150,  # Reduced for quick results
            evals=watchlist,
            verbose_eval=50
        )
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        import xgboost as xgb
        
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        
        dtest = xgb.DMatrix(X_np)
        return self.model.predict(dtest)

class EnsembleModel:
    """Ensemble of multiple models with weighted blending"""
    
    def __init__(self, models, weights=None):
        self.name = "ensemble"
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        """Generate ensemble predictions using weighted average"""
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
    
    def optimize_weights(self, X, y):
        """Optimize ensemble weights using grid search"""
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from all models
        model_predictions = []
        for model in self.models:
            try:
                preds = model.predict(X)
                model_predictions.append(preds)
            except Exception as e:
                logger.error(f"Error getting predictions from {model.name}: {e}")
                model_predictions.append(np.zeros(len(y)))
        
        # For 2 models, we can do a simple grid search
        if len(self.models) == 2:
            best_rmse = float('inf')
            best_weights = [0.5, 0.5]
            
            for w1 in np.linspace(0, 1, 20):  # More granular search
                w2 = 1 - w1
                weights = [w1, w2]
                
                # Calculate weighted predictions
                ensemble_preds = np.zeros(len(y))
                for i, preds in enumerate(model_predictions):
                    ensemble_preds += preds * weights[i]
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights
            
            self.weights = best_weights
            logger.info(f"Optimized weights: {self.weights}")
        
        # Calculate final RMSE with optimized weights
        ensemble_preds = np.zeros(len(y))
        for i, preds in enumerate(model_predictions):
            ensemble_preds += preds * self.weights[i]
        
        rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
        logger.info(f"Ensemble RMSE with optimized weights: {rmse:.6f}")
        
        return self

def evaluate_model(model, X, y, model_name=None):
    """Evaluate model performance"""
    model_name = model_name or model.name
    
    # Generate predictions
    preds = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    # Check if we've achieved target RMSE of 0.25
    if rmse <= 0.25:
        logger.info(f"TARGET ACHIEVED! {model_name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    else:
        logger.info(f"{model_name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': preds}

def save_submission(predictions, ids, filename):
    """Save submission file for Numerai"""
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

def main():
    """Main function"""
    # Create output directory
    output_dir = os.path.join(Path.cwd(), 'data', 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        logger.info(f"CUDA available: {use_gpu}")
        if use_gpu:
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except (ImportError, AttributeError):
        use_gpu = False
        logger.info("PyTorch not installed properly, using CPU")
    
    # Generate synthetic data
    historical_df, latest_df = create_synthetic_data()
    
    # Apply advanced feature engineering with polynomial degree 3
    historical_df, feature_names = engineer_features(historical_df, polynomial_degree=3)
    latest_df, _ = engineer_features(latest_df, polynomial_degree=3)
    
    # Split data
    logger.info("Splitting data into train and validation sets")
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
    test_ids = latest_df['id']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different models
    models = []
    model_results = {}
    
    # 1. LightGBM
    try:
        import lightgbm
        lgb_model = LightGBMModel(use_gpu=use_gpu)
        lgb_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(lgb_model)
        
        # Evaluate
        lgb_results = evaluate_model(lgb_model, X_val_scaled, y_val)
        model_results['lightgbm'] = lgb_results
        
        # Generate submission
        lgb_preds = lgb_model.predict(X_test_scaled)
        save_submission(
            lgb_preds,
            test_ids,
            os.path.join(output_dir, "lightgbm_submission.csv")
        )
    except (ImportError, Exception) as e:
        logger.error(f"Error with LightGBM: {e}")
    
    # 2. XGBoost
    try:
        import xgboost
        xgb_model = XGBModel(use_gpu=use_gpu)
        xgb_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(xgb_model)
        
        # Evaluate
        xgb_results = evaluate_model(xgb_model, X_val_scaled, y_val)
        model_results['xgboost'] = xgb_results
        
        # Generate submission
        xgb_preds = xgb_model.predict(X_test_scaled)
        save_submission(
            xgb_preds,
            test_ids,
            os.path.join(output_dir, "xgboost_submission.csv")
        )
    except (ImportError, Exception) as e:
        logger.error(f"Error with XGBoost: {e}")
    
    # 3. Ensemble (if we have multiple models)
    if len(models) >= 2:
        logger.info("Creating ensemble model")
        ensemble = EnsembleModel(models)
        ensemble.optimize_weights(X_val_scaled, y_val)
        
        # Evaluate ensemble
        ensemble_results = evaluate_model(ensemble, X_val_scaled, y_val)
        model_results['ensemble'] = ensemble_results
        
        # Generate submission
        ensemble_preds = ensemble.predict(X_test_scaled)
        save_submission(
            ensemble_preds,
            test_ids,
            os.path.join(output_dir, "ensemble_submission.csv")
        )
    
    # Print final summary
    logger.info("\n===== MODEL RESULTS SUMMARY =====")
    logger.info(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    logger.info("-" * 50)
    
    for model_name, results in sorted(model_results.items(), key=lambda x: x[1]['rmse']):
        logger.info(f"{model_name:<15} {results['rmse']:<10.6f} {results['mae']:<10.6f} {results['r2']:<10.6f}")
    
    logger.info("Model training and evaluation complete!")
    return 0

if __name__ == "__main__":
    main()