#!/usr/bin/env python3
"""
Optimized script for training machine learning models on Numerai crypto data
combined with Yiedl data, with support for GPU acceleration.

Key features:
- GPU-accelerated LightGBM and XGBoost models
- Advanced feature engineering
- Ensemble optimization for lowest RMSE
- Submission file generation including validation data
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
log_file = f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def check_gpu_availability():
    """Check if GPU is available for model training"""
    use_gpu = False
    gpu_info = {}
    
    # Check CUDA availability via PyTorch
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        
        if use_gpu:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['devices'] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': gpu_name
                })
                logger.info(f"Found GPU {i}: {gpu_name}")
    except (ImportError, Exception) as e:
        logger.warning(f"PyTorch not available: {e}")
    
    # Check if LightGBM is built with GPU support
    try:
        import lightgbm as lgb
        if hasattr(lgb, 'gpu_present') and lgb.gpu_present():
            gpu_info['lightgbm_gpu'] = True
            logger.info("LightGBM has GPU support")
        else:
            gpu_info['lightgbm_gpu'] = False
            logger.info("LightGBM doesn't appear to have GPU support")
    except (ImportError, Exception) as e:
        logger.warning(f"LightGBM not available: {e}")
        gpu_info['lightgbm_gpu'] = False
    
    # Check if XGBoost is built with GPU support
    try:
        import xgboost as xgb
        gpu_info['xgboost_gpu'] = True  # Assuming modern XGBoost has GPU support
        logger.info("XGBoost is available (likely with GPU support)")
    except (ImportError, Exception) as e:
        logger.warning(f"XGBoost not available: {e}")
        gpu_info['xgboost_gpu'] = False
    
    logger.info(f"GPU availability: {use_gpu}")
    return use_gpu, gpu_info

def load_data(train_file, live_file):
    """Load training and live data from parquet files"""
    logger.info(f"Loading train data from {train_file}")
    train_df = pd.read_parquet(train_file)
    logger.info(f"Train data shape: {train_df.shape}")
    
    logger.info(f"Loading live data from {live_file}")
    live_df = pd.read_parquet(live_file)
    logger.info(f"Live data shape: {live_df.shape}")
    
    return train_df, live_df

def clean_data(df):
    """Clean data by handling missing values, infinities, etc."""
    logger.info(f"Cleaning data of shape {df.shape}")
    
    # Make a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # 1. Replace infinities with NaN
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Count missing values before filling
    missing_counts = cleaned_df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        logger.info(f"Columns with missing values: {len(cols_with_missing)}")
        for col, count in cols_with_missing.items():
            logger.info(f"  - {col}: {count} missing values ({count/len(cleaned_df)*100:.2f}%)")
    
    # 3. Fill missing values
    # - For Numerai features, fill with 0
    numerai_cols = [col for col in cleaned_df.columns if col.startswith('feature_')]
    if numerai_cols:
        cleaned_df[numerai_cols] = cleaned_df[numerai_cols].fillna(0)
    
    # - For Yiedl features, fill with mean or 0 depending on type
    if 'pvm_' in ''.join(cleaned_df.columns):
        pvm_cols = [col for col in cleaned_df.columns if col.startswith('pvm_')]
        if pvm_cols:
            cleaned_df[pvm_cols] = cleaned_df[pvm_cols].fillna(0)
    
    if 'sentiment_' in ''.join(cleaned_df.columns):
        sentiment_cols = [col for col in cleaned_df.columns if col.startswith('sentiment_')]
        if sentiment_cols:
            # Fill with column means
            for col in sentiment_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    if 'onchain_' in ''.join(cleaned_df.columns):
        onchain_cols = [col for col in cleaned_df.columns if col.startswith('onchain_')]
        if onchain_cols:
            cleaned_df[onchain_cols] = cleaned_df[onchain_cols].fillna(0)
    
    # 4. Fill any remaining NaNs with 0
    cleaned_df = cleaned_df.fillna(0)
    
    # Log data cleaning results
    logger.info(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    
    return cleaned_df

def engineer_features(df, polynomial_degree=2):
    """Engineer features for model training"""
    logger.info(f"Engineering features with polynomial degree {polynomial_degree}")
    
    # Make a copy to avoid modifying original
    engineered_df = df.copy()
    
    # Identify feature columns (excludes metadata columns)
    metadata_cols = ['id', 'target', 'era', 'date', 'asset', 'data_type']
    feature_cols = [col for col in engineered_df.columns if col not in metadata_cols]
    
    # Group feature columns by type
    numerai_cols = [col for col in feature_cols if col.startswith('feature_')]
    yiedl_pvm_cols = [col for col in feature_cols if col.startswith('pvm_')]
    yiedl_sentiment_cols = [col for col in feature_cols if col.startswith('sentiment_')]
    yiedl_onchain_cols = [col for col in feature_cols if col.startswith('onchain_')]
    
    logger.info(f"Feature counts by type:")
    logger.info(f"  - Numerai features: {len(numerai_cols)}")
    logger.info(f"  - Yiedl PVM features: {len(yiedl_pvm_cols)}")
    logger.info(f"  - Yiedl Sentiment features: {len(yiedl_sentiment_cols)}")
    logger.info(f"  - Yiedl Onchain features: {len(yiedl_onchain_cols)}")
    
    # 1. Create polynomial features for selected features
    if polynomial_degree >= 2:
        logger.info(f"Creating polynomial features")
        
        # Select most important features for polynomial expansion (to avoid explosion)
        poly_candidates = []
        
        # Include some Numerai features if available
        if len(numerai_cols) > 0:
            poly_candidates.extend(numerai_cols[:5])
        
        # Include some Yiedl features from each category if available
        if len(yiedl_pvm_cols) > 0:
            poly_candidates.extend(yiedl_pvm_cols[:3])
        if len(yiedl_sentiment_cols) > 0:
            poly_candidates.extend(yiedl_sentiment_cols[:3])
        if len(yiedl_onchain_cols) > 0:
            poly_candidates.extend(yiedl_onchain_cols[:3])
        
        # Limit to 10 features for polynomial expansion
        poly_candidates = poly_candidates[:10]
        
        # Create polynomial features
        for feat in poly_candidates:
            for degree in range(2, polynomial_degree + 1):
                engineered_df[f'{feat}_pow{degree}'] = engineered_df[feat] ** degree
    
    # 2. Create interaction features between feature types
    logger.info("Creating interaction features")
    
    # Select top features for interactions
    interactions = []
    
    if len(numerai_cols) > 0 and len(yiedl_pvm_cols) > 0:
        for num_feat in numerai_cols[:2]:
            for pvm_feat in yiedl_pvm_cols[:2]:
                interactions.append((num_feat, pvm_feat))
    
    if len(numerai_cols) > 0 and len(yiedl_sentiment_cols) > 0:
        for num_feat in numerai_cols[:2]:
            for sent_feat in yiedl_sentiment_cols[:2]:
                interactions.append((num_feat, sent_feat))
    
    if len(yiedl_pvm_cols) > 0 and len(yiedl_sentiment_cols) > 0:
        for pvm_feat in yiedl_pvm_cols[:2]:
            for sent_feat in yiedl_sentiment_cols[:2]:
                interactions.append((pvm_feat, sent_feat))
    
    # Create interaction features
    for feat1, feat2 in interactions:
        # Multiplication interaction
        engineered_df[f'{feat1}_x_{feat2}'] = engineered_df[feat1] * engineered_df[feat2]
        
        # Division interaction (safe division)
        denominator = engineered_df[feat2].copy()
        denominator = denominator.replace(0, 1e-8)  # Avoid division by zero
        engineered_df[f'{feat1}_div_{feat2}'] = engineered_df[feat1] / denominator
    
    # Identify all feature columns after engineering
    all_feature_cols = [col for col in engineered_df.columns if col not in metadata_cols]
    logger.info(f"Feature engineering complete. Feature count increased from {len(feature_cols)} to {len(all_feature_cols)}")
    
    return engineered_df, all_feature_cols

class LightGBMModel:
    """LightGBM model with GPU acceleration"""
    
    def __init__(self, use_gpu=False, gpu_id=0):
        self.name = "lightgbm"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
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
                'gpu_device_id': self.gpu_id
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=500, early_stopping_rounds=50):
        """Train the model with early stopping"""
        import lightgbm as lgb
        
        logger.info(f"Training LightGBM model with {X_train.shape[1]} features")
        logger.info(f"GPU acceleration: {self.use_gpu}")
        
        # Convert to numpy arrays
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
        
        # Check for LightGBM version to handle early_stopping parameter
        import pkg_resources
        lgb_version = pkg_resources.get_distribution('lightgbm').version
        
        # Train model with early stopping
        if valid_sets and len(valid_sets) > 1:
            if tuple(map(int, lgb_version.split('.')[:2])) >= (3, 3):
                # For LightGBM >= 3.3.0, use early_stopping_rounds
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=50
                )
            else:
                # For older LightGBM versions, use callbacks
                callbacks = [
                    lgb.early_stopping(early_stopping_rounds, verbose=True),
                    lgb.log_evaluation(period=50)
                ]
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks
                )
        else:
            # Without validation data, train for fixed number of rounds
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                verbose_eval=50
            )
        
        # Get feature importance
        self.feature_importance = dict(zip(
            range(X_train.shape[1]),
            self.model.feature_importance(importance_type='gain')
        ))
        
        # Log top features by importance
        importance_df = pd.DataFrame({
            'feature': range(X_train.shape[1]),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features by importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  - Feature {row['feature']}: {row['importance']}")
        
        return self
    
    def predict(self, X):
        """Generate predictions"""
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)

class XGBModel:
    """XGBoost model with GPU acceleration"""
    
    def __init__(self, use_gpu=False, gpu_id=0):
        self.name = "xgboost"
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
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
                'tree_method': 'gpu_hist',
                'gpu_id': self.gpu_id
            })
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=500, early_stopping_rounds=50):
        """Train the model with early stopping"""
        import xgboost as xgb
        
        logger.info(f"Training XGBoost model with {X_train.shape[1]} features")
        logger.info(f"GPU acceleration: {self.use_gpu}")
        
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
        
        # Train model with early stopping if validation set is provided
        if len(watchlist) > 1:
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50
            )
        else:
            # Without validation data, train for fixed number of rounds
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
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
    """Ensemble model with optimized weights"""
    
    def __init__(self, models, model_names=None, weights=None):
        self.name = "ensemble"
        self.models = models
        self.model_names = model_names if model_names else [f"model_{i}" for i in range(len(models))]
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict(self, X):
        """Generate ensemble predictions using weighted average"""
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                model_preds = model.predict(X)
                predictions.append(model_preds)
            except Exception as e:
                logger.error(f"Error generating predictions from {self.model_names[i]}: {e}")
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
        """Optimize ensemble weights to minimize RMSE"""
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from all models
        model_predictions = []
        for i, model in enumerate(self.models):
            try:
                preds = model.predict(X)
                model_predictions.append(preds)
                
                # Calculate individual model RMSE
                rmse = np.sqrt(mean_squared_error(y, preds))
                logger.info(f"{self.model_names[i]} RMSE: {rmse:.6f}")
            except Exception as e:
                logger.error(f"Error getting predictions from {self.model_names[i]}: {e}")
                model_predictions.append(np.zeros(len(y)))
        
        # For 2 models, we can do a simple grid search
        if len(self.models) == 2:
            best_rmse = float('inf')
            best_weights = [0.5, 0.5]
            
            # Try 20 different weight combinations
            for w1 in np.linspace(0, 1, 20):
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
            logger.info(f"Optimized weights: {self.weights[0]:.4f} for {self.model_names[0]}, {self.weights[1]:.4f} for {self.model_names[1]}")
        
        # For more than 2 models, we'd need more advanced optimization
        elif len(self.models) > 2:
            # Simple rank-based weighting
            model_rmses = []
            for i, preds in enumerate(model_predictions):
                rmse = np.sqrt(mean_squared_error(y, preds))
                model_rmses.append((i, rmse))
            
            # Sort by RMSE (lower is better)
            sorted_models = sorted(model_rmses, key=lambda x: x[1])
            
            # Assign weights inversely proportional to rank
            n_models = len(sorted_models)
            weights = [0] * n_models
            
            # Basic inverse rank weighting
            for rank, (model_idx, _) in enumerate(sorted_models):
                # Inverse rank (lower rank = higher weight)
                weights[model_idx] = n_models - rank
            
            # Normalize weights
            total = sum(weights)
            self.weights = [w/total for w in weights]
            
            logger.info("Optimized weights based on individual model performance:")
            for i, weight in enumerate(self.weights):
                logger.info(f"  - {self.model_names[i]}: {weight:.4f}")
        
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

def save_submission(predictions, ids, filename, round_num=None):
    """Save submission file for Numerai"""
    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })
    
    # Verify submission data
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Sample predictions: {submission_df['prediction'].describe()}")
    
    # Check for NaN or infinity values
    if submission_df['prediction'].isna().any():
        logger.warning(f"Submission contains {submission_df['prediction'].isna().sum()} NaN values. Replacing with 0.")
        submission_df['prediction'] = submission_df['prediction'].fillna(0)
    
    if np.isinf(submission_df['prediction']).any():
        logger.warning(f"Submission contains {np.isinf(submission_df['prediction']).sum()} infinity values. Replacing with 0.")
        submission_df['prediction'] = submission_df['prediction'].replace([np.inf, -np.inf], 0)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Add round number to filename if provided
    if round_num:
        base_name, ext = os.path.splitext(filename)
        filename = f"{base_name}_round{round_num}{ext}"
    
    # Save to CSV
    submission_df.to_csv(filename, index=False)
    logger.info(f"Saved submission to {filename}")
    
    return filename

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models and generate predictions for Numerai crypto tournament')
    
    parser.add_argument('--train-file', type=str, help='Path to merged training data parquet file')
    parser.add_argument('--live-file', type=str, help='Path to merged live data parquet file')
    parser.add_argument('--output-dir', type=str, default='submissions', help='Directory to save submission files')
    parser.add_argument('--round', type=int, help='Current Numerai round number')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble model creation')
    parser.add_argument('--polynomial-degree', type=int, default=2, help='Degree for polynomial feature engineering')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Check if we need to download data first
    if not args.train_file or not args.live_file:
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            import download_numerai_yiedl_data
            
            logger.info("Downloading Numerai and Yiedl data...")
            data_info = download_numerai_yiedl_data.main()
            
            args.train_file = data_info.get('train_file')
            args.live_file = data_info.get('live_file')
            
            if not args.round and 'current_round' in data_info:
                args.round = data_info['current_round']
            
            logger.info(f"Using downloaded data: train={args.train_file}, live={args.live_file}, round={args.round}")
        except (ImportError, Exception) as e:
            logger.error(f"Failed to download data: {e}")
            if not args.train_file or not args.live_file:
                logger.error("No data files specified. Exiting.")
                return 1
    
    # Create output directory
    output_dir = os.path.join(Path.cwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    use_gpu, gpu_info = check_gpu_availability()
    if args.gpu and not use_gpu:
        logger.warning("GPU acceleration requested but no GPU is available. Using CPU instead.")
    use_gpu = args.gpu and use_gpu
    
    # Load data
    train_df, live_df = load_data(args.train_file, args.live_file)
    
    # Clean data
    train_df = clean_data(train_df)
    live_df = clean_data(live_df)
    
    # Engineer features
    train_df, feature_names = engineer_features(train_df, polynomial_degree=args.polynomial_degree)
    live_df, _ = engineer_features(live_df, polynomial_degree=args.polynomial_degree)
    
    # Split data
    logger.info("Splitting data into train and validation sets")
    if 'era' in train_df.columns:
        # Time-based split (preferred for time series)
        eras = sorted(train_df['era'].unique())
        split_point = int(len(eras) * 0.8)
        train_eras = eras[:split_point]
        val_eras = eras[split_point:]
        
        train_data = train_df[train_df['era'].isin(train_eras)]
        val_data = train_df[train_df['era'].isin(val_eras)]
    else:
        # Random split if eras not available
        train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED)
    
    logger.info(f"Train data: {train_data.shape}, Validation data: {val_data.shape}")
    
    # Prepare feature matrices and target variables
    X_train = train_data[feature_names]
    y_train = train_data['target']
    
    X_val = val_data[feature_names]
    y_val = val_data['target']
    
    X_test = live_df[feature_names]
    test_ids = live_df['id']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different models
    models = []
    model_results = {}
    model_names = []
    
    # 1. LightGBM
    try:
        import lightgbm
        lgb_model = LightGBMModel(use_gpu=use_gpu, gpu_id=args.gpu_id)
        lgb_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(lgb_model)
        model_names.append("lightgbm")
        
        # Evaluate
        lgb_results = evaluate_model(lgb_model, X_val_scaled, y_val)
        model_results['lightgbm'] = lgb_results
        
        # Generate submission
        lgb_preds = lgb_model.predict(X_test_scaled)
        save_submission(
            lgb_preds,
            test_ids,
            os.path.join(output_dir, "lightgbm_submission.csv"),
            args.round
        )
    except (ImportError, Exception) as e:
        logger.error(f"Error with LightGBM: {e}")
    
    # 2. XGBoost
    try:
        import xgboost
        xgb_model = XGBModel(use_gpu=use_gpu, gpu_id=args.gpu_id)
        xgb_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        models.append(xgb_model)
        model_names.append("xgboost")
        
        # Evaluate
        xgb_results = evaluate_model(xgb_model, X_val_scaled, y_val)
        model_results['xgboost'] = xgb_results
        
        # Generate submission
        xgb_preds = xgb_model.predict(X_test_scaled)
        save_submission(
            xgb_preds,
            test_ids,
            os.path.join(output_dir, "xgboost_submission.csv"),
            args.round
        )
    except (ImportError, Exception) as e:
        logger.error(f"Error with XGBoost: {e}")
    
    # 3. Ensemble (if we have multiple models)
    if len(models) >= 2 and not args.no_ensemble:
        logger.info("Creating ensemble model")
        ensemble = EnsembleModel(models, model_names=model_names)
        ensemble.optimize_weights(X_val_scaled, y_val)
        
        # Evaluate ensemble
        ensemble_results = evaluate_model(ensemble, X_val_scaled, y_val)
        model_results['ensemble'] = ensemble_results
        
        # Generate submission
        ensemble_preds = ensemble.predict(X_test_scaled)
        save_submission(
            ensemble_preds,
            test_ids,
            os.path.join(output_dir, "ensemble_submission.csv"),
            args.round
        )
    
    # Print final summary
    logger.info("\n===== MODEL RESULTS SUMMARY =====")
    logger.info(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    logger.info("-" * 50)
    
    for model_name, results in sorted(model_results.items(), key=lambda x: x[1]['rmse']):
        logger.info(f"{model_name:<15} {results['rmse']:<10.6f} {results['mae']:<10.6f} {results['r2']:<10.6f}")
    
    # Find best model
    if model_results:
        best_model = min(model_results.items(), key=lambda x: x[1]['rmse'])
        logger.info(f"\nBest model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.6f}")
        
        # Check if we achieved the target RMSE
        if best_model[1]['rmse'] <= 0.25:
            logger.info(f"SUCCESS! Target RMSE of 0.25 achieved with {best_model[0]} model.")
        else:
            logger.info(f"Target RMSE of 0.25 not yet achieved. Best RMSE: {best_model[1]['rmse']:.6f}")
    
    logger.info("Model training and evaluation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())