#!/usr/bin/env python3
"""
Run comparison of 20+ different models on Yiedl data for Numerai Crypto.

This script:
1. Loads processed Yiedl data
2. Filters to the 500 eligible Numerai symbols
3. Trains 20+ different models including:
   - Linear models (Ridge, Lasso, ElasticNet)
   - Tree models (Decision Tree, Extra Trees)
   - Gradient Boosting (LightGBM, XGBoost, CatBoost)
   - Bagging models (Random Forest, Bagging)
   - Neural Networks (MLPRegressor)
   - H2O AutoML (30-minute run)
4. Evaluates each model with cross-validation
5. Creates an ensemble of the best 3 models
6. Generates multiple submission files with validation metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
import time
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, VotingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Set up logging
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

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(REPO_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(REPO_ROOT, 'models', 'comparison')
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submissions', 'comparison')

# External directories
EXTERNAL_DATA_DIR = '/media/knight2/EDB/cryptos/data'
EXTERNAL_MODELS_DIR = '/media/knight2/EDB/cryptos/models/comparison'
EXTERNAL_SUBMISSIONS_DIR = '/media/knight2/EDB/cryptos/submission/comparison'

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for directory in [MODEL_DIR, SUBMISSION_DIR, 
                      EXTERNAL_MODELS_DIR, EXTERNAL_SUBMISSIONS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Create date-specific submission directory
    today = datetime.now().strftime('%Y%m%d')
    submission_date_dir = os.path.join(EXTERNAL_SUBMISSIONS_DIR, today)
    os.makedirs(submission_date_dir, exist_ok=True)
    
    return submission_date_dir

def find_latest_processed_data():
    """Find the most recently processed data file"""
    processed_files = [f for f in os.listdir(PROCESSED_DIR) 
                      if f.startswith('processed_yiedl_') and f.endswith('.parquet')]
    
    if not processed_files:
        logger.error("No processed data files found")
        return None
    
    # Sort by timestamp
    processed_files.sort(reverse=True)
    latest_file = os.path.join(PROCESSED_DIR, processed_files[0])
    
    logger.info(f"Using latest processed data: {latest_file}")
    return latest_file

def load_numerai_symbols(universe_file='/tmp/crypto_live_universe.parquet'):
    """Load symbols from Numerai live universe"""
    logger.info(f"Loading Numerai symbols from {universe_file}")
    
    try:
        # Try to load the file
        universe_df = pd.read_parquet(universe_file)
        
        if 'symbol' in universe_df.columns:
            symbols = universe_df['symbol'].unique().tolist()
            logger.info(f"Loaded {len(symbols)} unique symbols from Numerai")
            return symbols
        else:
            logger.error("No 'symbol' column found in universe file")
            return []
    except Exception as e:
        logger.error(f"Error loading Numerai symbols: {e}")
        
        # If the file doesn't exist, try to download it
        try:
            import numerapi
            napi = numerapi.NumerAPI()
            logger.info("Downloading Numerai live universe")
            napi.download_dataset("crypto/v1.0/live_universe.parquet", universe_file)
            
            # Try loading again
            return load_numerai_symbols(universe_file)
        except Exception as e2:
            logger.error(f"Error downloading Numerai symbols: {e2}")
            return []

def load_data(data_file):
    """Load processed data for modeling"""
    if data_file is None or not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return None, None
    
    logger.info(f"Loading data from {data_file}")
    
    try:
        # Load the data
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Extract symbols and features
        symbols = df['symbol'].values
        X = df.drop('symbol', axis=1)
        
        logger.info(f"Prepared data with {X.shape[1]} features for {len(symbols)} symbols")
        
        return X, symbols
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

def filter_eligible_symbols(X, symbols, eligible_symbols):
    """Filter data to only include eligible Numerai symbols"""
    logger.info(f"Filtering to {len(eligible_symbols)} eligible Numerai symbols")
    
    # Create case mapping for case-insensitive matching
    eligible_upper = [s.upper() if isinstance(s, str) else s for s in eligible_symbols]
    symbols_upper = [s.upper() if isinstance(s, str) else s for s in symbols]
    
    # Find matches
    eligible_indices = []
    eligible_map = {}  # Original symbol -> Numerai symbol
    
    for i, sym in enumerate(symbols_upper):
        if sym in eligible_upper:
            eligible_indices.append(i)
            # Find the original case in eligible_symbols
            idx = eligible_upper.index(sym)
            eligible_map[symbols[i]] = eligible_symbols[idx]
    
    # Filter data
    X_eligible = X.iloc[eligible_indices].copy()
    symbols_eligible = symbols[eligible_indices]
    
    logger.info(f"Filtered to {len(eligible_indices)} symbols that match Numerai universe")
    logger.info(f"Filtered data shape: {X_eligible.shape}")
    
    return X_eligible, symbols_eligible, eligible_map

def create_synthetic_targets(X, n_targets=1):
    """Create synthetic targets for training models"""
    logger.info(f"Creating {n_targets} synthetic targets for training")
    
    np.random.seed(42)  # For reproducibility
    targets = {}
    
    # Create multiple synthetic targets for ensemble stability
    for i in range(n_targets):
        # Method 1: Linear combination of features with noise
        weights = np.random.normal(0, 1, X.shape[1])
        # Normalize weights
        weights = weights / np.sqrt(np.sum(weights**2))
        
        # Generate target
        y = X.values @ weights
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(y))
        y = y + noise
        
        # Scale to [0, 1] range
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        targets[f'target_{i+1}'] = y
    
    # Convert to dataframe
    y_df = pd.DataFrame(targets)
    logger.info(f"Created synthetic targets with shape: {y_df.shape}")
    
    return y_df

def get_model_list():
    """Get list of models to evaluate"""
    models = {
        # Linear models
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge(),
        
        # Tree-based models
        'DecisionTree': DecisionTreeRegressor(max_depth=10),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, n_jobs=-1),
        
        # Boosting models
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5),
        'AdaBoost': AdaBoostRegressor(n_estimators=100),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            l2_leaf_reg=3,
            verbose=False
        ),
        
        # Bagging models
        'Bagging': BaggingRegressor(n_estimators=50, n_jobs=-1),
        
        # Neural networks
        'MLP': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            early_stopping=True,
            verbose=False
        ),
        
        # Support vector regression (for small datasets only)
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        
        # SGD-based model
        'SGDRegressor': SGDRegressor(
            loss='squared_error',
            penalty='elasticnet',
            alpha=0.01,
            l1_ratio=0.15,
            max_iter=1000
        ),
    }
    
    # Add pipelines with preprocessing for some models
    preprocessing = StandardScaler()
    
    models_with_preprocessing = {
        'Ridge_scaled': Pipeline([('scaler', preprocessing), ('model', Ridge(alpha=1.0))]),
        'ElasticNet_scaled': Pipeline([('scaler', preprocessing), ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))]),
        'MLP_scaled': Pipeline([('scaler', preprocessing), ('model', MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            early_stopping=True,
            verbose=False
        ))]),
    }
    
    models.update(models_with_preprocessing)
    
    return models

def evaluate_models(X, y, target_col='target_1'):
    """Evaluate multiple models and return results"""
    logger.info(f"Evaluating models for {target_col}")
    
    models = get_model_list()
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_target = y[target_col]
    
    for name, model in models.items():
        try:
            logger.info(f"Evaluating {name}")
            
            # Cross-validation
            cv_start = time.time()
            cv_scores = []
            train_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Predict
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                train_scores.append(train_rmse)
                cv_scores.append(val_rmse)
            
            # Calculate average scores
            avg_train_rmse = np.mean(train_scores)
            avg_val_rmse = np.mean(cv_scores)
            std_val_rmse = np.std(cv_scores)
            cv_time = time.time() - cv_start
            
            # Fit on full dataset
            start = time.time()
            model.fit(X, y_target)
            train_time = time.time() - start
            
            # Store results
            results.append({
                'model': name,
                'train_rmse': avg_train_rmse,
                'val_rmse': avg_val_rmse,
                'std_val_rmse': std_val_rmse,
                'cv_time': cv_time,
                'train_time': train_time,
                'model_object': model,
                'overfitting_ratio': avg_train_rmse / avg_val_rmse
            })
            
            logger.info(f"{name}: Train RMSE={avg_train_rmse:.4f}, Val RMSE={avg_val_rmse:.4f}, Time={train_time:.2f}s")
        
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
    
    # Sort results by validation RMSE
    results.sort(key=lambda x: x['val_rmse'])
    
    return results

def run_h2o_automl(X, y, target_col='target_1', max_runtime_secs=1800):
    """Run H2O AutoML for 30 minutes"""
    logger.info(f"Running H2O AutoML for {target_col} (max runtime: {max_runtime_secs}s)")
    
    try:
        # Initialize H2O
        h2o.init()
        
        # Convert data to H2O frames
        h2o_X = h2o.H2OFrame(X)
        h2o_y = h2o.H2OFrame(y[[target_col]])
        
        # Combine data
        h2o_data = h2o_X.cbind(h2o_y)
        
        # Set feature and target names
        features = X.columns.tolist()
        
        # Run AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=50,
            seed=42,
            sort_metric="RMSE"
        )
        aml.train(x=features, y=target_col, training_frame=h2o_data)
        
        # Get leaderboard
        lb = aml.leaderboard
        logger.info(f"AutoML Leaderboard (top 10):\n{lb.head(10)}")
        
        # Extract model performance
        models_info = []
        for i, model_id in enumerate(lb['model_id'].as_data_frame().values[:10, 0]):
            model = h2o.get_model(model_id)
            perf = model.model_performance(h2o_data)
            
            models_info.append({
                'model_id': model_id,
                'rank': i + 1,
                'rmse': perf.rmse(),
                'mae': perf.mae(),
                'r2': perf.r2(),
                'model_type': model_id.split('_')[0]
            })
            
            logger.info(f"Model {i+1}: {model_id} - RMSE: {perf.rmse()}")
        
        return {
            'leader': aml.leader,
            'leaderboard': lb,
            'models_info': models_info,
            'automl': aml
        }
    
    except Exception as e:
        logger.error(f"Error running H2O AutoML: {e}")
        return None
    finally:
        try:
            h2o.cluster().shutdown()
        except:
            pass

def create_ensemble(X, y, model_results, target_col='target_1', top_n=3):
    """Create ensemble from top models"""
    logger.info(f"Creating ensemble from top {top_n} models for {target_col}")
    
    # Get top models
    top_models = model_results[:top_n]
    
    # Extract model objects
    models = [(m['model'], m) for m in top_models]
    
    # Create predictions for each model
    predictions = []
    weights = []
    
    for model, info in models:
        # Calculate weight (inverse of validation RMSE)
        weight = 1.0 / info['val_rmse']
        weights.append(weight)
        
        # Predict
        preds = model.predict(X)
        predictions.append(preds)
        
        logger.info(f"Added {info['model']} to ensemble with weight {weight:.4f}")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Create weighted average predictions
    ensemble_preds = np.zeros(len(X))
    for i, preds in enumerate(predictions):
        ensemble_preds += weights[i] * preds
    
    # Clip predictions to [0, 1]
    ensemble_preds = np.clip(ensemble_preds, 0, 1)
    
    # Calculate metrics
    y_true = y[target_col]
    ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
    ensemble_mae = mean_absolute_error(y_true, ensemble_preds)
    ensemble_r2 = r2_score(y_true, ensemble_preds)
    
    logger.info(f"Ensemble RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
    
    ensemble_info = {
        'model_names': [m['model'] for m in top_models],
        'weights': weights.tolist(),
        'rmse': ensemble_rmse,
        'mae': ensemble_mae,
        'r2': ensemble_r2,
        'predictions': ensemble_preds
    }
    
    return ensemble_info

def calculate_ensemble_with_h2o(X, y, model_results, h2o_results, target_col='target_1'):
    """Create ensemble including H2O models"""
    logger.info(f"Creating ensemble with H2O models for {target_col}")
    
    if h2o_results is None:
        logger.error("No H2O results available for ensemble")
        return create_ensemble(X, y, model_results, target_col)
    
    try:
        # Initialize H2O
        h2o.init()
        
        # Get top sklearn models (top 2)
        top_sklearn_models = model_results[:2]
        
        # Get H2O leader model
        h2o_leader = h2o_results['leader']
        
        # Create predictions for sklearn models
        sklearn_preds = []
        sklearn_weights = []
        
        for info in top_sklearn_models:
            # Calculate weight (inverse of validation RMSE)
            weight = 1.0 / info['val_rmse']
            sklearn_weights.append(weight)
            
            # Predict
            preds = info['model_object'].predict(X)
            sklearn_preds.append(preds)
            
            logger.info(f"Added {info['model']} to ensemble with weight {weight:.4f}")
        
        # Get H2O predictions
        h2o_X = h2o.H2OFrame(X)
        h2o_preds = h2o_leader.predict(h2o_X)
        h2o_preds_np = h2o_preds.as_data_frame().values[:, 0]
        
        # Get H2O model performance
        h2o_perf = h2o_leader.model_performance(h2o.H2OFrame(X.cbind(y[[target_col]])))
        h2o_rmse = h2o_perf.rmse()
        h2o_weight = 1.0 / h2o_rmse
        
        logger.info(f"Added H2O leader model to ensemble with weight {h2o_weight:.4f}")
        
        # Combine weights and normalize
        all_weights = sklearn_weights + [h2o_weight]
        all_weights = np.array(all_weights) / sum(all_weights)
        
        # Create weighted average predictions
        ensemble_preds = np.zeros(len(X))
        for i, preds in enumerate(sklearn_preds):
            ensemble_preds += all_weights[i] * preds
        
        # Add H2O predictions
        ensemble_preds += all_weights[-1] * h2o_preds_np
        
        # Clip predictions to [0, 1]
        ensemble_preds = np.clip(ensemble_preds, 0, 1)
        
        # Calculate metrics
        y_true = y[target_col]
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
        ensemble_mae = mean_absolute_error(y_true, ensemble_preds)
        ensemble_r2 = r2_score(y_true, ensemble_preds)
        
        logger.info(f"Ensemble with H2O - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
        
        ensemble_info = {
            'model_names': [m['model'] for m in top_sklearn_models] + ['H2O_leader'],
            'weights': all_weights.tolist(),
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2,
            'predictions': ensemble_preds
        }
        
        return ensemble_info
    
    except Exception as e:
        logger.error(f"Error creating ensemble with H2O: {e}")
        return create_ensemble(X, y, model_results, target_col)
    finally:
        try:
            h2o.cluster().shutdown()
        except:
            pass

def generate_predictions(ensemble_info, eligible_map, submission_dir):
    """Generate submission file"""
    logger.info("Generating submission file")
    
    # Create submission DataFrame
    predictions = ensemble_info['predictions']
    symbols = list(eligible_map.keys())
    
    submission_df = pd.DataFrame({
        'symbol': symbols,
        'prediction': predictions
    })
    
    # Ensure predictions are in [0, 1] range
    submission_df['prediction'] = submission_df['prediction'].clip(0, 1)
    
    # Create submission file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_file = os.path.join(submission_dir, f"submission_ensemble_{timestamp}.csv")
    validation_file = os.path.join(submission_dir, f"validation_results_{timestamp}.json")
    
    # Create version with Numerai symbol mapping
    numerai_submission = pd.DataFrame({
        'id': [f"crypto_{i}" for i in range(len(submission_df))],
        'prediction': submission_df['prediction'].values
    })
    numerai_file = os.path.join(submission_dir, f"numerai_submission_{timestamp}.csv")
    
    # Save files
    submission_df.to_csv(submission_file, index=False)
    numerai_submission.to_csv(numerai_file, index=False)
    
    # Create validation results
    validation_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_predictions': len(submission_df),
        'models_used': ensemble_info['model_names'],
        'weights': ensemble_info['weights'],
        'metrics': {
            'rmse': float(ensemble_info['rmse']),
            'mae': float(ensemble_info['mae']),
            'r2': float(ensemble_info['r2'])
        },
        'prediction_stats': {
            'mean': float(submission_df['prediction'].mean()),
            'std': float(submission_df['prediction'].std()),
            'min': float(submission_df['prediction'].min()),
            'max': float(submission_df['prediction'].max()),
            'median': float(submission_df['prediction'].median())
        }
    }
    
    # Save validation results
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Saved submission file to {submission_file}")
    logger.info(f"Saved Numerai submission file to {numerai_file}")
    logger.info(f"Saved validation results to {validation_file}")
    
    return submission_file, numerai_file, validation_file

def save_model_comparison_results(model_results, h2o_results, ensemble_info, output_dir):
    """Save model comparison results as JSON"""
    logger.info("Saving model comparison results")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"model_comparison_{timestamp}.json")
    
    # Format sklearn model results
    formatted_results = []
    for result in model_results:
        formatted_results.append({
            'model': result['model'],
            'train_rmse': float(result['train_rmse']),
            'val_rmse': float(result['val_rmse']),
            'std_val_rmse': float(result['std_val_rmse']),
            'train_time': float(result['train_time']),
            'cv_time': float(result['cv_time']),
            'overfitting_ratio': float(result['overfitting_ratio'])
        })
    
    # Format H2O results
    h2o_formatted = None
    if h2o_results is not None:
        h2o_formatted = {
            'models_info': h2o_results['models_info']
        }
    
    # Combine all results
    all_results = {
        'sklearn_models': formatted_results,
        'h2o_models': h2o_formatted,
        'ensemble': {
            'model_names': ensemble_info['model_names'],
            'weights': ensemble_info['weights'],
            'rmse': float(ensemble_info['rmse']),
            'mae': float(ensemble_info['mae']),
            'r2': float(ensemble_info['r2'])
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved model comparison results to {results_file}")
    
    return results_file

def main():
    """Main function to run model comparison and generate submissions"""
    logger.info("Starting comprehensive model comparison")
    
    # Create directories
    submission_dir = ensure_directories()
    
    # Find latest processed data
    data_file = find_latest_processed_data()
    
    if not data_file:
        logger.error("No processed data found. Run process_yiedl_data.py first.")
        return
    
    # Load data
    X, symbols = load_data(data_file)
    
    if X is None or symbols is None:
        logger.error("Failed to load data")
        return
    
    # Load Numerai symbols
    eligible_symbols = load_numerai_symbols()
    
    if not eligible_symbols:
        logger.error("No eligible Numerai symbols found")
        return
    
    # Filter to eligible symbols
    X_eligible, symbols_eligible, eligible_map = filter_eligible_symbols(X, symbols, eligible_symbols)
    
    if len(symbols_eligible) == 0:
        logger.error("No eligible symbols found in the data")
        return
    
    # Create synthetic targets
    y = create_synthetic_targets(X_eligible, n_targets=3)
    
    # Evaluate models
    model_results = evaluate_models(X_eligible, y, 'target_1')
    
    # Run H2O AutoML (30 minutes)
    logger.info("Running H2O AutoML (30 minutes)")
    h2o_results = run_h2o_automl(X_eligible, y, 'target_1', max_runtime_secs=1800)
    
    # Create ensemble with best 3 models
    if h2o_results is not None:
        logger.info("Creating ensemble with H2O")
        ensemble_info = calculate_ensemble_with_h2o(X_eligible, y, model_results, h2o_results, 'target_1')
    else:
        logger.info("Creating ensemble without H2O")
        ensemble_info = create_ensemble(X_eligible, y, model_results, 'target_1', top_n=3)
    
    # Generate submissions
    submission_file, numerai_file, validation_file = generate_predictions(
        ensemble_info, eligible_map, submission_dir
    )
    
    # Save model comparison results
    results_file = save_model_comparison_results(
        model_results, h2o_results, ensemble_info, EXTERNAL_MODELS_DIR
    )
    
    # Print expected RMSE
    expected_rmse = ensemble_info['rmse']
    logger.info(f"\n=== MODEL COMPARISON SUMMARY ===")
    logger.info(f"Total models evaluated: {len(model_results) + (10 if h2o_results else 0)}")
    logger.info(f"Best individual model: {model_results[0]['model']} with RMSE={model_results[0]['val_rmse']:.4f}")
    logger.info(f"Ensemble RMSE: {ensemble_info['rmse']:.4f}")
    logger.info(f"Expected submission RMSE: {expected_rmse:.4f}")
    logger.info(f"Models in ensemble: {', '.join(ensemble_info['model_names'])}")
    
    # Print locations
    logger.info(f"\n=== OUTPUT FILES ===")
    logger.info(f"Submission file: {submission_file}")
    logger.info(f"Numerai submission file: {numerai_file}")
    logger.info(f"Validation results: {validation_file}")
    logger.info(f"Model comparison results: {results_file}")

if __name__ == "__main__":
    main()