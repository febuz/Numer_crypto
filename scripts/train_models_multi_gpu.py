#!/usr/bin/env python3
"""
Multi-GPU Model Training for Numerai Crypto

This script distributes model training across all available GPUs for maximum performance.
Each GPU trains a different model variant simultaneously.
"""

import os
import sys
import time
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, date
import glob

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Configure GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "4"

# Default paths
GPU_FEATURES_FILE = "/media/knight2/EDB/numer_crypto_temp/data/features/gpu_features.parquet"
MODELS_DIR = "/media/knight2/EDB/numer_crypto_temp/models"
PREDICTIONS_DIR = "/media/knight2/EDB/numer_crypto_temp/predictions"

def check_models_exist_today(model_type, models_dir=MODELS_DIR):
    """Check if models of the specified type already exist from today"""
    today_str = date.today().strftime("%Y%m%d")
    
    # Pattern to match model files created today
    if model_type == 'all':
        # Check for any model type
        patterns = [
            f"*lightgbm*gpu*{today_str}*.pkl",
            f"*xgboost*gpu*{today_str}*.pkl", 
            f"*simple*gpu*{today_str}*.pkl",
        ]
    else:
        # Check for specific model type
        patterns = [f"*{model_type}*gpu*{today_str}*.pkl"]
    
    existing_models = []
    for pattern in patterns:
        search_path = os.path.join(models_dir, pattern)
        matching_files = glob.glob(search_path)
        existing_models.extend(matching_files)
    
    if existing_models:
        logger.info(f"Found {len(existing_models)} existing multi-GPU models from today for type '{model_type}':")
        for model in existing_models:
            logger.info(f"  - {os.path.basename(model)}")
        return True
    else:
        logger.info(f"No existing multi-GPU models from today found for type '{model_type}'")
        return False

def load_and_prepare_data():
    """Load and prepare data for training"""
    import polars as pl
    
    if not os.path.exists(GPU_FEATURES_FILE):
        logger.error(f"GPU features file not found: {GPU_FEATURES_FILE}")
        return None, None
    
    logger.info(f"Loading data from {GPU_FEATURES_FILE}")
    df = pl.read_parquet(GPU_FEATURES_FILE)
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Prepare features and target
    if 'target' not in df.columns:
        logger.error("Target column 'target' not found in data")
        return None, None
    
    # Get numeric columns and exclude non-feature columns
    excluded_cols = ['target', 'Symbol', 'symbol', 'Prediction', 'prediction', 'date', 'era', 'id', 'asset']
    
    numeric_cols = []
    for col in df.columns:
        if col not in excluded_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            numeric_cols.append(col)
    
    if not numeric_cols:
        logger.error("No numeric feature columns found")
        return None, None
    
    # Prepare features
    X_df = df.select(numeric_cols).fill_null(0)
    y_series = df.select('target').fill_null(0)['target']
    
    # Convert to numpy arrays
    X = X_df.to_numpy().astype(np.float32)
    y = y_series.to_numpy().astype(np.float32)
    
    logger.info(f"Prepared data with {X.shape[1]} features")
    return X, y

def train_lightgbm_worker(gpu_id, X_train, y_train, model_name, output_dir):
    """Train LightGBM model on specific GPU"""
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"Worker {gpu_id} training LightGBM model: {model_name}")
    
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # GPU-optimized parameters with variation per GPU
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 10,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': 42 + gpu_id,
            'force_row_wise': True,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0  # Will be 0 since we set CUDA_VISIBLE_DEVICES
        }
        
        # Vary parameters by GPU to create model diversity
        if gpu_id == 0:
            base_params.update({
                'num_leaves': 128,
                'max_depth': 8,
                'min_gain_to_split': 0.02
            })
        elif gpu_id == 1:
            base_params.update({
                'num_leaves': 64,
                'max_depth': 10,
                'min_gain_to_split': 0.01
            })
        else:  # gpu_id == 2
            base_params.update({
                'num_leaves': 256,
                'max_depth': 6,
                'min_gain_to_split': 0.03
            })
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with {base_params['num_leaves']} leaves, depth {base_params['max_depth']}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train_split, label=y_train_split)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        start_time = time.time()
        
        # Train model
        model = lgb.train(
            base_params,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200)
            ]
        )
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed LightGBM training in {training_time:.2f} seconds")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        # Get training metrics
        best_score = model.best_score['valid']['rmse']
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
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def train_xgboost_worker(gpu_id, X_train, y_train, model_name, output_dir):
    """Train XGBoost model on specific GPU"""
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"Worker {gpu_id} training XGBoost model: {model_name}")
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # GPU-optimized parameters with variation per GPU (updated for XGBoost 2.0+)
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'device': 'cuda',  # Use new device parameter instead of gpu_hist
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42 + gpu_id,
            'n_jobs': 4
        }
        
        # Vary parameters by GPU
        if gpu_id == 0:
            base_params.update({
                'max_depth': 6,
                'min_child_weight': 3,
                'gamma': 0.1
            })
        elif gpu_id == 1:
            base_params.update({
                'max_depth': 8,
                'min_child_weight': 1,
                'gamma': 0.05
            })
        else:  # gpu_id == 2
            base_params.update({
                'max_depth': 4,
                'min_child_weight': 5,
                'gamma': 0.2
            })
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with depth {base_params['max_depth']}, min_child_weight {base_params['min_child_weight']}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up evaluation
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        start_time = time.time()
        
        # Train model
        model = xgb.train(
            base_params,
            dtrain,
            num_boost_round=1500,
            evals=evallist,
            early_stopping_rounds=100,
            verbose_eval=200
        )
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed XGBoost training in {training_time:.2f} seconds")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        # Get training metrics
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
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def train_simple_worker(gpu_id, X_train, y_train, model_name, output_dir):
    """Train simple GPU-accelerated model using CuML"""
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"Worker {gpu_id} training simple GPU model: {model_name}")
    
    try:
        # Try CuML first (GPU-accelerated), fallback to sklearn
        try:
            import cuml
            from cuml.ensemble import RandomForestRegressor as CuMLRandomForest
            from sklearn.model_selection import train_test_split
            use_gpu = True
            logger.info(f"GPU {gpu_id} using CuML GPU-accelerated Random Forest")
        except ImportError:
            logger.warning(f"GPU {gpu_id} CuML not available, falling back to sklearn")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            use_gpu = False
        
        # Split data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 + gpu_id
        )
        
        # Train parameters with variation
        if use_gpu:
            # CuML Random Forest parameters
            base_params = {
                'random_state': 42 + gpu_id,
                'split_criterion': 'mse',
                'bootstrap': True
            }
            
            if gpu_id == 0:
                base_params.update({
                    'n_estimators': 100,
                    'max_depth': 10,
                    'max_features': 0.8
                })
            elif gpu_id == 1:
                base_params.update({
                    'n_estimators': 150,
                    'max_depth': 8,
                    'max_features': 0.6
                })
            else:
                base_params.update({
                    'n_estimators': 200,
                    'max_depth': 12,
                    'max_features': 0.9
                })
            
            model = CuMLRandomForest(**base_params)
        else:
            # Sklearn parameters
            base_params = {
                'random_state': 42 + gpu_id,
                'n_jobs': 4
            }
            
            if gpu_id == 0:
                base_params.update({
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                })
            elif gpu_id == 1:
                base_params.update({
                    'n_estimators': 150,
                    'max_depth': 8,
                    'min_samples_split': 3
                })
            else:
                base_params.update({
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 7
                })
            
            model = RandomForestRegressor(**base_params)
        
        logger.info(f"GPU {gpu_id} using device: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"GPU {gpu_id} training with {base_params['n_estimators']} estimators, depth {base_params['max_depth']}")
        
        start_time = time.time()
        
        # Train model
        model.fit(X_train_split, y_train_split)
        
        training_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} completed {'GPU' if use_gpu else 'CPU'} simple model training in {training_time:.2f} seconds")
        
        # Evaluate
        if use_gpu:
            # CuML prediction
            val_pred = model.predict(X_val)
            # Calculate R² score manually for CuML
            from sklearn.metrics import r2_score
            val_score = r2_score(y_val, val_pred.get() if hasattr(val_pred, 'get') else val_pred)
        else:
            val_score = model.score(X_val, y_val)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_gpu{gpu_id}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"GPU {gpu_id} saved model to {model_path}")
        
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'model_path': model_path,
            'training_time': training_time,
            'val_score': val_score,
            'use_gpu': use_gpu,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"GPU {gpu_id} simple model training failed: {e}")
        return {
            'gpu_id': gpu_id,
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Model Training for Numerai Crypto')
    parser.add_argument('--model-type', type=str, default='all', 
                       choices=['lightgbm', 'xgboost', 'simple', 'all'],
                       help='Model type to train')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--output-dir', type=str, default=MODELS_DIR,
                       help='Output directory for models')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force model retraining even if models exist from today')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Check if models already exist from today (unless force-retrain is specified)
    if not args.force_retrain:
        if check_models_exist_today(args.model_type, args.output_dir):
            logger.info(f"Multi-GPU models for '{args.model_type}' already exist from today.")
            logger.info("Skipping training. Use --force-retrain to retrain existing models.")
            return True
    else:
        logger.info("Force retrain specified - will train multi-GPU models even if they exist from today")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, y_train = load_and_prepare_data()
    
    if X_train is None or y_train is None:
        logger.error("Failed to load data")
        return False
    
    logger.info(f"Training with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # Train models based on type
    if args.model_type in ['lightgbm', 'all']:
        logger.info("Starting multi-GPU LightGBM training...")
        
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_ids):
                future = executor.submit(
                    train_lightgbm_worker,
                    gpu_id,
                    X_train,
                    y_train,
                    f"lightgbm",
                    args.output_dir
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ LightGBM GPU {result['gpu_id']}: {result['training_time']:.2f}s, RMSE: {result['best_score']:.6f}")
                else:
                    logger.error(f"❌ LightGBM GPU {result['gpu_id']}: {result['error']}")
    
    if args.model_type in ['xgboost', 'all']:
        logger.info("Starting multi-GPU XGBoost training...")
        
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_ids):
                future = executor.submit(
                    train_xgboost_worker,
                    gpu_id,
                    X_train,
                    y_train,
                    f"xgboost",
                    args.output_dir
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    logger.info(f"✅ XGBoost GPU {result['gpu_id']}: {result['training_time']:.2f}s, RMSE: {result['best_score']:.6f}")
                else:
                    logger.error(f"❌ XGBoost GPU {result['gpu_id']}: {result['error']}")
    
    if args.model_type in ['simple', 'all']:
        logger.info("Starting multi-worker simple model training...")
        
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_ids):
                future = executor.submit(
                    train_simple_worker,
                    gpu_id,
                    X_train,
                    y_train,
                    f"simple",
                    args.output_dir
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                all_results.append(result)
                if result['status'] == 'success':
                    gpu_type = "GPU" if result.get('use_gpu', False) else "CPU"
                    logger.info(f"✅ Simple {gpu_type} {result['gpu_id']}: {result['training_time']:.2f}s, Score: {result['val_score']:.6f}")
                else:
                    logger.error(f"❌ Simple Worker {result['gpu_id']}: {result['error']}")
    
    # Summary
    successful_models = [r for r in all_results if r['status'] == 'success']
    failed_models = [r for r in all_results if r['status'] == 'failed']
    
    logger.info(f"\n=== Training Summary ===")
    logger.info(f"✅ Successful models: {len(successful_models)}")
    logger.info(f"❌ Failed models: {len(failed_models)}")
    
    if successful_models:
        total_time = max([r['training_time'] for r in successful_models])
        avg_time = sum([r['training_time'] for r in successful_models]) / len(successful_models)
        logger.info(f"🕒 Total training time: {total_time:.2f}s (parallel)")
        logger.info(f"🕒 Average training time: {avg_time:.2f}s per model")
        
        logger.info("\nSuccessful models:")
        for result in successful_models:
            logger.info(f"  {result['model_name']}_gpu{result['gpu_id']}: {result['model_path']}")
    
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)