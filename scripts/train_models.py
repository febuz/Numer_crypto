#!/usr/bin/env python3
"""
train_models.py - Train models for Numerai Crypto

This script trains machine learning models for Numerai Crypto predictions.
"""
import os
import sys
import logging
import argparse
import polars as pl
import numpy as np
import pickle
from datetime import datetime, date
import glob

# CRITICAL: Configure temp directories immediately to prevent / disk usage
H2O_TEMP_DIR = '/media/knight2/EDB/tmp/h2o'
os.makedirs(H2O_TEMP_DIR, mode=0o755, exist_ok=True)

# Set environment variables before any imports that might use temp space
os.environ['TMPDIR'] = H2O_TEMP_DIR
os.environ['TMP'] = H2O_TEMP_DIR  
os.environ['TEMP'] = H2O_TEMP_DIR
os.environ['JAVA_OPTS'] = f'-Djava.io.tmpdir={H2O_TEMP_DIR}'
os.environ['_JAVA_OPTIONS'] = f'-Djava.io.tmpdir={H2O_TEMP_DIR}'

# Optional import for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
MODELS_DIR = "/media/knight2/EDB/numer_crypto_temp/models"
GPU_FEATURES_FILE = "/media/knight2/EDB/numer_crypto_temp/data/features/gpu_features.parquet"

def check_models_exist_today(model_type, models_dir=MODELS_DIR):
    """Check if models of the specified type already exist from today"""
    today_str = date.today().strftime("%Y%m%d")
    
    # Pattern to match model files created today
    if model_type == 'all':
        # Check for any model type
        patterns = [
            f"*simple*{today_str}*.pkl",
            f"*lightgbm*{today_str}*.pkl", 
            f"*xgboost*{today_str}*.pkl",
            f"*h2o*{today_str}*",
        ]
    else:
        # Check for specific model type
        patterns = [f"*{model_type}*{today_str}*.pkl"]
        if model_type == 'h2o':
            patterns.extend([f"*h2o*{today_str}*"])
    
    existing_models = []
    for pattern in patterns:
        search_path = os.path.join(models_dir, pattern)
        matching_files = glob.glob(search_path)
        existing_models.extend(matching_files)
    
    if existing_models:
        logger.info(f"Found {len(existing_models)} existing models from today for type '{model_type}':")
        for model in existing_models:
            logger.info(f"  - {os.path.basename(model)}")
        return True
    else:
        logger.info(f"No existing models from today found for type '{model_type}'")
        return False

def train_simple_model(X_train, y_train, use_gpu=False, parallel=False):
    """Train a simple model for demonstration purposes"""
    logger.info(f"Training simple model with {X_train.shape[1]} features")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Train parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Add parallel processing if requested
        if parallel:
            params['n_jobs'] = -1
            logger.info("Using parallel processing")
        
        # Create and train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def prepare_data(file_path):
    """Prepare data for model training using Polars"""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None, None
    
    # Only support Parquet files for performance
    if not file_path.endswith('.parquet'):
        logger.error(f"Only Parquet files are supported. Got: {file_path}")
        return None, None
    
    # Load data using Polars for better performance
    df = pl.read_parquet(file_path)
    logger.info(f"Loaded data from {file_path} with shape {df.shape}")
    
    # Prepare features and target
    if 'target' not in df.columns:
        logger.error("Target column 'target' not found in data")
        return None, None
    
    # Get numeric columns and exclude non-feature columns
    excluded_cols = ['target', 'Symbol', 'symbol', 'Prediction', 'prediction', 'date', 'era', 'id', 'asset']
    
    # Select all numeric columns except excluded ones
    numeric_cols = []
    for col in df.columns:
        if col not in excluded_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            numeric_cols.append(col)
    
    if not numeric_cols:
        logger.error("No numeric feature columns found")
        return None, None
    
    # Prepare features using Polars operations
    X_df = df.select(numeric_cols).fill_null(0)  # Handle missing values
    y_series = df.select('target').fill_null(0)['target']
    
    # Convert to numpy arrays for model training compatibility
    X = X_df.to_numpy().astype(np.float32)
    y = y_series.to_numpy().astype(np.float32)
    
    logger.info(f"Prepared data with {X.shape[1]} features using Polars")
    return X, y

def train_lightgbm_model(X_train, y_train, use_gpu=False, parallel=False):
    """Train an optimized LightGBM model"""
    logger.info(f"Training LightGBM model with {X_train.shape[1]} features")
    
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # Split data for validation to enable early stopping
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=None
        )
        
        # Optimized parameters for financial time series
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 128,  # Increased for better model complexity
            'learning_rate': 0.02,  # Lower for better convergence
            'feature_fraction': 0.8,  # Reduced for better generalization
            'bagging_fraction': 0.7,  # Reduced for better generalization
            'bagging_freq': 10,  # Increased frequency
            'min_data_in_leaf': 100,  # Prevent overfitting
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
            'max_depth': 8,  # Limit tree depth
            'min_gain_to_split': 0.02,  # Minimum gain to split
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True  # Better for wide datasets
        }
        
        # Add GPU support if requested
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            logger.info("Using GPU for LightGBM training")
        
        # Add parallel processing if requested  
        if parallel:
            params['num_threads'] = -1
            logger.info("Using parallel processing for LightGBM")
        
        # Create datasets with validation split
        train_data = lgb.Dataset(X_train_split, label=y_train_split)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with validation and early stopping
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=2000,  # Increased rounds
            valid_sets=[train_data, valid_data], 
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(100),  # Increased patience
                lgb.log_evaluation(100)   # Log every 100 rounds
            ]
        )
        
        logger.info(f"LightGBM training completed with {model.num_trees()} trees")
        logger.info(f"Best validation RMSE: {model.best_score['valid']['rmse']:.6f}")
        
        logger.info("LightGBM model training completed")
        return model
        
    except ImportError:
        logger.error("LightGBM not installed. Please install with: pip install lightgbm")
        return None
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        return None

def train_xgboost_model(X_train, y_train, use_gpu=False, parallel=False):
    """Train an optimized XGBoost model"""
    logger.info(f"Training XGBoost model with {X_train.shape[1]} features")
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        # Split data for validation to enable early stopping
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Optimized parameters for financial time series
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.02,  # Lower for better convergence
            'max_depth': 8,  # Increased for better model complexity
            'subsample': 0.7,  # Reduced for better generalization
            'colsample_bytree': 0.7,  # Reduced for better generalization
            'colsample_bylevel': 0.8,  # Additional regularization
            'alpha': 0.1,  # L1 regularization
            'lambda': 0.1,  # L2 regularization
            'min_child_weight': 10,  # Prevent overfitting
            'gamma': 0.1,  # Minimum loss reduction
            'random_state': 42,
            'verbosity': 0  # Reduce output
        }
        
        # Add GPU support if requested (fix deprecation warnings)
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'  # New way to specify GPU
            logger.info("Using GPU for XGBoost training (CUDA)")
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'
        
        # Add parallel processing if requested
        if parallel:
            params['n_jobs'] = -1
            logger.info("Using parallel processing for XGBoost")
        
        # Create DMatrix with validation split
        dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with validation and early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=2000,  # Increased rounds
            evals=evals, 
            early_stopping_rounds=100,  # Increased patience
            verbose_eval=100  # Log every 100 rounds
        )
        
        logger.info(f"XGBoost training completed with {model.num_boosted_rounds()} rounds")
        logger.info(f"Best validation RMSE: {model.best_score:.6f}")
        
        logger.info("XGBoost model training completed")
        return model
        
    except ImportError:
        logger.error("XGBoost not installed. Please install with: pip install xgboost")
        return None
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        return None

def optimize_lightgbm_hyperparams(X_train, y_train, use_gpu=False, n_trials=30):
    """Optimize LightGBM hyperparameters using Optuna"""
    logger.info(f"Optimizing LightGBM hyperparameters with {n_trials} trials")
    
    try:
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping hyperparameter optimization.")
            return None
            
        import lightgbm as lgb
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 64, 256),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 5, 15),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.3),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.3),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'verbose': -1,
                'random_state': 42,
                'force_row_wise': True
            }
            
            if use_gpu:
                params['device'] = 'gpu'
            
            # Use cross-validation for robust evaluation
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            rmse_scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=300,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        # Run optimization with reduced verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Best RMSE: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
        
    except ImportError:
        logger.warning("Optuna not available. Using default parameters.")
        return None
    except Exception as e:
        logger.error(f"Error optimizing LightGBM hyperparameters: {e}")
        return None

def train_h2o_model(X_train, y_train, use_gpu=False, parallel=False, h2o_time_limit=300):
    """Train an H2O Sparkling Water model with GPU acceleration"""
    logger.info(f"Training H2O Sparkling Water model with {X_train.shape[1]} features")
    
    # Set environment variables to force H2O and Java to use NVMe disk
    os.environ['TMPDIR'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['TMP'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['TEMP'] = '/media/knight2/EDB/tmp/h2o'
    os.environ['JAVA_OPTS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
    os.environ['_JAVA_OPTIONS'] = '-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o'
    
    # Ensure the temp directory exists
    os.makedirs('/media/knight2/EDB/tmp/h2o', mode=0o755, exist_ok=True)
    logger.info("Set H2O/Java temp directories to NVMe disk")
    
    try:
        # Import H2O and check for Sparkling Water
        import h2o
        from h2o.automl import H2OAutoML
        
        try:
            import pysparkling
            from pysparkling import H2OContext
            from pyspark.sql import SparkSession
            import pyspark
            logger.info("H2O Sparkling Water available - enabling GPU acceleration")
            sparkling_water = True
        except ImportError as e:
            logger.info(f"H2O Sparkling Water not available ({e}), using regular H2O")
            sparkling_water = False
        
        # Initialize Spark session with GPU support if using Sparkling Water
        if sparkling_water and use_gpu:
            spark = SparkSession.builder \
                .appName("H2O-Sparkling-Water-GPU") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            # Initialize H2O context
            hc = H2OContext.getOrCreate()
            logger.info("H2O Sparkling Water context initialized with GPU support")
        else:
            # Initialize regular H2O cluster
            cluster = h2o.cluster()
            if cluster is None or not cluster.is_running():
                init_params = {
                    "nthreads": -1 if parallel else 1,
                    "max_mem_size": "8G",
                    "verbose": False,
                    "ice_root": "/media/knight2/EDB/tmp/h2o",
                    "jvm_custom_args": ["-Djava.io.tmpdir=/media/knight2/EDB/tmp/h2o"]
                }
                if use_gpu:
                    init_params["enable_assertions"] = False
                h2o.init(**init_params)
                logger.info("H2O cluster initialized")
            else:
                logger.info("H2O cluster already running")
        
        # Convert numpy arrays to H2O frame
        logger.info(f"Converting {X_train.shape} features and {y_train.shape} targets to H2O frame")
        logger.info(f"Using temp directory: {os.environ.get('TMPDIR', '/tmp')} for H2O operations")
        
        # Ensure H2O uses correct temp directory for this operation
        h2o_temp_dir = '/media/knight2/EDB/tmp/h2o'
        os.environ['TMPDIR'] = h2o_temp_dir
        os.environ['TMP'] = h2o_temp_dir
        os.environ['TEMP'] = h2o_temp_dir
        
        # Create a combined array with features and target
        combined_data = np.column_stack([X_train, y_train])
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        column_names = feature_names + ['target']
        
        # Create H2O frame with proper column types
        # This is the critical operation that was using / disk
        logger.info("Creating H2O frame - this should use NVMe disk for temp files")
        h2o_frame = h2o.H2OFrame(combined_data, column_names=column_names)
        logger.info(f"Created H2O frame with shape: {h2o_frame.shape}")
        
        # Ensure target is treated as numeric for regression
        h2o_frame['target'] = h2o_frame['target'].asnumeric()
        
        # Set target column
        target = 'target'
        features = [col for col in h2o_frame.columns if col != target]
        logger.info(f"Using {len(features)} features for H2O AutoML")
        
        # Configure AutoML with proper time limit and GPU algorithms
        automl_params = {
            "max_models": 10,
            "seed": 42, 
            "max_runtime_secs": h2o_time_limit,
            "sort_metric": "RMSE",
            "verbosity": "info"
        }
        
        if use_gpu:
            # Include GPU-accelerated algorithms
            automl_params["include_algos"] = ["GBM", "XGBoost", "GLM", "DRF"]
            logger.info(f"Using GPU-accelerated algorithms with {h2o_time_limit}s time limit")
        else:
            # Exclude complex algorithms for CPU-only runs
            automl_params["exclude_algos"] = ["DeepLearning"]
            logger.info(f"Using CPU algorithms with {h2o_time_limit}s time limit")
        
        aml = H2OAutoML(**automl_params)
        aml.train(x=features, y=target, training_frame=h2o_frame)
        logger.info(f"H2O AutoML completed. Best model: {aml.leader.model_id}")
        
        logger.info("H2O model training completed")
        return aml.leader
        
    except ImportError:
        logger.error("H2O not installed. Please install with: pip install h2o")
        return None
    except Exception as e:
        logger.error(f"Error training H2O model: {e}")
        return None

def save_model(model, model_type):
    """Save trained model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        if model_type == 'h2o':
            # H2O models need special handling
            model_path = os.path.join(MODELS_DIR, f"h2o_model_{timestamp}")
            import h2o
            h2o.save_model(model=model, path=MODELS_DIR, filename=f"h2o_model_{timestamp}", force=True)
        else:
            # Standard pickle save for other models
            model_path = os.path.join(MODELS_DIR, f"{model_type}_model_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Register model in model store
        try:
            from utils.model.model_store import ModelStore
            model_store = ModelStore()
            model_store.register_model(
                model_path=model_path,
                model_type=model_type,
                feature_set_id="gpu_features",
                version="1",
                description=f"{model_type} model trained on GPU features"
            )
            logger.info(f"Model registered in model store")
        except Exception as e:
            logger.warning(f"Failed to register model in model store: {e}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None


def setup_h2o_environment():
    """Setup H2O environment to use NVMe disk for all temp files"""
    h2o_temp_dir = '/media/knight2/EDB/tmp/h2o'
    
    # Create directory if it doesn't exist
    os.makedirs(h2o_temp_dir, mode=0o755, exist_ok=True)
    
    # Set ALL possible temp directory environment variables
    temp_env_vars = {
        'TMPDIR': h2o_temp_dir,
        'TMP': h2o_temp_dir,
        'TEMP': h2o_temp_dir,
        'JAVA_OPTS': f'-Djava.io.tmpdir={h2o_temp_dir}',
        '_JAVA_OPTIONS': f'-Djava.io.tmpdir={h2o_temp_dir}',
        'H2O_TEMP_DIR': h2o_temp_dir,
        # Additional Java system properties
        'JVM_ARGS': f'-Djava.io.tmpdir={h2o_temp_dir} -Dh2o.temp.dir={h2o_temp_dir}',
    }
    
    for var, value in temp_env_vars.items():
        os.environ[var] = value
        logger.info(f"Set {var}={value}")
    
    logger.info(f"H2O environment configured to use {h2o_temp_dir}")
    return h2o_temp_dir

def main():
    parser = argparse.ArgumentParser(description='Train models for Numerai Crypto')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--h2o-time-limit', type=int, default=300, help='Time limit for H2O models in seconds')
    parser.add_argument('--model-type', type=str, default='simple', help='Model type to train: simple, lightgbm, xgboost, h2o, or all')
    parser.add_argument('--multi-train', action='store_true', help='Train multiple model variants')
    parser.add_argument('--optimize-hyperparams', action='store_true', help='Optimize hyperparameters using Optuna')
    parser.add_argument('--optimization-trials', type=int, default=30, help='Number of optimization trials')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining even if models exist from today')
    
    args = parser.parse_args()
    
    logger.info("Starting train_models.py")
    
    # CRITICAL: Setup H2O environment BEFORE any H2O operations
    setup_h2o_environment()
    
    # Check if models already exist from today (unless force-retrain is specified)
    if not args.force_retrain:
        if check_models_exist_today(args.model_type, MODELS_DIR):
            logger.info(f"Models for '{args.model_type}' already exist from today.")
            logger.info("Skipping training. Use --force-retrain to retrain existing models.")
            return True
    else:
        logger.info("Force retrain specified - will train models even if they exist from today")
    
    # Prepare training data - only use GPU-generated Parquet features
    if not os.path.exists(GPU_FEATURES_FILE):
        logger.error(f"GPU features file not found: {GPU_FEATURES_FILE}")
        logger.error("Please run GPU feature generation first to create the required Parquet file")
        return False
    
    logger.info(f"Using GPU-generated features from {GPU_FEATURES_FILE}")
    X_train, y_train = prepare_data(GPU_FEATURES_FILE)
    
    if X_train is None or y_train is None:
        logger.error("Failed to prepare training data")
        return False
    
    # Optimize hyperparameters if requested
    if args.optimize_hyperparams:
        logger.info("Hyperparameter optimization enabled")
        if args.model_type in ['lightgbm', 'all']:
            logger.info("Optimizing LightGBM hyperparameters...")
            best_lgb_params = optimize_lightgbm_hyperparams(X_train, y_train, args.use_gpu, args.optimization_trials)
    
    # Train model based on model type
    model = None
    if args.model_type == 'simple':
        model = train_simple_model(X_train, y_train, args.use_gpu, args.parallel)
    elif args.model_type == 'lightgbm':
        model = train_lightgbm_model(X_train, y_train, args.use_gpu, args.parallel)
    elif args.model_type == 'xgboost':
        model = train_xgboost_model(X_train, y_train, args.use_gpu, args.parallel)
    elif args.model_type == 'h2o':
        model = train_h2o_model(X_train, y_train, args.use_gpu, args.parallel, args.h2o_time_limit)
    elif args.model_type == 'all':
        # Train all model types
        models = {}
        models['simple'] = train_simple_model(X_train, y_train, args.use_gpu, args.parallel)
        models['lightgbm'] = train_lightgbm_model(X_train, y_train, args.use_gpu, args.parallel)
        models['xgboost'] = train_xgboost_model(X_train, y_train, args.use_gpu, args.parallel)
        models['h2o'] = train_h2o_model(X_train, y_train, args.use_gpu, args.parallel, args.h2o_time_limit)
        
        # Save all models
        for model_name, trained_model in models.items():
            if trained_model is not None:
                save_model(trained_model, model_name)
        
        logger.info("All model training completed successfully")
        return True
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return False
    
    if model is None:
        logger.error("Model training failed")
        return False
    
    # Save model
    model_path = save_model(model, args.model_type)
    if model_path:
        logger.info("Model saved successfully")
    else:
        logger.error("Failed to save model")
        return False
    
    logger.info("Model training completed successfully")
    return True

if __name__ == "__main__":
    main()
