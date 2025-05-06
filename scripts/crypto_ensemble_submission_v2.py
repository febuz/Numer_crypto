#!/usr/bin/env python3
"""
Crypto Ensemble Submission (Version 2) - Enhanced with Feature Store & Checkpoints

An advanced ensemble pipeline for Numerai Crypto competition with:
- Feature Store for caching generated features
- Model Checkpoint system for saving and loading models
- Integration with functional and performance tests
- Multi-GPU support with dynamic allocation
- Feature engineering with polynomial expansions
- Ensemble of diverse ML models

Usage:
    python crypto_ensemble_submission_v2.py [--download] [--test-first]
                                         [--feature-count NUM] [--use-cache]
                                         [--output OUTPUT_FILE]
"""
import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import project modules
from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.models.lightgbm_model import LightGBMModel
from numer_crypto.models.xgboost_model import H2OXGBoostModel
from numer_crypto.utils.gpu_utils import (
    get_available_gpus, select_best_gpu, get_gpu_model_params, print_gpu_status
)
from numer_crypto.utils.data_utils import (
    convert_pandas_to_h2o, convert_polars_to_h2o, convert_spark_to_h2o
)
from numer_crypto.utils.rapids_utils import (
    check_rapids_availability, pandas_to_cudf, cudf_to_pandas,
    polars_to_cudf, cudf_to_polars, with_cudf, enable_spark_rapids
)
from numer_crypto.config.settings import MODELS_DIR, DATA_DIR

# Import feature store and model checkpoint
sys.path.append(str(script_dir))
from feature_store import FeatureStore
from model_checkpoint import ModelCheckpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"crypto_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize feature store and model checkpoint
feature_store_dir = project_root / "data" / "feature_store"
feature_store = FeatureStore(str(feature_store_dir))

model_checkpoint_dir = project_root / "models" / "checkpoints"
model_checkpoint = ModelCheckpoint(str(model_checkpoint_dir))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Ensemble for Numerai Crypto')
    parser.add_argument('--download', action='store_true', help='Download latest data')
    parser.add_argument('--test-first', action='store_true', 
                        help='Run functional tests before starting')
    parser.add_argument('--feature-count', type=int, default=5000, 
                        help='Number of polynomial features to generate')
    parser.add_argument('--poly-degree', type=int, default=2, 
                        help='Degree of polynomial features')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU acceleration (auto-detected if not specified)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached features when available')
    parser.add_argument('--ensemble-size', type=int, default=5, 
                        help='Number of models in the ensemble')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for predictions (defaults to timestamp)')
    parser.add_argument('--output2', type=str, default=None,
                        help='Second output file for alternative predictions')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--use-spark', action='store_true', 
                        help='Use Spark for distributed processing')
    parser.add_argument('--iterative-pruning', action='store_true', 
                        help='Use iterative feature pruning')
    parser.add_argument('--prune-pct', type=float, default=0.5, 
                        help='Percentage of features to keep in each iteration')
    parser.add_argument('--no-h2o', action='store_true',
                        help='Disable H2O models in ensemble')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Skip feature engineering (use original features only)')
    parser.add_argument('--feature-set', type=str, default=None,
                        help='Use specific feature set from feature store')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load specific model from checkpoint')
    parser.add_argument('--best-model', action='store_true',
                        help='Use best model from checkpoint based on validation metrics')
    return parser.parse_args()

def run_tests():
    """Run functional tests before starting the pipeline"""
    logger.info("Running functional tests before starting...")
    
    test_script = script_dir / "run_functional_tests.py"
    if not test_script.exists():
        logger.error(f"Test script not found at {test_script}")
        return False
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(test_script), "--test-type", "basic"],
            check=False,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0
        
        if success:
            logger.info("Functional tests passed. Proceeding with ensemble pipeline.")
        else:
            logger.error("Functional tests failed. Pipeline may not work correctly.")
            logger.error(f"Test output: {result.stdout}")
            logger.error(f"Test errors: {result.stderr}")
        
        return success
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def init_h2o_and_spark(args):
    """Initialize H2O and Spark environments if available"""
    h2o_context = None
    spark = None
    
    # Skip H2O initialization if requested
    if args.no_h2o:
        logger.info("Skipping H2O initialization as requested")
        return h2o_context, spark
    
    # Initialize H2O
    try:
        import h2o
        logger.info("Initializing H2O...")
        h2o.init(nthreads=-1, max_mem_size="8G")
        
        # Initialize Spark and H2O-Sparkling Water if requested
        if args.use_spark:
            try:
                from pyspark.sql import SparkSession
                from pysparkling import H2OContext
                
                logger.info("Initializing Spark and H2O-Sparkling Water...")
                
                # Create Spark session
                builder = SparkSession.builder \
                    .appName("NumeraiCryptoEnsemble") \
                    .config("spark.executor.memory", "4g") \
                    .config("spark.driver.memory", "4g") \
                    .config("spark.executor.cores", "2")
                
                # Enable RAPIDS for Spark if available
                if args.gpu:
                    try:
                        import importlib.util
                        if importlib.util.find_spec("rapids.spark") is not None:
                            logger.info("Enabling RAPIDS Accelerator for Spark...")
                            builder = builder \
                                .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
                                .config("spark.rapids.sql.enabled", "true") \
                                .config("spark.rapids.sql.explain", "ALL")
                    except ImportError:
                        logger.warning("RAPIDS for Spark not available")
                
                spark = builder.getOrCreate()
                
                # Initialize H2O context
                h2o_context = H2OContext.getOrCreate(spark)
                logger.info(f"Spark version: {spark.version}")
                logger.info(f"H2O context version: {h2o_context.getSparklingWaterVersion()}")
            except ImportError:
                logger.warning("PySparkling not available. Skipping Spark+H2O integration.")
    except ImportError:
        logger.warning("H2O not available. Some models will be disabled.")
    
    return h2o_context, spark

def generate_or_load_features(args, data_type, df):
    """
    Generate polynomial features or load from feature store if available
    
    Args:
        args: Command line arguments
        data_type: Type of data ('train', 'valid', 'tournament')
        df: DataFrame to process
        
    Returns:
        Pandas DataFrame with features
    """
    # If skipping feature engineering, return original features
    if args.skip_feature_engineering:
        logger.info(f"Skipping feature engineering for {data_type} data")
        return df
    
    # If a specific feature set is requested, try to load it
    if args.feature_set:
        if feature_store.feature_set_exists(f"{args.feature_set}_{data_type}"):
            logger.info(f"Loading {data_type} features from feature store: {args.feature_set}")
            loaded_df = feature_store.get_features(f"{args.feature_set}_{data_type}")
            if loaded_df is not None:
                return loaded_df
            logger.warning(f"Failed to load {args.feature_set}_{data_type} from feature store")
    
    # Generate a feature set name
    feature_set_name = f"poly{args.poly_degree}_{args.feature_count}_{data_type}"
    
    # Check cache if requested
    if args.use_cache and feature_store.feature_set_exists(feature_set_name):
        logger.info(f"Loading cached features for {data_type} from feature store")
        cached_df = feature_store.get_features(feature_set_name)
        if cached_df is not None:
            return cached_df
        logger.warning("Cached features not found or corrupted")
    
    # Generate features
    logger.info(f"Generating polynomial features for {data_type} data")
    
    # Convert to polars for initial processing
    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df)
    elif not isinstance(df, pl.DataFrame):
        pl_df = pl.DataFrame(df)
    else:
        pl_df = df
        
    # Get original feature columns (excluding target and id columns)
    feature_cols = [col for col in pl_df.columns if col not in ['target', 'id', 'era', 'data_type']]
    
    # Use GPU if requested and available
    use_gpu = args.gpu
    if use_gpu:
        try:
            import importlib.util
            rapids_available = importlib.util.find_spec("cudf") is not None
            if not rapids_available:
                logger.warning("RAPIDS (cuDF) not available. Using CPU for feature generation.")
                use_gpu = False
        except ImportError:
            use_gpu = False
    
    # Generate features with GPU if available
    if use_gpu:
        try:
            logger.info("Using RAPIDS (cuDF) for polynomial feature generation")
            import cudf
            
            # Convert to cuDF
            cudf_df = polars_to_cudf(pl_df)
            
            # Select original features only
            features_df = cudf_df[feature_cols]
            
            # Generate polynomial features
            poly_features = []
            for i, col1 in enumerate(feature_cols):
                # Add original feature
                poly_features.append(features_df[col1])
                
                # Add squares if degree >= 2
                if args.poly_degree >= 2:
                    poly_features.append(features_df[col1] ** 2)
                
                # Add interactions and higher powers
                for j in range(i + 1, min(len(feature_cols), i + 50)):  # Limit interactions
                    col2 = feature_cols[j]
                    # Interaction term
                    poly_features.append(features_df[col1] * features_df[col2])
                    
                    # Cubic terms if degree >= 3
                    if args.poly_degree >= 3:
                        poly_features.append(features_df[col1] ** 3)
                        poly_features.append(features_df[col1] ** 2 * features_df[col2])
                        poly_features.append(features_df[col1] * features_df[col2] ** 2)
                
                # Stop if we have enough features
                if len(poly_features) >= args.feature_count:
                    break
                    
            # Trim to max features
            poly_features = poly_features[:args.feature_count]
            
            # Create column names for polynomial features
            poly_names = [f'poly_{i}' for i in range(len(poly_features))]
            
            # Create new dataframe with polynomial features
            poly_df = cudf.DataFrame(dict(zip(poly_names, poly_features)))
            
            # Add non-feature columns back
            for col in pl_df.columns:
                if col not in feature_cols:
                    poly_df[col] = cudf_df[col]
            
            # Add random features for robustness
            np.random.seed(args.random_seed)
            for i in range(10):
                poly_df[f'random_{i}'] = np.random.randn(len(poly_df))
            
            # Convert back to pandas
            result_df = cudf_to_pandas(poly_df)
            logger.info(f"Generated {len(poly_names)} polynomial features using GPU")
            
        except Exception as e:
            logger.warning(f"Error generating polynomial features with GPU: {e}")
            logger.warning("Falling back to CPU implementation")
            use_gpu = False
    
    # CPU implementation if GPU failed or unavailable
    if not use_gpu:
        logger.info("Using Polars for polynomial feature generation")
        
        # Select original features only
        features_df = pl_df.select(feature_cols)
        
        # Generate polynomial features
        poly_features = []
        poly_names = []
        
        for i, col1 in enumerate(feature_cols):
            # Add original feature
            poly_features.append(features_df[col1])
            poly_names.append(f"poly_{len(poly_names)}")
            
            # Add squares if degree >= 2
            if args.poly_degree >= 2:
                poly_features.append(features_df[col1] ** 2)
                poly_names.append(f"poly_{len(poly_names)}")
            
            # Add interactions and higher powers
            for j in range(i + 1, min(len(feature_cols), i + 30)):  # Limit interactions
                col2 = feature_cols[j]
                # Interaction term
                poly_features.append(features_df[col1] * features_df[col2])
                poly_names.append(f"poly_{len(poly_names)}")
                
                # Cubic terms if degree >= 3
                if args.poly_degree >= 3:
                    poly_features.append(features_df[col1] ** 3)
                    poly_names.append(f"poly_{len(poly_names)}")
                    poly_features.append(features_df[col1] ** 2 * features_df[col2])
                    poly_names.append(f"poly_{len(poly_names)}")
                    poly_features.append(features_df[col1] * features_df[col2] ** 2)
                    poly_names.append(f"poly_{len(poly_names)}")
            
            # Stop if we have enough features
            if len(poly_features) >= args.feature_count:
                break
        
        # Trim to max features
        poly_features = poly_features[:args.feature_count]
        poly_names = poly_names[:args.feature_count]
        
        # Create new dataframe with polynomial features
        poly_df = pl.DataFrame()
        for name, expr in zip(poly_names, poly_features):
            poly_df = poly_df.with_columns(expr.alias(name))
        
        # Add non-feature columns back
        for col in pl_df.columns:
            if col not in feature_cols:
                poly_df = poly_df.with_columns(pl_df[col])
        
        # Add random features for robustness
        np.random.seed(args.random_seed)
        for i in range(10):
            poly_df = poly_df.with_columns(
                pl.lit(np.random.randn(len(poly_df))).alias(f'random_{i}')
            )
        
        # Convert to pandas
        result_df = poly_df.to_pandas()
        logger.info(f"Generated {len(poly_names)} polynomial features using CPU")
    
    # Store in feature store
    logger.info(f"Storing {data_type} features in feature store: {feature_set_name}")
    feature_store.store_features(
        result_df, 
        feature_set_name, 
        metadata={
            'type': data_type,
            'poly_degree': args.poly_degree,
            'feature_count': args.feature_count,
            'timestamp': datetime.now().isoformat(),
            'random_seed': args.random_seed
        }
    )
    
    return result_df

def select_features_iteratively(train_df, valid_df, feature_cols, keep_pct=0.5, iterations=3, gpu_id=None, seed=42):
    """Iteratively select features by training models and keeping the most important ones"""
    logger.info(f"Performing iterative feature selection ({iterations} iterations, keeping {keep_pct*100}% each time)")
    
    current_features = feature_cols.copy()
    
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations} - Features: {len(current_features)}")
        
        # Train a model
        model = train_lightgbm_model(train_df, valid_df, current_features, gpu_id, seed+i)
        
        # Get feature importance
        feature_imp = model.get_feature_importance()
        
        # Store feature importance in feature store
        feature_set_name = f"train_poly{args.poly_degree}_{args.feature_count}"
        if feature_store.feature_set_exists(feature_set_name):
            feature_store.store_feature_importance(
                feature_set_name,
                feature_imp,
                f"lightgbm_iteration_{i+1}"
            )
        
        # Sort features by importance
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate how many features to keep
        keep_count = max(int(len(current_features) * keep_pct), 50)  # Keep at least 50 features
        
        # Select top features
        top_features = [f[0] for f in sorted_features[:keep_count]]
        
        logger.info(f"Keeping {len(top_features)} features for next iteration")
        current_features = top_features
    
    return current_features

def train_lightgbm_model(train_df, valid_df, feature_cols, gpu_id=None, seed=42):
    """Train a LightGBM model optionally using GPU"""
    model_name = f"lightgbm_gpu{gpu_id if gpu_id is not None else 'cpu'}_{int(time.time())}"
    logger.info(f"Training LightGBM model '{model_name}' (GPU: {gpu_id is not None})")
    
    # Get best GPU if not specified and GPU is available
    if gpu_id is None and args.gpu:
        gpu_id = select_best_gpu()
    
    # Get optimized params for GPU/CPU
    params = get_gpu_model_params('lightgbm', gpu_id) if gpu_id is not None else None
    
    # Add custom params
    if params is None:
        params = {}
    params.update({
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'seed': seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1
    })
    
    # Create and train model
    model = LightGBMModel(
        params=params,
        use_gpu=(gpu_id is not None),
        gpu_device_id=gpu_id if gpu_id is not None else 0,
        seed=seed,
        name=model_name
    )
    
    # Ensure dataframes are pandas
    train_pandas = train_df.to_pandas() if hasattr(train_df, 'to_pandas') else train_df
    valid_pandas = valid_df.to_pandas() if hasattr(valid_df, 'to_pandas') else valid_df
    
    # Extract features and target
    X_train = train_pandas[feature_cols]
    y_train = train_pandas['target']
    X_valid = valid_pandas[feature_cols]
    y_valid = valid_pandas['target']
    
    # Train model
    start_time = time.time()
    train_result = model.train(
        X_train, y_train, X_valid, y_valid,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    training_time = time.time() - start_time
    
    # Get metrics
    metrics = {
        'training_time': training_time,
        'best_iteration': train_result['best_iteration'],
        'best_score': train_result['best_score']
    }
    
    logger.info(f"LightGBM training complete - Training time: {training_time:.2f}s, Best score: {train_result['best_score']}")
    
    # Save model checkpoint
    feature_importance = model.get_feature_importance()
    model_checkpoint.save_model(
        model,
        model_name,
        model_type='lightgbm',
        metrics=metrics,
        params=params,
        feature_names=feature_cols,
        feature_importance=feature_importance
    )
    
    return model

def train_h2o_xgboost_model(train_df, valid_df, feature_cols, h2o_context=None, seed=42):
    """Train an H2O XGBoost model"""
    model_name = f"h2o_xgboost_{int(time.time())}"
    logger.info(f"Training H2O XGBoost model '{model_name}'")
    
    # Configure parameters
    params = {
        'ntrees': 500,
        'max_depth': 6,
        'learn_rate': 0.05,
        'sample_rate': 0.8,
        'col_sample_rate': 0.8,
        'seed': seed
    }
    
    # Create model
    model = H2OXGBoostModel(params=params)
    
    # Convert to H2O frames if not already
    if isinstance(train_df, pd.DataFrame):
        h2o_train = convert_pandas_to_h2o(train_df)
        h2o_valid = convert_pandas_to_h2o(valid_df)
    elif h2o_context is not None and hasattr(train_df, '_jdf'):  # Spark DataFrame
        h2o_train = h2o_context.asH2OFrame(train_df)
        h2o_valid = h2o_context.asH2OFrame(valid_df)
    else:
        h2o_train = train_df
        h2o_valid = valid_df
    
    # Train model
    start_time = time.time()
    model.train(
        train_df=h2o_train,
        valid_df=h2o_valid,
        feature_cols=feature_cols,
        target_col='target',
        model_id=model_name
    )
    training_time = time.time() - start_time
    
    # Get metrics
    metrics = {
        'training_time': training_time
    }
    
    # Try to get auc
    try:
        metrics['validation_auc'] = model.model.auc(valid=True)
    except:
        pass
    
    logger.info(f"H2O XGBoost training complete - Training time: {training_time:.2f}s")
    
    # Save model checkpoint
    try:
        # Get feature importance if available
        varimp = model.model.varimp(use_pandas=True)
        feature_importance = {}
        if varimp is not None:
            for i, row in varimp.iterrows():
                feature_importance[row['variable']] = float(row['relative_importance'])
        
        model_checkpoint.save_model(
            model,
            model_name,
            model_type='h2o_xgboost',
            metrics=metrics,
            params=params,
            feature_names=feature_cols,
            feature_importance=feature_importance
        )
    except Exception as e:
        logger.warning(f"Could not save H2O XGBoost model checkpoint: {e}")
    
    return model

def train_h2o_automl(train_df, valid_df, feature_cols, max_runtime_secs=600, h2o_context=None, seed=42):
    """Train an H2O AutoML model"""
    model_name = f"h2o_automl_{int(time.time())}"
    logger.info(f"Training H2O AutoML model '{model_name}' (max_runtime: {max_runtime_secs} seconds)")
    
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        # Convert to H2O frames if needed
        if isinstance(train_df, pd.DataFrame):
            h2o_train = convert_pandas_to_h2o(train_df)
            h2o_valid = convert_pandas_to_h2o(valid_df)
        elif h2o_context is not None and hasattr(train_df, '_jdf'):  # Spark DataFrame
            h2o_train = h2o_context.asH2OFrame(train_df)
            h2o_valid = h2o_context.asH2OFrame(valid_df)
        else:
            h2o_train = train_df
            h2o_valid = valid_df
        
        # Set up AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=seed,
            sort_metric="AUC"
        )
        
        # Train AutoML model
        start_time = time.time()
        aml.train(
            x=feature_cols,
            y="target",
            training_frame=h2o_train,
            validation_frame=h2o_valid
        )
        training_time = time.time() - start_time
        
        # Get best model
        best_model = aml.leader
        
        # Save metrics
        metrics = {
            'training_time': training_time,
            'validation_auc': best_model.auc(valid=True)
        }
        
        logger.info(f"H2O AutoML complete - Best model: {best_model.model_id}, AUC: {metrics['validation_auc']}")
        
        # Save model checkpoint
        try:
            # Get feature importance if available
            feature_importance = {}
            if hasattr(best_model, 'varimp') and callable(getattr(best_model, 'varimp')):
                varimp = best_model.varimp(use_pandas=True)
                if varimp is not None:
                    for i, row in varimp.iterrows():
                        feature_importance[row['variable']] = float(row['relative_importance'])
            
            model_checkpoint.save_model(
                best_model,
                model_name,
                model_type='h2o_automl',
                metrics=metrics,
                feature_names=feature_cols,
                feature_importance=feature_importance
            )
        except Exception as e:
            logger.warning(f"Could not save H2O AutoML model checkpoint: {e}")
        
        return best_model
    
    except Exception as e:
        logger.error(f"Error training H2O AutoML: {e}")
        return None

def load_best_model():
    """Load the best model from checkpoint based on validation metrics"""
    logger.info("Loading best model from checkpoint based on validation metrics")
    
    # Find best model
    best_model_name = model_checkpoint.get_best_model(metric='validation_auc', higher_is_better=True)
    
    if not best_model_name:
        logger.warning("No models found with validation metrics. Cannot load best model.")
        return None, None
    
    # Load the model
    model, metadata = model_checkpoint.load_model(best_model_name)
    
    if model is None:
        logger.warning(f"Failed to load best model '{best_model_name}'")
        return None, None
    
    logger.info(f"Loaded best model: {best_model_name}")
    if metadata and 'metrics' in metadata:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metadata['metrics'].items()])
        logger.info(f"Model metrics: {metrics_str}")
    
    return model, metadata

def train_models_multi_gpu(train_df, valid_df, feature_cols, h2o_context=None, seed=42):
    """Train multiple models on multiple GPUs if available"""
    models = []
    
    # Check GPU availability
    gpus = []
    if args.gpu:
        gpus = get_available_gpus()
    
    if not gpus:
        logger.warning("No GPUs available for multi-GPU training. Using CPU instead.")
        
        # Train LightGBM on CPU
        lgbm_model = train_lightgbm_model(train_df, valid_df, feature_cols, None, seed)
        models.append({
            'model': lgbm_model,
            'model_type': 'lightgbm',
            'gpu_id': None,
            'name': lgbm_model.name
        })
        
        # Train H2O XGBoost if available
        if not args.no_h2o:
            try:
                import h2o
                h2o_model = train_h2o_xgboost_model(train_df, valid_df, feature_cols, h2o_context, seed)
                models.append({
                    'model': h2o_model,
                    'model_type': 'h2o_xgboost',
                    'gpu_id': None,
                    'name': h2o_model.model.model_id
                })
            except ImportError:
                logger.warning("H2O not available for CPU training")
        
        return models
    
    # Determine models to train
    ensemble_size = min(args.ensemble_size, len(gpus) * 2)  # Limit by GPU count
    
    # Mix of model types
    model_types = []
    for i in range(ensemble_size):
        if i % 3 == 0 and not args.no_h2o:
            model_types.append('h2o_xgboost')
        else:
            model_types.append('lightgbm')
    
    # Assign GPUs to models (ensure balance)
    gpu_assignments = []
    gpu_indices = list(range(len(gpus)))
    
    # Round-robin assignment
    for i in range(len(model_types)):
        gpu_idx = gpu_indices[i % len(gpu_indices)]
        gpu_id = gpus[gpu_idx]['index']
        gpu_assignments.append(gpu_id)
    
    logger.info(f"Training {len(model_types)} models on {len(gpus)} GPUs")
    for i, (model_type, gpu_id) in enumerate(zip(model_types, gpu_assignments)):
        logger.info(f"  Model {i+1}: {model_type} on GPU {gpu_id}")
    
    # Train models sequentially to avoid conflicts
    for i, (model_type, gpu_id) in enumerate(zip(model_types, gpu_assignments)):
        try:
            # Set environment variable for this GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            if model_type == 'lightgbm':
                logger.info(f"Training LightGBM model ({i+1}/{len(model_types)}) on GPU {gpu_id}")
                model = train_lightgbm_model(train_df, valid_df, feature_cols, 0, seed+i)  # 0 because of CUDA_VISIBLE_DEVICES
                models.append({
                    'model': model,
                    'model_type': 'lightgbm',
                    'gpu_id': gpu_id,
                    'name': model.name
                })
            
            elif model_type == 'h2o_xgboost':
                logger.info(f"Training H2O XGBoost model ({i+1}/{len(model_types)}) on GPU {gpu_id}")
                model = train_h2o_xgboost_model(train_df, valid_df, feature_cols, h2o_context, seed+i)
                models.append({
                    'model': model,
                    'model_type': 'h2o_xgboost',
                    'gpu_id': gpu_id,
                    'name': model.model.model_id
                })
        
        except Exception as e:
            logger.error(f"Error training {model_type} on GPU {gpu_id}: {e}")
    
    # Add H2O AutoML if requested and available
    if not args.no_h2o and len(models) < args.ensemble_size:
        try:
            import h2o
            logger.info("Training H2O AutoML model")
            # Reset CUDA_VISIBLE_DEVICES to use all GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu['index']) for gpu in gpus)
            automl_model = train_h2o_automl(
                train_df, 
                valid_df, 
                feature_cols, 
                max_runtime_secs=600,
                h2o_context=h2o_context,
                seed=seed+len(models)
            )
            if automl_model:
                models.append({
                    'model': automl_model,
                    'model_type': 'h2o_automl',
                    'gpu_id': None,
                    'name': automl_model.model_id
                })
        except ImportError:
            logger.warning("H2O not available for AutoML")
    
    logger.info(f"Trained {len(models)} models successfully")
    return models

def predict_with_ensemble(models, test_df, feature_cols, h2o_context=None):
    """Generate predictions using an ensemble of models"""
    logger.info(f"Generating predictions with ensemble of {len(models)} models")
    
    all_predictions = []
    model_weights = []  # Default weights (equal)
    
    # Generate predictions from each model
    for model_info in models:
        model = model_info['model']
        model_type = model_info['model_type']
        model_name = model_info.get('name', str(model_type))
        
        logger.info(f"Generating predictions for {model_type} model: {model_name}")
        
        try:
            if model_type == 'lightgbm':
                # Ensure we have pandas DataFrame with the right features
                test_features = test_df[feature_cols] if isinstance(test_df, pd.DataFrame) else test_df.select(feature_cols).to_pandas()
                preds = model.predict(test_features)
                all_predictions.append(preds)
                model_weights.append(1.0)  # Default weight
            
            elif model_type == 'h2o_xgboost':
                # Convert to H2O frame if needed
                if isinstance(test_df, pd.DataFrame):
                    h2o_test = convert_pandas_to_h2o(test_df)
                elif h2o_context is not None and hasattr(test_df, '_jdf'):  # Spark DataFrame
                    h2o_test = h2o_context.asH2OFrame(test_df)
                else:
                    h2o_test = test_df
                
                # Make predictions
                preds_df = model.predict(h2o_test)
                preds = preds_df['predict'].as_data_frame().values.flatten()
                all_predictions.append(preds)
                model_weights.append(1.0)  # Default weight
            
            elif model_type == 'h2o_automl':
                # Convert to H2O frame if needed
                if isinstance(test_df, pd.DataFrame):
                    h2o_test = convert_pandas_to_h2o(test_df)
                elif h2o_context is not None and hasattr(test_df, '_jdf'):  # Spark DataFrame
                    h2o_test = h2o_context.asH2OFrame(test_df)
                else:
                    h2o_test = test_df
                
                # Make predictions
                preds_df = model.predict(h2o_test)
                preds = preds_df['p1'].as_data_frame().values.flatten()  # p1 for binary classification
                all_predictions.append(preds)
                model_weights.append(1.0)  # Default weight
                
        except Exception as e:
            logger.error(f"Error generating predictions for {model_type} model: {e}")
    
    if not all_predictions:
        logger.error("No valid predictions from any model!")
        return None
    
    # Normalize weights
    total_weight = sum(model_weights)
    if total_weight > 0:
        model_weights = [w / total_weight for w in model_weights]
    
    # Weighted ensemble
    ensemble_preds = np.zeros_like(all_predictions[0])
    for preds, weight in zip(all_predictions, model_weights):
        ensemble_preds += preds * weight
    
    # Create predictions dataframe
    if isinstance(test_df, pd.DataFrame):
        ids = test_df['id'].values
    else:
        ids = test_df.select(['id']).to_pandas()['id'].values
    
    predictions_df = pd.DataFrame({
        'id': ids,
        'prediction': ensemble_preds
    })
    
    return predictions_df

def create_randomized_predictions(predictions_df, randomization_pct=0.5, seed=42):
    """Create a second set of predictions with some randomization"""
    logger.info(f"Creating second submission with {randomization_pct*100}% randomization")
    
    # Make a copy
    predictions2 = predictions_df.copy()
    
    # Set random seed
    np.random.seed(seed)
    
    # Add small random noise
    std_dev = randomization_pct / 100.0  # Convert percentage to value (e.g. 0.5% -> 0.005)
    noise = np.random.normal(0, std_dev, size=len(predictions2))
    predictions2['prediction'] = predictions2['prediction'] + noise
    
    # Clip to valid range [0, 1]
    predictions2['prediction'] = np.clip(predictions2['prediction'], 0, 1)
    
    return predictions2

def save_predictions(predictions_df, output_file=None):
    """Save predictions to CSV file"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATA_DIR, "submissions", f"crypto_ensemble_predictions_{timestamp}.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    predictions_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    return output_file

def main():
    """Main function to run the Numerai Crypto ensemble pipeline"""
    # Get command line arguments
    global args
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # Print system info
    logger.info("=" * 80)
    logger.info("NUMERAI CRYPTO ENSEMBLE SUBMISSION (VERSION 2)")
    logger.info("=" * 80)
    
    # Check GPU availability
    gpus = []
    if args.gpu:
        gpus = get_available_gpus()
        if gpus:
            logger.info(f"Found {len(gpus)} GPUs:")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.get('name', 'Unknown')}")
        else:
            logger.warning("No GPUs found despite --gpu flag. Using CPU for computation.")
    
    # Run tests if requested
    if args.test_first:
        if not run_tests():
            logger.warning("Functional tests failed, but continuing with pipeline")
    
    # Create output paths
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(DATA_DIR, "submissions"), exist_ok=True)
        args.output = os.path.join(DATA_DIR, "submissions", f"crypto_ensemble_predictions_{timestamp}.csv")
    
    if not args.output2:
        args.output2 = os.path.splitext(args.output)[0] + "_v2.csv"
    
    # Initialize H2O and Spark
    h2o_context, spark = init_h2o_and_spark(args)
    
    # Create data retriever
    data_retriever = NumeraiDataRetriever(use_spark=args.use_spark and spark is not None)
    
    # Download data if requested
    if args.download:
        logger.info("Downloading latest data...")
        data_retriever.download_current_dataset(tournament='crypto')
    
    # Load datasets
    logger.info("Loading datasets...")
    train_df = data_retriever.load_dataset('training')
    valid_df = data_retriever.load_dataset('validation')
    tournament_df = data_retriever.load_dataset('tournament')
    
    # Get original feature names
    original_features = data_retriever.get_feature_names()
    logger.info(f"Loaded {len(original_features)} original features")
    
    # Process features
    train_processed = generate_or_load_features(args, 'train', train_df)
    valid_processed = generate_or_load_features(args, 'valid', valid_df)
    tournament_processed = generate_or_load_features(args, 'tournament', tournament_df)
    
    # Get all feature columns
    feature_cols = [col for col in train_processed.columns 
                   if col not in ['id', 'era', 'data_type', 'target']]
    logger.info(f"Total features after processing: {len(feature_cols)}")
    
    # Iterative feature selection if requested
    if args.iterative_pruning:
        selected_features = select_features_iteratively(
            train_processed, 
            valid_processed, 
            feature_cols, 
            keep_pct=args.prune_pct, 
            iterations=3, 
            gpu_id=select_best_gpu() if gpus else None,
            seed=args.random_seed
        )
        feature_cols = selected_features
        logger.info(f"Selected {len(feature_cols)} features after iterative pruning")
    
    # Train or load models
    models = []
    
    if args.load_model:
        # Load a specific model
        logger.info(f"Loading model from checkpoint: {args.load_model}")
        model, metadata = model_checkpoint.load_model(args.load_model)
        if model:
            model_type = metadata.get('model_type', 'generic')
            models.append({
                'model': model,
                'model_type': model_type,
                'gpu_id': None,
                'name': args.load_model
            })
            logger.info(f"Loaded model: {args.load_model} (type: {model_type})")
        else:
            logger.warning(f"Failed to load model: {args.load_model}")
    
    elif args.best_model:
        # Load best model
        model, metadata = load_best_model()
        if model:
            model_type = metadata.get('model_type', 'generic')
            models.append({
                'model': model,
                'model_type': model_type,
                'gpu_id': None,
                'name': metadata.get('name', 'best_model')
            })
            logger.info(f"Using best model from checkpoint (type: {model_type})")
        else:
            logger.warning("Failed to load best model, training new models")
    
    # Train models if none were loaded or if loading failed
    if not models:
        models = train_models_multi_gpu(
            train_processed, 
            valid_processed, 
            feature_cols, 
            h2o_context, 
            args.random_seed
        )
    
    # Generate predictions
    predictions_df = predict_with_ensemble(models, tournament_processed, feature_cols, h2o_context)
    
    if predictions_df is None:
        logger.error("Failed to generate predictions!")
        return 1
    
    # Save the first set of predictions
    save_path = save_predictions(predictions_df, args.output)
    
    # Create second randomized predictions for diversification
    predictions2 = create_randomized_predictions(
        predictions_df,
        randomization_pct=0.5,  # 0.5% randomization
        seed=args.random_seed+42
    )
    
    # Save second predictions file
    save_path2 = save_predictions(predictions2, args.output2)
    
    # Clean up
    if h2o_context and not args.no_h2o:
        import h2o
        h2o.shutdown(prompt=False)
    
    if spark:
        spark.stop()
    
    logger.info("=" * 80)
    logger.info("PROCESS COMPLETE")
    logger.info(f"Submission 1: {save_path}")
    logger.info(f"Submission 2: {save_path2}")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())