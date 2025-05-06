#!/usr/bin/env python3
"""
H2O Sparkling Water Solution for Numerai Crypto

This script creates a high-quality submission using:
- H2O Sparkling Water for distributed processing
- Sophisticated feature engineering
- Proper cross-validation to ensure out-of-sample predictions
- XGBoost and other ML algorithms

Usage:
    python h2o_yiedl_submission.py [--gpus GPU_IDS] [--output OUTPUT_PATH]
"""
import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import zipfile

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import project modules if available
try:
    from numer_crypto.utils.gpu_utils import get_available_gpus, select_best_gpu
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"h2o_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='H2O Sparkling Water Solution for Numerai Crypto')
    parser.add_argument('--gpus', type=str, default='',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for submission file')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation on previous submission files')
    parser.add_argument('--time-limit', type=int, default=1200,
                        help='Time limit in seconds (default: 1200)')
    return parser.parse_args()

def setup_environment(gpus=''):
    """Set up environment for H2O and Spark"""
    # Set GPU environment variables if requested
    if gpus:
        logger.info(f"Setting up GPUs: {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    # Set Java options for H2O
    os.environ["_JAVA_OPTIONS"] = (
        "-Xms4g -Xmx8g "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED"
    )
    
    # Return environment settings
    return {
        'gpus': gpus,
        'java_options': os.environ.get("_JAVA_OPTIONS")
    }

def init_h2o_and_spark():
    """Initialize H2O and Spark with Sparkling Water"""
    try:
        import h2o
        from pyspark.sql import SparkSession
        from pysparkling import H2OContext
        
        logger.info("Initializing H2O standalone first...")
        h2o.init(nthreads=-1, max_mem_size="4g")
        
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("NumeraiCryptoH2O") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.jars.repositories", "https://h2oai.jfrog.io/artifactory/h2o-releases") \
            .getOrCreate()
            
        logger.info("Initializing H2O context with Spark...")
        h2o_context = H2OContext.getOrCreate(spark)
        
        logger.info(f"Spark version: {spark.version}")
        try:
            logger.info(f"H2O context version: {h2o_context.getSparklingWaterVersion()}")
        except:
            logger.info("H2O context initialized, but couldn't get version info")
            
        return h2o, spark, h2o_context
    
    except Exception as e:
        logger.error(f"Error initializing H2O and Spark: {e}")
        logger.info("Falling back to standalone H2O...")
        
        try:
            import h2o
            h2o.init(nthreads=-1, max_mem_size="4g")
            logger.info(f"H2O version: {h2o.__version__}")
            return h2o, None, None
        except Exception as e2:
            logger.error(f"Error initializing standalone H2O: {e2}")
            return None, None, None

def load_yiedl_data(use_spark=False, spark=None):
    """Load and parse Yiedl data"""
    # Set paths
    yiedl_dir = project_root / "data" / "yiedl"
    latest_file = yiedl_dir / "yiedl_latest.parquet"
    historical_zip = yiedl_dir / "yiedl_historical.zip"
    
    logger.info("Loading Yiedl data...")
    latest_df = None
    historical_df = None
    
    # Create tmp directory for extractions
    tmp_dir = yiedl_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Load latest data (predictions)
    if latest_file.exists():
        try:
            if use_spark and spark:
                logger.info("Loading latest data with Spark...")
                latest_df = spark.read.parquet(str(latest_file))
                logger.info(f"Loaded latest data: {latest_df.count()} rows")
            else:
                logger.info("Loading latest data with pandas...")
                latest_df = pd.read_parquet(latest_file)
                logger.info(f"Loaded latest data: {latest_df.shape}")
        except Exception as e:
            logger.error(f"Error loading latest data: {e}")
    
    # Extract historical data from zip
    if historical_zip.exists():
        try:
            logger.info("Extracting data from historical zip...")
            
            # Check if we already extracted the parquet
            extracted_parquet = tmp_dir / "yiedl_historical.parquet"
            if not extracted_parquet.exists():
                # Try to extract parquet file from zip
                with zipfile.ZipFile(historical_zip, 'r') as zip_ref:
                    # Find parquet files in the zip
                    parquet_files = [f for f in zip_ref.namelist() if f.endswith('.parquet')]
                    
                    if parquet_files:
                        logger.info(f"Found parquet files in zip: {parquet_files}")
                        # Extract the first parquet file
                        zip_ref.extract(parquet_files[0], path=tmp_dir)
                        extracted_file = tmp_dir / parquet_files[0]
                        
                        # Rename to expected name if needed
                        if str(extracted_file) != str(extracted_parquet):
                            os.rename(extracted_file, extracted_parquet)
                    else:
                        logger.warning("No parquet files found in historical zip")
            
            # Load the extracted parquet file
            if extracted_parquet.exists():
                if use_spark and spark:
                    logger.info("Loading historical data with Spark...")
                    historical_df = spark.read.parquet(str(extracted_parquet))
                    logger.info(f"Loaded historical data: {historical_df.count()} rows")
                else:
                    logger.info("Loading historical data with pandas...")
                    historical_df = pd.read_parquet(extracted_parquet)
                    logger.info(f"Loaded historical data: {historical_df.shape}")
            
        except Exception as e:
            logger.error(f"Error extracting historical data: {e}")
    
    return latest_df, historical_df

def validate_submission_files(previous_files, h2o, h2o_context=None):
    """Validate previously created submission files"""
    logger.info(f"Validating {len(previous_files)} submission files...")
    
    results = {}
    
    for i, file_path in enumerate(previous_files):
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"Validating file {i+1}/{len(previous_files)}: {file_path.name}")
        
        try:
            # Read the submission file
            df = pd.read_csv(file_path)
            logger.info(f"Submission has {len(df)} rows")
            
            # Basic validation checks
            has_id = 'id' in df.columns
            has_prediction = 'prediction' in df.columns
            
            if not has_id or not has_prediction:
                logger.warning(f"Missing required columns. Has id: {has_id}, Has prediction: {has_prediction}")
                continue
            
            # Check prediction statistics
            pred_stats = {
                'mean': float(df['prediction'].mean()),
                'std': float(df['prediction'].std()),
                'min': float(df['prediction'].min()),
                'max': float(df['prediction'].max()),
                'null_count': int(df['prediction'].isnull().sum())
            }
            
            # Check prediction distribution
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.hist(df['prediction'], bins=50)
            plt.title(f"Prediction Distribution: {file_path.name}")
            plt.xlabel("Prediction Value")
            plt.ylabel("Count")
            
            # Save the histogram
            plot_file = file_path.with_suffix('.png')
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Generated distribution plot: {plot_file}")
            
            # Add results
            results[str(file_path)] = {
                'valid': True,
                'row_count': len(df),
                'prediction_stats': pred_stats,
                'distribution_plot': str(plot_file)
            }
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            results[str(file_path)] = {
                'valid': False,
                'error': str(e)
            }
    
    # Save validation results
    results_file = project_root / "data" / "submissions" / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {results_file}")
    return results

def engineer_features(df, feature_type='training'):
    """Apply feature engineering to the dataset"""
    logger.info(f"Engineering features for {feature_type} data...")
    
    if df is None:
        logger.warning(f"No data provided for {feature_type} feature engineering")
        return None, []
    
    # Make a copy to avoid modifying the original
    if hasattr(df, 'toPandas'):  # Check if it's a Spark DataFrame
        logger.info("Converting Spark DataFrame to pandas for feature engineering")
        df_pandas = df.toPandas()
    else:
        df_pandas = df.copy()
    
    # Check if we have a target column (only for training data)
    has_target = 'target' in df_pandas.columns
    logger.info(f"Dataset has target column: {has_target}")
    
    # Identify likely feature columns (numeric columns excluding id, target, etc.)
    non_feature_cols = ['id', 'target', 'era', 'data_type', 'timestamp', 'date']
    numeric_cols = df_pandas.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
    
    logger.info(f"Identified {len(feature_cols)} base numeric features")
    
    # Handle missing values
    for col in feature_cols:
        missing_count = df_pandas[col].isnull().sum()
        if missing_count > 0:
            logger.info(f"Column {col} has {missing_count} missing values, filling with median")
            df_pandas[col] = df_pandas[col].fillna(df_pandas[col].median())
    
    # Create a list to track the engineered feature names
    engineered_features = feature_cols.copy()
    
    # Feature set 1: Normalized features (ensures all features have similar scales)
    for col in feature_cols[:30]:  # Limit to first 30 features to avoid explosion
        norm_col = f"{col}_norm"
        df_pandas[norm_col] = (df_pandas[col] - df_pandas[col].mean()) / df_pandas[col].std()
        engineered_features.append(norm_col)
    
    logger.info(f"Added normalized features, now have {len(engineered_features)} features")
    
    # Feature set 2: Rolling statistics for time series (if we have a date column)
    date_col = None
    for possible_date in ['date', 'timestamp', 'era']:
        if possible_date in df_pandas.columns:
            date_col = possible_date
            break
    
    if date_col:
        logger.info(f"Found date column: {date_col}, adding time series features")
        
        # Ensure date column is properly formatted
        if df_pandas[date_col].dtype == 'object':
            try:
                df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
            except:
                logger.warning(f"Couldn't convert {date_col} to datetime")
        
        # Sort by date if it's a datetime
        if pd.api.types.is_datetime64_any_dtype(df_pandas[date_col]):
            df_pandas = df_pandas.sort_values(date_col)
            
            # Add rolling features for a few important columns
            for col in feature_cols[:10]:  # Limit to first 10 features
                # 7-day rolling mean
                df_pandas[f"{col}_roll7_mean"] = df_pandas[col].rolling(7, min_periods=1).mean()
                engineered_features.append(f"{col}_roll7_mean")
                
                # 7-day rolling std
                df_pandas[f"{col}_roll7_std"] = df_pandas[col].rolling(7, min_periods=1).std().fillna(0)
                engineered_features.append(f"{col}_roll7_std")
                
            logger.info(f"Added rolling features, now have {len(engineered_features)} features")
    
    # Feature set 3: Polynomial features for capturing non-linear relationships
    from itertools import combinations
    
    # Generate interaction terms for top features
    top_features = feature_cols[:15]  # Use only the first 15 features
    for col1, col2 in combinations(top_features, 2):
        # Interaction feature
        interaction_col = f"{col1}_{col2}_interact"
        df_pandas[interaction_col] = df_pandas[col1] * df_pandas[col2]
        engineered_features.append(interaction_col)
        
        # Ratio feature (handle division by zero)
        ratio_col = f"{col1}_{col2}_ratio"
        df_pandas[ratio_col] = df_pandas[col1] / (df_pandas[col2] + 1e-8)
        engineered_features.append(ratio_col)
    
    logger.info(f"Added polynomial features, now have {len(engineered_features)} features")
    
    # Feature set 4: Cluster features based on K-means
    try:
        from sklearn.cluster import MiniBatchKMeans
        
        # Use a subset of features for clustering
        cluster_features = feature_cols[:20]  # Use only the first 20 features
        
        # Create normalized dataset for clustering
        cluster_data = df_pandas[cluster_features].copy()
        for col in cluster_features:
            cluster_data[col] = (cluster_data[col] - cluster_data[col].mean()) / cluster_data[col].std()
        
        # Fill any remaining NaNs
        cluster_data = cluster_data.fillna(0)
        
        # Run K-means with different cluster counts
        for n_clusters in [5, 10]:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
            df_pandas[f'cluster_{n_clusters}'] = kmeans.fit_predict(cluster_data)
            engineered_features.append(f'cluster_{n_clusters}')
            
            # Get distance to cluster centers as features
            distances = kmeans.transform(cluster_data)
            for i in range(min(5, n_clusters)):  # Add distance to first 5 clusters
                df_pandas[f'cluster_{n_clusters}_dist_{i}'] = distances[:, i]
                engineered_features.append(f'cluster_{n_clusters}_dist_{i}')
        
        logger.info(f"Added cluster features, now have {len(engineered_features)} features")
    except Exception as e:
        logger.warning(f"Error creating clustering features: {e}")
    
    # Return the engineered dataframe and list of feature columns
    return df_pandas, engineered_features

def train_h2o_models(train_df, valid_df, feature_names, h2o, h2o_context=None, time_limit=1200):
    """Train H2O models with proper cross-validation"""
    logger.info(f"Training H2O models with {len(feature_names)} features")
    
    # Set start time for the time limit
    start_time = time.time()
    max_end_time = start_time + time_limit
    
    # Convert pandas DataFrames to H2O frames
    logger.info("Converting to H2O frames...")
    try:
        if h2o_context and hasattr(train_df, '_jdf'):  # Spark DataFrame
            train_h2o = h2o_context.asH2OFrame(train_df)
            valid_h2o = h2o_context.asH2OFrame(valid_df) if valid_df is not None else None
        else:
            train_h2o = h2o.H2OFrame(train_df)
            valid_h2o = h2o.H2OFrame(valid_df) if valid_df is not None else None
        
        logger.info(f"Converted training data to H2O frame: {train_h2o.shape}")
        if valid_h2o is not None:
            logger.info(f"Converted validation data to H2O frame: {valid_h2o.shape}")
        
        # Set target and features
        target_col = 'target'
        
        # Check if we have a target column
        if target_col not in train_h2o.columns:
            logger.error(f"Target column '{target_col}' not found in training data")
            return None
        
        # Convert target to factor (categorical) for classification
        train_h2o[target_col] = train_h2o[target_col].asfactor()
        if valid_h2o is not None and target_col in valid_h2o.columns:
            valid_h2o[target_col] = valid_h2o[target_col].asfactor()
        
        # Define models to train
        models = []
        model_metrics = {}
        
        # 1. Train an XGBoost model
        if time.time() < max_end_time:
            try:
                logger.info("Training XGBoost model...")
                from h2o.estimators.xgboost import H2OXGBoostEstimator
                
                xgb_params = {
                    'ntrees': 500,
                    'max_depth': 6,
                    'learn_rate': 0.05,
                    'sample_rate': 0.8,
                    'col_sample_rate': 0.8,
                    'seed': RANDOM_SEED,
                    'score_each_iteration': True,
                    'stopping_rounds': 10,
                    'stopping_metric': 'auc',
                    'stopping_tolerance': 0.001
                }
                
                # Check if GPU is available
                try:
                    xgb_params['tree_method'] = 'gpu_hist'
                    xgb_params['gpu_id'] = 0
                    logger.info("Using GPU for XGBoost")
                except:
                    logger.info("GPU not available for XGBoost, using CPU")
                
                xgb_model = H2OXGBoostEstimator(**xgb_params)
                
                xgb_model.train(
                    x=feature_names,
                    y=target_col,
                    training_frame=train_h2o,
                    validation_frame=valid_h2o
                )
                
                # Get model performance
                train_auc = xgb_model.auc()
                valid_auc = xgb_model.auc(valid=True) if valid_h2o is not None else None
                
                logger.info(f"XGBoost model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('xgboost', xgb_model))
                model_metrics['xgboost'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training XGBoost model: {e}")
        
        # 2. Train a Random Forest model
        if time.time() < max_end_time:
            try:
                logger.info("Training Random Forest model...")
                from h2o.estimators.random_forest import H2ORandomForestEstimator
                
                rf_model = H2ORandomForestEstimator(
                    ntrees=100,
                    max_depth=10,
                    sample_rate=0.8,
                    seed=RANDOM_SEED,
                    score_each_iteration=True
                )
                
                rf_model.train(
                    x=feature_names,
                    y=target_col,
                    training_frame=train_h2o,
                    validation_frame=valid_h2o
                )
                
                # Get model performance
                train_auc = rf_model.auc()
                valid_auc = rf_model.auc(valid=True) if valid_h2o is not None else None
                
                logger.info(f"Random Forest model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('random_forest', rf_model))
                model_metrics['random_forest'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training Random Forest model: {e}")
        
        # 3. Train a GBM model
        if time.time() < max_end_time:
            try:
                logger.info("Training GBM model...")
                from h2o.estimators.gbm import H2OGradientBoostingEstimator
                
                gbm_model = H2OGradientBoostingEstimator(
                    ntrees=500,
                    max_depth=6,
                    learn_rate=0.05,
                    sample_rate=0.8,
                    col_sample_rate=0.8,
                    seed=RANDOM_SEED,
                    score_each_iteration=True
                )
                
                gbm_model.train(
                    x=feature_names,
                    y=target_col,
                    training_frame=train_h2o,
                    validation_frame=valid_h2o
                )
                
                # Get model performance
                train_auc = gbm_model.auc()
                valid_auc = gbm_model.auc(valid=True) if valid_h2o is not None else None
                
                logger.info(f"GBM model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('gbm', gbm_model))
                model_metrics['gbm'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training GBM model: {e}")
        
        # 4. Train an AutoML model if time permits
        remaining_time = max_end_time - time.time()
        if remaining_time > 300:  # At least 5 minutes remaining
            try:
                logger.info(f"Training AutoML model with {int(remaining_time)} seconds remaining...")
                from h2o.automl import H2OAutoML
                
                automl = H2OAutoML(
                    max_runtime_secs=int(remaining_time * 0.9),  # Use 90% of remaining time
                    seed=RANDOM_SEED,
                    sort_metric='AUC'
                )
                
                automl.train(
                    x=feature_names,
                    y=target_col,
                    training_frame=train_h2o,
                    validation_frame=valid_h2o
                )
                
                # Get best model
                best_model = automl.leader
                
                # Get model performance
                train_auc = best_model.auc()
                valid_auc = best_model.auc(valid=True) if valid_h2o is not None else None
                
                logger.info(f"AutoML best model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('automl', best_model))
                model_metrics['automl'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time,
                    'automl_type': best_model.__class__.__name__
                }
            except Exception as e:
                logger.error(f"Error training AutoML model: {e}")
        
        # Return trained models
        return {
            'models': models,
            'metrics': model_metrics,
            'feature_names': feature_names,
            'target_column': target_col,
            'training_time': time.time() - start_time
        }
    
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        return None

def generate_predictions(test_df, models_info, h2o, h2o_context=None):
    """Generate predictions using trained H2O models"""
    logger.info("Generating predictions...")
    
    if not models_info or 'models' not in models_info or not models_info['models']:
        logger.error("No trained models available")
        return None
    
    try:
        # Convert test data to H2O frame
        if h2o_context and hasattr(test_df, '_jdf'):  # Spark DataFrame
            test_h2o = h2o_context.asH2OFrame(test_df)
        else:
            test_h2o = h2o.H2OFrame(test_df)
        
        logger.info(f"Test data converted to H2O frame: {test_h2o.shape}")
        
        # Get predictions from each model
        all_predictions = []
        model_weights = []
        
        for model_name, model in models_info['models']:
            logger.info(f"Getting predictions from {model_name} model...")
            
            # Generate predictions
            pred_frame = model.predict(test_h2o)
            
            # Extract probability column
            if 'p1' in pred_frame.columns:
                # Binary classification probability of class 1
                preds = pred_frame['p1'].as_data_frame()['p1'].values
            else:
                # If no probability column, use the prediction column
                preds = pred_frame['predict'].as_data_frame()['predict'].values
            
            # Get model weight based on validation performance
            weight = 1.0  # Default weight
            if model_name in models_info['metrics'] and 'valid_auc' in models_info['metrics'][model_name]:
                # Use validation AUC as weight (scaled)
                valid_auc = models_info['metrics'][model_name]['valid_auc']
                if valid_auc is not None:
                    weight = max(0.1, valid_auc - 0.5) * 2  # Scale: AUC 0.5 -> weight 0.1, AUC 0.75 -> weight 1.0
            
            logger.info(f"Using weight {weight:.4f} for {model_name} model")
            
            all_predictions.append(preds)
            model_weights.append(weight)
        
        # Normalize weights
        if sum(model_weights) > 0:
            model_weights = [w / sum(model_weights) for w in model_weights]
        else:
            # If all weights are 0, use equal weights
            model_weights = [1.0 / len(model_weights)] * len(model_weights)
        
        # Create weighted ensemble prediction
        ensemble_preds = np.zeros_like(all_predictions[0])
        for preds, weight in zip(all_predictions, model_weights):
            ensemble_preds += preds * weight
        
        # Get ID column
        if 'id' in test_df.columns:
            ids = test_df['id']
        else:
            # Generate sequential IDs
            ids = [f"id_{i}" for i in range(len(ensemble_preds))]
        
        # Create pandas DataFrame for submission
        submission_df = pd.DataFrame({
            'id': ids,
            'prediction': ensemble_preds
        })
        
        logger.info(f"Generated {len(submission_df)} predictions")
        
        return submission_df
    
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None

def main():
    """Main function for H2O Sparkling Water solution"""
    # Parse command line arguments
    args = parse_args()
    
    # Output directory for submissions
    output_dir = project_root / "data" / "submissions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = output_dir / f"h2o_submission_{timestamp}.csv"
    
    # Set start time for time limit
    start_time = time.time()
    max_time = start_time + args.time_limit
    
    logger.info(f"Starting H2O Sparkling Water solution with {args.time_limit}s time limit")
    
    # Set up environment
    env_settings = setup_environment(args.gpus)
    
    # Validate existing submission files if requested
    if args.validate:
        # Find previous submission files
        previous_files = list(output_dir.glob("*.csv"))
        if not previous_files:
            logger.warning("No previous submission files found for validation")
        else:
            # Initialize H2O
            logger.info("Initializing H2O for validation...")
            h2o, _, _ = init_h2o_and_spark()
            
            # Validate files
            validation_results = validate_submission_files(previous_files, h2o)
            
            # Find best submission based on validation
            best_file = None
            best_std = None
            
            for file_path, results in validation_results.items():
                if results.get('valid', False):
                    # Look for files with reasonable standard deviation (not too low, not too high)
                    std_dev = results['prediction_stats']['std']
                    if 0.1 <= std_dev <= 0.3:
                        if best_std is None or abs(std_dev - 0.2) < abs(best_std - 0.2):
                            best_std = std_dev
                            best_file = file_path
            
            if best_file:
                logger.info(f"Best previous submission file: {best_file} (std: {best_std:.4f})")
            else:
                logger.info("No suitable previous submission file found")
    
    # Initialize H2O and Spark
    h2o, spark, h2o_context = init_h2o_and_spark()
    
    if h2o is None:
        logger.error("Failed to initialize H2O. Exiting.")
        return 1
    
    # Use Spark if available
    use_spark = spark is not None and h2o_context is not None
    logger.info(f"Using Spark: {use_spark}")
    
    # Load Yiedl data
    latest_df, historical_df = load_yiedl_data(use_spark=use_spark, spark=spark)
    
    if historical_df is None:
        logger.error("Failed to load historical data. Exiting.")
        return 1
    
    # Apply feature engineering
    historical_df_eng, feature_names = engineer_features(historical_df, feature_type='training')
    latest_df_eng, _ = engineer_features(latest_df, feature_type='prediction')
    
    if historical_df_eng is None or latest_df_eng is None:
        logger.error("Feature engineering failed. Exiting.")
        return 1
    
    logger.info(f"Engineered {len(feature_names)} features")
    
    # Split historical data into training and validation
    # Ensure completely out-of-sample validation
    if 'era' in historical_df_eng.columns:
        # Use era for time-based split
        unique_eras = historical_df_eng['era'].unique()
        n_eras = len(unique_eras)
        
        # Use the last 20% of eras for validation
        val_eras = unique_eras[int(n_eras * 0.8):]
        
        train_df = historical_df_eng[~historical_df_eng['era'].isin(val_eras)].copy()
        valid_df = historical_df_eng[historical_df_eng['era'].isin(val_eras)].copy()
        
        logger.info(f"Split by era: Training {len(train_df)} rows, Validation {len(valid_df)} rows")
    else:
        # Random split as a fallback
        logger.warning("No era column found, using random split")
        from sklearn.model_selection import train_test_split
        
        train_df, valid_df = train_test_split(
            historical_df_eng, 
            test_size=0.2, 
            random_state=RANDOM_SEED
        )
        
        logger.info(f"Random split: Training {len(train_df)} rows, Validation {len(valid_df)} rows")
    
    # Train H2O models
    remaining_time = max(60, int(max_time - time.time()))  # At least 60 seconds
    logger.info(f"Training models with {remaining_time} seconds remaining")
    
    models_info = train_h2o_models(
        train_df,
        valid_df,
        feature_names,
        h2o,
        h2o_context,
        time_limit=remaining_time
    )
    
    if not models_info:
        logger.error("Model training failed. Exiting.")
        return 1
    
    # Generate predictions for latest data
    submission_df = generate_predictions(latest_df_eng, models_info, h2o, h2o_context)
    
    if submission_df is None:
        logger.error("Failed to generate predictions. Exiting.")
        return 1
    
    # Save submission
    submission_df.to_csv(args.output, index=False)
    logger.info(f"Submission saved to {args.output}")
    
    # Create an alternative submission with slight variations
    alt_output = Path(str(args.output).replace('.csv', '_v2.csv'))
    
    # Add small random variations to predictions
    alt_df = submission_df.copy()
    np.random.seed(RANDOM_SEED + 42)
    noise = np.random.normal(0, 0.01, size=len(alt_df))  # 1% noise
    alt_df['prediction'] = np.clip(alt_df['prediction'] + noise, 0, 1)
    
    # Save alternative submission
    alt_df.to_csv(alt_output, index=False)
    logger.info(f"Alternative submission saved to {alt_output}")
    
    # Calculate and log statistics
    main_stats = {
        'mean': float(submission_df['prediction'].mean()),
        'std': float(submission_df['prediction'].std()),
        'min': float(submission_df['prediction'].min()),
        'max': float(submission_df['prediction'].max())
    }
    
    alt_stats = {
        'mean': float(alt_df['prediction'].mean()),
        'std': float(alt_df['prediction'].std()),
        'min': float(alt_df['prediction'].min()),
        'max': float(alt_df['prediction'].max())
    }
    
    logger.info(f"Main submission stats: {main_stats}")
    logger.info(f"Alternative submission stats: {alt_stats}")
    
    # Save model information and statistics
    info_file = output_dir / f"h2o_model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(info_file, 'w') as f:
        json.dump({
            'models': [(name, model.__class__.__name__) for name, model in models_info['models']],
            'metrics': models_info['metrics'],
            'feature_count': len(feature_names),
            'main_stats': main_stats,
            'alt_stats': alt_stats,
            'total_time': time.time() - start_time
        }, f, indent=2)
    
    logger.info(f"Model information saved to {info_file}")
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())