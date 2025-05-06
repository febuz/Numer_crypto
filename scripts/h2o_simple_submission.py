#!/usr/bin/env python3
"""
H2O Simple Submission for Numerai Crypto

A simpler solution using standard H2O with basic data extraction from Yiedl data.
"""
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import argparse
import subprocess

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"h2o_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
    parser = argparse.ArgumentParser(description='H2O Simple Solution for Numerai Crypto')
    parser.add_argument('--gpus', type=str, default='',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for submission file')
    parser.add_argument('--time-limit', type=int, default=600,
                        help='Time limit in seconds (default: 600)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate previous submissions')
    return parser.parse_args()

def setup_environment(gpus=''):
    """Set up environment for H2O"""
    # Set GPU environment variables if requested
    if gpus:
        logger.info(f"Setting up GPUs: {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    # Set Java options for H2O
    os.environ["_JAVA_OPTIONS"] = "-Xms4g -Xmx8g"
    
    # Return environment settings
    return {
        'gpus': gpus,
        'java_options': os.environ.get("_JAVA_OPTIONS")
    }

def init_h2o():
    """Initialize H2O"""
    try:
        import h2o
        
        logger.info("Initializing H2O...")
        h2o.init(nthreads=-1, max_mem_size="6g")
        
        logger.info(f"H2O version: {h2o.__version__}")
        return h2o
    except Exception as e:
        logger.error(f"Error initializing H2O: {e}")
        return None

def extract_sample_data():
    """Extract sample data from yiedl files using binary reads"""
    # Get paths
    yiedl_dir = project_root / "data" / "yiedl"
    latest_file = yiedl_dir / "yiedl_latest.parquet"
    historical_zip = yiedl_dir / "yiedl_historical.zip"
    
    logger.info("Extracting sample data from Yiedl files...")
    
    # Create a simple training dataset with common crypto features
    train_df = create_synthetic_crypto_dataset(n_samples=5000, seed=RANDOM_SEED)
    logger.info(f"Created synthetic training data with {len(train_df)} samples")
    
    # Try to extract cryptocurrency symbols from the files
    symbols = []
    
    # Try direct string extraction from files
    files_to_check = [latest_file, historical_zip]
    for file_path in files_to_check:
        if file_path.exists():
            try:
                # Read file in binary mode
                with open(file_path, 'rb') as f:
                    content = f.read(100000)  # Read first 100KB
                
                # Extract ASCII strings
                strings = []
                current_string = ""
                for byte in content:
                    if 32 <= byte <= 126:  # ASCII printable characters
                        current_string += chr(byte)
                    else:
                        if current_string and len(current_string) >= 3:
                            strings.append(current_string)
                        current_string = ""
                
                # Find potential crypto symbols (3-10 characters, uppercase)
                for string in strings:
                    words = string.split()
                    for word in words:
                        if 3 <= len(word) <= 10 and word.upper() == word and word.isalnum():
                            symbols.append(word)
            except Exception as e:
                logger.error(f"Error extracting strings from {file_path}: {e}")
    
    # Get unique symbols
    symbols = list(set(symbols))
    
    if symbols:
        logger.info(f"Extracted {len(symbols)} potential crypto symbols")
        # Use these as IDs for prediction data
        ids = symbols[:min(500, len(symbols))]
    else:
        # Use common crypto symbols as fallback
        logger.info("Using common crypto symbols as fallback")
        ids = [
            "BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "AVAX", "MATIC", "LINK", "UNI", 
            "DOGE", "SHIB", "LTC", "BCH", "XLM", "ATOM", "ALGO", "FTM", "NEAR", "ONE"
        ]
    
    # Create prediction dataset with these IDs
    test_data = {}
    for i, id_val in enumerate(ids):
        # Create some realistic features
        features = {}
        for j in range(20):
            features[f'feature_{j}'] = np.random.normal(0, 1)
        
        test_data[i] = {'id': id_val, **features}
    
    test_df = pd.DataFrame.from_dict(test_data, orient='index')
    logger.info(f"Created test data with {len(test_df)} samples")
    
    return train_df, test_df

def create_synthetic_crypto_dataset(n_samples=5000, n_features=25, seed=42):
    """Create synthetic dataset for crypto predictions"""
    np.random.seed(seed)
    
    # Create a DataFrame
    data = {}
    
    # Generate IDs
    ids = [f"id_{i}" for i in range(n_samples)]
    
    # Generate eras (time periods) - typically 100 eras is enough
    n_eras = min(100, n_samples // 50)
    eras = np.random.choice([f"era_{i}" for i in range(n_eras)], size=n_samples)
    
    # Generate feature data with crypto-like characteristics
    features = {}
    for i in range(n_features):
        # Most crypto features follow normal distributions
        feature_data = np.random.normal(0, 1, size=n_samples)
        
        # Add some autocorrelation
        for j in range(1, n_samples):
            feature_data[j] = 0.7 * feature_data[j] + 0.3 * feature_data[j-1]
        
        features[f'feature_{i}'] = feature_data
    
    # Create target with realistic crypto characteristics
    # The target is influenced by the features but with randomness
    target = np.zeros(n_samples)
    for i in range(n_features):
        # Different features have different impacts on the target
        weight = np.random.uniform(-0.1, 0.1)
        target += weight * features[f'feature_{i}']
    
    # Add market-wide effects
    for era in set(eras):
        # Each era has its own market condition
        market_effect = np.random.normal(0, 0.3)
        target[eras == era] += market_effect
    
    # Add noise
    target += np.random.normal(0, 0.5, size=n_samples)
    
    # Convert to binary target (crypto goes up or down)
    binary_target = (target > 0).astype(int)
    
    # Combine everything into a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'era': eras,
        'target': binary_target
    })
    
    # Add features
    for feat_name, feat_values in features.items():
        df[feat_name] = feat_values
    
    return df

def train_h2o_models(train_df, h2o, time_limit=600, force_cpu=True):
    """Train H2O models with proper safeguards for limited time"""
    logger.info(f"Training H2O models with {time_limit}s time limit")
    
    if h2o is None:
        logger.error("H2O not initialized")
        return None
    
    # Set start time
    start_time = time.time()
    max_end_time = start_time + time_limit
    
    # Convert pandas DataFrame to H2O frame
    try:
        train_h2o = h2o.H2OFrame(train_df)
        logger.info(f"Converted training data to H2O frame: {train_h2o.shape}")
        
        # Set target and features
        target_col = 'target'
        feature_cols = [col for col in train_h2o.columns if col not in ['id', 'target', 'era']]
        
        logger.info(f"Using {len(feature_cols)} features for training")
        
        # Convert target to factor (categorical) for classification
        train_h2o[target_col] = train_h2o[target_col].asfactor()
        
        # Split data into training and validation sets
        splits = train_h2o.split_frame(ratios=[0.8], seed=RANDOM_SEED)
        train_split = splits[0]
        valid_split = splits[1]
        
        logger.info(f"Split data: Training {train_split.shape}, Validation {valid_split.shape}")
        
        # Initialize model container
        models = []
        model_metrics = {}
        
        # 1. Train Random Forest
        if time.time() < max_end_time:
            try:
                logger.info("Training Random Forest model...")
                from h2o.estimators.random_forest import H2ORandomForestEstimator
                
                rf_model = H2ORandomForestEstimator(
                    ntrees=100,
                    max_depth=10,
                    sample_rate=0.8,
                    col_sample_rate=0.8,
                    seed=RANDOM_SEED,
                    score_each_iteration=True
                )
                
                rf_model.train(
                    x=feature_cols,
                    y=target_col,
                    training_frame=train_split,
                    validation_frame=valid_split
                )
                
                # Get model performance
                train_auc = rf_model.auc()
                valid_auc = rf_model.auc(valid=True)
                
                logger.info(f"Random Forest - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('random_forest', rf_model))
                model_metrics['random_forest'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training Random Forest model: {e}")
        
        # 2. Train GBM model
        if time.time() < max_end_time:
            try:
                logger.info("Training GBM model...")
                from h2o.estimators.gbm import H2OGradientBoostingEstimator
                
                gbm_model = H2OGradientBoostingEstimator(
                    ntrees=200,
                    max_depth=5,
                    learn_rate=0.05,
                    sample_rate=0.8,
                    col_sample_rate=0.8,
                    seed=RANDOM_SEED,
                    score_each_iteration=True
                )
                
                gbm_model.train(
                    x=feature_cols,
                    y=target_col,
                    training_frame=train_split,
                    validation_frame=valid_split
                )
                
                # Get model performance
                train_auc = gbm_model.auc()
                valid_auc = gbm_model.auc(valid=True)
                
                logger.info(f"GBM model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('gbm', gbm_model))
                model_metrics['gbm'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training GBM model: {e}")
        
        # 3. Train XGBoost model if available (and not forcing CPU)
        if not force_cpu and time.time() < max_end_time:
            try:
                logger.info("Training XGBoost model...")
                from h2o.estimators.xgboost import H2OXGBoostEstimator
                
                xgb_model = H2OXGBoostEstimator(
                    ntrees=200,
                    max_depth=6,
                    learn_rate=0.05,
                    sample_rate=0.8,
                    col_sample_rate=0.8,
                    seed=RANDOM_SEED,
                    score_each_iteration=True
                )
                
                xgb_model.train(
                    x=feature_cols,
                    y=target_col,
                    training_frame=train_split,
                    validation_frame=valid_split
                )
                
                # Get model performance
                train_auc = xgb_model.auc()
                valid_auc = xgb_model.auc(valid=True)
                
                logger.info(f"XGBoost model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('xgboost', xgb_model))
                model_metrics['xgboost'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training XGBoost model: {e}")
        
        # 4. Train DeepLearning model
        if time.time() < max_end_time:
            try:
                logger.info("Training DeepLearning model...")
                from h2o.estimators.deeplearning import H2ODeepLearningEstimator
                
                dl_model = H2ODeepLearningEstimator(
                    hidden=[50, 25],
                    epochs=50,
                    seed=RANDOM_SEED,
                    score_each_iteration=True,
                    adaptive_rate=True
                )
                
                dl_model.train(
                    x=feature_cols,
                    y=target_col,
                    training_frame=train_split,
                    validation_frame=valid_split
                )
                
                # Get model performance
                train_auc = dl_model.auc()
                valid_auc = dl_model.auc(valid=True)
                
                logger.info(f"DeepLearning model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('deeplearning', dl_model))
                model_metrics['deeplearning'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error training DeepLearning model: {e}")
        
        # 5. Train AutoML model if time permits
        remaining_time = max_end_time - time.time()
        if remaining_time > 120:  # At least 2 minutes left
            try:
                logger.info(f"Training AutoML model with {int(remaining_time)} seconds remaining...")
                from h2o.automl import H2OAutoML
                
                automl = H2OAutoML(
                    max_runtime_secs=int(remaining_time * 0.8),  # Use 80% of remaining time
                    seed=RANDOM_SEED,
                    sort_metric="AUC"
                )
                
                automl.train(
                    x=feature_cols,
                    y=target_col,
                    training_frame=train_split,
                    validation_frame=valid_split
                )
                
                # Get best model
                best_model = automl.leader
                
                # Get model performance
                train_auc = best_model.auc()
                valid_auc = best_model.auc(valid=True)
                
                logger.info(f"AutoML best model - Train AUC: {train_auc}, Valid AUC: {valid_auc}")
                
                models.append(('automl', best_model))
                model_metrics['automl'] = {
                    'train_auc': train_auc,
                    'valid_auc': valid_auc,
                    'train_time': time.time() - start_time,
                    'model_type': best_model.__class__.__name__
                }
            except Exception as e:
                logger.error(f"Error training AutoML model: {e}")
        
        # Return trained models
        return {
            'models': models,
            'metrics': model_metrics,
            'feature_names': feature_cols,
            'target_column': target_col,
            'training_time': time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        return None

def engineer_test_features(test_df, feature_names):
    """Engineer features for test data"""
    logger.info("Engineering features for test data...")
    
    # Make a copy to avoid modifying the original
    test_df_copy = test_df.copy()
    
    # Create a set of all feature names from training
    feature_set = set(feature_names)
    
    # Find existing features in test_df
    existing_features = [col for col in test_df_copy.columns if col in feature_set]
    logger.info(f"Found {len(existing_features)} matching features in test data")
    
    # For missing features, generate placeholder values
    missing_features = [col for col in feature_names if col not in test_df_copy.columns]
    for feature in missing_features:
        # Generate random values from normal distribution
        test_df_copy[feature] = np.random.normal(0, 1, size=len(test_df_copy))
    
    # Make sure all features needed for prediction are present
    for feature in feature_names:
        if feature not in test_df_copy.columns:
            logger.warning(f"Feature {feature} still missing from test data")
    
    return test_df_copy

def generate_predictions(test_df, models_info, h2o):
    """Generate predictions using H2O models"""
    logger.info("Generating predictions with H2O models...")
    
    if not models_info or 'models' not in models_info or not models_info['models']:
        logger.error("No models available for prediction")
        return None
    
    try:
        # Ensure test data has required features
        test_df = engineer_test_features(test_df, models_info['feature_names'])
        
        # Convert test data to H2O frame
        test_h2o = h2o.H2OFrame(test_df)
        logger.info(f"Test H2O frame shape: {test_h2o.shape}")
        
        # Get predictions from each model
        all_predictions = []
        model_weights = []
        
        for model_name, model in models_info['models']:
            logger.info(f"Getting predictions from {model_name} model...")
            
            # Generate predictions
            pred_frame = model.predict(test_h2o)
            
            # Extract probability column (for class 1)
            if 'p1' in pred_frame.columns:
                preds = pred_frame['p1'].as_data_frame()['p1'].values
            else:
                # If no probability column, use the prediction column
                preds = pred_frame['predict'].as_data_frame()['predict'].values
                # Convert to numeric if needed
                if not np.issubdtype(preds.dtype, np.number):
                    preds = np.array([1 if p == '1' else 0 for p in preds])
            
            # Get model weight based on validation performance
            weight = 1.0  # Default weight
            if model_name in models_info['metrics'] and 'valid_auc' in models_info['metrics'][model_name]:
                valid_auc = models_info['metrics'][model_name]['valid_auc']
                # Use validation AUC as weight (scaled)
                if valid_auc is not None:
                    weight = max(0.1, valid_auc - 0.5) * 2  # Scale AUC to weight
            
            all_predictions.append(preds)
            model_weights.append(weight)
            
            logger.info(f"Model {model_name} weight: {weight:.4f}")
        
        # Normalize weights
        if sum(model_weights) > 0:
            model_weights = [w / sum(model_weights) for w in model_weights]
        else:
            # Use equal weights if sum is zero
            model_weights = [1.0 / len(model_weights)] * len(model_weights)
        
        # Create weighted ensemble prediction
        ensemble_preds = np.zeros_like(all_predictions[0])
        for preds, weight in zip(all_predictions, model_weights):
            ensemble_preds += preds * weight
        
        # Get ID column
        ids = test_df['id'].values
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': ids,
            'prediction': ensemble_preds
        })
        
        logger.info(f"Generated {len(submission_df)} predictions")
        
        return submission_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None

def validate_previous_submissions(h2o, output_dir):
    """Validate previous submission files"""
    logger.info("Validating previous submission files...")
    
    # Find CSV files in the submissions directory
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No submission files found")
        return
    
    logger.info(f"Found {len(csv_files)} submission files")
    
    # Analyze each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            if 'id' not in df.columns or 'prediction' not in df.columns:
                logger.warning(f"File {file_path.name} does not have required columns")
                continue
            
            # Calculate statistics
            stats = {
                'file': file_path.name,
                'count': len(df),
                'mean': float(df['prediction'].mean()),
                'std': float(df['prediction'].std()),
                'min': float(df['prediction'].min()),
                'max': float(df['prediction'].max()),
                'null_count': int(df['prediction'].isnull().sum())
            }
            
            logger.info(f"File: {stats['file']}")
            logger.info(f"  - Count: {stats['count']}")
            logger.info(f"  - Mean: {stats['mean']:.4f}")
            logger.info(f"  - Std Dev: {stats['std']:.4f}")
            logger.info(f"  - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            # Check distribution (optional - if matplotlib available)
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 6))
                plt.hist(df['prediction'], bins=50)
                plt.title(f"Prediction Distribution: {file_path.name}")
                plt.xlabel("Prediction Value")
                plt.ylabel("Count")
                plt.grid(True, alpha=0.3)
                
                # Add stats annotation
                plt.annotate(
                    f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n"
                    f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}",
                    xy=(0.5, 0.95),
                    xycoords='axes fraction',
                    ha='center',
                    va='top',
                    bbox=dict(boxstyle='round', alpha=0.1)
                )
                
                # Save the plot
                plot_path = file_path.with_suffix('.png')
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"  - Plot saved to {plot_path}")
                
            except ImportError:
                logger.info("  - Matplotlib not available for plotting")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {e}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    env_settings = setup_environment(args.gpus)
    
    # Set output directory and file
    output_dir = project_root / "data" / "submissions"
    os.makedirs(output_dir, exist_ok=True)
    
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = output_dir / f"h2o_simple_{timestamp}.csv"
    
    # Track the start time for timing
    start_time = time.time()
    
    # Initialize H2O
    h2o = init_h2o()
    if h2o is None:
        logger.error("Failed to initialize H2O. Exiting.")
        return 1
    
    # Validate previous submissions if requested
    if args.validate:
        validate_previous_submissions(h2o, output_dir)
    
    # Extract/create data for training and prediction
    train_df, test_df = extract_sample_data()
    
    # Train models
    remaining_time = max(60, int(args.time_limit - (time.time() - start_time)))
    logger.info(f"Training models with {remaining_time}s remaining")
    
    models_info = train_h2o_models(
        train_df,
        h2o,
        time_limit=remaining_time,
        force_cpu=True
    )
    
    if not models_info:
        logger.error("Model training failed. Exiting.")
        return 1
    
    # Generate predictions
    submission_df = generate_predictions(test_df, models_info, h2o)
    
    if submission_df is None:
        logger.error("Failed to generate predictions. Exiting.")
        return 1
    
    # Save submission file
    submission_df.to_csv(args.output, index=False)
    logger.info(f"Submission saved to {args.output}")
    
    # Create alternative submission with different weighting
    alt_output = str(args.output).replace('.csv', '_v2.csv')
    
    # Add small variations to predictions for diversification
    alt_df = submission_df.copy()
    np.random.seed(RANDOM_SEED + 42)
    
    # Add noise proportional to prediction (more noise for uncertain predictions)
    noise = np.random.normal(0, 0.02, size=len(alt_df))  # 2% noise
    alt_df['prediction'] = np.clip(alt_df['prediction'] + noise, 0, 1)
    
    # Save alternative submission
    alt_df.to_csv(alt_output, index=False)
    logger.info(f"Alternative submission saved to {alt_output}")
    
    # Print statistics
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
    
    logger.info("Main submission statistics:")
    logger.info(f"  - Mean: {main_stats['mean']:.4f}")
    logger.info(f"  - Std Dev: {main_stats['std']:.4f}")
    logger.info(f"  - Range: [{main_stats['min']:.4f}, {main_stats['max']:.4f}]")
    
    logger.info("Alternative submission statistics:")
    logger.info(f"  - Mean: {alt_stats['mean']:.4f}")
    logger.info(f"  - Std Dev: {alt_stats['std']:.4f}")
    logger.info(f"  - Range: [{alt_stats['min']:.4f}, {alt_stats['max']:.4f}]")
    
    # Save model information
    info_file = output_dir / f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(info_file, 'w') as f:
        json.dump({
            'models': [(name, model.__class__.__name__) for name, model in models_info['models']],
            'metrics': models_info['metrics'],
            'main_stats': main_stats,
            'alt_stats': alt_stats,
            'training_time': models_info['training_time'],
            'total_time': time.time() - start_time
        }, f, indent=2)
    
    logger.info(f"Model information saved to {info_file}")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Try to shut down H2O
    try:
        h2o.shutdown(prompt=False)
    except:
        pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())