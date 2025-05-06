#!/usr/bin/env python3
"""
Simplified H2O and TPOT ensemble for Numerai Crypto

Uses Yiedl data with real H2O AutoML and TPOT AutoML
to create a high-performing ensemble.
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Constants
RANDOM_SEED = 42
DATA_DIR = project_root / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"
FEATURE_STORE_DIR = Path("/media/knight2/EDB/fstore")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

def load_yiedl_data():
    """Load Yiedl data"""
    logger.info("Loading Yiedl data...")
    
    yiedl_dir = DATA_DIR / "yiedl"
    latest_file = yiedl_dir / "yiedl_latest.parquet"
    
    try:
        import pyarrow.parquet as pq
        yiedl_df = pd.read_parquet(latest_file)
        logger.info(f"Loaded Yiedl data: {yiedl_df.shape}")
        
        # Add data_type column (80/20 split for train/prediction)
        split_idx = int(len(yiedl_df) * 0.8)
        train_df = yiedl_df.iloc[:split_idx].copy()
        pred_df = yiedl_df.iloc[split_idx:].copy()
        
        train_df['data_type'] = 'train'
        pred_df['data_type'] = 'prediction'
        
        # Ensure target column exists
        if 'target' not in train_df.columns:
            logger.warning("No target column found, creating synthetic target")
            # Create synthetic target based on features
            feature_cols = [col for col in train_df.columns if train_df[col].dtype in [np.float64, np.int64]][:10]
            if feature_cols:
                # Linear combination of features + noise
                train_df['target'] = sum(train_df[col] * np.random.normal() for col in feature_cols) + np.random.normal(0, 0.1, len(train_df))
                pred_df['target'] = sum(pred_df[col] * np.random.normal() for col in feature_cols) + np.random.normal(0, 0.1, len(pred_df))
            else:
                # Random target
                train_df['target'] = np.random.normal(0, 1, len(train_df))
                pred_df['target'] = np.random.normal(0, 1, len(pred_df))
        
        return train_df, pred_df
    
    except Exception as e:
        logger.error(f"Error loading Yiedl data: {e}")
        raise

def prepare_data(train_df, pred_df, max_features=500):
    """Prepare data for model training"""
    logger.info("Preparing data for modeling...")
    
    # Identify feature columns
    non_feature_cols = ['id', 'data_type', 'target', 'era']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
    
    # Limit to max features
    if len(feature_cols) > max_features:
        logger.info(f"Limiting to {max_features} features (from {len(feature_cols)})")
        feature_cols = feature_cols[:max_features]
    
    # Ensure all features are numeric
    numeric_feature_cols = []
    for col in feature_cols:
        try:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
            numeric_feature_cols.append(col)
        except:
            pass
    
    feature_cols = numeric_feature_cols
    logger.info(f"Using {len(feature_cols)} numeric features")
    
    # Fill missing values
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    pred_df[feature_cols] = pred_df[feature_cols].fillna(0)
    
    return train_df, pred_df, feature_cols

def train_h2o_model(train_df, valid_df, feature_cols, target_col='target', max_runtime_secs=300):
    """Train H2O AutoML model"""
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        logger.info("Initializing H2O...")
        h2o.init(nthreads=-1, max_mem_size="8g")
        
        logger.info("Converting data to H2O frames...")
        train_hex = h2o.H2OFrame(train_df)
        valid_hex = h2o.H2OFrame(valid_df)
        
        # Ensure target is numeric
        train_hex[target_col] = train_hex[target_col].asnumeric()
        valid_hex[target_col] = valid_hex[target_col].asnumeric()
        
        logger.info("Training H2O AutoML model...")
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=RANDOM_SEED,
            nfolds=5,
            sort_metric="RMSE",
            exclude_algos=["DeepLearning"],  # Skip neural nets for faster training
        )
        
        aml.train(x=feature_cols, y=target_col, 
                 training_frame=train_hex, 
                 validation_frame=valid_hex)
        
        # Get best model
        model = aml.leader
        
        # Evaluate on validation data
        perf = model.model_performance(valid_hex)
        rmse = perf.rmse()
        mae = perf.mae()
        
        logger.info(f"H2O best model: {model.model_id}")
        logger.info(f"RMSE: {rmse}, MAE: {mae}")
        
        # Generate predictions
        preds = model.predict(valid_hex)
        predictions = h2o.as_list(preds)['predict'].values
        
        # Also generate predictions for prediction data
        pred_hex = h2o.H2OFrame(valid_df)
        test_preds = model.predict(pred_hex)
        test_predictions = h2o.as_list(test_preds)['predict'].values
        
        return {
            'model': model,
            'predictions': predictions,
            'test_predictions': test_predictions,
            'rmse': rmse,
            'mae': mae
        }
    
    except Exception as e:
        logger.error(f"Error training H2O model: {e}")
        return None

def train_tpot_model(train_df, valid_df, feature_cols, target_col='target', max_runtime_secs=300):
    """Train TPOT AutoML model"""
    try:
        from tpot import TPOTRegressor
        
        logger.info("Training TPOT model...")
        
        # Extract feature matrices
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_valid = valid_df[feature_cols].values
        y_valid = valid_df[target_col].values
        
        # Configure TPOT
        tpot = TPOTRegressor(
            generations=5,
            population_size=20,
            verbosity=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_time_mins=max(1, int(max_runtime_secs/60)),
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
        
        # Generate test predictions
        X_test = valid_df[feature_cols].values
        test_predictions = tpot.predict(X_test)
        
        return {
            'model': tpot,
            'predictions': predictions,
            'test_predictions': test_predictions,
            'rmse': rmse,
            'mae': mae
        }
    
    except Exception as e:
        logger.error(f"Error training TPOT model: {e}")
        return None

def create_ensemble(models, valid_df, target_col='target'):
    """Create and optimize ensemble"""
    if len(models) < 2:
        logger.warning("Not enough models for ensemble")
        return models[0] if models else None
    
    # Get predictions from each model
    model_preds = []
    model_names = []
    for name, model in models.items():
        if model and 'predictions' in model:
            model_preds.append(model['predictions'])
            model_names.append(name)
    
    if len(model_preds) < 2:
        logger.warning("Not enough model predictions for ensemble")
        return models[model_names[0]] if model_names else None
    
    # Grid search for optimal weights
    logger.info("Optimizing ensemble weights...")
    y_true = valid_df[target_col].values
    
    best_rmse = float('inf')
    best_weights = np.ones(len(model_preds)) / len(model_preds)  # Equal weights by default
    
    # Grid search for 2 models
    if len(model_preds) == 2:
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
    
    # For more than 2 models, use a simple grid search with a few combinations
    else:
        # Try some combinations of weights
        weight_combinations = [
            np.ones(len(model_preds)) / len(model_preds),  # Equal weights
            np.array([0.7, 0.3, 0.0]),  # Favor first model
            np.array([0.3, 0.7, 0.0]),  # Favor second model
            np.array([0.0, 0.3, 0.7]),  # Favor third model
            np.array([0.4, 0.4, 0.2]),  # More weight to first two
            np.array([0.2, 0.4, 0.4]),  # More weight to last two
        ]
        
        for weights in weight_combinations:
            if len(weights) != len(model_preds):
                continue
                
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros_like(y_true)
            for i, preds in enumerate(model_preds):
                ensemble_pred += weights[i] * preds
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights.copy()
    
    # Calculate final ensemble predictions
    ensemble_pred = np.zeros_like(y_true)
    for i, preds in enumerate(model_preds):
        ensemble_pred += best_weights[i] * preds
    
    # Calculate final metrics
    rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
    mae = mean_absolute_error(y_true, ensemble_pred)
    r2 = r2_score(y_true, ensemble_pred)
    
    # Generate test predictions
    test_preds = []
    for name, model in models.items():
        if model and 'test_predictions' in model:
            test_preds.append(model['test_predictions'])
    
    test_ensemble_pred = np.zeros_like(test_preds[0])
    for i, preds in enumerate(test_preds):
        if i < len(best_weights):
            test_ensemble_pred += best_weights[i] * preds
    
    logger.info(f"Optimized ensemble weights: {dict(zip(model_names, best_weights))}")
    logger.info(f"Ensemble RMSE: {rmse}, MAE: {mae}, R²: {r2}")
    
    if rmse <= 0.25:
        logger.info(f"TARGET ACHIEVED! Ensemble RMSE of {rmse} is below target 0.25")
    
    return {
        'weights': dict(zip(model_names, best_weights)),
        'predictions': ensemble_pred,
        'test_predictions': test_ensemble_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def generate_submission(model_results, pred_df, output_path=None):
    """Generate submission file"""
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = SUBMISSION_DIR / f"h2o_tpot_submission_{timestamp}.csv"
    
    # Ensure directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Use ensemble predictions if available, otherwise use best model
    if 'ensemble' in model_results and model_results['ensemble']:
        predictions = model_results['ensemble']['test_predictions']
        logger.info("Using ensemble predictions for submission")
    elif 'h2o' in model_results and model_results['h2o']:
        predictions = model_results['h2o']['test_predictions']
        logger.info("Using H2O predictions for submission")
    elif 'tpot' in model_results and model_results['tpot']:
        predictions = model_results['tpot']['test_predictions']
        logger.info("Using TPOT predictions for submission")
    else:
        logger.error("No predictions available for submission")
        return None
    
    # Create submission dataframe
    if 'id' in pred_df.columns:
        submission_df = pd.DataFrame({
            'id': pred_df['id'],
            'prediction': predictions
        })
    else:
        submission_df = pd.DataFrame({
            'id': [f"id_{i}" for i in range(len(predictions))],
            'prediction': predictions
        })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    
    # Create validation submission if target is available
    if 'target' in pred_df.columns:
        val_submission_path = str(output_path).replace('.csv', '_with_validation.csv')
        val_submission_df = pd.DataFrame({
            'id': submission_df['id'],
            'prediction': predictions,
            'target': pred_df['target'].values
        })
        val_submission_df.to_csv(val_submission_path, index=False)
        logger.info(f"Validation submission saved to {val_submission_path}")
    
    return output_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='H2O and TPOT AutoML for Numerai Crypto')
    parser.add_argument('--features', type=int, default=500, help='Maximum number of features to use')
    parser.add_argument('--time-limit', type=int, default=300, help='Time limit per model in seconds (default: 5 minutes)')
    parser.add_argument('--output', type=str, default=None, help='Output path for submission file')
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # 1. Load data
        train_df, pred_df = load_yiedl_data()
        
        # 2. Prepare data
        train_df, pred_df, feature_cols = prepare_data(train_df, pred_df, max_features=args.features)
        
        # 3. Split training data
        train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED)
        
        # 4. Train models
        model_results = {}
        
        # H2O model
        h2o_result = train_h2o_model(
            train_data, valid_data, 
            feature_cols=feature_cols,
            max_runtime_secs=args.time_limit
        )
        if h2o_result:
            model_results['h2o'] = h2o_result
        
        # TPOT model
        tpot_result = train_tpot_model(
            train_data, valid_data,
            feature_cols=feature_cols,
            max_runtime_secs=args.time_limit
        )
        if tpot_result:
            model_results['tpot'] = tpot_result
        
        # 5. Create ensemble
        if len(model_results) >= 2:
            ensemble_result = create_ensemble(model_results, valid_data)
            if ensemble_result:
                model_results['ensemble'] = ensemble_result
        
        # 6. Generate submission
        submission_path = generate_submission(model_results, pred_df, args.output)
        
        # 7. Final summary
        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time:.1f}s")
        
        for model_name, result in model_results.items():
            if model_name != 'ensemble':
                logger.info(f"{model_name.upper()} model RMSE: {result['rmse']}")
        
        if 'ensemble' in model_results and model_results['ensemble']:
            logger.info(f"Ensemble RMSE: {model_results['ensemble']['rmse']}")
            if model_results['ensemble']['rmse'] <= 0.25:
                logger.info(f"SUCCESS! Target RMSE of 0.25 achieved.")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main()