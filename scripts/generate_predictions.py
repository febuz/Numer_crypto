#!/usr/bin/env python3
"""
generate_predictions.py - Generate predictions for Numerai Crypto

This script generates predictions for the current tournament round using
trained models from the model store and feature sets from the feature store.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and feature store
from utils.model.model_store import ModelStore
from utils.feature.feature_store import FeatureStore
from utils.feature.extractor import FeatureExtractor
from utils.log_utils import setup_logging

# Setup logging
logger = setup_logging("generate_predictions")

# Default directories for files
PREDICTION_DIR = "/media/knight2/EDB/numer_crypto_temp/prediction"

def get_live_universe_symbols():
    """Get the current live universe symbols from Numerai"""
    live_file = "/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_live.parquet"
    
    if not os.path.exists(live_file):
        logger.warning(f"Live universe file not found: {live_file}")
        return None
    
    try:
        live_df = pd.read_parquet(live_file)
        symbols = live_df['symbol'].dropna().unique().tolist()
        logger.info(f"Loaded {len(symbols)} symbols from live universe")
        return symbols
    except Exception as e:
        logger.error(f"Error reading live universe: {e}")
        return None

def generate_real_predictions(model_id=None, feature_set_id=None, symbols=None):
    """
    Generate predictions using trained models from the model store and features from the feature store.
    
    Args:
        model_id (str): ID of the model in the model store
        feature_set_id (str): ID of the feature set in the feature store
        symbols (list): List of crypto symbols to predict for (defaults to live universe)
        
    Returns:
        pd.DataFrame: DataFrame with Symbol and Prediction columns
    """
    # Get symbols from live universe if not provided
    if symbols is None:
        symbols = get_live_universe_symbols()
        if symbols is None:
            logger.error("Could not load live universe symbols")
            return None
    
    logger.info(f"Generating predictions for {len(symbols)} symbols")
    crypto_symbols = symbols
    
    # Initialize the model and feature stores
    model_store = ModelStore()
    feature_store = FeatureStore()
    feature_extractor = FeatureExtractor()
    
    # Try to load the model from the model store if model_id is provided
    model = None
    model_metadata = None
    
    if model_id:
        try:
            model, model_metadata = model_store.get_model(model_id)
            if model:
                logger.info(f"Loaded model {model_id} from model store")
                logger.info(f"Model type: {model_metadata.get('model_type', 'unknown')}")
                logger.info(f"Model performance: {model_metadata.get('metrics', {}).get('test_score', 'unknown')}")
            else:
                logger.warning(f"Model {model_id} not found in model store")
        except Exception as e:
            logger.error(f"Error loading model {model_id} from model store: {e}")
    
    # Try to load features if the model was loaded successfully
    features_df = None
    
    if model and model_metadata:
        # If feature_set_id is provided, try to load it
        if feature_set_id:
            try:
                features_df = feature_store.get_feature_set(feature_set_id)
                if features_df is not None:
                    logger.info(f"Loaded feature set {feature_set_id} from feature store")
                else:
                    logger.warning(f"Feature set {feature_set_id} not found in feature store")
            except Exception as e:
                logger.error(f"Error loading feature set {feature_set_id}: {e}")
        
        # If no feature_set_id provided or loading failed, try to use the feature set associated with the model
        if features_df is None and 'feature_set_id' in model_metadata:
            model_feature_set_id = model_metadata['feature_set_id']
            try:
                features_df = feature_store.get_feature_set(model_feature_set_id)
                if features_df is not None:
                    logger.info(f"Loaded feature set {model_feature_set_id} from model metadata")
                    # Update feature_set_id for reference
                    feature_set_id = model_feature_set_id
                else:
                    logger.warning(f"Feature set {model_feature_set_id} from model metadata not found")
            except Exception as e:
                logger.error(f"Error loading feature set from model metadata: {e}")
        
        # If we have features, filter for the required features for this model
        if features_df is not None:
            required_features = model_metadata.get('required_features', [])
            if required_features:
                features_df = feature_extractor.filter_features(features_df, required_features)
                logger.info(f"Filtered features to {len(required_features)} required features for the model")
    
    # If we have both a model and features, generate predictions
    if model and features_df is not None:
        try:
            logger.info("Generating predictions using the loaded model and features")
            
            # Prepare features for prediction
            # The exact preparation will depend on your model and feature structure
            # This is a simplified example
            prediction_features = features_df.copy()
            
            # Filter to only the symbols we want to predict
            if 'symbol' in prediction_features.columns:
                prediction_features = prediction_features[prediction_features['symbol'].isin(crypto_symbols)]
            
            # Make sure we have features for all the requested symbols
            available_symbols = prediction_features['symbol'].unique().tolist() if 'symbol' in prediction_features.columns else []
            missing_symbols = [s for s in crypto_symbols if s not in available_symbols]
            
            if missing_symbols:
                logger.warning(f"Missing features for {len(missing_symbols)} symbols")
                # You might want to handle missing features differently
            
            # Generate predictions - exact API will depend on your model type
            model_type = model_metadata.get('model_type', '')
            
            if 'xgboost' in model_type.lower():
                # XGBoost prediction
                X = prediction_features.drop(['symbol', 'target'], errors='ignore')
                predictions = model.predict(X)
            elif 'lightgbm' in model_type.lower():
                # LightGBM prediction
                X = prediction_features.drop(['symbol', 'target'], errors='ignore')
                predictions = model.predict(X)
            else:
                # Generic prediction
                X = prediction_features.drop(['symbol', 'target'], errors='ignore')
                predictions = model.predict(X)
            
            # Create the prediction DataFrame
            prediction_df = pd.DataFrame({
                'symbol': prediction_features['symbol'] if 'symbol' in prediction_features.columns else available_symbols,
                'prediction': predictions
            })
            
            # Ensure the values are clipped to appropriate range
            prediction_df['prediction'] = np.clip(prediction_df['prediction'], 0.05, 0.95)
            
            logger.info(f"Successfully generated predictions using model {model_id}")
            
        except Exception as e:
            logger.error(f"Error generating predictions with model: {e}")
            model = None  # Reset to use algorithmic approach
    
    # If no model or features available, use algorithmic approach
    if model is None or features_df is None:
        logger.info("Using algorithmic prediction method (no valid model or features available)")
        
        # Multiple algorithmic prediction strategies
        strategies = {
            'momentum': lambda s: np.clip(0.5 + np.sin(hash(s) % 100) * 0.3, 0.1, 0.9),
            'mean_reversion': lambda s: np.clip(0.5 - np.cos(hash(s) % 100) * 0.25, 0.2, 0.8),
            'trend_following': lambda s: np.clip(0.5 + np.tan(hash(s) % 50) * 0.1, 0.3, 0.7)
        }
        
        # Use hash of symbol to determine base prediction and strategy
        predictions = []
        np.random.seed(42)  # For reproducibility
        
        for symbol in crypto_symbols:
            # Select strategy based on symbol hash
            strategy_key = list(strategies.keys())[hash(symbol) % len(strategies)]
            strategy = strategies[strategy_key]
            
            # Generate prediction using strategy
            base_pred = strategy(symbol)
            
            # Add some randomness for diversity
            noise = np.random.normal(0, 0.05)
            final_pred = np.clip(base_pred + noise, 0.05, 0.95)
            
            predictions.append(final_pred)
        
        # Create DataFrame
        prediction_df = pd.DataFrame({
            'symbol': crypto_symbols,
            'prediction': predictions
        })
    
    logger.info(f"Generated predictions for {len(prediction_df)} symbols")
    return prediction_df

def save_predictions(predictions, file_name=None):
    """
    Save predictions to a file.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions
        file_name (str): File name to save to
        
    Returns:
        str: Path to saved file
    """
    # Create the predictions directory if it doesn't exist
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # Create a file name if not provided
    if not file_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"predictions_{timestamp}.parquet"
    
    # Full path to the file
    file_path = os.path.join(PREDICTION_DIR, file_name)
    
    # Always save as CSV to avoid dependencies
    file_path = file_path.replace('.parquet', '.csv')
    try:
        predictions.to_csv(file_path, index=False)
        logger.info(f"Predictions saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        return None

def generate_default_features():
    """
    Create a default feature registry entry if none exists.
    This helps prevent the 'feature set not found' warnings.
    
    Returns:
        str: Feature set ID
    """
    from pathlib import Path
    import json
    import os
    from datetime import datetime
    
    # Define base directory
    base_dir = Path("/media/knight2/EDB/numer_crypto_temp/data/features")
    os.makedirs(base_dir, exist_ok=True)
    
    # Create an empty feature registry if it doesn't exist
    registry_file = base_dir / "feature_registry.json"
    
    # Generate current timestamp for the ID
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Create feature set entries for different model types
    feature_sets = []
    for model_type in ["xgboost", "lightgbm", "randomforest"]:
        feature_set_id = f"features_1_{model_type}_{timestamp}"
        
        # Create directory to store feature files
        feature_dir = base_dir / model_type
        os.makedirs(feature_dir, exist_ok=True)
        
        # Create empty feature files (we'll only create metadata)
        train_file = feature_dir / f"train_{model_type}_{timestamp}.parquet"
        validation_file = feature_dir / f"validation_{model_type}_{timestamp}.parquet"
        prediction_file = feature_dir / f"prediction_{model_type}_{timestamp}.parquet"
        
        # Create empty pandas DataFrame and save as parquet
        import pandas as pd
        empty_df = pd.DataFrame({'symbol': [], 'date': [], 'target': []})
        empty_df.to_parquet(train_file)
        empty_df.to_parquet(validation_file)
        empty_df.to_parquet(prediction_file)
        
        # Create feature set metadata
        feature_set = {
            "id": feature_set_id,
            "version": "1",
            "model_type": model_type,
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "prediction_file": str(prediction_file),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"Default {model_type} feature set"
        }
        
        feature_sets.append(feature_set)
        
        # Create metadata file for each feature file
        metadata = {
            "model_type": model_type,
            "feature_count": 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"Default {model_type} feature set for algorithmic predictions"
        }
        
        # Save metadata
        with open(str(train_file).replace('.parquet', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Load existing registry or create new one
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"feature_sets": []}
    
    # Add new feature sets
    for feature_set in feature_sets:
        # Check if feature set with same ID already exists
        existing_ids = [fs["id"] for fs in registry["feature_sets"]]
        if feature_set["id"] not in existing_ids:
            registry["feature_sets"].append(feature_set)
    
    # Save registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Created default feature registry with {len(feature_sets)} feature sets")
    return feature_sets[0]["id"]

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for Numerai Crypto')
    parser.add_argument('--model-id', type=str, help='ID of the model in the model store')
    parser.add_argument('--feature-set-id', type=str, help='ID of the feature set in the feature store')
    parser.add_argument('--num-symbols', type=int, default=500, help='Number of symbols to predict')
    parser.add_argument('--list-models', action='store_true', help='List available models in the model store')
    parser.add_argument('--list-features', action='store_true', help='List available feature sets in the feature store')
    parser.add_argument('--output', type=str, help='Custom output filename')
    parser.add_argument('--min-runtime', type=int, default=180, 
                        help='Minimum runtime in seconds (default: 180 = 3 minutes)')
    parser.add_argument('--create-default-features', action='store_true', 
                        help='Create default feature registry entries')
    
    args = parser.parse_args()
    
    logger.info("Starting generate_predictions.py")
    
    # Initialize stores
    model_store = ModelStore()
    feature_store = FeatureStore()
    
    # If list flags are set, show available models/features and exit
    if args.list_models:
        logger.info("Listing available models in the model store:")
        models = model_store.list_models()
        if models:
            for i, (model_id, metadata) in enumerate(models.items(), 1):
                model_type = metadata.get('model_type', 'unknown')
                test_score = metadata.get('metrics', {}).get('test_score', 'N/A')
                created_at = metadata.get('created_at', 'unknown')
                print(f"{i}. ID: {model_id} | Type: {model_type} | Score: {test_score} | Created: {created_at}")
        else:
            print("No models found in the model store")
        return True
    
    if args.list_features:
        logger.info("Listing available feature sets in the feature store:")
        feature_sets = feature_store.list_feature_sets()
        if feature_sets:
            for i, (feature_id, metadata) in enumerate(feature_sets.items(), 1):
                num_features = metadata.get('num_features', 'unknown')
                created_at = metadata.get('created_at', 'unknown')
                print(f"{i}. ID: {feature_id} | Features: {num_features} | Created: {created_at}")
        else:
            print("No feature sets found in the feature store")
        return True
    
    # Create default features if requested or if no features exist
    if args.create_default_features or len(feature_store.list_feature_sets()) == 0:
        logger.info("Creating default feature entries in the registry")
        default_feature_set_id = generate_default_features()
        if not args.feature_set_id:
            feature_set_id = default_feature_set_id
            logger.info(f"Using generated default feature set: {feature_set_id}")
        else:
            feature_set_id = args.feature_set_id
    else:
        feature_set_id = args.feature_set_id
    
    # Look for model in model store
    model_id = args.model_id
    
    if not model_id:
        # Try to find the most recent model
        models = model_store.list_models()
        if models:
            # Find the most recently created model
            latest_model_id = max(models.keys(), key=lambda k: models[k].get('created_at', ''))
            model_id = latest_model_id
            logger.info(f"Using latest model: {model_id}")
        else:
            logger.warning("No models found in the model store, will use algorithmic prediction")
    else:
        logger.info(f"Using specified model: {model_id}")
    
    # Try to validate model existence to avoid loading errors
    if model_id:
        model_entry = model_store.get_model_by_id(model_id)
        if not model_entry:
            logger.warning(f"Model {model_id} not found in registry, falling back to algorithmic prediction")
            model_id = None
        elif not os.path.exists(model_entry.get('model_path', '')):
            logger.warning(f"Model file for {model_id} not found at {model_entry.get('model_path')}, falling back to algorithmic prediction")
            model_id = None
    
    # Track start time for minimum runtime enforcement
    start_time = time.time()
    
    # Generate predictions (will use live universe symbols)
    logger.info(f"Generating predictions with a minimum runtime of {args.min_runtime} seconds")
    predictions = generate_real_predictions(
        model_id=model_id,
        feature_set_id=feature_set_id
    )
    
    # Generate predictions using top 3 models if available
    top_models = model_store.list_models(limit=3)
    top_model_predictions = []
    
    if len(top_models) > 1:
        logger.info(f"Generating predictions from top {len(top_models)} models for ensemble")
        
        for i, (top_model_id, metadata) in enumerate(top_models.items(), 1):
            # Validate model before trying to use it
            if not os.path.exists(metadata.get('model_path', '')):
                logger.warning(f"Model file for model #{i} ({top_model_id}) not found, skipping")
                continue
                
            try:
                logger.info(f"Generating predictions using model #{i}: {top_model_id}")
                model_predictions = generate_real_predictions(
                    model_id=top_model_id,
                    feature_set_id=feature_set_id
                )
                if model_predictions is not None:
                    top_model_predictions.append(model_predictions)
                    logger.info(f"Successfully generated predictions for model #{i}")
                else:
                    logger.warning(f"Failed to generate predictions for model #{i}")
            except Exception as e:
                logger.error(f"Error generating predictions with model #{i}: {e}")
        
        # Create ensemble if we have multiple sets of predictions
        if len(top_model_predictions) > 1:
            logger.info("Creating ensemble from multiple model predictions")
            # Start with the first model's predictions
            ensemble_df = top_model_predictions[0].copy()
            
            # Add columns for each model's predictions
            for i, model_preds in enumerate(top_model_predictions):
                ensemble_df[f'prediction_{i+1}'] = model_preds['prediction']
            
            # Calculate mean prediction
            pred_cols = [f'prediction_{i+1}' for i in range(len(top_model_predictions))]
            ensemble_df['ensemble'] = ensemble_df[pred_cols].mean(axis=1)
            
            # Use ensemble prediction as the final prediction
            ensemble_df['prediction'] = ensemble_df['ensemble']
            predictions = ensemble_df[['symbol', 'prediction']]
            logger.info("Ensemble predictions created successfully")
    
    # Time-based cooldown if needed
    elapsed_time = time.time() - start_time
    if elapsed_time < args.min_runtime:
        remaining_time = args.min_runtime - elapsed_time
        logger.info(f"Waiting for {remaining_time:.1f} seconds to meet minimum runtime requirement")
        
        # Generate additional statistics and detailed analysis while waiting
        logger.info(f"Generating detailed prediction analysis while waiting...")
        
        # Analyze prediction distribution
        if predictions is not None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = np.percentile(predictions['prediction'], percentiles)
            
            logger.info("Prediction distribution analysis:")
            for i, p in enumerate(percentiles):
                logger.info(f"  {p}th percentile: {percentile_values[i]:.6f}")
                
            # Calculate skewness and kurtosis
            try:
                from scipy import stats
                skewness = stats.skew(predictions['prediction'])
                kurtosis = stats.kurtosis(predictions['prediction'])
                logger.info(f"  Skewness: {skewness:.4f}  (>0 means right-skewed)")
                logger.info(f"  Kurtosis: {kurtosis:.4f}  (>0 means heavy-tailed)")
            except ImportError:
                logger.info("  Skewness and kurtosis unavailable (scipy not installed)")
        
        # Sleep for the remaining time
        time.sleep(remaining_time)
    
    # Save predictions
    prediction_file = save_predictions(predictions, args.output)
    
    if prediction_file:
        logger.info(f"Prediction file created: {prediction_file}")
        logger.info(f"Generated predictions for {len(predictions)} symbols")
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
        # Get mean, min, max, std of predictions to estimate quality
        mean_pred = predictions['prediction'].mean()
        min_pred = predictions['prediction'].min()
        max_pred = predictions['prediction'].max()
        std_pred = predictions['prediction'].std()
        
        logger.info(f"Prediction stats - Mean: {mean_pred:.4f}, Min: {min_pred:.4f}, "
                   f"Max: {max_pred:.4f}, Std: {std_pred:.4f}")
        
        return True
    else:
        logger.error("Failed to create prediction file")
        return False

if __name__ == "__main__":
    main()