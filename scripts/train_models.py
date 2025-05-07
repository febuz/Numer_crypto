#!/usr/bin/env python3
"""
train_models.py - Train models for Numerai Crypto

This script trains machine learning models for Numerai Crypto predictions.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
PROCESSED_DATA_DIR = "/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
MODELS_DIR = "/numer_crypto_temp/models"

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
    """Prepare data for model training"""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None, None
    
    # Load data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data from {file_path} with shape {df.shape}")
    
    # Prepare features and target
    if 'target' not in df.columns:
        logger.error("Target column 'target' not found in data")
        return None, None
    
    # Remove non-numeric columns and the target column from features
    X = df.select_dtypes(include=[np.number])
    excluded_cols = ['target', 'Symbol', 'symbol', 'Prediction', 'prediction']
    feature_cols = [col for col in X.columns if col not in excluded_cols]
    
    # Handle missing values
    X = X[feature_cols].fillna(0)
    y = df['target'].fillna(0)
    
    logger.info(f"Prepared data with {X.shape[1]} features")
    return X, y

def save_model(model, model_path):
    """Save the trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train models for Numerai Crypto')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    args = parser.parse_args()
    
    logger.info("Starting train_models.py")
    
    # Prepare training data
    train_file = os.path.join(TRAIN_DIR, "train_data_featured.csv")
    X_train, y_train = prepare_data(train_file)
    
    if X_train is None or y_train is None:
        logger.error("Failed to prepare training data")
        return False
    
    # Train model
    model = train_simple_model(X_train, y_train, args.use_gpu, args.parallel)
    
    if model is None:
        logger.error("Model training failed")
        return False
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pkl")
    
    if save_model(model, model_path):
        logger.info("Model saved successfully")
    else:
        logger.error("Failed to save model")
        return False
    
    logger.info("Model training completed successfully")
    return True

if __name__ == "__main__":
    main()
