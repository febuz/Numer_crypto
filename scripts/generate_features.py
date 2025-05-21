#!/usr/bin/env python3
"""
generate_features.py - Generate features for Numerai Crypto

This script generates time series features for training and prediction.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
PROCESSED_DATA_DIR = "/media/knight2/EDB/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
PREDICTION_DIR = os.path.join(PROCESSED_DATA_DIR, "prediction")

def generate_time_series_features(df, generate_ts_features=True):
    """Generate time series features for the given DataFrame"""
    logger.info(f"Generating features for DataFrame with shape {df.shape}")
    
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    if not generate_ts_features:
        logger.info("Time series feature generation disabled, returning original data")
        return result_df
    
    # Example of simple time series features
    
    # Get numeric columns for feature generation
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    date_col = 'date' if 'date' in result_df.columns else None
    group_col = 'symbol' if 'symbol' in result_df.columns else ('asset' if 'asset' in result_df.columns else None)
    
    if date_col is None or group_col is None:
        logger.warning("Missing date or group column, cannot generate time series features")
        return result_df
    
    # Convert date column to datetime
    result_df[date_col] = pd.to_datetime(result_df[date_col])
    
    # Sort by group and date
    result_df = result_df.sort_values([group_col, date_col])
    
    # Generate rolling mean features
    for col in numeric_cols:
        # Skip columns that aren't meaningful for time series features
        if col in [group_col, 'target']:
            continue
            
        # Rolling mean with window sizes 3, 7
        for window in [3, 7]:
            result_df[f'{col}_rolling_mean_{window}'] = result_df.groupby(group_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Generate lag features
    for col in numeric_cols:
        # Skip columns that aren't meaningful for time series features
        if col in [group_col, 'target']:
            continue
            
        # Lag features with lag 1, 2
        for lag in [1, 2]:
            result_df[f'{col}_lag_{lag}'] = result_df.groupby(group_col)[col].shift(lag)
    
    logger.info(f"Generated features, new shape: {result_df.shape}")
    
    return result_df

def process_dataset(input_file, output_file, generate_ts_features=True):
    """Process a dataset file, generating features and saving the result"""
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded data from {input_file} with shape {df.shape}")
    
    # Generate features
    result_df = generate_time_series_features(df, generate_ts_features)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save result
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved featured data to {output_file} with shape {result_df.shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate features for Numerai Crypto')
    parser.add_argument('--timeseries', action='store_true', help='Generate time series features')
    parser.add_argument('--max-features', type=int, default=10000, help='Maximum number of features to generate')
    parser.add_argument('--cache', action='store_true', help='Cache intermediate results')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for feature generation if available')
    
    args = parser.parse_args()
    
    logger.info("Starting generate_features.py")
    
    # Process train data
    train_input = os.path.join(TRAIN_DIR, "train_data.csv")
    train_output = os.path.join(TRAIN_DIR, "train_data_featured.csv")
    if process_dataset(train_input, train_output, args.timeseries):
        logger.info("Train data feature generation completed successfully")
    else:
        logger.error("Train data feature generation failed")
        return False
    
    # Process validation data
    val_input = os.path.join(VALIDATION_DIR, "validation_data.csv")
    val_output = os.path.join(VALIDATION_DIR, "validation_data_featured.csv")
    if process_dataset(val_input, val_output, args.timeseries):
        logger.info("Validation data feature generation completed successfully")
    else:
        logger.error("Validation data feature generation failed")
        return False
    
    # Process prediction data
    pred_input = os.path.join(PREDICTION_DIR, "prediction_data.csv")
    pred_output = os.path.join(PREDICTION_DIR, "prediction_data_featured.csv")
    if process_dataset(pred_input, pred_output, args.timeseries):
        logger.info("Prediction data feature generation completed successfully")
    else:
        logger.error("Prediction data feature generation failed")
        return False
    
    logger.info("Feature generation completed successfully")
    return True

if __name__ == "__main__":
    main()
