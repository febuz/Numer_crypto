#!/usr/bin/env python3
"""
process_data.py - Process data for Numerai Crypto

This script processes raw data and creates train/validation/prediction splits.
"""
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
RAW_DATA_DIR = "/media/knight2/EDB/numer_crypto_temp/data/raw"
PROCESSED_DATA_DIR = "/media/knight2/EDB/numer_crypto_temp/data/processed"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATA_DIR, "validation")
PREDICTION_DIR = os.path.join(PROCESSED_DATA_DIR, "prediction")

def process_numerai_data(use_historical=False, skip_historical=False, pit_date=None):
    """Process Numerai data"""
    logger.info("Processing Numerai data...")
    
    # Load raw data
    numerai_file = os.path.join(RAW_DATA_DIR, "numerai_sample_data.csv")
    if not os.path.exists(numerai_file):
        logger.error(f"Numerai data file not found: {numerai_file}")
        return False
    
    numerai_data = pd.read_csv(numerai_file)
    logger.info(f"Loaded Numerai data with shape: {numerai_data.shape}")
    
    return numerai_data

def process_yiedl_data(use_historical=False, skip_historical=False, pit_date=None):
    """Process Yiedl data"""
    logger.info("Processing Yiedl data...")
    
    # Load raw data
    yiedl_file = os.path.join(RAW_DATA_DIR, "yiedl_sample_data.csv")
    if not os.path.exists(yiedl_file):
        logger.error(f"Yiedl data file not found: {yiedl_file}")
        return False
    
    yiedl_data = pd.read_csv(yiedl_file)
    logger.info(f"Loaded Yiedl data with shape: {yiedl_data.shape}")
    
    # Load historical data if requested
    if use_historical:
        historical_file = os.path.join(RAW_DATA_DIR, "yiedl_historical_data.csv")
        if os.path.exists(historical_file):
            historical_data = pd.read_csv(historical_file)
            logger.info(f"Loaded historical Yiedl data with shape: {historical_data.shape}")
            
            # Combine with current data
            yiedl_data = pd.concat([historical_data, yiedl_data], ignore_index=True)
            logger.info(f"Combined Yiedl data shape: {yiedl_data.shape}")
    
    return yiedl_data

def create_data_splits(numerai_data, yiedl_data):
    """Create train/validation/prediction splits"""
    logger.info("Creating data splits...")
    
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Merge data (simple example - in a real scenario this would be more complex)
    merged_data = pd.merge(
        numerai_data, 
        yiedl_data, 
        left_on=['date', 'symbol'], 
        right_on=['date', 'asset'],
        how='inner'
    )
    
    logger.info(f"Merged data shape: {merged_data.shape}")
    
    # Create splits (70% train, 15% validation, 15% prediction)
    train_size = int(len(merged_data) * 0.7)
    val_size = int(len(merged_data) * 0.15)
    
    train_data = merged_data.iloc[:train_size]
    val_data = merged_data.iloc[train_size:train_size+val_size]
    pred_data = merged_data.iloc[train_size+val_size:]
    
    # Save splits
    train_file = os.path.join(TRAIN_DIR, "train_data.csv")
    val_file = os.path.join(VALIDATION_DIR, "validation_data.csv")
    pred_file = os.path.join(PREDICTION_DIR, "prediction_data.csv")
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    pred_data.to_csv(pred_file, index=False)
    
    logger.info(f"Saved train data to {train_file} with shape {train_data.shape}")
    logger.info(f"Saved validation data to {val_file} with shape {val_data.shape}")
    logger.info(f"Saved prediction data to {pred_file} with shape {pred_data.shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process data for Numerai Crypto')
    parser.add_argument('--use-historical', action='store_true', help='Use historical data')
    parser.add_argument('--skip-historical', action='store_true', help='Skip downloading historical data')
    parser.add_argument('--pit', type=str, help='Point-in-time date for data (format: YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if files exist')
    
    args = parser.parse_args()
    
    logger.info("Starting process_data.py")
    
    # Process Numerai data
    numerai_data = process_numerai_data(
        use_historical=args.use_historical,
        skip_historical=args.skip_historical,
        pit_date=args.pit
    )
    if numerai_data is False:
        logger.error("Numerai data processing failed")
        return False
    
    # Process Yiedl data
    yiedl_data = process_yiedl_data(
        use_historical=args.use_historical,
        skip_historical=args.skip_historical,
        pit_date=args.pit
    )
    if yiedl_data is False:
        logger.error("Yiedl data processing failed")
        return False
    
    # Check for force flag or if files don't exist
    crypto_train_file = os.path.join(PROCESSED_DATA_DIR, "crypto_train.parquet")
    crypto_test_file = os.path.join(PROCESSED_DATA_DIR, "crypto_test.parquet")
    crypto_live_file = os.path.join(PROCESSED_DATA_DIR, "crypto_live.parquet")
    
    if args.force or not (os.path.exists(crypto_train_file) and 
                       os.path.exists(crypto_test_file) and
                       os.path.exists(crypto_live_file)):
        # Create data splits
        if create_data_splits(numerai_data, yiedl_data):
            logger.info("Data splits created successfully")
        else:
            logger.error("Data splits creation failed")
            return False
    else:
        logger.info("Using existing processed data files")
        return True
    
    logger.info("Data processing completed successfully")
    return True

if __name__ == "__main__":
    main()
