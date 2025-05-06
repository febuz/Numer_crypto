#!/usr/bin/env python3
"""
Script to download and prepare Numerai crypto data and Yiedl data
for model training and submission.

This script uses numerapi to download Numerai crypto tournament data
and retrieves Yiedl data files, storing them with date stamps.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime
from pathlib import Path
import logging
import zipfile
import tempfile

# Try to import numerapi
try:
    import numerapi
except ImportError:
    print("numerapi package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numerapi"])
    import numerapi

# Set up logging
log_file = f"data_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths for data storage
DATA_DIR = os.path.join(Path.cwd(), 'data')
NUMERAI_DIR = os.path.join(DATA_DIR, 'numerai')
YIEDL_DIR = os.path.join(DATA_DIR, 'yiedl')
MERGED_DIR = os.path.join(DATA_DIR, 'merged')

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NUMERAI_DIR, exist_ok=True)
    os.makedirs(YIEDL_DIR, exist_ok=True)
    os.makedirs(MERGED_DIR, exist_ok=True)

def download_numerai_crypto_data():
    """Download Numerai crypto tournament data using numerapi"""
    logger.info("Downloading Numerai crypto tournament data")
    
    # Initialize Numerai API client
    napi = numerapi.NumerAPI()
    
    # Get current round for filename
    current_round = napi.get_current_round()
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Download datasets
    train_file = os.path.join(NUMERAI_DIR, f"train_targets_r{current_round}_{date_str}.parquet")
    live_file = os.path.join(NUMERAI_DIR, f"live_universe_r{current_round}_{date_str}.parquet")
    
    logger.info(f"Downloading train targets to {train_file}")
    napi.download_dataset("crypto/v1.0/train_targets.parquet", train_file)
    
    logger.info(f"Downloading live universe to {live_file}")
    napi.download_dataset("crypto/v1.0/live_universe.parquet", live_file)
    
    # Also download training data
    train_data_file = os.path.join(NUMERAI_DIR, f"train_data_r{current_round}_{date_str}.parquet")
    logger.info(f"Downloading train data to {train_data_file}")
    napi.download_dataset("crypto/v1.0/train.parquet", train_data_file)
    
    return {
        'train_targets': train_file,
        'live_universe': live_file,
        'train_data': train_data_file,
        'current_round': current_round
    }

def download_yiedl_data():
    """Download latest Yiedl data"""
    logger.info("Downloading latest Yiedl data")
    
    # Date string for filename
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Download latest data
    latest_url = 'https://api.yiedl.ai/yiedl/v1/downloadDataset?type=latest'
    latest_file = os.path.join(YIEDL_DIR, f"yiedl_latest_{date_str}.parquet")
    
    logger.info(f"Downloading latest Yiedl data to {latest_file}")
    response = requests.get(latest_url)
    
    if response.status_code == 200:
        with open(latest_file, 'wb') as f:
            f.write(response.content)
        logger.info("Latest Yiedl data downloaded successfully")
    else:
        logger.error(f"Failed to download latest Yiedl data: {response.status_code}")
        latest_file = None
    
    # Download historical data (if not already downloaded)
    historical_zip = os.path.join(YIEDL_DIR, 'yiedl_historical.zip')
    historical_file = os.path.join(YIEDL_DIR, f"yiedl_historical_{date_str}.parquet")
    
    if not os.path.exists(historical_zip):
        historical_url = 'https://api.yiedl.ai/yiedl/v1/downloadDataset?type=historical'
        logger.info("Downloading historical Yiedl data")
        response = requests.get(historical_url)
        
        if response.status_code == 200:
            with open(historical_zip, 'wb') as f:
                f.write(response.content)
            logger.info("Historical Yiedl data (zip) downloaded successfully")
        else:
            logger.error(f"Failed to download historical Yiedl data: {response.status_code}")
            historical_zip = None
    
    # Extract historical data if zip exists
    if historical_zip and os.path.exists(historical_zip):
        logger.info(f"Extracting historical data to {historical_file}")
        try:
            with zipfile.ZipFile(historical_zip, 'r') as zip_ref:
                temp_dir = tempfile.mkdtemp()
                zip_ref.extractall(temp_dir)
                
                # Find the parquet file in extracted contents
                parquet_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
                if parquet_files:
                    source_file = os.path.join(temp_dir, parquet_files[0])
                    # Copy to our dated file
                    with open(source_file, 'rb') as src, open(historical_file, 'wb') as dst:
                        dst.write(src.read())
                    logger.info("Historical Yiedl data extracted successfully")
                else:
                    logger.error("No parquet file found in the zip archive")
                    historical_file = None
        except Exception as e:
            logger.error(f"Error extracting historical Yiedl data: {e}")
            historical_file = None
    else:
        historical_file = None
    
    return {
        'latest': latest_file,
        'historical': historical_file
    }

def load_numerai_data(numerai_files):
    """Load Numerai data from downloaded files"""
    logger.info("Loading Numerai data")
    
    numerai_data = {}
    
    # Load train targets
    if os.path.exists(numerai_files['train_targets']):
        numerai_data['train_targets'] = pd.read_parquet(numerai_files['train_targets'])
        logger.info(f"Loaded train targets: {numerai_data['train_targets'].shape}")
    else:
        logger.error(f"Train targets file not found: {numerai_files['train_targets']}")
    
    # Load live universe
    if os.path.exists(numerai_files['live_universe']):
        numerai_data['live_universe'] = pd.read_parquet(numerai_files['live_universe'])
        logger.info(f"Loaded live universe: {numerai_data['live_universe'].shape}")
    else:
        logger.error(f"Live universe file not found: {numerai_files['live_universe']}")
    
    # Load train data
    if os.path.exists(numerai_files['train_data']):
        numerai_data['train_data'] = pd.read_parquet(numerai_files['train_data'])
        logger.info(f"Loaded train data: {numerai_data['train_data'].shape}")
    else:
        logger.error(f"Train data file not found: {numerai_files['train_data']}")
    
    return numerai_data

def load_yiedl_data(yiedl_files):
    """Load Yiedl data from downloaded files"""
    logger.info("Loading Yiedl data")
    
    yiedl_data = {}
    
    # Load latest data
    if yiedl_files['latest'] and os.path.exists(yiedl_files['latest']):
        yiedl_data['latest'] = pd.read_parquet(yiedl_files['latest'])
        logger.info(f"Loaded latest Yiedl data: {yiedl_data['latest'].shape}")
    else:
        logger.error(f"Latest Yiedl data file not found or specified")
    
    # Load historical data
    if yiedl_files['historical'] and os.path.exists(yiedl_files['historical']):
        yiedl_data['historical'] = pd.read_parquet(yiedl_files['historical'])
        logger.info(f"Loaded historical Yiedl data: {yiedl_data['historical'].shape}")
    else:
        logger.warning("Historical Yiedl data file not found or specified")
    
    return yiedl_data

def create_merged_datasets(numerai_data, yiedl_data, current_round):
    """Create merged datasets for training and prediction"""
    logger.info("Creating merged datasets")
    
    date_str = datetime.now().strftime('%Y%m%d')
    merged_data = {}
    
    # Check we have the necessary data
    if ('train_targets' not in numerai_data or 
        'train_data' not in numerai_data or 
        'live_universe' not in numerai_data or 
        'latest' not in yiedl_data):
        logger.error("Missing required data for merging")
        return merged_data
    
    # 1. Merge train data
    logger.info("Merging training data")
    train_data = numerai_data['train_data']
    train_targets = numerai_data['train_targets']
    yiedl_historical = yiedl_data.get('historical')
    
    # Merge train data with targets
    if 'id' in train_data.columns and 'id' in train_targets.columns:
        train_merged = pd.merge(train_data, train_targets, on='id', how='inner')
        logger.info(f"Merged train data with targets: {train_merged.shape}")
        
        # Add Yiedl historical data if available
        if yiedl_historical is not None:
            # Extract asset and date from id
            if 'id' in train_merged.columns:
                # Assuming id format: asset_date
                train_merged['asset'] = train_merged['id'].str.split('_').str[0]
                train_merged['date'] = train_merged['id'].str.split('_').str[1]
                
                # Convert to common format for joining
                if 'asset' in yiedl_historical.columns and 'date' in yiedl_historical.columns:
                    # Merge on asset and date
                    train_with_yiedl = pd.merge(
                        train_merged, 
                        yiedl_historical,
                        on=['asset', 'date'],
                        how='left'
                    )
                    
                    logger.info(f"Merged train data with Yiedl historical: {train_with_yiedl.shape}")
                    merged_data['train'] = train_with_yiedl
                else:
                    logger.warning("Yiedl historical data missing asset or date columns")
                    merged_data['train'] = train_merged
            else:
                logger.warning("Training data missing id column")
                merged_data['train'] = train_merged
        else:
            merged_data['train'] = train_merged
    else:
        logger.error("Training data or targets missing id column")
    
    # 2. Merge live data for predictions
    logger.info("Merging live data")
    live_universe = numerai_data['live_universe']
    yiedl_latest = yiedl_data['latest']
    
    # Extract asset and date from id
    if 'id' in live_universe.columns:
        # Assuming id format: asset_date
        live_universe['asset'] = live_universe['id'].str.split('_').str[0]
        live_universe['date'] = live_universe['id'].str.split('_').str[1]
        
        # Convert to common format for joining
        if 'asset' in yiedl_latest.columns and 'date' in yiedl_latest.columns:
            # Merge on asset and date
            live_with_yiedl = pd.merge(
                live_universe, 
                yiedl_latest,
                on=['asset', 'date'],
                how='left'
            )
            
            logger.info(f"Merged live data with Yiedl latest: {live_with_yiedl.shape}")
            merged_data['live'] = live_with_yiedl
        else:
            logger.warning("Yiedl latest data missing asset or date columns")
            merged_data['live'] = live_universe
    else:
        logger.error("Live universe missing id column")
        merged_data['live'] = live_universe
    
    # Save merged datasets
    if 'train' in merged_data:
        train_file = os.path.join(MERGED_DIR, f"train_merged_r{current_round}_{date_str}.parquet")
        merged_data['train'].to_parquet(train_file)
        logger.info(f"Saved merged train data to {train_file}")
        merged_data['train_file'] = train_file
    
    if 'live' in merged_data:
        live_file = os.path.join(MERGED_DIR, f"live_merged_r{current_round}_{date_str}.parquet")
        merged_data['live'].to_parquet(live_file)
        logger.info(f"Saved merged live data to {live_file}")
        merged_data['live_file'] = live_file
    
    return merged_data

def main():
    """Main function to download and prepare data"""
    logger.info("Starting data download and preparation")
    
    # Create directories
    ensure_directories()
    
    # Download Numerai crypto data
    numerai_files = download_numerai_crypto_data()
    current_round = numerai_files['current_round']
    
    # Download Yiedl data
    yiedl_files = download_yiedl_data()
    
    # Load data
    numerai_data = load_numerai_data(numerai_files)
    yiedl_data = load_yiedl_data(yiedl_files)
    
    # Create merged datasets
    merged_data = create_merged_datasets(numerai_data, yiedl_data, current_round)
    
    # Report summary
    logger.info("\n===== DATA PREPARATION SUMMARY =====")
    logger.info(f"Numerai current round: {current_round}")
    
    if 'train_targets' in numerai_data:
        logger.info(f"Numerai train targets shape: {numerai_data['train_targets'].shape}")
    if 'live_universe' in numerai_data:
        logger.info(f"Numerai live universe shape: {numerai_data['live_universe'].shape}")
    if 'latest' in yiedl_data:
        logger.info(f"Yiedl latest data shape: {yiedl_data['latest'].shape}")
    if 'historical' in yiedl_data:
        logger.info(f"Yiedl historical data shape: {yiedl_data['historical'].shape}")
    
    if 'train' in merged_data:
        logger.info(f"Merged train data shape: {merged_data['train'].shape}")
    if 'live' in merged_data:
        logger.info(f"Merged live data shape: {merged_data['live'].shape}")
    
    # Print dataset locations
    logger.info("\n===== DATASET LOCATIONS =====")
    if 'train_file' in merged_data:
        logger.info(f"Merged train data: {merged_data['train_file']}")
    if 'live_file' in merged_data:
        logger.info(f"Merged live data: {merged_data['live_file']}")
    
    # Return merged data paths for the calling script
    return {
        'train_file': merged_data.get('train_file'),
        'live_file': merged_data.get('live_file'),
        'current_round': current_round
    }

if __name__ == "__main__":
    main()