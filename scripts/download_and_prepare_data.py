#!/usr/bin/env python3
"""
Download and prepare data for crypto prediction:
1. Download Numerai crypto data using numerapi
2. Retrieve Yiedl data files from the internet
3. Store files with date in external data folder
4. Prepare merged dataset for model training

This script creates data files for online validation on Numerai.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import requests
import tempfile
import shutil
import time

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Configure logging
log_file = f"data_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default paths
DATA_DIR = project_root / "data"
YIEDL_DIR = DATA_DIR / "yiedl"
NUMERAI_DIR = DATA_DIR / "numerai"
EXTERNAL_DATA_DIR = Path("/media/knight2/EDB/data/crypto_data")
TODAY = datetime.now().strftime("%Y%m%d")

def download_numerai_data():
    """
    Download Numerai crypto tournament data via numerapi
    """
    logger.info("Downloading Numerai crypto data...")
    
    try:
        import numerapi
        
        # Initialize Numerapi client without credentials for public data access
        napi = numerapi.NumerAPI()
        
        # Create directories if they don't exist
        os.makedirs(NUMERAI_DIR, exist_ok=True)
        
        # Download training data (includes targets)
        train_path = NUMERAI_DIR / "train_targets.parquet"
        logger.info(f"Downloading train data to {train_path}")
        napi.download_dataset("crypto/v1.0/train_targets.parquet", str(train_path))
        
        # Download validation data
        validation_path = NUMERAI_DIR / "validation.parquet"
        logger.info(f"Downloading validation data to {validation_path}")
        napi.download_dataset("crypto/v1.0/validation.parquet", str(validation_path))
        
        # Download live data (current round for predictions)
        live_path = NUMERAI_DIR / "live.parquet"
        logger.info(f"Downloading live data to {live_path}")
        napi.download_dataset("crypto/v1.0/live_universe.parquet", str(live_path))
        
        # Create dated copies in external data directory
        os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
        dated_dir = EXTERNAL_DATA_DIR / TODAY
        os.makedirs(dated_dir, exist_ok=True)
        
        # Copy files with date in filename
        shutil.copy(train_path, dated_dir / f"numerai_train_targets_{TODAY}.parquet")
        shutil.copy(validation_path, dated_dir / f"numerai_validation_{TODAY}.parquet")
        shutil.copy(live_path, dated_dir / f"numerai_live_{TODAY}.parquet")
        
        logger.info(f"Numerai data downloaded successfully to {NUMERAI_DIR}")
        logger.info(f"Dated copies saved to {dated_dir}")
        
        # Return paths to downloaded files
        return {
            'train': train_path,
            'validation': validation_path, 
            'live': live_path,
            'dated_dir': dated_dir
        }
    
    except ImportError:
        logger.error("numerapi package not installed. Install with: pip install numerapi")
        return None
    except Exception as e:
        logger.error(f"Error downloading Numerai data: {e}")
        return None

def retrieve_yiedl_data():
    """
    Retrieve Yiedl data files from the internet
    
    The data can be obtained from:
    - Original source: https://yiedl.com/data
    - Alternative sources (GitHub, etc.)
    
    If data cannot be retrieved, we use the existing local files
    """
    logger.info("Retrieving Yiedl data...")
    
    # Create directories if they don't exist
    os.makedirs(YIEDL_DIR, exist_ok=True)
    
    # URL for Yiedl data (placeholder - update with actual URLs)
    # These URLs will need to be replaced with the actual data sources
    yiedl_urls = {
        'latest': "https://example.com/yiedl_latest.parquet",
        'historical': "https://example.com/yiedl_historical.zip"
    }
    
    # Check if local files already exist
    latest_file = YIEDL_DIR / "yiedl_latest.parquet"
    historical_zip = YIEDL_DIR / "yiedl_historical.zip"
    
    downloaded = {}
    
    # Try to download latest data
    if not latest_file.exists():
        try:
            logger.info(f"Attempting to download latest Yiedl data from {yiedl_urls['latest']}")
            
            # This is a placeholder for the actual download - replace with real code
            # response = requests.get(yiedl_urls['latest'], stream=True)
            # if response.status_code == 200:
            #     with open(latest_file, 'wb') as f:
            #         for chunk in response.iter_content(chunk_size=8192):
            #             f.write(chunk)
            #     downloaded['latest'] = True
            # else:
            #     logger.warning(f"Failed to download latest data: HTTP {response.status_code}")
            
            # For now, just check if the file exists and log the outcome
            if latest_file.exists():
                logger.info(f"Using existing latest Yiedl data at {latest_file}")
                downloaded['latest'] = True
            else:
                logger.warning("No latest Yiedl data file found")
                downloaded['latest'] = False
        except Exception as e:
            logger.error(f"Error downloading latest Yiedl data: {e}")
            downloaded['latest'] = False
    else:
        logger.info(f"Using existing latest Yiedl data at {latest_file}")
        downloaded['latest'] = True
    
    # Try to download historical data
    if not historical_zip.exists():
        try:
            logger.info(f"Attempting to download historical Yiedl data from {yiedl_urls['historical']}")
            
            # This is a placeholder for the actual download - replace with real code
            # response = requests.get(yiedl_urls['historical'], stream=True)
            # if response.status_code == 200:
            #     with open(historical_zip, 'wb') as f:
            #         for chunk in response.iter_content(chunk_size=8192):
            #             f.write(chunk)
            #     downloaded['historical'] = True
            # else:
            #     logger.warning(f"Failed to download historical data: HTTP {response.status_code}")
            
            # For now, just check if the file exists and log the outcome
            if historical_zip.exists():
                logger.info(f"Using existing historical Yiedl data at {historical_zip}")
                downloaded['historical'] = True
            else:
                logger.warning("No historical Yiedl data file found")
                downloaded['historical'] = False
        except Exception as e:
            logger.error(f"Error downloading historical Yiedl data: {e}")
            downloaded['historical'] = False
    else:
        logger.info(f"Using existing historical Yiedl data at {historical_zip}")
        downloaded['historical'] = True
    
    # Create dated copies in external data directory if available
    if any(downloaded.values()):
        os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
        dated_dir = EXTERNAL_DATA_DIR / TODAY
        os.makedirs(dated_dir, exist_ok=True)
        
        if downloaded.get('latest') and latest_file.exists():
            dated_latest = dated_dir / f"yiedl_latest_{TODAY}.parquet"
            shutil.copy(latest_file, dated_latest)
            logger.info(f"Created dated copy at {dated_latest}")
        
        if downloaded.get('historical') and historical_zip.exists():
            dated_historical = dated_dir / f"yiedl_historical_{TODAY}.zip"
            shutil.copy(historical_zip, dated_historical)
            logger.info(f"Created dated copy at {dated_historical}")
    
    # Return paths to the files
    return {
        'latest': latest_file if latest_file.exists() else None,
        'historical': historical_zip if historical_zip.exists() else None,
        'dated_dir': EXTERNAL_DATA_DIR / TODAY if any(downloaded.values()) else None
    }

def load_numerai_data(paths):
    """
    Load Numerai data from downloaded files
    """
    logger.info("Loading Numerai data...")
    
    try:
        # Load training data
        train_df = pd.read_parquet(paths['train'])
        logger.info(f"Loaded training data: {train_df.shape}")
        
        # Load validation data
        validation_df = pd.read_parquet(paths['validation'])
        logger.info(f"Loaded validation data: {validation_df.shape}")
        
        # Load live data
        live_df = pd.read_parquet(paths['live'])
        logger.info(f"Loaded live data: {live_df.shape}")
        
        # Add data type column to identify source
        train_df['data_type'] = 'train'
        validation_df['data_type'] = 'validation'
        live_df['data_type'] = 'live'
        
        return {
            'train': train_df,
            'validation': validation_df,
            'live': live_df
        }
    except Exception as e:
        logger.error(f"Error loading Numerai data: {e}")
        return None

def load_yiedl_data(paths):
    """
    Load Yiedl data from available files
    """
    logger.info("Loading Yiedl data...")
    
    try:
        latest_df = None
        historical_df = None
        
        # Try to load latest data
        if paths.get('latest') and paths['latest'].exists():
            try:
                import pyarrow.parquet as pq
                latest_df = pd.read_parquet(paths['latest'])
                logger.info(f"Loaded Yiedl latest data: {latest_df.shape}")
            except Exception as e:
                logger.error(f"Error loading Yiedl latest data: {e}")
        
        # Try to extract and load historical data from ZIP
        if paths.get('historical') and paths['historical'].exists():
            try:
                import zipfile
                extracted_dir = YIEDL_DIR / "extracted"
                os.makedirs(extracted_dir, exist_ok=True)
                
                with zipfile.ZipFile(paths['historical'], 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                
                # Look for parquet files in extracted directory
                parquet_files = list(extracted_dir.glob("**/*.parquet"))
                
                if parquet_files:
                    dfs = []
                    for file in parquet_files:
                        df = pd.read_parquet(file)
                        dfs.append(df)
                    
                    if dfs:
                        historical_df = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded Yiedl historical data: {historical_df.shape}")
            except Exception as e:
                logger.error(f"Error extracting/loading Yiedl historical data: {e}")
        
        # Add data type column to identify source
        if latest_df is not None:
            latest_df['data_type'] = 'latest'
        
        if historical_df is not None:
            historical_df['data_type'] = 'historical'
        
        return {
            'latest': latest_df,
            'historical': historical_df
        }
    except Exception as e:
        logger.error(f"Error loading Yiedl data: {e}")
        return None

def create_merged_dataset(numerai_data, yiedl_data):
    """
    Create merged datasets for each data type
    """
    logger.info("Creating merged datasets...")
    
    merged_datasets = {}
    
    try:
        # Merge training data
        if numerai_data.get('train') is not None:
            train_merge = numerai_data['train'].copy()
            
            # Add Yiedl features if available
            if yiedl_data.get('historical') is not None:
                # For demonstration purposes, we'll add a prefix to avoid column name conflicts
                yiedl_features = yiedl_data['historical'].copy()
                # Exclude overlapping columns
                overlap_cols = set(train_merge.columns).intersection(set(yiedl_features.columns))
                yiedl_cols = [col for col in yiedl_features.columns if col not in overlap_cols]
                
                # Add a prefix to Yiedl columns to avoid conflicts
                yiedl_features = yiedl_features[yiedl_cols].copy()
                yiedl_features = yiedl_features.add_prefix('yiedl_')
                
                # Merge by common ID if available, or just concatenate features
                # This is a simplified approach and would need customization for real data
                train_merge = pd.concat([train_merge, yiedl_features.iloc[:len(train_merge)].reset_index(drop=True)], axis=1)
            
            merged_datasets['train'] = train_merge
            logger.info(f"Created merged training dataset: {train_merge.shape}")
        
        # Merge validation data
        if numerai_data.get('validation') is not None:
            val_merge = numerai_data['validation'].copy()
            
            # Add Yiedl features if available
            if yiedl_data.get('historical') is not None:
                # Similar approach as above
                yiedl_features = yiedl_data['historical'].copy()
                overlap_cols = set(val_merge.columns).intersection(set(yiedl_features.columns))
                yiedl_cols = [col for col in yiedl_features.columns if col not in overlap_cols]
                
                yiedl_features = yiedl_features[yiedl_cols].copy()
                yiedl_features = yiedl_features.add_prefix('yiedl_')
                
                val_merge = pd.concat([val_merge, yiedl_features.iloc[:len(val_merge)].reset_index(drop=True)], axis=1)
            
            merged_datasets['validation'] = val_merge
            logger.info(f"Created merged validation dataset: {val_merge.shape}")
        
        # Merge live data
        if numerai_data.get('live') is not None:
            live_merge = numerai_data['live'].copy()
            
            # Add Yiedl features if available
            if yiedl_data.get('latest') is not None:
                # Similar approach as above
                yiedl_features = yiedl_data['latest'].copy()
                overlap_cols = set(live_merge.columns).intersection(set(yiedl_features.columns))
                yiedl_cols = [col for col in yiedl_features.columns if col not in overlap_cols]
                
                yiedl_features = yiedl_features[yiedl_cols].copy()
                yiedl_features = yiedl_features.add_prefix('yiedl_')
                
                live_merge = pd.concat([live_merge, yiedl_features.iloc[:len(live_merge)].reset_index(drop=True)], axis=1)
            
            merged_datasets['live'] = live_merge
            logger.info(f"Created merged live dataset: {live_merge.shape}")
        
        # Save merged datasets to the dated directory
        dated_dir = EXTERNAL_DATA_DIR / TODAY
        os.makedirs(dated_dir, exist_ok=True)
        
        for data_type, df in merged_datasets.items():
            file_path = dated_dir / f"merged_{data_type}_{TODAY}.parquet"
            df.to_parquet(file_path)
            logger.info(f"Saved merged {data_type} dataset to {file_path}")
        
        return merged_datasets
    
    except Exception as e:
        logger.error(f"Error creating merged datasets: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download and prepare data for crypto prediction')
    parser.add_argument('--skip-numerai', action='store_true', help='Skip Numerai data download')
    parser.add_argument('--skip-yiedl', action='store_true', help='Skip Yiedl data retrieval')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()
    
    # Override output directory if specified
    global EXTERNAL_DATA_DIR
    if args.output_dir:
        EXTERNAL_DATA_DIR = Path(args.output_dir)
    
    # Download Numerai data
    numerai_paths = None
    if not args.skip_numerai:
        numerai_paths = download_numerai_data()
    else:
        logger.info("Skipping Numerai data download")
        # Use existing data if available
        numerai_paths = {
            'train': NUMERAI_DIR / "train_targets.parquet",
            'validation': NUMERAI_DIR / "validation.parquet",
            'live': NUMERAI_DIR / "live.parquet",
            'dated_dir': EXTERNAL_DATA_DIR / TODAY
        }
    
    # Retrieve Yiedl data
    yiedl_paths = None
    if not args.skip_yiedl:
        yiedl_paths = retrieve_yiedl_data()
    else:
        logger.info("Skipping Yiedl data retrieval")
        # Use existing data if available
        yiedl_paths = {
            'latest': YIEDL_DIR / "yiedl_latest.parquet",
            'historical': YIEDL_DIR / "yiedl_historical.zip",
            'dated_dir': EXTERNAL_DATA_DIR / TODAY
        }
    
    # Load and process data if available
    if numerai_paths or yiedl_paths:
        # Load Numerai data
        numerai_data = None
        if numerai_paths and all(numerai_paths.get(k) and numerai_paths[k].exists() for k in ['train', 'validation', 'live']):
            numerai_data = load_numerai_data(numerai_paths)
        
        # Load Yiedl data
        yiedl_data = None
        if yiedl_paths and (yiedl_paths.get('latest') or yiedl_paths.get('historical')):
            yiedl_data = load_yiedl_data(yiedl_paths)
        
        # Create merged datasets
        if numerai_data or yiedl_data:
            merged_datasets = create_merged_dataset(numerai_data or {}, yiedl_data or {})
            
            if merged_datasets:
                logger.info("Data download and preparation completed successfully")
                return 0
    
    logger.error("Failed to prepare data")
    return 1

if __name__ == "__main__":
    sys.exit(main())