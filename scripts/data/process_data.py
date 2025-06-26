#!/usr/bin/env python3
"""
process_data_polars.py - Memory-optimized data processing using Polars
"""
import os
import sys
import logging
import argparse
import re
import polars as pl
from datetime import datetime
from pathlib import Path

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

def reduce_precision(df):
    """Reduce numeric precision to save memory"""
    logger.info("Reducing numeric precision to save memory")
    
    for col in df.columns:
        if df[col].dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        elif df[col].dtype == pl.Int64:
            # Check if values can fit in smaller int types
            if df[col].min() >= -32768 and df[col].max() <= 32767:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
    
    return df

def process_numerai_data(use_historical=False, skip_historical=False, pit_date=None):
    """Process Numerai data using Polars"""
    logger.info("Processing Numerai data with Polars...")
    
    # Check for parquet files first (preferred format)
    potential_files = [
        os.path.join(RAW_DATA_DIR, "numerai_train.parquet"),
        os.path.join(RAW_DATA_DIR, "numerai_live.parquet"),
        os.path.join(RAW_DATA_DIR, "numerai_targets.parquet")
    ]
    
    # Look for most recent files in date folders
    for pattern in ["*/train_data_r*.parquet", "*/live_universe_r*.parquet", "*/train_targets_r*.parquet"]:
        latest_files = sorted(Path(RAW_DATA_DIR).parent.glob(f"numerai/{pattern}"), 
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if latest_files:
            potential_files.append(str(latest_files[0]))
    
    # Check if we have all needed files
    numerai_files = {
        'train_data': None,
        'live_universe': None,
        'train_targets': None
    }
    
    # Identify files by name pattern
    for file_path in potential_files:
        if os.path.exists(file_path):
            if 'train_data' in file_path or 'numerai_train' in file_path:
                numerai_files['train_data'] = file_path
            elif 'live_universe' in file_path or 'numerai_live' in file_path:
                numerai_files['live_universe'] = file_path
            elif 'train_targets' in file_path or 'numerai_targets' in file_path:
                numerai_files['train_targets'] = file_path
    
    # Check if we have the minimum required files
    if not (numerai_files['live_universe'] and numerai_files['train_targets']):
        logger.error("Missing essential Numerai files (live_universe and/or train_targets)")
        return False
    
    # Load the files using Polars
    numerai_data = {}
    for key, file_path in numerai_files.items():
        if file_path and os.path.exists(file_path):
            try:
                df = pl.read_parquet(file_path)
                df = reduce_precision(df)
                numerai_data[key] = df
                logger.info(f"Loaded {key} with shape: {df.shape}")
                numerai_data[f"{key}_path"] = file_path
            except Exception as e:
                logger.error(f"Error loading {key} from {file_path}: {e}")
    
    # Add current round if available
    current_round = None
    for file_path in numerai_files.values():
        if file_path:
            match = re.search(r'r(\d+)', file_path)
            if match:
                current_round = int(match.group(1))
                break
    
    if current_round:
        numerai_data['current_round'] = current_round
        logger.info(f"Detected tournament round: {current_round}")
    
    return numerai_data

def process_yiedl_data(use_historical=False, skip_historical=False, pit_date=None):
    """Process Yiedl data using Polars"""
    logger.info("Processing Yiedl data with Polars...")
    
    # Check for parquet files first (preferred format)
    potential_files = {
        'latest': os.path.join(RAW_DATA_DIR, "yiedl_latest.parquet"),
        'historical': os.path.join(RAW_DATA_DIR, "yiedl_historical.parquet")
    }
    
    # Look for most recent files in date folders
    for prefix in ["latest", "historical"]:
        for pattern in [f"*/{prefix}*.parquet", f"*/{prefix}*.csv"]:
            latest_files = sorted(Path(RAW_DATA_DIR).parent.glob(f"yiedl/{pattern}"), 
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if latest_files:
                potential_files[prefix] = str(latest_files[0])
    
    # Check if we have the latest data file
    if not os.path.exists(potential_files['latest']):
        logger.error(f"Yiedl latest data file not found: {potential_files['latest']}")
        return False
    
    # Load the latest data
    yiedl_data = {'latest': None, 'historical': None}
    
    # Load latest data
    try:
        if potential_files['latest'].endswith('.parquet'):
            latest_df = pl.read_parquet(potential_files['latest'])
        else:
            latest_df = pl.read_csv(potential_files['latest'])
        
        latest_df = reduce_precision(latest_df)
        yiedl_data['latest'] = latest_df
        yiedl_data['latest_path'] = potential_files['latest']
        logger.info(f"Loaded Yiedl latest data with shape: {latest_df.shape}")
    except Exception as e:
        logger.error(f"Error loading Yiedl latest data: {e}")
        return False
    
    # Load historical data if requested and available
    if use_historical and potential_files['historical'] and os.path.exists(potential_files['historical']):
        try:
            if potential_files['historical'].endswith('.parquet'):
                historical_df = pl.read_parquet(potential_files['historical'])
            else:
                historical_df = pl.read_csv(potential_files['historical'])
            
            historical_df = reduce_precision(historical_df)
            yiedl_data['historical'] = historical_df
            yiedl_data['historical_path'] = potential_files['historical']
            logger.info(f"Loaded Yiedl historical data with shape: {historical_df.shape}")
        except Exception as e:
            logger.error(f"Error loading Yiedl historical data: {e}")
    elif use_historical:
        logger.warning(f"Historical data requested but file not found: {potential_files['historical']}")
    
    return yiedl_data

def prepare_dataframe_for_merge(df, data_name):
    """Prepare dataframe for merging - handle symbol/asset columns and dates"""
    logger.info(f"Preparing {data_name} for merge")
    
    # Handle newer format with just 'symbol' column instead of 'id'
    if 'symbol' in df.columns and 'id' not in df.columns:
        logger.info(f"Found new format {data_name} with 'symbol' column")
        
        # Check if date column exists - if not, use today's date
        if 'date' not in df.columns:
            from datetime import date
            today = date.today().strftime('%Y-%m-%d')
            logger.info(f"Adding date column with today's date: {today}")
            df = df.with_columns(pl.lit(today).alias('date'))
            
        # Set asset column to symbol
        df = df.with_columns(pl.col('symbol').alias('asset'))
        logger.info(f"Set asset column equal to symbol. Shape: {df.shape}")
        
    # Fallback to old format with 'id' column
    elif 'id' in df.columns:
        df = df.with_columns([
            pl.col('id').str.split('_').list.get(0).alias('asset'),
            pl.col('id').str.split('_').list.get(1).alias('date')
        ])
        logger.info(f"Extracted asset and date from {data_name} ID. Shape: {df.shape}")
    else:
        logger.error(f"{data_name} missing both 'id' and 'symbol' columns, cannot proceed")
        return None
    
    # Ensure we have both symbol and asset columns
    if 'symbol' not in df.columns and 'asset' in df.columns:
        df = df.with_columns(pl.col('asset').alias('symbol'))
    elif 'asset' not in df.columns and 'symbol' in df.columns:
        df = df.with_columns(pl.col('symbol').alias('asset'))
    
    # Ensure dates are strings
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Utf8))
    
    return df

def create_data_splits(numerai_data, yiedl_data):
    """Create train/validation/prediction splits using Polars"""
    logger.info("Creating data splits with Polars...")
    
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Get date stamp for filenames
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Get current round if available
    current_round = numerai_data.get('current_round', 'unknown')
    if current_round != 'unknown':
        logger.info(f"Detected tournament round: {current_round}")
    else:
        logger.warning("Could not detect tournament round")
    
    # Extract dataframes
    train_df = numerai_data.get('train_data')
    targets_df = numerai_data.get('train_targets')
    live_df = numerai_data.get('live_universe')
    yiedl_latest_df = yiedl_data.get('latest')
    yiedl_historical_df = yiedl_data.get('historical')
    
    # Check if we have required data
    if not (live_df is not None and targets_df is not None and yiedl_latest_df is not None):
        logger.error("Missing required dataframes for merging")
        return False
    
    # Prepare dataframes for merging
    live_df = prepare_dataframe_for_merge(live_df, "live universe")
    targets_df = prepare_dataframe_for_merge(targets_df, "targets")
    yiedl_latest_df = prepare_dataframe_for_merge(yiedl_latest_df, "Yiedl latest")
    
    if train_df is not None:
        train_df = prepare_dataframe_for_merge(train_df, "train data")
    
    if yiedl_historical_df is not None:
        yiedl_historical_df = prepare_dataframe_for_merge(yiedl_historical_df, "Yiedl historical")
    
    if any(df is None for df in [live_df, targets_df, yiedl_latest_df]):
        logger.error("Failed to prepare dataframes for merging")
        return False
    
    # Create training dataset
    logger.info("Creating training dataset with merged Numerai and Yiedl data...")
    
    # First merge train with targets (if train is available) 
    if train_df is not None:
        train_with_targets = train_df.join(targets_df, on='id', how='inner')
        logger.info(f"Merged train data with targets. Shape: {train_with_targets.shape}")
    else:
        train_with_targets = targets_df
        logger.info(f"Using targets as base for training data. Shape: {train_with_targets.shape}")
    
    # Merge with Yiedl historical data if available, otherwise use latest
    train_merged = None
    yiedl_for_training = yiedl_historical_df if yiedl_historical_df is not None else yiedl_latest_df
    
    if yiedl_for_training is not None:
        logger.info("Merging training data with Yiedl data (using Yiedl as base to preserve features)")
        
        # Try different merge strategies
        merge_strategies = [
            (['asset', 'date'], 'asset_date'),
            (['symbol', 'date'], 'symbol_date'),
        ]
        
        for join_cols, strategy_name in merge_strategies:
            try:
                logger.info(f"Trying merge strategy: {strategy_name}")
                
                # Check if both dataframes have the required columns
                yiedl_cols = set(yiedl_for_training.columns)
                train_cols = set(train_with_targets.columns)
                
                if all(col in yiedl_cols for col in join_cols) and all(col in train_cols for col in join_cols):
                    train_merged = yiedl_for_training.join(
                        train_with_targets,
                        on=join_cols,
                        how='left'
                    )
                    
                    # Check if merge was successful
                    target_cols = [col for col in train_merged.columns if col.startswith('target')]
                    if target_cols and train_merged.shape[0] > 100000:
                        logger.info(f"Successful merge using {strategy_name}. Shape: {train_merged.shape}")
                        break
                    else:
                        logger.warning(f"Merge {strategy_name} didn't add sufficient target data")
                        train_merged = None
                        
            except Exception as e:
                logger.warning(f"Error with merge strategy {strategy_name}: {e}")
                train_merged = None
    
    if train_merged is None:
        logger.error("All training data merge attempts failed")
        return False
    
    # Create live prediction dataset
    logger.info("Creating prediction dataset with merged Numerai and Yiedl data...")
    
    # Merge live data with Yiedl latest
    try:
        live_merged = live_df.join(
            yiedl_latest_df,
            on=['asset', 'date'],
            how='left'
        )
        logger.info(f"Merged live data with Yiedl latest. Shape: {live_merged.shape}")
    except Exception as e:
        logger.warning(f"Error merging on asset and date: {e}")
        try:
            live_merged = live_df.join(
                yiedl_latest_df,
                left_on='asset',
                right_on='symbol',
                how='left'
            )
            logger.info(f"Merged live data using asset->symbol join. Shape: {live_merged.shape}")
        except Exception as e2:
            logger.error(f"Failed to merge live data: {e2}")
            return False
    
    # Validate column count
    num_columns_train = len(train_merged.columns)
    num_columns_live = len(live_merged.columns)
    logger.info(f"Number of columns in train merged: {num_columns_train}")
    logger.info(f"Number of columns in live merged: {num_columns_live}")
    
    MIN_COLUMNS = 3000
    if num_columns_train < MIN_COLUMNS:
        logger.error(f"ERROR: Merged training data has only {num_columns_train} columns, but should have at least {MIN_COLUMNS}")
    
    # Create train/validation split (80/20 split)
    logger.info("Splitting training data into train and validation sets...")
    train_size = int(len(train_merged) * 0.8)
    
    train_final = train_merged.slice(0, train_size)
    val_final = train_merged.slice(train_size)
    
    # Save all datasets with memory-efficient parquet writing
    logger.info("Saving datasets to parquet files...")
    
    # Configure parquet writing for memory efficiency
    parquet_config = {
        'compression': 'zstd',  # Good compression ratio with fast decompression
        'compression_level': 6,
        'row_group_size': 100000,  # Smaller row groups for better memory usage
    }
    
    # Save training data
    train_file = os.path.join(PROCESSED_DATA_DIR, f"train_merged_r{current_round}_{date_str}.parquet")
    train_final.write_parquet(train_file, **parquet_config)
    logger.info(f"Saved merged train data to {train_file} with shape {train_final.shape}")
    
    # Save in TRAIN_DIR
    train_dir_file = os.path.join(TRAIN_DIR, "train_data.parquet")
    train_final.write_parquet(train_dir_file, **parquet_config)
    logger.info(f"Saved train data parquet to {train_dir_file}")
    
    # Save validation data
    val_file = os.path.join(PROCESSED_DATA_DIR, f"validation_merged_r{current_round}_{date_str}.parquet")
    val_final.write_parquet(val_file, **parquet_config)
    logger.info(f"Saved merged validation data to {val_file} with shape {val_final.shape}")
    
    # Save in VALIDATION_DIR
    val_dir_file = os.path.join(VALIDATION_DIR, "validation_data.parquet")
    val_final.write_parquet(val_dir_file, **parquet_config)
    logger.info(f"Saved validation data parquet to {val_dir_file}")
    
    # Save prediction/live data
    pred_file = os.path.join(PROCESSED_DATA_DIR, f"live_merged_r{current_round}_{date_str}.parquet")
    live_merged.write_parquet(pred_file, **parquet_config)
    logger.info(f"Saved merged prediction data to {pred_file} with shape {live_merged.shape}")
    
    # Save in PREDICTION_DIR
    pred_dir_file = os.path.join(PREDICTION_DIR, "prediction_data.parquet")
    live_merged.write_parquet(pred_dir_file, **parquet_config)
    logger.info(f"Saved prediction data parquet to {pred_dir_file}")
    
    # Create symlinks for standard filenames
    logger.info("Creating symlinks for standard filenames...")
    
    crypto_train_file = os.path.join(PROCESSED_DATA_DIR, "crypto_train.parquet")
    crypto_val_file = os.path.join(PROCESSED_DATA_DIR, "crypto_validation.parquet")
    crypto_live_file = os.path.join(PROCESSED_DATA_DIR, "crypto_live.parquet")
    
    # Create symlinks (overwrite if exists)
    for target, link in [(train_file, crypto_train_file), 
                        (val_file, crypto_val_file), 
                        (pred_file, crypto_live_file)]:
        if os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
    
    logger.info("All data splits created and symlinked successfully")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process data for Numerai Crypto using Polars')
    parser.add_argument('--use-historical', action='store_true', help='Use historical data')
    parser.add_argument('--skip-historical', action='store_true', help='Skip downloading historical data')
    parser.add_argument('--pit', type=str, help='Point-in-time date for data (format: YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if files exist')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting process_data_polars.py")
    
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
        use_historical=True,
        skip_historical=args.skip_historical,
        pit_date=args.pit
    )
    if yiedl_data is False:
        logger.error("Yiedl data processing failed")
        return False
    
    # Check files exist
    crypto_train_file = os.path.join(PROCESSED_DATA_DIR, "crypto_train.parquet")
    crypto_test_file = os.path.join(PROCESSED_DATA_DIR, "crypto_validation.parquet")
    crypto_live_file = os.path.join(PROCESSED_DATA_DIR, "crypto_live.parquet")
    
    files_exist = (os.path.exists(crypto_train_file) and 
                   os.path.exists(crypto_test_file) and
                   os.path.exists(crypto_live_file))
    
    if args.force or not files_exist:
        if args.force:
            logger.info("Force flag specified - reprocessing data")
            for file_path in [crypto_train_file, crypto_test_file, crypto_live_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed existing file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}")
        
        # Create data splits
        if create_data_splits(numerai_data, yiedl_data):
            logger.info("Data splits created successfully")
        else:
            logger.error("Data splits creation failed")
            return False
    else:
        logger.info("Using existing processed data files")
    
    logger.info("Data processing completed successfully")
    return True

if __name__ == "__main__":
    main()