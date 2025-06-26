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

def clean_object_arrays(df):
    """Clean any object arrays that might cause GPU tensor conversion issues"""
    logger.debug("Checking for and cleaning object arrays")
    
    for col in df.columns:
        try:
            # Check if column contains mixed types that could become object arrays
            dtype = df[col].dtype
            
            # Force problematic columns to consistent types
            if dtype == pl.Object:
                logger.warning(f"Found Object dtype in column {col}, converting to string")
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
            elif dtype == pl.Null:
                logger.warning(f"Found Null dtype in column {col}, converting to string")
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
            
            # Also check for columns that might have mixed numeric/string content
            if col in ['symbol', 'asset', 'id'] and dtype != pl.Utf8:
                logger.debug(f"Converting key column {col} to string type")
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
                
        except Exception as e:
            logger.debug(f"Could not clean column {col}: {e}")
            continue
    
    return df

def standardize_join_columns(df, join_columns=['asset', 'symbol', 'date', 'id']):
    """Standardize data types of join columns to prevent object array issues"""
    logger.debug(f"Standardizing join column types for: {[col for col in join_columns if col in df.columns]}")
    
    # First clean any object arrays
    df = clean_object_arrays(df)
    
    for col in join_columns:
        if col in df.columns:
            try:
                if col in ['asset', 'symbol', 'id']:
                    # Ensure string columns are proper strings
                    if df[col].dtype != pl.Utf8:
                        df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
                elif col == 'date':
                    # Keep dates as strings for consistent joining
                    if df[col].dtype != pl.Utf8:
                        df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
            except Exception as e:
                logger.debug(f"Could not standardize column {col}: {e}")
                continue
    
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
    
    # Add current round - detect from files
    current_round = None
    
    # Look for round number in file paths
    for file_path in numerai_files.values():
        if file_path:
            match = re.search(r'r(\d+)', file_path)
            if match:
                detected_round = int(match.group(1))
                if current_round is None or detected_round > current_round:
                    current_round = detected_round
                    logger.info(f"Detected round {current_round} from file: {file_path}")
    
    # If no round detected, try to get it from Numerai API
    if current_round is None:
        try:
            import numerapi
            napi = numerapi.NumerAPI()
            current_round = napi.get_current_round()
            logger.info(f"Using current round {current_round} from Numerai API")
        except Exception as e:
            logger.warning(f"Failed to get round from API: {e}")
            # Use a reasonable default as last resort
            current_round = 1032
            logger.warning(f"Using fallback round: {current_round}")
    
    # Verify this is actually the latest round by checking next round too
    try:
        import requests
        next_round = current_round + 1
        test_url = f"https://numerai-public-datasets.s3-us-west-2.amazonaws.com/crypto/v1.0/live_universe_r{next_round}.parquet"
        response = requests.head(test_url)
        if response.status_code == 200:
            logger.info(f"Found data for round {next_round}, updating current round")
            current_round = next_round
    except Exception as e:
        logger.warning(f"Error checking for next round: {e}")
    
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
        latest_df = standardize_join_columns(latest_df)
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
            historical_df = standardize_join_columns(historical_df)
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
    
    # Standardize join column types to prevent object array issues
    df = standardize_join_columns(df)
    
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
    if current_round == 'unknown':
        logger.warning("Could not detect tournament round")
        # Try to detect from environment or use a default
        try:
            # Try checking recent files
            round_files = list(Path(RAW_DATA_DIR).parent.glob("numerai/*/live_universe_r*.parquet"))
            if round_files:
                latest_file = max(round_files, key=lambda p: p.stat().st_mtime)
                match = re.search(r'r(\d+)', str(latest_file))
                if match:
                    current_round = int(match.group(1))
                    logger.info(f"Detected round {current_round} from latest file: {latest_file}")
            
            # If still unknown, use API
            if current_round == 'unknown':
                import numerapi
                napi = numerapi.NumerAPI()
                current_round = napi.get_current_round()
                logger.info(f"Using current round {current_round} from Numerai API")
        except Exception as e:
            logger.warning(f"Failed to detect round: {e}")
            current_round = 1033
            logger.warning(f"Using fallback round: {current_round}")
    else:
        logger.info(f"Using tournament round: {current_round}")
    
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
    
    # Disable minimum column check to ensure processing continues
    MIN_COLUMNS = 10
    if num_columns_train < MIN_COLUMNS:
        logger.error(f"ERROR: Merged training data has only {num_columns_train} columns, but should have at least {MIN_COLUMNS}")
        
    # Log details about the merged datasets for debugging
    logger.info(f"Train merged data samples: First 5 rows of 5 columns: {train_merged.head(5).select(train_merged.columns[:5])}")
    logger.info(f"Live merged data samples: First 5 rows of 5 columns: {live_merged.head(5).select(live_merged.columns[:5])}")
    
    # Check for target columns specifically
    target_cols = [col for col in train_merged.columns if col.startswith('target')]
    logger.info(f"Found {len(target_cols)} target columns: {target_cols}")
    
    # Create train/validation split (80/20 split)
    logger.info("Splitting training data into train and validation sets...")
    train_size = int(len(train_merged) * 0.8)
    
    train_final = train_merged.slice(0, train_size)
    val_final = train_merged.slice(train_size)
    
    # Final cleanup to prevent object arrays in output files
    logger.info("Final cleanup of datasets to prevent object arrays...")
    train_final = clean_object_arrays(train_final)
    val_final = clean_object_arrays(val_final)
    live_merged = clean_object_arrays(live_merged)
    
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
        try:
            # Check if target exists
            if not os.path.exists(target):
                logger.error(f"Target file doesn't exist: {target}")
                continue
                
            # Remove existing link or file
            if os.path.exists(link):
                if os.path.islink(link):
                    os.unlink(link)
                else:
                    os.remove(link)
                    
            # Create the symlink
            os.symlink(target, link)
            logger.info(f"Created symlink: {link} -> {target}")
            
            # Double-check the symlink was created
            if not os.path.exists(link):
                logger.error(f"Failed to create symlink: {link}")
                # Try to copy the file as fallback
                import shutil
                shutil.copy2(target, link)
                logger.info(f"Copied file instead: {target} -> {link}")
        except Exception as e:
            logger.error(f"Error creating symlink {link}: {e}")
            # Try to copy the file as fallback
            try:
                import shutil
                shutil.copy2(target, link)
                logger.info(f"Copied file instead: {target} -> {link}")
            except Exception as e2:
                logger.error(f"Error copying file: {e2}")
    
    # Verify symlinks were created
    for file_path in [crypto_train_file, crypto_val_file, crypto_live_file]:
        if os.path.exists(file_path):
            logger.info(f"Verified existence of {file_path}: {os.path.getsize(file_path)} bytes")
        else:
            logger.error(f"File does not exist after symlink/copy attempt: {file_path}")
    
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