#!/usr/bin/env python3
"""
process_data.py - Process data for Numerai Crypto

This script processes raw data and creates train/validation/prediction splits.
"""
import os
import sys
import logging
import argparse
import re
import pandas as pd
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

def process_numerai_data(use_historical=False, skip_historical=False, pit_date=None):
    """Process Numerai data"""
    logger.info("Processing Numerai data...")
    
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
    
    # Load the files
    numerai_data = {}
    for key, file_path in numerai_files.items():
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                numerai_data[key] = df
                logger.info(f"Loaded {key} with shape: {df.shape}")
                # Also store the file path for reference
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
    """Process Yiedl data"""
    logger.info("Processing Yiedl data...")
    
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
            latest_df = pd.read_parquet(potential_files['latest'])
        else:
            latest_df = pd.read_csv(potential_files['latest'])
        
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
                historical_df = pd.read_parquet(potential_files['historical'])
            else:
                historical_df = pd.read_csv(potential_files['historical'])
            
            yiedl_data['historical'] = historical_df
            yiedl_data['historical_path'] = potential_files['historical']
            logger.info(f"Loaded Yiedl historical data with shape: {historical_df.shape}")
        except Exception as e:
            logger.error(f"Error loading Yiedl historical data: {e}")
    elif use_historical:
        logger.warning(f"Historical data requested but file not found: {potential_files['historical']}")
    
    return yiedl_data

def create_data_splits(numerai_data, yiedl_data):
    """Create train/validation/prediction splits"""
    logger.info("Creating data splits...")
    
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Get date stamp for filenames
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Get current round if available
    current_round = None
    if 'current_round' in numerai_data:
        current_round = numerai_data['current_round']
    else:
        # Try to extract from file paths
        for key in ['live_universe_path', 'train_data_path', 'train_targets_path']:
            if key in numerai_data:
                file_path = numerai_data[key]
                match = re.search(r'r(\d+)', file_path)
                if match:
                    current_round = int(match.group(1))
                    break
    
    if current_round:
        logger.info(f"Detected tournament round: {current_round}")
    else:
        current_round = "unknown"
        logger.warning("Could not detect tournament round")
    
    # Extract necessary dataframes from the input dictionaries
    # Handle both dictionary formats - direct DataFrames or nested in a dict
    logger.debug(f"Numerai data keys: {numerai_data.keys()}")
    logger.debug(f"Yiedl data keys: {yiedl_data.keys()}")
    
    # Get data frames by checking for both formats
    if isinstance(numerai_data.get('train_data', None), pd.DataFrame):
        train_df = numerai_data.get('train_data', None)
        targets_df = numerai_data.get('train_targets', None)
        live_df = numerai_data.get('live_universe', None)
    else:
        # If we get a dictionary of dictionaries instead
        train_df = None
        for key in numerai_data:
            if key.startswith('train_data') and isinstance(numerai_data[key], pd.DataFrame):
                train_df = numerai_data[key]
                break
                
        targets_df = None
        for key in numerai_data:
            if key.startswith('train_targets') and isinstance(numerai_data[key], pd.DataFrame):
                targets_df = numerai_data[key]
                break
                
        live_df = None
        for key in numerai_data:
            if key.startswith('live_universe') and isinstance(numerai_data[key], pd.DataFrame):
                live_df = numerai_data[key]
                break
    
    # Get Yiedl dataframes
    if isinstance(yiedl_data.get('latest', None), pd.DataFrame):
        yiedl_latest_df = yiedl_data.get('latest', None)
        yiedl_historical_df = yiedl_data.get('historical', None)
    else:
        # If we get a dictionary of dictionaries instead
        yiedl_latest_df = None
        for key in yiedl_data:
            if isinstance(yiedl_data[key], pd.DataFrame) and key == 'latest':
                yiedl_latest_df = yiedl_data[key]
                break
                
        yiedl_historical_df = None
        for key in yiedl_data:
            if isinstance(yiedl_data[key], pd.DataFrame) and key == 'historical':
                yiedl_historical_df = yiedl_data[key]
                break
    
    # Check if we have required data
    if not (live_df is not None and targets_df is not None and yiedl_latest_df is not None):
        logger.error("Missing required dataframes for merging")
        if train_df is None:
            logger.error("Missing Numerai train data")
        if targets_df is None:
            logger.error("Missing Numerai target data")
        if live_df is None:
            logger.error("Missing Numerai live universe data")
        if yiedl_latest_df is None:
            logger.error("Missing Yiedl latest data")
        return False
    
    # 1. Process Numerai data - Extract asset and date from id
    logger.info("Processing Numerai data for merging...")
    
    # Process live data (always required)
    logger.debug(f"Live universe columns: {live_df.columns}")
    
    # Handle newer format with just 'symbol' column instead of 'id'
    if 'symbol' in live_df.columns and 'id' not in live_df.columns:
        logger.info("Found new format live universe file with 'symbol' column")
        
        # Check if date column exists - if not, use today's date
        if 'date' not in live_df.columns:
            from datetime import date
            today = date.today().strftime('%Y-%m-%d')
            logger.info(f"Adding date column with today's date: {today}")
            live_df['date'] = today
            
        # Set asset column to symbol
        live_df['asset'] = live_df['symbol']
        logger.info(f"Set asset column equal to symbol. Shape: {live_df.shape}")
    # Fallback to old format with 'id' column
    elif 'id' in live_df.columns:
        live_df['asset'] = live_df['id'].str.split('_').str[0]
        live_df['date'] = live_df['id'].str.split('_').str[1]
        logger.info(f"Extracted asset and date from live universe ID. Shape: {live_df.shape}")
    else:
        logger.error("Live universe missing both 'id' and 'symbol' columns, cannot proceed")
        # List available columns to help diagnose
        logger.error(f"Available columns: {list(live_df.columns)}")
        return False
    
    # Process train data (if available)
    if train_df is not None:
        logger.debug(f"Train data columns: {train_df.columns}")
        
        # Handle newer format with direct symbol and date columns
        if 'symbol' in train_df.columns and 'date' in train_df.columns:
            logger.info("Found new format train file with 'symbol' and 'date' columns")
            # Set asset column to symbol for consistent naming
            train_df['asset'] = train_df['symbol']
            logger.info(f"Set asset column equal to symbol. Shape: {train_df.shape}")
        # Fallback to old format with 'id' column
        elif 'id' in train_df.columns:
            train_df['asset'] = train_df['id'].str.split('_').str[0]
            train_df['date'] = train_df['id'].str.split('_').str[1]
            logger.info(f"Extracted asset and date from train ID. Shape: {train_df.shape}")
        else:
            logger.warning("Train data missing required columns. Will use targets data as base.")
    
    # Process targets data (required)
    logger.debug(f"Targets data columns: {targets_df.columns}")
    
    # Handle newer format with direct symbol and date columns
    if 'symbol' in targets_df.columns and 'date' in targets_df.columns:
        logger.info("Found new format targets file with 'symbol' and 'date' columns")
        # Set asset column to symbol for consistent naming
        targets_df['asset'] = targets_df['symbol']
        logger.info(f"Set asset column equal to symbol. Shape: {targets_df.shape}")
    # Fallback to old format with 'id' column
    elif 'id' in targets_df.columns:
        targets_df['asset'] = targets_df['id'].str.split('_').str[0]
        targets_df['date'] = targets_df['id'].str.split('_').str[1]
        logger.info(f"Extracted asset and date from targets ID. Shape: {targets_df.shape}")
    else:
        logger.error("Targets data missing required columns. Need either 'id' or both 'symbol' and 'date' columns.")
        # List available columns to help diagnose
        logger.error(f"Available columns: {list(targets_df.columns)}")
        return False
    
    # 2. Create train/validation merged dataset
    logger.info("Creating training dataset with merged Numerai and Yiedl data...")
    
    # First merge train with targets (if train is available) 
    train_with_targets = None
    if train_df is not None:
        train_with_targets = pd.merge(train_df, targets_df, on='id', how='inner')
        logger.info(f"Merged train data with targets. Shape: {train_with_targets.shape}")
    else:
        # If no train data, use targets as base
        train_with_targets = targets_df
        logger.info(f"Using targets as base for training data. Shape: {train_with_targets.shape}")
    
    # Now merge with Yiedl historical data if available
    # IMPORTANT: Use Yiedl historical data as BASE to preserve more feature data
    train_merged = None
    if yiedl_historical_df is not None:
        logger.info("Using Yiedl historical data as BASE for merging to preserve more feature data")
        
        # Process Yiedl historical data to ensure it has both 'asset' and 'symbol' columns
        if 'symbol' in yiedl_historical_df.columns and 'asset' not in yiedl_historical_df.columns:
            logger.info("Adding 'asset' column to Yiedl historical data (copied from 'symbol')")
            # Create a copy of 'symbol' as 'asset' but preserve the original 'symbol' column
            yiedl_historical_df['asset'] = yiedl_historical_df['symbol'].copy()
        
        # Add a symbol column from asset column if needed
        if 'asset' in yiedl_historical_df.columns and 'symbol' not in yiedl_historical_df.columns:
            logger.info("Adding 'symbol' column to Yiedl historical data (copied from 'asset')")
            # Create a copy of 'asset' as 'symbol' but preserve the original 'asset' column
            yiedl_historical_df['symbol'] = yiedl_historical_df['asset'].copy()
            
        # Ensure dates are in consistent format (string)
        if 'date' in yiedl_historical_df.columns:
            logger.info(f"Converting Yiedl historical dates to string format for consistent joining")
            # Get a sample of the date values to detect type
            date_sample = yiedl_historical_df['date'].iloc[0] if len(yiedl_historical_df) > 0 else None
            if date_sample is not None:
                logger.info(f"Yiedl historical date sample: {date_sample}, type: {type(date_sample)}")
                
                # Convert to string format if it's not already
                if not isinstance(date_sample, str):
                    try:
                        # Convert date to string format YYYY-MM-DD
                        yiedl_historical_df['date'] = yiedl_historical_df['date'].apply(
                            lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                        )
                        logger.info(f"Converted Yiedl historical dates to string format")
                    except Exception as e:
                        logger.error(f"Error converting Yiedl historical dates: {e}")
        
        # Ensure train data dates are strings too
        if 'date' in train_with_targets.columns:
            date_sample = train_with_targets['date'].iloc[0]
            if not isinstance(date_sample, str):
                try:
                    train_with_targets['date'] = train_with_targets['date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                    )
                    logger.info(f"Converted training data dates to string format")
                except Exception as e:
                    logger.error(f"Error converting training data dates: {e}")
        
        # Log the column names to assist with debugging
        logger.info(f"Train data columns: {sorted(train_with_targets.columns)}")
        logger.info(f"Yiedl historical columns: {sorted(yiedl_historical_df.columns)}")
        
        # Try multiple merge strategies - USE YIEDL AS BASE TO PRESERVE MORE ROWS
        merge_attempts = [
            # 1. First try direct merge on asset and date (Yiedl as base)
            {'on': ['asset', 'date'], 'method': 'direct'},
            # 2. Then try left_on = symbol, right_on = asset (Yiedl as base)
            {'left_on': ['symbol', 'date'], 'right_on': ['asset', 'date'], 'method': 'symbol_to_asset'},
            # 3. Then try left_on = asset, right_on = symbol (Yiedl as base)
            {'left_on': ['asset', 'date'], 'right_on': ['symbol', 'date'], 'method': 'asset_to_symbol'},
            # 4. Finally try direct merge on symbol and date (Yiedl as base)
            {'on': ['symbol', 'date'], 'method': 'symbol_direct'}
        ]
        
        for attempt in merge_attempts:
            logger.info(f"Trying merge method: {attempt['method']} (using Yiedl as base)")
            try:
                if 'on' in attempt:
                    train_merged = pd.merge(
                        yiedl_historical_df,
                        train_with_targets,
                        on=attempt['on'],
                        how='left'
                    )
                else:
                    train_merged = pd.merge(
                        yiedl_historical_df,
                        train_with_targets,
                        left_on=attempt['left_on'],
                        right_on=attempt['right_on'],
                        how='left'
                    )
                
                # Check if merge was successful by checking target column presence and row count
                yiedl_original_cols = len(yiedl_historical_df.columns)
                merged_cols = len(train_merged.columns)
                added_cols = merged_cols - yiedl_original_cols
                
                # Check if we have target columns (indicating successful merge)
                has_targets = any(col.startswith('target') for col in train_merged.columns)
                rows_preserved = len(train_merged)
                
                logger.info(f"Merge attempt '{attempt['method']}' results:")
                logger.info(f"  - Added {added_cols} columns from targets")
                logger.info(f"  - Has target columns: {has_targets}")
                logger.info(f"  - Rows preserved: {rows_preserved:,}")
                
                if has_targets and rows_preserved > 1000000:  # Success if we have targets and preserved lots of rows
                    logger.info(f"Successful merge using method '{attempt['method']}'. Shape: {train_merged.shape}")
                    
                    # Clean up duplicate symbol columns after merge
                    if 'symbol_x' in train_merged.columns and 'symbol_y' in train_merged.columns:
                        # Keep symbol_x (from Yiedl) as the main symbol column
                        train_merged['symbol'] = train_merged['symbol_x']
                        train_merged = train_merged.drop(columns=['symbol_x', 'symbol_y'])
                        logger.info("Cleaned up duplicate symbol columns after merge")
                    elif 'symbol_x' in train_merged.columns and 'symbol' not in train_merged.columns:
                        train_merged['symbol'] = train_merged['symbol_x']
                        train_merged = train_merged.drop(columns=['symbol_x'])
                        logger.info("Renamed symbol_x to symbol")
                    elif 'symbol_y' in train_merged.columns and 'symbol' not in train_merged.columns:
                        train_merged['symbol'] = train_merged['symbol_y']
                        train_merged = train_merged.drop(columns=['symbol_y'])
                        logger.info("Renamed symbol_y to symbol")
                    
                    break
                else:
                    logger.warning(f"Merge added only {added_cols} columns - trying next method")
                    train_merged = None
            except Exception as e:
                logger.warning(f"Error with merge method '{attempt['method']}': {e}")
                train_merged = None
        
        # If all merge attempts failed, fall back to manual merging (still using Yiedl as base)
        if train_merged is None:
            logger.warning("All merge attempts failed - falling back to manual symbol-based merging with Yiedl as base")
            
            # Get the target columns from Numerai
            target_cols = [c for c in train_with_targets.columns if c.startswith('target') or c in ['date', 'symbol', 'asset']]
            logger.info(f"Found {len(target_cols)} target columns in Numerai data")
            
            # Create a dictionary of target data by symbol and date
            symbol_date_targets = {}
            for _, row in train_with_targets.iterrows():
                key = (row['symbol'], row['date'])
                symbol_date_targets[key] = {col: row[col] for col in target_cols if col in row}
            
            # Start with Yiedl historical data and add targets where available
            train_merged = yiedl_historical_df.copy()
            
            # Add target columns (initialize with NaN)
            for col in target_cols:
                if col not in ['date', 'symbol', 'asset']:
                    train_merged[col] = pd.NA
            
            # Fill in target values where we have them
            for i, row in train_merged.iterrows():
                key = (row['symbol'], row['date'])
                if key in symbol_date_targets:
                    for col, value in symbol_date_targets[key].items():
                        if col not in ['date', 'symbol', 'asset']:
                            train_merged.at[i, col] = value
            
            logger.info(f"Created symbol-matched merged data with Yiedl as base. Shape: {train_merged.shape}")
            logger.info(f"Rows with targets: {train_merged[target_cols[0]].notna().sum() if target_cols else 0}")
    else:
        # If no Yiedl historical data is available, use latest data
        logger.warning("No Yiedl historical data available, trying to use Yiedl latest data")
        try:
            # Ensure dates are consistent format
            if 'date' in train_with_targets.columns and 'date' in yiedl_latest_df.columns:
                # Make sure both dates are in string format
                if not isinstance(train_with_targets['date'].iloc[0], str):
                    train_with_targets['date'] = train_with_targets['date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                    )
                    
                if not isinstance(yiedl_latest_df['date'].iloc[0], str):
                    yiedl_latest_df['date'] = yiedl_latest_df['date'].apply(
                        lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                    )
                
            train_merged = pd.merge(
                yiedl_latest_df,
                train_with_targets,
                on=['asset', 'date'],
                how='left'
            )
            logger.info(f"Merged training data with Yiedl latest as fallback (Yiedl as base). Shape: {train_merged.shape}")
        except Exception as e:
            logger.error(f"Error merging with Yiedl latest: {e}")
            # Last resort - manually insert target data into Yiedl based on matching symbols
            logger.warning("Attempting manual symbol-based data merging as last resort (using Yiedl as base)")
            
            # Get the target columns from Numerai
            target_cols = [c for c in train_with_targets.columns if c.startswith('target') or c in ['date', 'symbol', 'asset']]
            logger.info(f"Found {len(target_cols)} target columns in Numerai data")
            
            # Create a dictionary of target data by symbol
            symbol_targets = {}
            for symbol in train_with_targets['symbol'].unique():
                # Get the latest row for this symbol
                symbol_data = train_with_targets[train_with_targets['symbol'] == symbol].iloc[-1]
                symbol_targets[symbol] = {col: symbol_data[col] for col in target_cols if col in symbol_data}
            
            # Start with Yiedl latest data as base
            train_merged = yiedl_latest_df.copy()
            
            # Add target columns (initialize with NaN)
            for col in target_cols:
                if col not in ['date', 'symbol', 'asset'] and col not in train_merged.columns:
                    train_merged[col] = pd.NA
            
            # Fill in target values where we have them
            for symbol in train_merged['symbol'].unique():
                if symbol in symbol_targets:
                    mask = train_merged['symbol'] == symbol
                    for col, value in symbol_targets[symbol].items():
                        if col not in ['date', 'symbol', 'asset']:
                            train_merged.loc[mask, col] = value
            
            logger.info(f"Created symbol-matched merged data with Yiedl as base. Shape: {train_merged.shape}")
            target_col = next((c for c in target_cols if c.startswith('target')), None)
            if target_col:
                logger.info(f"Rows with targets: {train_merged[target_col].notna().sum()}")
    
    # 3. Create live prediction dataset
    logger.info("Creating prediction dataset with merged Numerai and Yiedl data...")
    
    # Process Yiedl data to ensure it has both 'asset' and 'symbol' columns
    if 'symbol' in yiedl_latest_df.columns and 'asset' not in yiedl_latest_df.columns:
        logger.info("Adding 'asset' column to Yiedl data (copied from 'symbol')")
        # Create a copy of 'symbol' as 'asset' but preserve the original 'symbol' column
        yiedl_latest_df['asset'] = yiedl_latest_df['symbol'].copy()
    elif 'asset' in yiedl_latest_df.columns and 'symbol' not in yiedl_latest_df.columns:
        logger.info("Adding 'symbol' column to Yiedl data (copied from 'asset')")
        # Create a copy of 'asset' as 'symbol' but preserve the original 'asset' column
        yiedl_latest_df['symbol'] = yiedl_latest_df['asset'].copy()
        
    # Ensure dates are in consistent format (string)
    if 'date' in yiedl_latest_df.columns:
        logger.info(f"Converting Yiedl dates to string format for consistent joining")
        # Get a sample of the date values to detect type
        date_sample = yiedl_latest_df['date'].iloc[0]
        logger.info(f"Yiedl date sample: {date_sample}, type: {type(date_sample)}")
        
        # Convert to string format if it's not already
        if not isinstance(date_sample, str):
            try:
                # Convert date to string format YYYY-MM-DD
                yiedl_latest_df['date'] = yiedl_latest_df['date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                )
                logger.info(f"Converted Yiedl dates to string format")
            except Exception as e:
                logger.error(f"Error converting Yiedl dates: {e}")
    
    # Also make sure the live_df dates are strings for consistency
    if 'date' in live_df.columns:
        date_sample = live_df['date'].iloc[0]
        if not isinstance(date_sample, str):
            try:
                live_df['date'] = live_df['date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                )
                logger.info(f"Converted live universe dates to string format")
            except Exception as e:
                logger.error(f"Error converting live universe dates: {e}")
    
    # Merge live data with Yiedl latest
    try:
        live_merged = pd.merge(
            live_df,
            yiedl_latest_df,
            on=['asset', 'date'],
            how='left'
        )
        logger.info(f"Merged on 'asset' and 'date' columns")
    except KeyError as e:
        logger.warning(f"Error merging on 'asset' column: {e}")
        logger.info(f"Yiedl columns: {yiedl_latest_df.columns[:10]}")
        logger.info(f"Trying alternate merge on 'symbol' column")
        # Try merging on symbol instead
        live_merged = pd.merge(
            live_df,
            yiedl_latest_df,
            left_on='asset',
            right_on='symbol', 
            how='left'
        )
    logger.info(f"Merged live data with Yiedl latest. Shape: {live_merged.shape}")
    
    # Clean up duplicate symbol columns in live data after merge
    if 'symbol_x' in live_merged.columns and 'symbol_y' in live_merged.columns:
        # Keep symbol_x (from live universe) as the main symbol column
        live_merged['symbol'] = live_merged['symbol_x']
        live_merged = live_merged.drop(columns=['symbol_x', 'symbol_y'])
        logger.info("Cleaned up duplicate symbol columns in live data after merge")
    elif 'symbol_x' in live_merged.columns and 'symbol' not in live_merged.columns:
        live_merged['symbol'] = live_merged['symbol_x']
        live_merged = live_merged.drop(columns=['symbol_x'])
        logger.info("Renamed symbol_x to symbol in live data")
    elif 'symbol_y' in live_merged.columns and 'symbol' not in live_merged.columns:
        live_merged['symbol'] = live_merged['symbol_y']
        live_merged = live_merged.drop(columns=['symbol_y'])
        logger.info("Renamed symbol_y to symbol in live data")
    
    # Get column counts
    num_columns_train = len(train_merged.columns)
    num_columns_live = len(live_merged.columns)
    logger.info(f"Number of columns in train merged: {num_columns_train}")
    logger.info(f"Number of columns in live merged: {num_columns_live}")
    
    # Validate column count - should have at least 3000 columns
    MIN_COLUMNS = 3000
    if num_columns_train < MIN_COLUMNS:
        logger.error(f"ERROR: Merged training data has only {num_columns_train} columns, but should have at least {MIN_COLUMNS}")
        logger.error("This indicates the data processing pipeline failed to properly merge Numerai and Yiedl data")
        # Continue processing but log the error
    
    # 4. Create train/validation split (80/20 split of train data)
    logger.info("Splitting training data into train and validation sets...")
    
    # Calculate train and validation sizes (80% train, 20% validation)
    train_size = int(len(train_merged) * 0.8)
    
    # Split the training data
    train_final = train_merged.iloc[:train_size]
    val_final = train_merged.iloc[train_size:]
    
    # 5. Save all datasets
    # Save training data
    train_file = os.path.join(PROCESSED_DATA_DIR, f"train_merged_r{current_round}_{date_str}.parquet")
    train_final.to_parquet(train_file)
    logger.info(f"Saved merged train data to {train_file} with shape {train_final.shape}")
    
    # Save as parquet in the TRAIN_DIR
    train_dir_file = os.path.join(TRAIN_DIR, "train_data.parquet")
    train_final.to_parquet(train_dir_file)
    logger.info(f"Saved train data parquet to {train_dir_file}")
    
    # Save validation data
    val_file = os.path.join(PROCESSED_DATA_DIR, f"validation_merged_r{current_round}_{date_str}.parquet")
    val_final.to_parquet(val_file)
    logger.info(f"Saved merged validation data to {val_file} with shape {val_final.shape}")
    
    # Save as parquet in the VALIDATION_DIR
    val_dir_file = os.path.join(VALIDATION_DIR, "validation_data.parquet")
    val_final.to_parquet(val_dir_file)
    logger.info(f"Saved validation data parquet to {val_dir_file}")
    
    # Save prediction/live data
    pred_file = os.path.join(PROCESSED_DATA_DIR, f"live_merged_r{current_round}_{date_str}.parquet")
    live_merged.to_parquet(pred_file)
    logger.info(f"Saved merged prediction data to {pred_file} with shape {live_merged.shape}")
    
    # Save as parquet in the PREDICTION_DIR
    pred_dir_file = os.path.join(PREDICTION_DIR, "prediction_data.parquet")
    live_merged.to_parquet(pred_dir_file)
    logger.info(f"Saved prediction data parquet to {pred_dir_file}")
    
    # Create symlinks for standard filenames that scripts will look for
    logger.info("Creating symlinks for standard filenames...")
    
    # Standard files expected by other scripts
    crypto_train_file = os.path.join(PROCESSED_DATA_DIR, "crypto_train.parquet")
    crypto_val_file = os.path.join(PROCESSED_DATA_DIR, "crypto_validation.parquet")
    crypto_live_file = os.path.join(PROCESSED_DATA_DIR, "crypto_live.parquet")
    
    # Create symlinks (overwrite if exists)
    if os.path.exists(crypto_train_file):
        os.remove(crypto_train_file)
    os.symlink(train_file, crypto_train_file)
    
    if os.path.exists(crypto_val_file):
        os.remove(crypto_val_file)
    os.symlink(val_file, crypto_val_file)
    
    if os.path.exists(crypto_live_file):
        os.remove(crypto_live_file)
    os.symlink(pred_file, crypto_live_file)
    
    logger.info("All data splits created and symlinked successfully")
    
    # Generate historical dataset with multiple dates for time-series feature generation
    logger.info("Generating historical dataset with multiple dates for time-series feature generation")
    
    # Get the targets dataframe to extract historical dates
    if targets_df is not None:
        logger.info("Using targets data to create historical dataset")
        
        # Get unique symbols from both datasets
        target_symbols = set(targets_df['symbol'].unique())
        if yiedl_latest_df is not None:
            yiedl_symbols = set(yiedl_latest_df['symbol'].unique())
            common_symbols = target_symbols.intersection(yiedl_symbols)
            logger.info(f"Found {len(common_symbols)} common symbols between datasets")
        else:
            common_symbols = target_symbols
            logger.info(f"Using {len(common_symbols)} symbols from targets data only")
        
        # Get all dates from targets
        all_dates = sorted(targets_df['date'].unique())
        logger.info(f"Found {len(all_dates)} unique dates in targets data")
        logger.info(f"Date range: {min(all_dates)} to {max(all_dates)}")
        
        # Filter targets dataframe to only include common symbols
        filtered_targets = targets_df[targets_df['symbol'].isin(common_symbols)].copy()
        logger.info(f"Filtered targets data to shape: {filtered_targets.shape}")
        
        # Extract feature columns from Yiedl data
        if yiedl_latest_df is not None:
            feature_cols = [c for c in yiedl_latest_df.columns 
                           if c not in ['date', 'symbol', 'asset', 'id']]
            logger.info(f"Using {len(feature_cols)} feature columns from Yiedl data")
            
            # Create feature dataframe with one row per symbol
            logger.info("Creating feature dataframe with all symbols")
            feature_data = {}
            
            # Create empty dataframe with all feature columns at once
            for symbol in common_symbols:
                # Get the features for this symbol
                symbol_data = yiedl_latest_df[yiedl_latest_df['symbol'] == symbol]
                if len(symbol_data) > 0:
                    # Use the latest row if multiple exist
                    row = symbol_data.iloc[-1]
                    feature_data[symbol] = {col: row[col] for col in feature_cols if col in row}
            
            # Convert feature_data to DataFrame
            feature_df = pd.DataFrame.from_dict(feature_data, orient='index')
            logger.info(f"Created feature dataframe with shape: {feature_df.shape}")
            
            # Merge with filtered targets
            logger.info("Merging targets with features")
            historical_df = filtered_targets.merge(
                feature_df,
                left_on='symbol',
                right_index=True,
                how='left'
            )
            
            logger.info(f"Created historical dataset with shape: {historical_df.shape}")
            
            # Verify the dataset has multiple dates
            unique_dates = historical_df['date'].unique()
            logger.info(f"Final dataset has {len(unique_dates)} unique dates")
            logger.info(f"Date range: {min(unique_dates)} to {max(unique_dates)}")
            
            # Save as crypto_train.parquet (overwriting the symlink)
            if os.path.exists(crypto_train_file) and os.path.islink(crypto_train_file):
                os.remove(crypto_train_file)
            
            historical_df.to_parquet(crypto_train_file)
            logger.info(f"Saved historical dataset to {crypto_train_file}")
        else:
            logger.warning("No Yiedl data available for historical dataset, skipping")
    else:
        logger.warning("No targets data available for historical dataset, skipping")
    
    # Validate the final datasets
    try:
        # Import validation function
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline_utils import PipelineUtils
        
        validation_result = PipelineUtils.validate_merged_dataset()
        if validation_result:
            logger.info("Merged dataset validation passed")
        else:
            logger.error("Merged dataset validation failed")
            
    except ImportError:
        logger.warning("Could not import validation utilities, skipping validation")
    except Exception as e:
        logger.error(f"Error validating merged dataset: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process data for Numerai Crypto')
    parser.add_argument('--use-historical', action='store_true', help='Use historical data')
    parser.add_argument('--skip-historical', action='store_true', help='Skip downloading historical data')
    parser.add_argument('--pit', type=str, help='Point-in-time date for data (format: YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if files exist')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up more detailed logging if debug mode is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Add a file handler for permanent debug logs
        debug_file = os.path.join("/media/knight2/EDB/numer_crypto_temp/log", "process_data_debug.log")
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.debug("Debug logging enabled")
    
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
        use_historical=True,  # Always use historical data when available
        skip_historical=args.skip_historical,
        pit_date=args.pit
    )
    if yiedl_data is False:
        logger.error("Yiedl data processing failed")
        return False
        
    # Log more detailed information about the loaded data
    logger.info(f"Yiedl data keys: {list(yiedl_data.keys())}")
    import pandas as pd  # Make sure pandas is imported in this scope
    for key, item in yiedl_data.items():
        if isinstance(item, pd.DataFrame):
            logger.info(f"Yiedl {key} dataframe: {item.shape}, columns: {list(item.columns[:5])}")
        elif key.endswith('_path') and item:
            logger.info(f"Yiedl {key}: {item}")
    
    # Check for force flag or if files don't exist
    crypto_train_file = os.path.join(PROCESSED_DATA_DIR, "crypto_train.parquet")
    crypto_test_file = os.path.join(PROCESSED_DATA_DIR, "crypto_validation.parquet")  # Updated file name
    crypto_live_file = os.path.join(PROCESSED_DATA_DIR, "crypto_live.parquet")
    
    files_exist = (os.path.exists(crypto_train_file) and 
                   os.path.exists(crypto_test_file) and
                   os.path.exists(crypto_live_file))
    
    # Check column count in existing training file if it exists
    if files_exist and not args.force:
        try:
            import pandas as pd
            df = pd.read_parquet(crypto_train_file)
            column_count = len(df.columns)
            logger.info(f"Existing training file has {column_count} columns")
            
            # If column count is too low, force reprocessing
            if column_count < 3000:
                logger.warning(f"Training file has only {column_count} columns (less than 3000). Forcing reprocessing.")
                args.force = True
        except Exception as e:
            logger.error(f"Error checking existing training file: {e}")
            args.force = True
    
    if args.force or not files_exist:
        if args.force:
            logger.info("Force flag specified - reprocessing data")
            
            # Remove existing files if they exist (to ensure clean processing)
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
            
            # Final check to ensure the merged datasets have sufficient columns
            try:
                import pandas as pd
                df = pd.read_parquet(crypto_train_file)
                column_count = len(df.columns)
                logger.info(f"Final training file has {column_count} columns")
                
                if column_count < 3000:
                    logger.error(f"WARNING: Final merged dataset has only {column_count} columns (less than 3000)")
                    logger.error("The merged dataset does not have enough features. This suggests the merging failed.")
                    
                    # Log input data shapes for debugging
                    if 'train_data' in numerai_data:
                        logger.info(f"Numerai train data shape: {numerai_data['train_data'].shape}")
                    if 'train_targets' in numerai_data:
                        logger.info(f"Numerai targets shape: {numerai_data['train_targets'].shape}")
                    if 'live_universe' in numerai_data:
                        logger.info(f"Numerai live universe shape: {numerai_data['live_universe'].shape}")
                    if 'latest' in yiedl_data:
                        logger.info(f"Yiedl latest data shape: {yiedl_data['latest'].shape}")
                    if 'historical' in yiedl_data:
                        logger.info(f"Yiedl historical data shape: {yiedl_data['historical'].shape}")
            except Exception as e:
                logger.error(f"Error checking final training file: {e}")
        else:
            logger.error("Data splits creation failed")
            return False
    else:
        logger.info("Using existing processed data files")
        
        # Check column count in existing training file
        try:
            import pandas as pd
            df = pd.read_parquet(crypto_train_file)
            column_count = len(df.columns)
            logger.info(f"Existing training file has {column_count} columns")
            
            # Provide a warning if column count is low
            if column_count < 3000:
                logger.warning(f"CAUTION: Existing training file has only {column_count} columns")
                logger.warning("This may cause issues with feature generation. Consider using --force to reprocess.")
        except Exception as e:
            logger.error(f"Error checking existing training file: {e}")
        
        return True
    
    logger.info("Data processing completed successfully")
    return True

if __name__ == "__main__":
    main()
