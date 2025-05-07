#!/usr/bin/env python3
"""
Data splitting utilities for time series data.

This module provides functions for creating train/validation/test splits
with proper handling of time series data, including temporal splits and
walk-forward validation.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
from datetime import datetime, timedelta

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from utils.memory_utils import optimize_dataframe_memory, log_memory_usage
from config.settings import TRAIN_DIR, VALIDATION_DIR, PREDICTION_DIR

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

def create_temporal_split(df: pd.DataFrame, 
                         date_col: str = 'date',
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         group_col: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Create temporal train/validation/test splits for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Column containing dates
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        test_ratio (float): Ratio of data for testing
        group_col (str, optional): Column to group by (e.g., 'symbol' or 'asset')
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train, validation, and test DataFrames
    """
    log_memory_usage("Before creating temporal split:")
    
    # Ensure the ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
        logger.warning(f"Adjusted ratios to sum to 1: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime. Cannot create temporal split.")
            return {'train': df_copy, 'validation': pd.DataFrame(), 'test': pd.DataFrame()}
    
    # Sort by date
    df_copy = df_copy.sort_values(date_col)
    
    # If group_col is provided, ensure each group has the same date range
    if group_col is not None and group_col in df_copy.columns:
        groups = df_copy[group_col].unique()
        logger.info(f"Creating temporal split with {len(groups)} groups")
        
        # Check date ranges for each group
        min_dates = {}
        max_dates = {}
        for group in groups:
            group_df = df_copy[df_copy[group_col] == group]
            min_dates[group] = group_df[date_col].min()
            max_dates[group] = group_df[date_col].max()
        
        # Find common date range
        common_min_date = max(min_dates.values())
        common_max_date = min(max_dates.values())
        
        # If common date range is valid
        if common_min_date <= common_max_date:
            logger.info(f"Common date range: {common_min_date} to {common_max_date}")
            df_copy = df_copy[(df_copy[date_col] >= common_min_date) & (df_copy[date_col] <= common_max_date)]
        else:
            logger.warning("No common date range across groups. Using full date range for each group.")
            # Process each group separately and then combine
            train_dfs = []
            val_dfs = []
            test_dfs = []
            
            for group in groups:
                group_df = df_copy[df_copy[group_col] == group]
                group_dates = sorted(group_df[date_col].unique())
                n_dates = len(group_dates)
                
                train_end_idx = int(n_dates * train_ratio)
                val_end_idx = int(n_dates * (train_ratio + val_ratio))
                
                train_dates = group_dates[:train_end_idx]
                val_dates = group_dates[train_end_idx:val_end_idx]
                test_dates = group_dates[val_end_idx:]
                
                train_dfs.append(group_df[group_df[date_col].isin(train_dates)])
                val_dfs.append(group_df[group_df[date_col].isin(val_dates)])
                test_dfs.append(group_df[group_df[date_col].isin(test_dates)])
            
            train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
            val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
            test_df = pd.concat(test_dfs, axis=0, ignore_index=True)
            
            log_memory_usage("After creating temporal split:")
            
            return {
                'train': optimize_dataframe_memory(train_df),
                'validation': optimize_dataframe_memory(val_df),
                'test': optimize_dataframe_memory(test_df)
            }
    
    # Get unique dates
    dates = sorted(df_copy[date_col].unique())
    n_dates = len(dates)
    
    # Calculate split indices
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    # Get date cutoffs
    train_end_date = dates[train_end_idx - 1]
    val_end_date = dates[val_end_idx - 1]
    
    logger.info(f"Split dates - Train: up to {train_end_date}, Validation: to {val_end_date}, Test: after {val_end_date}")
    
    # Create splits
    train_df = df_copy[df_copy[date_col] <= train_end_date]
    val_df = df_copy[(df_copy[date_col] > train_end_date) & (df_copy[date_col] <= val_end_date)]
    test_df = df_copy[df_copy[date_col] > val_end_date]
    
    logger.info(f"Split sizes - Train: {len(train_df)} rows, Validation: {len(val_df)} rows, Test: {len(test_df)} rows")
    log_memory_usage("After creating temporal split:")
    
    return {
        'train': optimize_dataframe_memory(train_df),
        'validation': optimize_dataframe_memory(val_df),
        'test': optimize_dataframe_memory(test_df)
    }

def create_prediction_dataset(df: pd.DataFrame, 
                             prediction_window: int = 30,
                             date_col: str = 'date') -> pd.DataFrame:
    """
    Create a dataset for prediction using the most recent data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        prediction_window (int): Number of most recent days to include
        date_col (str): Column containing dates
        
    Returns:
        pd.DataFrame: DataFrame with data for prediction
    """
    log_memory_usage("Before creating prediction dataset:")
    
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime. Using all data for prediction.")
            return df_copy
    
    # Sort by date
    df_copy = df_copy.sort_values(date_col)
    
    # Get the most recent date
    most_recent_date = df_copy[date_col].max()
    
    # Calculate cutoff date
    cutoff_date = most_recent_date - pd.Timedelta(days=prediction_window)
    
    # Filter to only include data after cutoff date
    prediction_df = df_copy[df_copy[date_col] > cutoff_date]
    
    logger.info(f"Created prediction dataset with {len(prediction_df)} rows from {cutoff_date} to {most_recent_date}")
    log_memory_usage("After creating prediction dataset:")
    
    return optimize_dataframe_memory(prediction_df)

def generate_walk_forward_folds(df: pd.DataFrame,
                              date_col: str = 'date',
                              n_folds: int = 5,
                              training_window: int = 365,
                              validation_window: int = 30,
                              group_col: Optional[str] = None) -> List[Dict[str, pd.DataFrame]]:
    """
    Generate walk-forward validation folds for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Column containing dates
        n_folds (int): Number of folds to generate
        training_window (int): Number of days for training in each fold
        validation_window (int): Number of days for validation in each fold
        group_col (str, optional): Column to group by (e.g., 'symbol' or 'asset')
        
    Returns:
        List[Dict[str, pd.DataFrame]]: List of dictionaries with train and validation DataFrames for each fold
    """
    log_memory_usage("Before generating walk-forward folds:")
    
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime. Cannot generate walk-forward folds.")
            return []
    
    # Sort by date
    df_copy = df_copy.sort_values(date_col)
    
    # Get unique dates
    dates = sorted(df_copy[date_col].unique())
    
    # If we don't have enough data for the specified windows and folds, adjust
    required_days = (training_window + validation_window) * n_folds
    if len(dates) < required_days:
        logger.warning(f"Not enough data for {n_folds} folds with specified windows. Adjusting parameters.")
        days_per_fold = len(dates) // n_folds
        training_window = int(days_per_fold * 0.8)
        validation_window = days_per_fold - training_window
        logger.info(f"Adjusted windows: training={training_window} days, validation={validation_window} days")
    
    # Generate folds
    folds = []
    
    # Start from the end of the data and work backwards
    end_idx = len(dates) - 1
    
    for fold in range(n_folds):
        # Calculate window indices
        val_start_idx = max(0, end_idx - validation_window + 1)
        train_start_idx = max(0, val_start_idx - training_window)
        
        # Get date ranges
        val_start_date = dates[val_start_idx]
        val_end_date = dates[end_idx]
        train_start_date = dates[train_start_idx]
        train_end_date = dates[val_start_idx - 1] if val_start_idx > 0 else dates[0]
        
        # Create train and validation sets
        train_df = df_copy[(df_copy[date_col] >= train_start_date) & (df_copy[date_col] <= train_end_date)]
        val_df = df_copy[(df_copy[date_col] >= val_start_date) & (df_copy[date_col] <= val_end_date)]
        
        # If group_col is provided, ensure each group has data in both train and validation sets
        if group_col is not None and group_col in df_copy.columns:
            train_groups = set(train_df[group_col].unique())
            val_groups = set(val_df[group_col].unique())
            common_groups = train_groups.intersection(val_groups)
            
            if len(common_groups) < len(train_groups):
                logger.warning(f"Fold {fold+1}: Only {len(common_groups)} out of {len(train_groups)} groups"
                              f" have data in both train and validation sets")
            
            # Filter to only include common groups
            train_df = train_df[train_df[group_col].isin(common_groups)]
            val_df = val_df[val_df[group_col].isin(common_groups)]
        
        # Add fold to list
        folds.append({
            'train': optimize_dataframe_memory(train_df),
            'validation': optimize_dataframe_memory(val_df),
            'fold': fold + 1,
            'train_start': train_start_date,
            'train_end': train_end_date,
            'val_start': val_start_date,
            'val_end': val_end_date
        })
        
        # Update end index for next fold
        end_idx = train_start_idx - 1
        
        # Break if we've used all the data
        if end_idx < 0:
            break
    
    # Log fold information
    for i, fold in enumerate(folds):
        logger.info(f"Fold {i+1}: Train: {fold['train_start']} to {fold['train_end']} ({len(fold['train'])} rows), "
                   f"Validation: {fold['val_start']} to {fold['val_end']} ({len(fold['validation'])} rows)")
    
    log_memory_usage("After generating walk-forward folds:")
    
    return folds

def save_split_datasets(splits: Dict[str, pd.DataFrame], asset_col: str = 'symbol') -> Dict[str, str]:
    """
    Save train/validation/test splits to disk.
    
    Args:
        splits (Dict[str, pd.DataFrame]): Dictionary with train, validation, and test DataFrames
        asset_col (str): Column containing asset/symbol information
        
    Returns:
        Dict[str, str]: Dictionary with paths to saved files
    """
    from utils.data.io import save_dataframe
    
    # Create directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    saved_paths = {}
    
    # Save train data
    if 'train' in splits and not splits['train'].empty:
        train_df = splits['train']
        train_path = os.path.join(TRAIN_DIR, f"train_data.parquet")
        try:
            save_dataframe(train_df, train_path)
            saved_paths['train'] = train_path
            logger.info(f"Saved train data to {train_path}")
            
            # Save train data by asset
            assets = train_df[asset_col].unique()
            for asset in assets:
                asset_df = train_df[train_df[asset_col] == asset]
                asset_path = os.path.join(TRAIN_DIR, f"{asset}.parquet")
                save_dataframe(asset_df, asset_path)
                saved_paths[f'train_{asset}'] = asset_path
            logger.info(f"Saved {len(assets)} individual asset train files")
        except Exception as e:
            logger.error(f"Error saving train data: {e}")
    
    # Save validation data
    if 'validation' in splits and not splits['validation'].empty:
        val_df = splits['validation']
        val_path = os.path.join(VALIDATION_DIR, f"validation_data.parquet")
        try:
            save_dataframe(val_df, val_path)
            saved_paths['validation'] = val_path
            logger.info(f"Saved validation data to {val_path}")
            
            # Save validation data by asset
            assets = val_df[asset_col].unique()
            for asset in assets:
                asset_df = val_df[val_df[asset_col] == asset]
                asset_path = os.path.join(VALIDATION_DIR, f"{asset}.parquet")
                save_dataframe(asset_df, asset_path)
                saved_paths[f'validation_{asset}'] = asset_path
            logger.info(f"Saved {len(assets)} individual asset validation files")
        except Exception as e:
            logger.error(f"Error saving validation data: {e}")
    
    # Save test/prediction data
    if 'test' in splits and not splits['test'].empty:
        test_df = splits['test']
        test_path = os.path.join(PREDICTION_DIR, f"prediction_data.parquet")
        try:
            save_dataframe(test_df, test_path)
            saved_paths['prediction'] = test_path
            logger.info(f"Saved prediction data to {test_path}")
            
            # Save prediction data by asset
            assets = test_df[asset_col].unique()
            for asset in assets:
                asset_df = test_df[test_df[asset_col] == asset]
                asset_path = os.path.join(PREDICTION_DIR, f"{asset}.parquet")
                save_dataframe(asset_df, asset_path)
                saved_paths[f'prediction_{asset}'] = asset_path
            logger.info(f"Saved {len(assets)} individual asset prediction files")
        except Exception as e:
            logger.error(f"Error saving prediction data: {e}")
    
    return saved_paths

if __name__ == "__main__":
    # Test data splitting
    import random
    
    # Create test data
    n_assets = 3
    n_days = 100
    assets = [f"ASSET_{i}" for i in range(1, n_assets + 1)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    data = []
    for asset in assets:
        price = 100
        volume = 10000
        for date in dates:
            # Random walk for price
            price *= (1 + random.uniform(-0.03, 0.03))
            # Random volume
            volume *= (1 + random.uniform(-0.2, 0.2))
            data.append({
                'date': date,
                'symbol': asset,
                'price': price,
                'volume': volume
            })
    
    test_df = pd.DataFrame(data)
    logger.info(f"Created test data with shape {test_df.shape}")
    
    # Test temporal split
    splits = create_temporal_split(test_df, date_col='date', group_col='symbol')
    logger.info(f"Temporal split sizes - Train: {len(splits['train'])} rows, "
               f"Validation: {len(splits['validation'])} rows, "
               f"Test: {len(splits['test'])} rows")
    
    # Test walk-forward folds
    folds = generate_walk_forward_folds(test_df, date_col='date', n_folds=3, group_col='symbol')
    logger.info(f"Generated {len(folds)} walk-forward folds")
    
    # Test prediction dataset
    prediction_df = create_prediction_dataset(test_df, prediction_window=10)
    logger.info(f"Prediction dataset size: {len(prediction_df)} rows")