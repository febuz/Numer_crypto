"""
Utility to create merged datasets from Numerai and Yiedl data.
"""
import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def create_merged_datasets(numerai_data, yiedl_data, merged_dir, current_round=None):
    """
    Create merged datasets from Numerai and Yiedl data
    
    Args:
        numerai_data: Dictionary with loaded Numerai data frames
        yiedl_data: Dictionary with loaded Yiedl data frames
        merged_dir: Directory to save merged datasets
        current_round: Current Numerai round (optional)
        
    Returns:
        dict: Information about merged datasets
    """
    logger.info("Creating merged datasets")
    
    # Create directory if it doesn't exist
    os.makedirs(merged_dir, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    merged_data = {}
    
    # Use current round if provided, otherwise try to get from numerai_data
    if current_round is None and 'current_round' in numerai_data:
        current_round = numerai_data['current_round']
    if current_round is None:
        current_round = "unknown"  # Fallback value
    
    # Check we have the necessary data
    if ('train_targets' not in numerai_data or 
        'train_data' not in numerai_data or 
        'live_universe' not in numerai_data or 
        'latest' not in yiedl_data):
        logger.error("Missing required data for merging")
        return merged_data
    
    # Get overlapping symbols between Numerai and Yiedl
    numerai_symbols = set()
    yiedl_symbols = set()
    
    # Extract Numerai symbols
    if 'symbols' in numerai_data:
        numerai_symbols = set(numerai_data['symbols'])
    elif 'live_universe' in numerai_data and 'id' in numerai_data['live_universe'].columns:
        numerai_data['live_universe']['asset'] = numerai_data['live_universe']['id'].str.split('_').str[0]
        numerai_symbols = set(numerai_data['live_universe']['asset'].unique())
    
    # Extract Yiedl symbols
    if 'symbols' in yiedl_data:
        yiedl_symbols = set(yiedl_data['symbols'])
    elif 'latest' in yiedl_data and 'asset' in yiedl_data['latest'].columns:
        yiedl_symbols = set(yiedl_data['latest']['asset'].unique())
    
    # Find overlapping symbols
    overlapping_symbols = numerai_symbols.intersection(yiedl_symbols)
    merged_data['overlapping_symbols'] = sorted(list(overlapping_symbols))
    logger.info(f"Found {len(overlapping_symbols)} overlapping symbols between Numerai and Yiedl")
    
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
        train_file = os.path.join(merged_dir, f"train_merged_r{current_round}_{date_str}.parquet")
        merged_data['train'].to_parquet(train_file)
        logger.info(f"Saved merged train data to {train_file}")
        merged_data['train_file'] = train_file
    
    if 'live' in merged_data:
        live_file = os.path.join(merged_dir, f"live_merged_r{current_round}_{date_str}.parquet")
        merged_data['live'].to_parquet(live_file)
        logger.info(f"Saved merged live data to {live_file}")
        merged_data['live_file'] = live_file
    
    return merged_data

def get_overlapping_symbols(numerai_data, yiedl_data):
    """
    Get the set of crypto symbols that overlap between Numerai and Yiedl datasets
    
    Args:
        numerai_data: Dictionary with loaded Numerai data frames
        yiedl_data: Dictionary with loaded Yiedl data frames
        
    Returns:
        list: Overlapping crypto symbols
    """
    # Extract Numerai symbols
    numerai_symbols = set()
    if 'symbols' in numerai_data:
        numerai_symbols = set(numerai_data['symbols'])
    elif 'live_universe' in numerai_data and 'id' in numerai_data['live_universe'].columns:
        numerai_data['live_universe']['asset'] = numerai_data['live_universe']['id'].str.split('_').str[0]
        numerai_symbols = set(numerai_data['live_universe']['asset'].unique())
    
    # Extract Yiedl symbols
    yiedl_symbols = set()
    if 'symbols' in yiedl_data:
        yiedl_symbols = set(yiedl_data['symbols'])
    elif 'latest' in yiedl_data and 'asset' in yiedl_data['latest'].columns:
        yiedl_symbols = set(yiedl_data['latest']['asset'].unique())
    
    # Find and return overlapping symbols
    overlapping_symbols = numerai_symbols.intersection(yiedl_symbols)
    return sorted(list(overlapping_symbols))

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # This would typically be called after loading Numerai and Yiedl data
    # See load_numerai.py and load_yiedl.py for examples