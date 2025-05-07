"""
Utility to load Numerai crypto data.
"""
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_numerai_data(numerai_files):
    """
    Load Numerai data from downloaded files
    
    Args:
        numerai_files: Dictionary with paths to Numerai data files
        
    Returns:
        dict: Loaded Numerai data frames
    """
    logger.info("Loading Numerai data")
    
    numerai_data = {}
    
    # Load train targets
    if 'train_targets' in numerai_files and os.path.exists(numerai_files['train_targets']):
        try:
            numerai_data['train_targets'] = pd.read_parquet(numerai_files['train_targets'])
            logger.info(f"Loaded train targets: {numerai_data['train_targets'].shape}")
        except Exception as e:
            logger.error(f"Error loading train targets: {e}")
    else:
        logger.error(f"Train targets file not found: {numerai_files.get('train_targets')}")
    
    # Load live universe
    if 'live_universe' in numerai_files and os.path.exists(numerai_files['live_universe']):
        try:
            numerai_data['live_universe'] = pd.read_parquet(numerai_files['live_universe'])
            logger.info(f"Loaded live universe: {numerai_data['live_universe'].shape}")
            
            # Extract symbols from live universe
            if 'id' in numerai_data['live_universe'].columns:
                # Assuming id format: asset_date
                numerai_data['live_universe']['asset'] = numerai_data['live_universe']['id'].str.split('_').str[0]
                # Get unique assets (crypto symbols)
                numerai_data['symbols'] = sorted(numerai_data['live_universe']['asset'].unique().tolist())
                logger.info(f"Extracted {len(numerai_data['symbols'])} unique symbols from live universe")
        except Exception as e:
            logger.error(f"Error loading live universe: {e}")
    else:
        logger.error(f"Live universe file not found: {numerai_files.get('live_universe')}")
    
    # Load train data
    if 'train_data' in numerai_files and os.path.exists(numerai_files['train_data']):
        try:
            numerai_data['train_data'] = pd.read_parquet(numerai_files['train_data'])
            logger.info(f"Loaded train data: {numerai_data['train_data'].shape}")
        except Exception as e:
            logger.error(f"Error loading train data: {e}")
    else:
        logger.error(f"Train data file not found: {numerai_files.get('train_data')}")
    
    # Store the current round
    if 'current_round' in numerai_files:
        numerai_data['current_round'] = numerai_files['current_round']
    
    return numerai_data

def get_eligible_crypto_symbols(numerai_data):
    """
    Extract the list of eligible crypto symbols from Numerai data
    
    Args:
        numerai_data: Dictionary with loaded Numerai data frames
        
    Returns:
        list: Eligible crypto symbols
    """
    if 'symbols' in numerai_data:
        return numerai_data['symbols']
    
    symbols = []
    
    # Try to extract from live universe
    if 'live_universe' in numerai_data and 'id' in numerai_data['live_universe'].columns:
        numerai_data['live_universe']['asset'] = numerai_data['live_universe']['id'].str.split('_').str[0]
        symbols = sorted(numerai_data['live_universe']['asset'].unique().tolist())
        logger.info(f"Extracted {len(symbols)} unique symbols from live universe")
    
    # If still empty, try train data
    if not symbols and 'train_data' in numerai_data and 'id' in numerai_data['train_data'].columns:
        numerai_data['train_data']['asset'] = numerai_data['train_data']['id'].str.split('_').str[0]
        symbols = sorted(numerai_data['train_data']['asset'].unique().tolist())
        logger.info(f"Extracted {len(symbols)} unique symbols from train data")
    
    return symbols

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example usage
    numerai_files = {
        'train_targets': 'data/numerai/train_targets.parquet',
        'live_universe': 'data/numerai/live_universe.parquet',
        'train_data': 'data/numerai/train.parquet',
        'current_round': 42
    }
    
    numerai_data = load_numerai_data(numerai_files)
    symbols = get_eligible_crypto_symbols(numerai_data)
    
    if symbols:
        logger.info(f"Found {len(symbols)} eligible crypto symbols")
        logger.info(f"First 10 symbols: {symbols[:10]}")