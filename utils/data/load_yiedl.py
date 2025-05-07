"""
Utility to load Yiedl data.
"""
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_yiedl_data(yiedl_files):
    """
    Load Yiedl data from downloaded files
    
    Args:
        yiedl_files: Dictionary with paths to Yiedl data files
        
    Returns:
        dict: Loaded Yiedl data frames
    """
    logger.info("Loading Yiedl data")
    
    yiedl_data = {}
    
    # Load latest data
    if 'latest' in yiedl_files and yiedl_files['latest'] and os.path.exists(yiedl_files['latest']):
        try:
            yiedl_data['latest'] = pd.read_parquet(yiedl_files['latest'])
            logger.info(f"Loaded latest Yiedl data: {yiedl_data['latest'].shape}")
            
            # Extract symbols from latest data
            if 'asset' in yiedl_data['latest'].columns:
                yiedl_data['symbols'] = sorted(yiedl_data['latest']['asset'].unique().tolist())
                logger.info(f"Extracted {len(yiedl_data['symbols'])} unique symbols from latest Yiedl data")
        except Exception as e:
            logger.error(f"Error loading latest Yiedl data: {e}")
    else:
        logger.error(f"Latest Yiedl data file not found or specified")
    
    # Load historical data
    if 'historical' in yiedl_files and yiedl_files['historical'] and os.path.exists(yiedl_files['historical']):
        try:
            yiedl_data['historical'] = pd.read_parquet(yiedl_files['historical'])
            logger.info(f"Loaded historical Yiedl data: {yiedl_data['historical'].shape}")
            
            # Extract symbols from historical if we don't have them yet
            if 'symbols' not in yiedl_data and 'asset' in yiedl_data['historical'].columns:
                yiedl_data['symbols'] = sorted(yiedl_data['historical']['asset'].unique().tolist())
                logger.info(f"Extracted {len(yiedl_data['symbols'])} unique symbols from historical Yiedl data")
        except Exception as e:
            logger.error(f"Error loading historical Yiedl data: {e}")
    else:
        logger.warning("Historical Yiedl data file not found or specified")
    
    return yiedl_data

def get_yiedl_crypto_symbols(yiedl_data):
    """
    Extract the list of crypto symbols from Yiedl data
    
    Args:
        yiedl_data: Dictionary with loaded Yiedl data frames
        
    Returns:
        list: Crypto symbols from Yiedl data
    """
    if 'symbols' in yiedl_data:
        return yiedl_data['symbols']
    
    symbols = []
    
    # Try to extract from latest data
    if 'latest' in yiedl_data and 'asset' in yiedl_data['latest'].columns:
        symbols = sorted(yiedl_data['latest']['asset'].unique().tolist())
        logger.info(f"Extracted {len(symbols)} unique symbols from latest Yiedl data")
    
    # If still empty, try historical data
    if not symbols and 'historical' in yiedl_data and 'asset' in yiedl_data['historical'].columns:
        symbols = sorted(yiedl_data['historical']['asset'].unique().tolist())
        logger.info(f"Extracted {len(symbols)} unique symbols from historical Yiedl data")
    
    return symbols

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example usage
    yiedl_files = {
        'latest': 'data/yiedl/yiedl_latest.parquet',
        'historical': 'data/yiedl/yiedl_historical.parquet'
    }
    
    yiedl_data = load_yiedl_data(yiedl_files)
    symbols = get_yiedl_crypto_symbols(yiedl_data)
    
    if symbols:
        logger.info(f"Found {len(symbols)} crypto symbols in Yiedl data")
        logger.info(f"First 10 symbols: {symbols[:10]}")