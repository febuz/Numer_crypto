"""
Utility to download Numerai crypto tournament data.
"""
import os
import sys
import logging
from datetime import datetime

# Try to import numerapi
try:
    import numerapi
except ImportError:
    logging.warning("numerapi package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numerapi"])
    import numerapi

logger = logging.getLogger(__name__)

def download_numerai_crypto_data(numerai_dir, api_key=None, api_secret=None):
    """
    Download Numerai crypto tournament data using numerapi
    
    Args:
        numerai_dir: Directory to save Numerai data
        api_key: Optional Numerai API key
        api_secret: Optional Numerai API secret
        
    Returns:
        dict: Paths to downloaded files and current round number
    """
    logger.info("Downloading Numerai crypto tournament data")
    
    # Create directory if it doesn't exist
    os.makedirs(numerai_dir, exist_ok=True)
    
    # Initialize Numerai API client
    if api_key and api_secret:
        napi = numerapi.NumerAPI(api_key, api_secret)
    else:
        napi = numerapi.NumerAPI()
    
    try:
        # Get current round for filename
        current_round = napi.get_current_round(tournament='crypto')
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Download datasets
        train_file = os.path.join(numerai_dir, f"train_targets_r{current_round}_{date_str}.parquet")
        live_file = os.path.join(numerai_dir, f"live_universe_r{current_round}_{date_str}.parquet")
        
        logger.info(f"Downloading train targets to {train_file}")
        napi.download_dataset("crypto/v1.0/train_targets.parquet", train_file)
        
        logger.info(f"Downloading live universe to {live_file}")
        napi.download_dataset("crypto/v1.0/live_universe.parquet", live_file)
        
        # Also download training data
        train_data_file = os.path.join(numerai_dir, f"train_data_r{current_round}_{date_str}.parquet")
        logger.info(f"Downloading train data to {train_data_file}")
        napi.download_dataset("crypto/v1.0/train.parquet", train_data_file)
        
        return {
            'train_targets': train_file,
            'live_universe': live_file,
            'train_data': train_data_file,
            'current_round': current_round
        }
    except Exception as e:
        logger.error(f"Error downloading Numerai data: {e}")
        return {
            'error': str(e)
        }

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example usage
    data_dir = os.path.join(os.getcwd(), 'data', 'numerai')
    result = download_numerai_crypto_data(data_dir)
    
    if 'error' not in result:
        logger.info(f"Downloaded Numerai data for round {result['current_round']}")
        for key, path in result.items():
            if key != 'current_round':
                logger.info(f"{key}: {path}")
    else:
        logger.error(f"Failed to download Numerai data: {result['error']}")