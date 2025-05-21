"""
Utility to download Numerai crypto tournament data.
"""
import os
import sys
import logging
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import tournament configuration
from config.tournament_config import get_tournament_name, get_tournament_endpoint, TOURNAMENT_NAME

# Try to import numerapi
try:
    import numerapi
except ImportError:
    logging.warning("numerapi package not found. Installing...")
    import subprocess
    try:
        # Try to install with --break-system-packages for Docker/root environments
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "numerapi"])
    except subprocess.CalledProcessError:
        # If that fails, try normal installation
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numerapi"])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install numerapi: {e}")
            raise ImportError("Could not install numerapi package")
    import numerapi

logger = logging.getLogger(__name__)

def download_numerai_crypto_data(numerai_dir, api_key=None, api_secret=None, pit_date=None, force=False):
    """
    Download Numerai crypto tournament data using numerapi
    
    Args:
        numerai_dir: Base directory to save Numerai data
        api_key: Optional Numerai API key
        api_secret: Optional Numerai API secret
        pit_date: Optional point-in-time date string (YYYYMMDD) for organizing data
        force: Force download even if data exists (for 15:00+ data updates)
        
    Returns:
        dict: Paths to downloaded files and current round number
    """
    logger.info("Downloading Numerai crypto tournament data" + (" (forced download)" if force else ""))
    
    # Use current date or specified point-in-time date
    if pit_date and len(pit_date) == 8 and pit_date.isdigit():
        date_str = pit_date
        logger.info(f"Using point-in-time date: {date_str}")
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        logger.info(f"Using current date: {date_str}")
    
    # Create date-specific directory
    date_dir = os.path.join(numerai_dir, date_str)
    
    # If force flag is set, remove existing directory to ensure clean download
    if force and os.path.exists(date_dir):
        logger.info(f"Force flag set - removing existing directory: {date_dir}")
        import shutil
        try:
            shutil.rmtree(date_dir)
        except Exception as e:
            logger.error(f"Failed to remove existing directory {date_dir}: {e}")
    
    os.makedirs(date_dir, exist_ok=True)
    logger.info(f"Storing Numerai data in: {date_dir} (force={force})")
    
    # Initialize Numerai API client
    if api_key and api_secret:
        napi = numerapi.NumerAPI(api_key, api_secret)
    else:
        napi = numerapi.NumerAPI()
    
    try:
        # Get current round for filename
        try:
            # The get_current_round API call might not support the tournament parameter
            # based on the error we saw. Just get the current round without parameter.
            current_round = napi.get_current_round()
            logger.info(f"Got current round: {current_round}")
            
            # For safety, verify this is for crypto tournament
            logger.info(f"Using round {current_round} for {TOURNAMENT_NAME} tournament")
        except Exception as e:
            # If getting the round fails for some reason, log the error
            logger.error(f"Failed to get current round: {e}")
            # Use a fallback round number - not ideal but keeps the pipeline running
            current_round = 1004  # Hardcoded fallback as a last resort
            logger.warning(f"Using fallback round number: {current_round}")
        
        logger.info(f"Using tournament round: {current_round} for {TOURNAMENT_NAME} tournament")
        
        # Download datasets
        train_file = os.path.join(date_dir, f"train_targets_r{current_round}.parquet")
        live_file = os.path.join(date_dir, f"live_universe_r{current_round}.parquet")
        
        logger.info(f"Downloading train targets to {train_file}")
        napi.download_dataset(get_tournament_endpoint("train_targets"), train_file)
        
        logger.info(f"Downloading live universe to {live_file}")
        napi.download_dataset(get_tournament_endpoint("live_universe"), live_file)
        
        # Also download training data - try both standard and alternate path formats
        train_data_file = os.path.join(date_dir, f"train_data_r{current_round}.parquet")
        logger.info(f"Downloading train data to {train_data_file}")
        
        # Try standard path first
        try:
            napi.download_dataset(get_tournament_endpoint("train_data"), train_data_file)
            logger.info("Successfully downloaded train data using standard path")
        except Exception as train_error:
            logger.warning(f"Error downloading train data with standard path: {train_error}")
            
            # Try alternate path format
            try:
                logger.info("Trying alternate train data path format")
                napi.download_dataset(get_tournament_endpoint("train_data_alt"), train_data_file)
                logger.info("Successfully downloaded train data using alternate path")
            except Exception as alt_error:
                logger.error(f"Error downloading train data with alternate path: {alt_error}")
                logger.warning("Could not download train data, submission may still work with other files")
        
        # Create a metadata file with timestamp and round info
        metadata_file = os.path.join(date_dir, "metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tournament: {TOURNAMENT_NAME}\n")
            f.write(f"Tournament Round: {current_round}\n")
            f.write(f"Files:\n")
            f.write(f"  - {os.path.basename(train_file)}\n")
            f.write(f"  - {os.path.basename(live_file)}\n")
            if os.path.exists(train_data_file):
                f.write(f"  - {os.path.basename(train_data_file)}\n")
            else:
                f.write(f"  - {os.path.basename(train_data_file)} (failed to download)\n")
        
        # Build result dictionary
        result = {
            'train_targets': train_file,
            'live_universe': live_file,
            'current_round': current_round,
            'date_dir': date_dir,
            'date_str': date_str
        }
        
        # Only include train_data if file exists
        if os.path.exists(train_data_file):
            result['train_data'] = train_data_file
        
        return result
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