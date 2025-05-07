"""
Utility to download Yiedl data.
"""
import os
import sys
import logging
import requests
import zipfile
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

def download_yiedl_data(yiedl_dir):
    """
    Download latest and historical Yiedl data
    
    Args:
        yiedl_dir: Directory to save Yiedl data
        
    Returns:
        dict: Paths to downloaded files
    """
    logger.info("Downloading Yiedl data")
    
    # Create directory if it doesn't exist
    os.makedirs(yiedl_dir, exist_ok=True)
    
    # Date string for filename
    date_str = datetime.now().strftime('%Y%m%d')
    result = {}
    
    # Download latest data
    latest_url = 'https://api.yiedl.ai/yiedl/v1/downloadDataset?type=latest'
    latest_file = os.path.join(yiedl_dir, f"yiedl_latest_{date_str}.parquet")
    
    logger.info(f"Downloading latest Yiedl data to {latest_file}")
    try:
        response = requests.get(latest_url)
        
        if response.status_code == 200:
            with open(latest_file, 'wb') as f:
                f.write(response.content)
            logger.info("Latest Yiedl data downloaded successfully")
            result['latest'] = latest_file
        else:
            logger.error(f"Failed to download latest Yiedl data: {response.status_code}")
            result['latest'] = None
    except Exception as e:
        logger.error(f"Error downloading latest Yiedl data: {e}")
        result['latest'] = None
    
    # Download historical data (if not already downloaded)
    historical_zip = os.path.join(yiedl_dir, 'yiedl_historical.zip')
    historical_file = os.path.join(yiedl_dir, f"yiedl_historical_{date_str}.parquet")
    
    if not os.path.exists(historical_zip):
        historical_url = 'https://api.yiedl.ai/yiedl/v1/downloadDataset?type=historical'
        logger.info("Downloading historical Yiedl data")
        try:
            response = requests.get(historical_url)
            
            if response.status_code == 200:
                with open(historical_zip, 'wb') as f:
                    f.write(response.content)
                logger.info("Historical Yiedl data (zip) downloaded successfully")
            else:
                logger.error(f"Failed to download historical Yiedl data: {response.status_code}")
                historical_zip = None
        except Exception as e:
            logger.error(f"Error downloading historical Yiedl data: {e}")
            historical_zip = None
    
    # Extract historical data if zip exists
    if historical_zip and os.path.exists(historical_zip):
        logger.info(f"Extracting historical data to {historical_file}")
        try:
            with zipfile.ZipFile(historical_zip, 'r') as zip_ref:
                temp_dir = tempfile.mkdtemp()
                zip_ref.extractall(temp_dir)
                
                # Find the parquet file in extracted contents
                parquet_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
                if parquet_files:
                    source_file = os.path.join(temp_dir, parquet_files[0])
                    # Copy to our dated file
                    with open(source_file, 'rb') as src, open(historical_file, 'wb') as dst:
                        dst.write(src.read())
                    logger.info("Historical Yiedl data extracted successfully")
                    result['historical'] = historical_file
                else:
                    logger.error("No parquet file found in the zip archive")
                    result['historical'] = None
        except Exception as e:
            logger.error(f"Error extracting historical Yiedl data: {e}")
            result['historical'] = None
    else:
        result['historical'] = None
    
    return result

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example usage
    data_dir = os.path.join(os.getcwd(), 'data', 'yiedl')
    result = download_yiedl_data(data_dir)
    
    for key, path in result.items():
        if path:
            logger.info(f"{key}: {path}")
        else:
            logger.warning(f"{key}: Not available")