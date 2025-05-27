"""
Utility to download Yiedl data.
"""
import os
import sys
import logging
import requests
import zipfile
import tempfile
from datetime import datetime, time

logger = logging.getLogger(__name__)

def download_yiedl_data(yiedl_dir, include_historical=True, pit_date=None):
    """
    Download latest and historical Yiedl data
    
    Args:
        yiedl_dir: Directory to save Yiedl data
        include_historical: Whether to download historical data (default: True)
        pit_date: Point-in-time date for organizing data (YYYYMMDD format)
        
    Returns:
        dict: Paths to downloaded files
    """
    logger.info("Downloading Yiedl data")
    
    # Create directory if it doesn't exist
    os.makedirs(yiedl_dir, exist_ok=True)
    
    # Date string for filename - use pit_date if provided, otherwise current date
    if pit_date:
        date_str = pit_date
        logger.info(f"Using point-in-time date: {date_str}")
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        logger.info(f"Using current date: {date_str}")
    
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
    
    # Download historical data (if requested and not already downloaded)
    if include_historical:
        historical_zip = os.path.join(yiedl_dir, 'yiedl_historical.zip')
        historical_file = os.path.join(yiedl_dir, f"yiedl_historical_{date_str}.parquet")
        
        # Check timing constraints for historical data download
        current_time = datetime.now().time()
        cutoff_time = time(15, 0)  # 15:00
        
        # Skip historical download if after 15:00 or if today's historical data already exists
        if current_time >= cutoff_time:
            logger.info("Skipping historical data download: current time is after 15:00")
            result['historical'] = historical_file if os.path.exists(historical_file) else None
        elif os.path.exists(historical_file):
            logger.info(f"Skipping historical data download: today's historical data already exists at {historical_file}")
            result['historical'] = historical_file
        elif not os.path.exists(historical_zip):
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
        
        
        # Extract historical data if zip exists and we haven't skipped due to timing/file existence
        if historical_zip and os.path.exists(historical_zip) and current_time < cutoff_time and not os.path.exists(historical_file):
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
        elif not ('historical' in result):
            result['historical'] = None
    else:
        logger.info("Skipping historical data download (include_historical=False)")
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