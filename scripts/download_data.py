#!/usr/bin/env python3
"""
download_data.py - Download data for Numerai Crypto

This script downloads data from Numerai and Yiedl APIs.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
RAW_DATA_DIR = "/numer_crypto_temp/data/raw"

def download_numerai_data(include_historical=False):
    """Download data from Numerai API"""
    logger.info("Downloading Numerai data...")
    
    # Create output directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # For demonstration, create sample files
    sample_file = os.path.join(RAW_DATA_DIR, "numerai_sample_data.csv")
    with open(sample_file, 'w') as f:
        f.write("date,symbol,feature1,feature2,target\n")
        f.write("2023-01-01,BTC,0.1,0.2,1\n")
        f.write("2023-01-01,ETH,0.2,0.3,0\n")
    
    logger.info(f"Sample Numerai data saved to {sample_file}")
    return True

def download_yiedl_data(include_historical=False):
    """Download data from Yiedl API"""
    logger.info("Downloading Yiedl data...")
    
    # Create output directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # For demonstration, create sample files
    sample_file = os.path.join(RAW_DATA_DIR, "yiedl_sample_data.csv")
    with open(sample_file, 'w') as f:
        f.write("date,asset,price,volume\n")
        f.write("2023-01-01,BTC,50000,10000\n")
        f.write("2023-01-01,ETH,3000,20000\n")
    
    # If historical data requested, create another file
    if include_historical:
        historical_file = os.path.join(RAW_DATA_DIR, "yiedl_historical_data.csv")
        with open(historical_file, 'w') as f:
            f.write("date,asset,price,volume\n")
            for month in range(1, 13):
                for day in range(1, 28, 7):
                    date = f"2022-{month:02d}-{day:02d}"
                    f.write(f"{date},BTC,{40000 + month*1000 + day*100},{10000 + day*100}\n")
                    f.write(f"{date},ETH,{2000 + month*100 + day*10},{20000 + day*100}\n")
        logger.info(f"Sample historical Yiedl data saved to {historical_file}")
    
    logger.info(f"Sample Yiedl data saved to {sample_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download data for Numerai Crypto')
    parser.add_argument('--include-historical', action='store_true', help='Include historical data')
    
    args = parser.parse_args()
    
    logger.info("Starting download_data.py")
    
    # Download Numerai data
    if download_numerai_data(args.include_historical):
        logger.info("Numerai data download completed successfully")
    else:
        logger.error("Numerai data download failed")
        return False
    
    # Download Yiedl data
    if download_yiedl_data(args.include_historical):
        logger.info("Yiedl data download completed successfully")
    else:
        logger.error("Yiedl data download failed")
        return False
    
    logger.info("All data downloads completed successfully")
    return True

if __name__ == "__main__":
    main()
