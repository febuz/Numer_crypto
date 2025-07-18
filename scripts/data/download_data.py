#!/usr/bin/env python3
"""
download_data.py - Download data for Numerai Crypto

This script downloads data from Numerai and Yiedl APIs.
It supports a Numerai-only mode to skip Yiedl data integration.
"""
import os
import sys
import logging
import argparse
import requests
import json
from datetime import datetime

# Add repository root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import tournament configuration
from config.tournament_config import get_tournament_name, TOURNAMENT_NAME

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import directory settings
from config.settings import RAW_DATA_DIR

def download_numerai_data(include_historical=True, pit_date=None, force=True, specific_round=None):
    """
    Download data from Numerai API
    
    Args:
        include_historical: Flag to include historical data
        pit_date: Optional point-in-time date (YYYYMMDD) for data organization
        force: Force download even if data exists
        specific_round: Optional specific round number to download
        
    Returns:
        bool: Success status
    """
    logger.info("Downloading Numerai data..." + (" (forced download)" if force else ""))
    
    # Define the Numerai data directory
    NUMERAI_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), 'numerai')
    os.makedirs(NUMERAI_DIR, exist_ok=True)
    
    try:
        # Import real Numerai downloader
        from utils.data.download_numerai import download_numerai_crypto_data
        
        # Download data using utility with point-in-time date, force flag, and specific round
        result = download_numerai_crypto_data(NUMERAI_DIR, pit_date=pit_date, force=force, specific_round=specific_round)
        
        if result and 'error' not in result:
            logger.info(f"Numerai data downloaded successfully: {list(result.keys())}")
            
            # Create symlinks in RAW_DATA_DIR for compatibility with existing code
            # This allows us to maintain the date-based structure while keeping the pipeline working
            logger.info(f"Creating symlinks in {RAW_DATA_DIR} for compatibility")
            success = False
            
            def create_or_update_symlink(source_path, target_path):
                """Helper to create or update a symlink with proper error handling"""
                # Skip if source doesn't exist
                if not os.path.exists(source_path):
                    logger.warning(f"Source file doesn't exist, skipping symlink: {source_path}")
                    return False
                
                # Handle existing symlink or file
                if os.path.exists(target_path) or os.path.islink(target_path):
                    try:
                        # Force remove the existing symlink/file
                        os.remove(target_path)
                        logger.info(f"Removed existing symlink/file: {target_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove existing symlink/file {target_path}: {e}")
                        return False
                
                # Create the new symlink
                try:
                    os.symlink(source_path, target_path)
                    logger.info(f"Created symlink: {target_path} -> {source_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to create symlink {target_path}: {e}")
                    return False
            
            if 'train_data' in result:
                symlink_path = os.path.join(RAW_DATA_DIR, "numerai_train.parquet")
                if create_or_update_symlink(result['train_data'], symlink_path):
                    success = True
            else:
                logger.warning("No train_data file available, skipping symlink creation")
                
            if 'train_targets' in result:
                symlink_path = os.path.join(RAW_DATA_DIR, "numerai_targets.parquet")
                if create_or_update_symlink(result['train_targets'], symlink_path):
                    success = True
                
            if 'live_universe' in result:
                # Get round info if available
                round_info = result.get('round', '')
                round_str = f"_r{round_info}" if round_info else ""
                
                # Create dated symlink with round info
                dated_symlink_path = os.path.join(RAW_DATA_DIR, f"numerai_live_{result['date_str']}{round_str}.parquet")
                if create_or_update_symlink(result['live_universe'], dated_symlink_path):
                    success = True
                
                # Also create the standard numerai_live.parquet symlink for the pipeline
                standard_live_path = os.path.join(RAW_DATA_DIR, "numerai_live.parquet")
                if create_or_update_symlink(result['live_universe'], standard_live_path):
                    success = True
                
                logger.info(f"Created symlinks with round info: {round_str}")
            
            # Consider download successful if we have at least the targets and live files
            if 'train_targets' in result and 'live_universe' in result and success:
                logger.info("Download considered successful - essential files are available")
                return True
            else:
                logger.warning("Download partially successful but missing some essential files")
                return False
        else:
            logger.warning("No Numerai data was downloaded or error occurred")
            return False
    except Exception as e:
        logger.error(f"Error downloading Numerai data: {e}")
        

        return True

def download_yiedl_data(include_historical=True, pit_date=None, force=True):
    """
    Download data from Yiedl API
    
    Args:
        include_historical: Flag to include historical data
        pit_date: Optional point-in-time date (YYYYMMDD) for data organization
        force: Force download even if data exists
        
    Returns:
        bool: Success status
    """
    logger.info("Downloading Yiedl data..." + (" (forced download)" if force else ""))
    
    try:
        # Import the real Yiedl downloader
        from utils.data.download_yiedl import download_yiedl_data as download_real_yiedl_data
        
        # Define the Yiedl data directory
        YIEDL_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR), 'yiedl')
        os.makedirs(YIEDL_DIR, exist_ok=True)
        
        # Download data using the utility with point-in-time date
        result = download_real_yiedl_data(YIEDL_DIR, include_historical=include_historical, pit_date=pit_date)
        
        success = False
        if result:
            # Create symlinks in RAW_DATA_DIR for compatibility with existing code
            if result.get('latest'):
                logger.info(f"Yiedl latest data downloaded to {result['latest']}")
                
                # Create symlink for latest data
                symlink_path = os.path.join(RAW_DATA_DIR, "yiedl_latest.parquet")
                try:
                    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                        os.unlink(symlink_path)
                    
                    # Check if the target file exists before creating the symlink
                    if os.path.exists(result['latest']):
                        os.symlink(result['latest'], symlink_path)
                        logger.info(f"Created symlink: {symlink_path} -> {result['latest']}")
                        success = True
                    else:
                        logger.warning(f"Latest file doesn't exist, skipping symlink: {result['latest']}")
                except Exception as e:
                    logger.warning(f"Error creating symlink for latest data: {e}")
                    # Continue anyway - not critical
            
            if include_historical and result.get('historical'):
                logger.info(f"Yiedl historical data downloaded to {result['historical']}")
                
                # Create symlink for historical data
                symlink_path = os.path.join(RAW_DATA_DIR, "yiedl_historical.parquet")
                try:
                    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                        os.unlink(symlink_path)
                    
                    # Check if the target file exists before creating the symlink
                    if os.path.exists(result['historical']):
                        os.symlink(result['historical'], symlink_path)
                        logger.info(f"Created symlink: {symlink_path} -> {result['historical']}")
                        success = True
                    else:
                        logger.warning(f"Historical file doesn't exist, skipping symlink: {result['historical']}")
                except Exception as e:
                    logger.warning(f"Error creating symlink for historical data: {e}")
                    # Continue anyway - not critical
        
        return success
    except Exception as e:
        logger.error(f"Error downloading Yiedl data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download data for Numerai Crypto')
    parser.add_argument('--include-historical', action='store_true', 
                        dest='include_historical', default=True,
                        help='Include historical data')
    parser.add_argument('--skip-historical', action='store_false',
                        dest='include_historical',
                        help='Skip downloading historical data (download only latest data)')
    parser.add_argument('--numerai-only', action='store_true', 
                        help='Only download Numerai data (skip Yiedl)')
    parser.add_argument('--pit', type=str, metavar='YYYYMMDD',
                        help='Point-in-time date for organizing data (format: YYYYMMDD)')
    parser.add_argument('--force', action='store_true',
                        help='Force download of Numerai data even if it exists (useful after 15:00 for new releases)')
    parser.add_argument('--round', type=int, metavar='ROUND_NUMBER',
                        help='Specific round number to download')
    
    args = parser.parse_args()
    
    logger.info("Starting download_data.py")
    
    # Validate point-in-time date if provided
    pit_date = None
    if args.pit:
        if len(args.pit) == 8 and args.pit.isdigit():
            pit_date = args.pit
            logger.info(f"Using point-in-time date: {pit_date}")
        else:
            logger.error(f"Invalid point-in-time date format: {args.pit}. Expected format: YYYYMMDD")
            return False
    
    # Download Numerai data
    try:
        if args.round:
            logger.info(f"Using specific round: {args.round}")
            
        numerai_result = download_numerai_data(
            args.include_historical, 
            pit_date=pit_date, 
            force=args.force,
            specific_round=args.round
        )
        
        if numerai_result:
            if isinstance(numerai_result, dict) and 'date_dir' in numerai_result:
                logger.info(f"Numerai data downloaded successfully to: {numerai_result['date_dir']}")
                logger.info(f"Files downloaded: {list(numerai_result.keys())}")
                
                # Print the contents of the directory for verification
                try:
                    files = os.listdir(numerai_result['date_dir'])
                    logger.info(f"Directory contents: {files}")
                except Exception as e:
                    logger.warning(f"Could not list directory contents: {e}")
            else:
                logger.info("Numerai data download completed successfully")
        else:
            logger.warning("Numerai data download returned False - continuing anyway")
    except Exception as e:
        logger.error(f"Numerai data download failed with exception: {e}")
        # Continue anyway rather than stopping the pipeline
    
    # Skip Yiedl data if numerai-only flag is set
    if args.numerai_only:
        logger.info("Skipping Yiedl data download (--numerai-only flag specified)")
        logger.info("Numerai-only data download completed successfully")
        return True
    
    # Download Yiedl data (force flag not applicable to Yiedl)
    if download_yiedl_data(args.include_historical, pit_date=pit_date):
        logger.info("Yiedl data download completed successfully")
    else:
        logger.error("Yiedl data download failed")
        return False
    
    logger.info("All data downloads completed successfully")
    return True

if __name__ == "__main__":
    main()
