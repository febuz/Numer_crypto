import os
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import numerapi

logger = logging.getLogger(__name__)

def download_numerai_crypto_data(output_dir, pit_date=None, force=False, specific_round=None):
    """
    Download Numerai Crypto data from the Numerai API
    
    Args:
        output_dir: Directory to save downloaded files
        pit_date: Optional point-in-time date for organizing data
        force: Force download even if files exist
        specific_round: Optional specific round number to download
        
    Returns:
        Dictionary with paths to downloaded files or None if error
    """
    logger.info(f"Downloading Numerai Crypto data to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set proper permissions for the output directory if running as owner
    try:
        import stat
        if os.access(output_dir, os.W_OK):
            os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777 permissions
    except Exception as e:
        logger.debug(f"Could not set permissions on output directory: {e}")  # Downgrade to debug level
    
    # Use current date if pit_date not provided
    if pit_date is None:
        date_str = datetime.now().strftime('%Y%m%d')
    else:
        date_str = pit_date
    
    # Create date directory
    date_dir = os.path.join(output_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    # Set proper permissions for the date directory if running as owner
    try:
        import stat
        if os.access(date_dir, os.W_OK):
            os.chmod(date_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777 permissions
    except Exception as e:
        logger.debug(f"Could not set permissions on date directory: {e}")  # Downgrade to debug level
    
    try:
        # Initialize Numerai API client
        napi = numerapi.NumerAPI()
        
        # Get current round number
        current_round = napi.get_current_round()
        
        # If specific_round is set, use it directly
        if specific_round is not None:
            logger.info(f"Using specified round: {specific_round}")
            round_to_use = specific_round
            # Skip all round availability checks
        else:
            # Check for latest available round (current or next)
            next_round = current_round + 1
            logger.info(f"Current round from API: {current_round}, checking if round {next_round} is available")
        
            # Define file paths - check for next round availability first
            try:
                # Try several methods to determine if the next round is available
                next_round_available = False
                
                # Method 1: Check dataset list
                try:
                    datasets = napi.list_datasets()
                    logger.info(f"API returned {len(datasets)} datasets")
                    logger.info(f"First dataset example: {datasets[0] if datasets else 'None'}")
                    logger.info(f"Dataset type: {type(datasets[0]).__name__ if datasets else 'None'}")
                    
                    # Check if datasets is a list of dictionaries or a list of strings
                    if datasets and isinstance(datasets[0], dict):
                        crypto_datasets = [d for d in datasets if d.get('name', '').startswith('crypto/v1.0/')]
                        # Look for next round in dataset names
                        next_round_available = any(f"r{next_round}" in d.get('name', '') for d in crypto_datasets)
                    else:
                        # If datasets is a list of strings, search directly
                        crypto_datasets = [d for d in datasets if d.startswith('crypto/v1.0/')]
                        # Look for next round in dataset names
                        next_round_available = any(f"r{next_round}" in d for d in crypto_datasets)
                    
                    logger.info(f"Method 1 result: next_round_available={next_round_available}")
                except Exception as e:
                    logger.warning(f"Method 1 failed: {e}")
                
                # Method 2: Try to get round status directly
                if not next_round_available:
                    try:
                        # Try to get status of the next round
                        round_status = napi.get_competition_dataset("crypto", next_round)
                        # If we got a response without error, the round exists
                        logger.info(f"Method 2: Got round status for round {next_round}")
                        next_round_available = True
                    except Exception as e:
                        logger.info(f"Method 2: Round {next_round} not available: {e}")
                
                # Method 3: Direct file check
                if not next_round_available:
                    try:
                        # Create a temporary file to test download
                        import tempfile
                        with tempfile.NamedTemporaryFile() as temp_file:
                            try:
                                # Try to download the next round live universe file
                                napi.download_dataset(f"crypto/v1.0/live_universe_r{next_round}.parquet", 
                                                    dest_path=temp_file.name)
                                # If we got here, the file exists
                                logger.info(f"Method 3: Successfully downloaded test file for round {next_round}")
                                next_round_available = True
                            except Exception as e:
                                logger.info(f"Method 3: Round {next_round} test file not available: {e}")
                    except Exception as e:
                        logger.warning(f"Method 3 failed: {e}")
                
                if next_round_available:
                    logger.info(f"Round {next_round} data is available, using it instead of round {current_round}")
                    round_to_use = next_round
                else:
                    # Try checking if data for higher rounds exist by direct detection
                    try:
                        # Try multiple potential rounds above the current one
                        potential_rounds = [current_round + i for i in range(1, 5)]
                        for potential_round in potential_rounds:
                            test_url = f"https://numerai-public-datasets.s3-us-west-2.amazonaws.com/crypto/v1.0/live_universe_r{potential_round}.parquet"
                            test_resp = napi._session.head(test_url)
                            if test_resp.status_code == 200:
                                logger.info(f"Found data for round {potential_round} via direct URL check")
                                round_to_use = potential_round
                                break
                        else:
                            # If no higher round was found, use current round
                            logger.info(f"No higher round data found, using current round {current_round}")
                            round_to_use = current_round
                    except Exception as e2:
                        logger.warning(f"Error checking direct URLs: {e2}")
                        round_to_use = current_round
            except Exception as e:
                logger.warning(f"Error checking for next round availability: {e}")
                # Use current round as fallback
                round_to_use = current_round
        
        # Define file paths with determined round number
        live_file = os.path.join(date_dir, f"live_universe_r{round_to_use}.parquet")
        targets_file = os.path.join(date_dir, f"train_targets_r{round_to_use}.parquet")
        
        # Check if files already exist and force flag is not set
        if not force and os.path.exists(live_file) and os.path.exists(targets_file):
            logger.info("Files already exist and force flag not set. Skipping download.")
            return {
                'live_universe': live_file,
                'train_targets': targets_file,
                'date_dir': date_dir,
                'date_str': date_str,
                'round': round_to_use
            }
        
        # Define possible endpoints
        live_endpoints = [
            f"crypto/v1.0/live_universe_r{round_to_use}.parquet",  # Specific round format
            "crypto/v1.0/live_universe.parquet",                   # Generic format
            f"v2/competitions/crypto/r{round_to_use}/live.parquet"  # Newer v2 format
        ]
        
        # Try each endpoint for live data
        logger.info(f"Downloading live universe data for round {round_to_use} to {live_file}")
        live_download_success = False
        
        for endpoint in live_endpoints:
            try:
                logger.info(f"Trying endpoint: {endpoint}")
                napi.download_dataset(endpoint, dest_path=live_file)
                logger.info(f"Successfully downloaded from {endpoint}")
                
                # Set proper permissions for the downloaded file if running as owner
                try:
                    import stat
                    if os.access(live_file, os.W_OK):
                        os.chmod(live_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)  # 0666 permissions
                except Exception as e:
                    logger.debug(f"Could not set permissions on live file: {e}")  # Downgrade to debug level
                
                live_download_success = True
                break
            except Exception as e:
                logger.warning(f"Error downloading from {endpoint}: {e}")
        
        if not live_download_success:
            logger.error("Failed to download live data from any endpoint")
        
        # If using a specific round, don't try to find alternative rounds
        if specific_round is not None:
            # We'll still try to verify if the round exists, but only for this specific round
            available_rounds = []
            
            # Try to check if this specific round exists
            test_endpoint = f"crypto/v1.0/train_targets_r{specific_round}.parquet"
            try:
                # Check if file exists using HEAD request
                url = f"https://numerai-public-datasets.s3-us-west-2.amazonaws.com/{test_endpoint}"
                response = napi._session.head(url)
                if response.status_code == 200:
                    logger.info(f"Verified that train targets for specified round {specific_round} exist")
                    available_rounds.append(specific_round)
                else:
                    logger.warning(f"Train targets for specified round {specific_round} do not seem to exist, but will try anyway")
            except Exception as e:
                logger.warning(f"Error checking specified round {specific_round}: {e}")
                
            # We will still use the specified round regardless of whether we found it
            logger.info(f"Using specified round: {specific_round}")
            # Keep round_to_use as specific_round, which was set earlier
        else:
            # Get available rounds by trying to find the most recent available train targets file
            max_rounds_to_check = 10  # Check up to 10 rounds back
            available_rounds = []
            
            logger.info(f"Checking for available target files in previous rounds (up to {max_rounds_to_check} rounds back)")
            for i in range(max_rounds_to_check):
                check_round = current_round - i
                if check_round < 1:
                    break
                    
                # Try downloading small file to test if round exists
                test_endpoint = f"crypto/v1.0/train_targets_r{check_round}.parquet"
                try:
                    # Check if file exists using HEAD request
                    url = f"https://numerai-public-datasets.s3-us-west-2.amazonaws.com/{test_endpoint}"
                    response = napi._session.head(url)
                    if response.status_code == 200:
                        logger.info(f"Found available train targets for round {check_round}")
                        available_rounds.append(check_round)
                except Exception as e:
                    logger.debug(f"Error checking round {check_round}: {e}")
            
            # If we found available rounds, use the most recent one
            if available_rounds:
                latest_available_round = max(available_rounds)
                logger.info(f"Using the most recent available round: {latest_available_round}")
                round_to_use = latest_available_round
            else:
                logger.warning(f"Could not find any available train target files in rounds {current_round} through {current_round - max_rounds_to_check + 1}")
                # Fall back to current round minus 1 as a best guess
                round_to_use = max(1, current_round - 1)
                logger.info(f"Falling back to round {round_to_use}")
        
        # Define possible endpoints for targets with additional fallbacks, prioritizing known good rounds
        targets_endpoints = [
            f"crypto/v1.0/train_targets_r{round_to_use}.parquet",  # Specific round format with detected round
        ]
        
        # For specific round requests, add fewer fallbacks to avoid using wrong data
        if specific_round is not None:
            # Add only a few fallbacks for specific round requests
            targets_endpoints.extend([
                f"v2/competitions/crypto/r{specific_round}/targets.parquet",  # Newer v2 format
                "crypto/v1.0/train_targets.parquet"  # Generic format as last resort
            ])
        else:
            # Add all available rounds we found as potential endpoints
            for avail_round in available_rounds:
                if avail_round != round_to_use:  # Skip the one we're already using
                    targets_endpoints.append(f"crypto/v1.0/train_targets_r{avail_round}.parquet")
            
            # Add generic and fallback endpoints
            targets_endpoints.extend([
                "crypto/v1.0/train_targets.parquet",                   # Generic format
                f"v2/competitions/crypto/r{round_to_use}/targets.parquet",  # Newer v2 format
                # Try generic endpoint with version variations
                "crypto/v1/train_targets.parquet",
                "crypto/latest/train_targets.parquet"
            ])
            
            # Add more endpoints for additional rounds not explicitly checked
            for i in range(1, 5):  # Check a few more rounds back
                check_round = round_to_use - i
                if check_round > 0 and check_round not in available_rounds:
                    targets_endpoints.append(f"crypto/v1.0/train_targets_r{check_round}.parquet")
                
        logger.info(f"Prioritized targets endpoints: {targets_endpoints[:3]}...")  # Show first few for logging
        
        # Filter out None values (from conditional expressions)
        targets_endpoints = [ep for ep in targets_endpoints if ep is not None]
        
        # Try each endpoint for targets data
        logger.info(f"Downloading train targets data for round {round_to_use} to {targets_file}")
        targets_download_success = False
        
        for endpoint in targets_endpoints:
            try:
                logger.info(f"Trying endpoint: {endpoint}")
                napi.download_dataset(endpoint, dest_path=targets_file)
                logger.info(f"Successfully downloaded from {endpoint}")
                
                # Set proper permissions for the downloaded file if running as owner
                try:
                    import stat
                    if os.access(targets_file, os.W_OK):
                        os.chmod(targets_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)  # 0666 permissions
                except Exception as e:
                    logger.debug(f"Could not set permissions on targets file: {e}")  # Downgrade to debug level
                
                targets_download_success = True
                break
            except Exception as e:
                logger.warning(f"Error downloading from {endpoint}: {e}")
        
        if not targets_download_success:
            logger.error("Failed to download targets data from any endpoint")
            
            # Fallback: Try to reuse existing targets file from previous round
            try:
                # Look for existing targets files in the directory structure
                import glob
                existing_targets = glob.glob(os.path.join(output_dir, "**/train_targets_*.parquet"), recursive=True)
                
                if existing_targets:
                    # Sort by modification time, newest first
                    existing_targets.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    newest_target = existing_targets[0]
                    
                    logger.warning(f"Using existing targets file as fallback: {newest_target}")
                    
                    # Copy the file to our current location
                    import shutil
                    shutil.copy2(newest_target, targets_file)
                    
                    # Set permissions if possible
                    try:
                        import stat
                        if os.access(targets_file, os.W_OK):
                            os.chmod(targets_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                    except Exception as e:
                        logger.debug(f"Could not set permissions on copied targets file: {e}")
                    
                    targets_download_success = True
                else:
                    logger.error("No existing targets files found to use as fallback")
            except Exception as e:
                logger.error(f"Error trying to use fallback targets file: {e}")
        
        # Verify files were downloaded and are valid
        live_file_valid = os.path.exists(live_file) and os.path.getsize(live_file) > 1000  # Must be at least 1KB
        targets_file_valid = os.path.exists(targets_file) and os.path.getsize(targets_file) > 1000  # Must be at least 1KB
        
        # Create metadata file regardless of download status
        metadata_file = os.path.join(date_dir, "metadata.txt")
        try:
            # First check if we can write to the directory
            if os.access(date_dir, os.W_OK):
                with open(metadata_file, 'w') as f:
                    f.write(f"Download date: {datetime.now().isoformat()}\n")
                    f.write(f"API round: {current_round}\n")
                    f.write(f"Used round: {round_to_use}\n")
                
                # Set proper permissions for the metadata file if running as owner
                try:
                    import stat
                    if os.access(metadata_file, os.W_OK):
                        os.chmod(metadata_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)  # 0666 permissions
                except Exception as e:
                    logger.debug(f"Could not set permissions on metadata file: {e}")  # Downgrade to debug level
            else:
                logger.warning(f"Directory not writable, skipping metadata file creation: {date_dir}")
        except Exception as e:
            logger.warning(f"Could not create metadata file: {e}")
            # Continue anyway - metadata file is not critical
        
        if live_file_valid and targets_file_valid:
            logger.info("Numerai data downloaded successfully")
            return {
                'live_universe': live_file,
                'train_targets': targets_file,
                'date_dir': date_dir,
                'date_str': date_str,
                'round': round_to_use
            }
        else:
            if not live_file_valid:
                logger.error(f"Live universe file is missing or invalid: {live_file}")
            if not targets_file_valid:
                logger.error(f"Train targets file is missing or invalid: {targets_file}")
            
            logger.error("Failed to download all required files")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading Numerai data: {e}")
        return None