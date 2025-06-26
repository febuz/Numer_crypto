#!/usr/bin/env python3
"""
Run the data processing pipeline with NaN values replaced by column means.

This script acts as a wrapper to run data processing with the improved NaN handling,
replacing NaN values with column means instead of zeros to improve statistical properties
and prevent hanging during processing.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a shell command and log output"""
    logger.info(f"Running {description}...")
    logger.info(f"Command: {cmd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor output in real-time
        while True:
            # Read stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                logger.info(stdout_line.strip())
                
            # Read stderr
            stderr_line = process.stderr.readline()
            if stderr_line:
                logger.warning(stderr_line.strip())
                
            # Check if process has finished
            return_code = process.poll()
            if return_code is not None:
                # Process any remaining output
                for line in process.stdout:
                    logger.info(line.strip())
                for line in process.stderr:
                    logger.warning(line.strip())
                
                if return_code != 0:
                    logger.error(f"{description} failed with return code {return_code}")
                    return False
                else:
                    logger.info(f"{description} completed successfully")
                    return True
                    
            # Small sleep to prevent CPU hogging
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return False

def main():
    """Main function to run the NaN-fixing data pipeline"""
    parser = argparse.ArgumentParser(description='Run data processing with NaN values replaced by column means')
    parser.add_argument('--input', type=str, help='Input parquet file with data')
    parser.add_argument('--output', type=str, help='Output directory for processed data')
    parser.add_argument('--skip-processing', action='store_true', help='Skip data processing and only fix NaNs')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature generation')
    
    args = parser.parse_args()
    
    # Find input file if not provided
    if args.input is None:
        # Look for data in standard locations
        potential_files = [
            "/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_latest.parquet",
            "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet"
        ]
        
        for file_path in potential_files:
            if os.path.exists(file_path):
                args.input = file_path
                logger.info(f"Using input file: {args.input}")
                break
                
        if args.input is None:
            logger.error("No input file provided and no default files found")
            return 1
    
    # Verify input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Set output directory if not provided
    if args.output is None:
        args.output = "/media/knight2/EDB/numer_crypto_temp/data/processed"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Run data processing if not skipped
    if not args.skip_processing:
        logger.info("Step 1: Running data processing with Polars")
        
        process_cmd = f"python -m scripts.data.process_data_polars --force"
        if not run_command(process_cmd, "Data processing"):
            logger.error("Data processing failed")
            return 1
    
    # Step 2: Fix NaN values in processed data
    logger.info("Step 2: Fixing NaN values by replacing with column means")
    
    # Find the processed file to fix
    if args.skip_processing:
        # Use input file directly
        file_to_fix = args.input
    else:
        # Look for the processed file
        file_to_fix = os.path.join(args.output, "crypto_train.parquet")
        if not os.path.exists(file_to_fix):
            logger.error(f"Processed file not found: {file_to_fix}")
            return 1
    
    # Run NaN fixing script
    fixed_output = os.path.join(args.output, "crypto_train_nan_fixed.parquet")
    fix_cmd = f"python -m scripts.data.process_data_nan_fix --input {file_to_fix} --output {fixed_output}"
    if not run_command(fix_cmd, "NaN fixing"):
        logger.error("NaN fixing failed")
        return 1
    
    # Step 3: Run feature generation if not skipped
    if not args.skip_features:
        logger.info("Step 3: Running feature generation with NaN-fixed data")
        
        feature_cmd = f"python -m scripts.features.polars_generator --input-file {fixed_output}"
        if not run_command(feature_cmd, "Feature generation"):
            logger.warning("Feature generation completed with warnings - check logs")
    
    logger.info("Processing completed successfully")
    logger.info(f"Fixed data file: {fixed_output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())