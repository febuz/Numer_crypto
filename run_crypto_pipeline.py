#!/usr/bin/env python3
"""
Runs the complete Numerai Crypto pipeline:
1. Process Yiedl data
2. Train models
3. Generate predictions
4. Create submission file

This script serves as the main entry point for the Numerai Crypto pipeline.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import argparse

# Set up logging
log_file = f"crypto_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description=None):
    """Run a command and log output"""
    if description:
        logger.info(f"STEP: {description}")
    
    logger.info(f"Running command: {cmd}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        # Read output and errors
        for stdout_line in iter(process.stdout.readline, ""):
            logger.info(stdout_line.strip())
        
        # Wait for process to complete
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            # Capture and log error output
            stderr = process.stderr.read()
            logger.error(f"Command failed with return code {return_code}")
            logger.error(f"Error output: {stderr}")
            return False
        
        logger.info(f"Command completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def process_yiedl_data():
    """Run the Yiedl data processing script"""
    return run_command(
        "python3 scripts/process_yiedl_data.py",
        "Processing Yiedl data"
    )

def train_and_predict():
    """Run the training and prediction script"""
    return run_command(
        "python3 scripts/train_predict_crypto.py",
        "Training models and generating predictions"
    )

def main():
    """Main function to run the complete pipeline"""
    logger.info("Starting Numerai Crypto pipeline")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Numerai Crypto pipeline")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/submissions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run pipeline steps
    steps_completed = 0
    steps_failed = 0
    
    # Step 1: Process Yiedl data
    if not args.skip_processing:
        logger.info("PIPELINE STEP 1: Process Yiedl data")
        if process_yiedl_data():
            steps_completed += 1
            logger.info("Data processing completed successfully")
        else:
            steps_failed += 1
            logger.error("Data processing failed")
            return
    else:
        logger.info("Skipping data processing step")
    
    # Step 2: Train models and generate predictions
    if not args.skip_training:
        logger.info("PIPELINE STEP 2: Train models and generate predictions")
        if train_and_predict():
            steps_completed += 1
            logger.info("Model training and prediction completed successfully")
        else:
            steps_failed += 1
            logger.error("Model training and prediction failed")
            return
    else:
        logger.info("Skipping model training step")
    
    # Pipeline summary
    logger.info("\n=== PIPELINE SUMMARY ===")
    logger.info(f"Steps completed successfully: {steps_completed}")
    logger.info(f"Steps failed: {steps_failed}")
    
    if steps_failed == 0:
        logger.info("Pipeline completed successfully!")
        # Show location of latest submission
        run_command("ls -lt data/submissions/ | head -5", "Latest submissions")
    else:
        logger.error("Pipeline failed")

if __name__ == "__main__":
    main()