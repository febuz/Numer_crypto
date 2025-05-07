#!/usr/bin/env python3
"""
Runs the complete Numerai Crypto pipeline:
1. Process Yiedl data with historical data for training
2. Generate time series features 
3. Train models using 600GB RAM and 3 GPUs
4. Create predictions using latest data
5. Submit predictions to external directory

This script serves as the main entry point for the Numerai Crypto pipeline.
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
import argparse

# Add parent directory to path if needed
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.log_utils import setup_logging
from utils.memory_utils import log_memory_usage, clear_memory
from utils.threading_utils import optimize_threadpool_settings
from utils.gpu.optimization import setup_multi_gpu_environment
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SUBMISSIONS_DIR, 
    TRAIN_DIR, VALIDATION_DIR, PREDICTION_DIR
)

# Set up logging to external directory
logger = setup_logging(name=__name__, level=logging.INFO)

def run_command(cmd, description=None, timeout=7200):  # Default 2 hour timeout
    """Run a command and log output"""
    if description:
        logger.info(f"STEP: {description}")
    
    logger.info(f"Running command: {cmd}")
    
    start_time = time.time()
    
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
        
        # Wait for process to complete with timeout
        process.stdout.close()
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"Command timed out after {timeout} seconds")
            return False
        
        if return_code != 0:
            # Capture and log error output
            stderr = process.stderr.read()
            logger.error(f"Command failed with return code {return_code}")
            logger.error(f"Error output: {stderr}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"Command completed successfully in {elapsed:.1f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def setup_environment():
    """Set up the environment for optimal performance"""
    logger.info("Setting up environment for optimal performance")
    
    # Optimize thread pool settings
    optimize_threadpool_settings()
    
    # Set up multi-GPU environment
    gpu_ids = setup_multi_gpu_environment()
    if gpu_ids:
        logger.info(f"Using GPUs: {gpu_ids}")
    else:
        logger.warning("No GPUs detected, using CPU only")
    
    # Create necessary directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # Log memory usage
    log_memory_usage("Initial memory:")
    
    return True

def download_data():
    """Download data from Numerai and Yiedl"""
    logger.info("Downloading data")
    
    # Run the download script with a timeout of 30 minutes
    return run_command(
        "python3 scripts/download_data.py --include-historical",
        "Downloading Numerai and Yiedl data",
        timeout=1800
    )

def process_data():
    """Process data and create train/validation/prediction splits"""
    logger.info("Processing data and creating splits")
    
    # Run the processing script with a timeout of 1 hour
    return run_command(
        "python3 scripts/process_data.py --use-historical",
        "Processing data and creating splits",
        timeout=3600
    )

def generate_features():
    """Generate time series features"""
    logger.info("Generating time series features")
    
    # Run the feature generation script with a timeout of 2 hours
    return run_command(
        "python3 scripts/generate_features.py --timeseries",
        "Generating time series features",
        timeout=7200
    )

def train_models():
    """Train models on the processed data"""
    logger.info("Training models")
    
    # Run the training script with a timeout of 3 hours
    return run_command(
        "python3 scripts/train_models.py --use-gpu --parallel",
        "Training models",
        timeout=10800
    )

def generate_predictions():
    """Generate predictions for the latest data"""
    logger.info("Generating predictions")
    
    # Run the prediction script with a timeout of 1 hour
    return run_command(
        "python3 scripts/generate_predictions.py",
        "Generating predictions",
        timeout=3600
    )

def create_submission():
    """Create submission file"""
    logger.info("Creating submission file")
    
    # Run the submission script with a timeout of 30 minutes
    return run_command(
        f"python3 scripts/create_submission.py --output-dir {SUBMISSIONS_DIR}",
        "Creating submission file",
        timeout=1800
    )

def check_script(script_path):
    """Check if a script exists and create if it doesn't"""
    if not os.path.exists(script_path):
        logger.warning(f"Script {script_path} not found, creating template")
        
        # Create basic script template
        with open(script_path, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
\"\"\"
{os.path.basename(script_path)} - Auto-generated script
\"\"\"
import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.log_utils import setup_logging
logger = setup_logging(name=__name__, level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='{os.path.basename(script_path)}')
    # Add arguments here
    args = parser.parse_args()
    
    logger.info("Starting {os.path.basename(script_path)}")
    
    # Implement functionality here
    logger.info("Completed {os.path.basename(script_path)}")
    return True

if __name__ == "__main__":
    main()
""")
        
        # Make executable
        os.chmod(script_path, 0o755)
        logger.info(f"Created template for {script_path}")
        return False
    return True

def check_required_scripts():
    """Check if all required scripts exist, create templates if they don't"""
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    
    required_scripts = [
        os.path.join(scripts_dir, 'download_data.py'),
        os.path.join(scripts_dir, 'process_data.py'),
        os.path.join(scripts_dir, 'generate_features.py'),
        os.path.join(scripts_dir, 'train_models.py'),
        os.path.join(scripts_dir, 'generate_predictions.py'),
        os.path.join(scripts_dir, 'create_submission.py')
    ]
    
    all_exist = True
    for script in required_scripts:
        if not check_script(script):
            all_exist = False
    
    return all_exist

def main():
    """Main function to run the complete pipeline"""
    start_time = time.time()
    logger.info("Starting Numerai Crypto pipeline")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Numerai Crypto pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download step")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature generation step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction generation step")
    parser.add_argument("--skip-submission", action="store_true", help="Skip submission creation step")
    parser.add_argument("--time-limit", type=int, default=14400, help="Time limit in seconds (default: 4 hours)")
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check required scripts
    check_required_scripts()
    
    # Initialize step tracking
    steps_completed = 0
    steps_failed = 0
    
    # Set time limit
    time_limit = args.time_limit
    end_time = start_time + time_limit
    
    def check_time_limit():
        """Check if we're approaching the time limit"""
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        logger.info(f"Time elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
        
        if remaining < 600:  # Less than 10 minutes left
            logger.warning(f"Approaching time limit, only {remaining:.1f} seconds left")
            return False
        
        return True
    
    # Step 1: Download data
    if not args.skip_download and check_time_limit():
        logger.info("PIPELINE STEP 1: Download data")
        if download_data():
            steps_completed += 1
            logger.info("Data download completed successfully")
        else:
            steps_failed += 1
            logger.error("Data download failed")
            return
    else:
        logger.info("Skipping data download step")
    
    # Step 2: Process data
    if not args.skip_processing and check_time_limit():
        logger.info("PIPELINE STEP 2: Process data")
        if process_data():
            steps_completed += 1
            logger.info("Data processing completed successfully")
        else:
            steps_failed += 1
            logger.error("Data processing failed")
            return
    else:
        logger.info("Skipping data processing step")
    
    # Step 3: Generate features
    if not args.skip_features and check_time_limit():
        logger.info("PIPELINE STEP 3: Generate features")
        if generate_features():
            steps_completed += 1
            logger.info("Feature generation completed successfully")
        else:
            steps_failed += 1
            logger.error("Feature generation failed")
            return
    else:
        logger.info("Skipping feature generation step")
    
    # Step 4: Train models
    if not args.skip_training and check_time_limit():
        logger.info("PIPELINE STEP 4: Train models")
        if train_models():
            steps_completed += 1
            logger.info("Model training completed successfully")
        else:
            steps_failed += 1
            logger.error("Model training failed")
            return
    else:
        logger.info("Skipping model training step")
    
    # Step 5: Generate predictions
    if not args.skip_prediction and check_time_limit():
        logger.info("PIPELINE STEP 5: Generate predictions")
        if generate_predictions():
            steps_completed += 1
            logger.info("Prediction generation completed successfully")
        else:
            steps_failed += 1
            logger.error("Prediction generation failed")
            return
    else:
        logger.info("Skipping prediction generation step")
    
    # Step 6: Create submission
    if not args.skip_submission and check_time_limit():
        logger.info("PIPELINE STEP 6: Create submission")
        if create_submission():
            steps_completed += 1
            logger.info("Submission creation completed successfully")
        else:
            steps_failed += 1
            logger.error("Submission creation failed")
            return
    else:
        logger.info("Skipping submission creation step")
    
    # Pipeline summary
    total_time = time.time() - start_time
    logger.info("\n=== PIPELINE SUMMARY ===")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Steps completed successfully: {steps_completed}")
    logger.info(f"Steps failed: {steps_failed}")
    
    if steps_failed == 0:
        logger.info("Pipeline completed successfully!")
        # Show location of latest submission
        run_command(f"ls -lt {SUBMISSIONS_DIR}/ | head -5", "Latest submissions")
    else:
        logger.error("Pipeline failed")

if __name__ == "__main__":
    main()
