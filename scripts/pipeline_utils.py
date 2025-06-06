#!/usr/bin/env python3
"""
Pipeline utilities for Numerai Crypto go_pipeline.sh
This module provides utility functions for the pipeline script, including:
- Progress tracking and status updates
- Data download
- Environment initialization and verification
- Helper functions for pipeline steps
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = "/media/knight2/EDB/numer_crypto_temp"
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/models"
PREDICTION_DIR = f"{BASE_DIR}/prediction"
SUBMISSION_DIR = f"{BASE_DIR}/submission"
STATUS_DIR = f"{BASE_DIR}/status"

class PipelineUtils:
    """Utility class for pipeline operations"""
    
    @staticmethod
    def create_progress_tracker(status_file_path=None):
        """
        Create a progress tracker for the pipeline.
        
        Args:
            status_file_path: Optional path to the status file.
            
        Returns:
            str: Path to the status file
        """
        try:
            from utils.progress import create_progress_tracker
            
            # Create new progress tracker (it will generate run-specific file automatically)
            progress = create_progress_tracker()
            
            # Return status file path
            return progress.status_file
            
        except ImportError as e:
            logger.error(f"ImportError: {str(e)}")
            logger.info("Installing psutil using pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"])
            
            # Try again after installing
            try:
                from utils.progress import create_progress_tracker
                progress = create_progress_tracker(status_file_path)
                return progress.status_file
            except Exception as e2:
                logger.error(f"Still failed after installing psutil: {str(e2)}")
                return f'{STATUS_DIR}/pipeline_status_fallback.json'
        except Exception as e:
            logger.error(f"Error initializing progress tracker: {str(e)}")
            return f'{STATUS_DIR}/pipeline_status_fallback.json'
    
    @staticmethod
    def update_progress(status_file, stage, status, details=None, progress=None):
        """
        Update progress for a pipeline stage using the new progress tracker format.
        
        Args:
            status_file: Path to the status JSON file
            stage: Pipeline stage name
            status: Status value (start, update, complete, fail, skip)
            details: Optional details message
            progress: Optional progress percentage (0-100)
            
        Returns:
            bool: Success status
        """
        try:
            from utils.progress import PipelineProgress
            
            # Try to use the new progress tracker format
            try:
                # Load the progress file using the progress tracker
                progress_tracker = PipelineProgress(status_file)
                
                # Update stage based on status
                if status == 'start':
                    progress_tracker.start_stage(stage, details)
                elif status == 'update':
                    progress_tracker.update_stage_progress(stage, progress or 0, details)
                elif status == 'complete':
                    progress_tracker.complete_stage(stage, details)
                elif status == 'fail':
                    progress_tracker.fail_stage(stage, details or "Stage failed")
                elif status == 'skip':
                    progress_tracker.skip_stage(stage, details or "Stage skipped")
                    
                return True
                
            except Exception as e1:
                # Fallback to old format if new format fails
                logger.warning(f"Failed to use new progress format, falling back: {str(e1)}")
                
                # Create status directory if it doesn't exist
                os.makedirs(os.path.dirname(status_file), exist_ok=True)
                
                # Load existing status file if it exists
                if os.path.exists(status_file):
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                else:
                    # Create new status file structure (old format)
                    status_data = {
                        'status': 'running',
                        'start_time': datetime.now().isoformat(),
                        'tasks': {}
                    }
                
                # Update the stage status
                if 'tasks' not in status_data:
                    status_data['tasks'] = {}
                    
                if stage not in status_data['tasks']:
                    status_data['tasks'][stage] = {}
                    
                status_data['tasks'][stage]['status'] = status
                if details:
                    status_data['tasks'][stage]['details'] = details
                if progress is not None:
                    status_data['tasks'][stage]['progress'] = progress
                status_data['tasks'][stage]['timestamp'] = datetime.now().isoformat()
                
                # Update overall status if a task failed
                if status == 'fail':
                    status_data['status'] = 'error'
                elif status == 'complete' and all(task.get('status') in ['complete', 'skip'] 
                                               for task in status_data['tasks'].values()):
                    status_data['status'] = 'complete'
                    status_data['end_time'] = datetime.now().isoformat()
            
            # Write updated status file
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error updating progress for stage {stage}: {str(e)}")
            return False
    
    @staticmethod
    def download_latest_data(base_dir=DATA_DIR, skip_historical=False):
        """
        Download the latest Numerai and Yiedl data.
        
        Args:
            base_dir: Base directory for data storage
            skip_historical: If True, skip downloading historical data
            
        Returns:
            bool: Success status
        """
        try:
            from data.retrieval import NumeraiDataRetriever
            from utils.progress import PipelineProgress
            
            progress = PipelineProgress()
            data_retriever = NumeraiDataRetriever(base_dir=base_dir)
            
            progress.update_stage_progress('data_download', 30, 'Downloading Numerai latest data')
            data_retriever.download_current_datasets(cleanup_zip=True, include_historical=not skip_historical)
            progress.update_stage_progress('data_download', 60, 'Downloaded data')
            
            progress.update_stage_progress('data_download', 75, 'Loading datasets')
            data_retriever.load_datasets()
            progress.update_stage_progress('data_download', 85, 'Preparing merged datasets')
            data_retriever.prepare_merged_datasets()
            
            symbols = data_retriever.get_eligible_symbols()
            symbol_count = len(symbols) if symbols else 0
            progress.update_stage_progress('data_download', 95, 
                                         f'Found {symbol_count} eligible cryptocurrency symbols')
            
            return True
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return False
            
    @staticmethod
    def verify_data_files():
        """
        Verify downloaded data files exist and are of expected size.
        
        Returns:
            dict: Status information about data files
        """
        result = {
            'success': True,
            'files': {}
        }
        
        # Check Numerai files
        numerai_file = f"{DATA_DIR}/raw/numerai_latest.parquet"
        if os.path.exists(numerai_file):
            file_size = os.path.getsize(numerai_file)
            result['files']['numerai_latest'] = {
                'exists': True,
                'size': file_size,
                'size_human': f"{file_size / (1024*1024):.2f} MB"
            }
        else:
            result['files']['numerai_latest'] = {
                'exists': False
            }
            result['success'] = False
            
        # Check Yiedl files
        yiedl_file = f"{DATA_DIR}/raw/yiedl_latest.parquet"
        if os.path.exists(yiedl_file):
            file_size = os.path.getsize(yiedl_file)
            result['files']['yiedl_latest'] = {
                'exists': True,
                'size': file_size,
                'size_human': f"{file_size / (1024*1024):.2f} MB" 
            }
        else:
            result['files']['yiedl_latest'] = {
                'exists': False
            }
            
        # Check Yiedl historical file
        yiedl_hist_file = f"{DATA_DIR}/raw/yiedl_historical.parquet"
        if os.path.exists(yiedl_hist_file):
            file_size = os.path.getsize(yiedl_hist_file)
            result['files']['yiedl_historical'] = {
                'exists': True,
                'size': file_size,
                'size_human': f"{file_size / (1024*1024):.2f} MB",
                'size_warning': file_size < 1073741824  # Warning if less than 1GB
            }
        else:
            result['files']['yiedl_historical'] = {
                'exists': False
            }
            
        # Check merged dataset files (processed data)
        try:
            import glob
            
            # Look for merged training file - check multiple patterns
            merged_files = []
            # First look for standard pattern files
            for ext in [".parquet", ".csv"]:
                merged_files.extend(glob.glob(os.path.join(f"{DATA_DIR}/processed", f"*train*merged*{ext}")))
                
            # Also check for crypto_train.parquet which is the standard expected file
            crypto_train = os.path.join(f"{DATA_DIR}/processed", "crypto_train.parquet")
            if os.path.exists(crypto_train):
                merged_files.append(crypto_train)
            
            if merged_files:
                # Sort by modification time to get the latest
                latest_merged_file = max(merged_files, key=os.path.getmtime)
                file_size = os.path.getsize(latest_merged_file)
                
                result['files']['merged_train'] = {
                    'exists': True,
                    'path': latest_merged_file,
                    'size': file_size,
                    'size_human': f"{file_size / (1024*1024):.2f} MB"
                }
                
                # Check column count
                import pandas as pd
                MIN_COLUMNS = 3000
                
                try:
                    # Try to load file header only for column count
                    if latest_merged_file.endswith('.parquet'):
                        df_sample = pd.read_parquet(latest_merged_file, engine='pyarrow')
                    else:
                        df_sample = pd.read_csv(latest_merged_file, nrows=1)
                    
                    num_columns = len(df_sample.columns)
                    result['files']['merged_train']['columns'] = num_columns
                    
                    # Warning if columns are less than minimum expected
                    if num_columns < MIN_COLUMNS:
                        result['files']['merged_train']['column_warning'] = True
                        result['files']['merged_train']['column_warning_msg'] = f"Expected at least {MIN_COLUMNS} columns, found only {num_columns}"
                        result['success'] = False
                        logger.error(f"ERROR: Merged training file has only {num_columns} columns, but should have at least {MIN_COLUMNS}. This indicates the data processing pipeline failed to properly merge Numerai and Yiedl data.")
                except Exception as e:
                    logger.error(f"Error checking merged dataset column count: {str(e)}")
                    result['files']['merged_train']['column_error'] = str(e)
            else:
                result['files']['merged_train'] = {
                    'exists': False,
                    'error': "No merged training file found"
                }
                result['success'] = False
                logger.error("ERROR: No merged training file found. The data processing pipeline has likely failed.")
        except Exception as e:
            logger.error(f"Error checking merged dataset files: {str(e)}")
            result['files']['merged_train'] = {
                'exists': False,
                'error': str(e)
            }
            
        return result
        
    @staticmethod
    def validate_merged_dataset():
        """
        Validate that the merged dataset exists and has the required number of columns.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        import os
        import pandas as pd
        import glob
        from pathlib import Path
        
        # Define data directories
        PROCESSED_DIR = f"{DATA_DIR}/processed"
        
        # Look for merged training file - check multiple patterns
        merged_files = []
        # First look for standard pattern files
        for ext in [".parquet", ".csv"]:
            merged_files.extend(glob.glob(os.path.join(PROCESSED_DIR, f"*train*merged*{ext}")))
            
        # Also check for crypto_train.parquet which is the standard expected file
        crypto_train = os.path.join(PROCESSED_DIR, "crypto_train.parquet")
        if os.path.exists(crypto_train):
            merged_files.append(crypto_train)
        
        if not merged_files:
            logger.error("ERROR: No merged training file found. The data processing pipeline has failed.")
            return False
            
        # Sort by modification time to get the latest
        latest_merged_file = max(merged_files, key=os.path.getmtime)
        logger.info(f"Found latest merged training file: {latest_merged_file}")
        
        # Check file size
        file_size_mb = os.path.getsize(latest_merged_file) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Required minimum 
        MIN_COLUMNS = 3000
        
        # Check column count
        try:
            # Try to read column names only to limit memory usage
            if latest_merged_file.endswith('.parquet'):
                df_sample = pd.read_parquet(latest_merged_file, engine='pyarrow')
            else:
                df_sample = pd.read_csv(latest_merged_file, nrows=1)
            
            num_columns = len(df_sample.columns)
            logger.info(f"Number of columns: {num_columns}")
            
            # Validate column count - merged file should have at least 3000 columns
            if num_columns < MIN_COLUMNS:
                error_msg = f"ERROR: Merged training file has only {num_columns} columns, but should have at least {MIN_COLUMNS}. This indicates the data processing pipeline failed to properly merge Numerai and Yiedl data."
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                return False
            
            # Check for target column
            if 'target' not in df_sample.columns:
                error_msg = "ERROR: Merged training file does not contain a 'target' column."
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                return False
            
            logger.info("Merged training file validation successful.")
            return True
            
        except Exception as e:
            error_msg = f"ERROR: Failed to validate merged training file: {str(e)}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            return False

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Pipeline Utilities for Numerai Crypto')
    parser.add_argument('--action', type=str, required=True, 
                        choices=['create-progress', 'update-progress', 'download-data', 'verify-data', 'validate-merged-dataset'],
                        help='Action to perform')
    parser.add_argument('--stage', type=str, help='Pipeline stage for update-progress')
    parser.add_argument('--status', type=str, help='Status value for update-progress')
    parser.add_argument('--details', type=str, help='Details message for update-progress')
    parser.add_argument('--progress', type=float, help='Progress percentage for update-progress')
    parser.add_argument('--status-file', type=str, help='Path to status file')
    parser.add_argument('--skip-historical', action='store_true', help='Skip downloading historical data')
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.action == 'create-progress':
        status_file = PipelineUtils.create_progress_tracker(args.status_file)
        print(status_file)
        return 0
        
    elif args.action == 'update-progress':
        if not args.stage or not args.status:
            logger.error("--stage and --status are required for update-progress action")
            return 1
            
        result = PipelineUtils.update_progress(
            args.status_file, args.stage, args.status, args.details, args.progress)
        return 0 if result else 1
        
    elif args.action == 'download-data':
        result = PipelineUtils.download_latest_data(
            skip_historical=args.skip_historical)
        return 0 if result else 1
        
    elif args.action == 'verify-data':
        result = PipelineUtils.verify_data_files()
        print(json.dumps(result, indent=2))
        return 0 if result['success'] else 1
        
    elif args.action == 'validate-merged-dataset':
        result = PipelineUtils.validate_merged_dataset()
        if result:
            print("VALID: Merged dataset validation passed.")
        else:
            print("ERROR: Merged dataset validation failed. Check logs for details.", file=sys.stderr)
        return 0 if result else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())