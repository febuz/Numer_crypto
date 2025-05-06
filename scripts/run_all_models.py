#!/usr/bin/env python3
"""
Script to run all available models for Numerai crypto prediction
and compare their performance.

This script orchestrates the execution of different model training scripts:
- LightGBM/XGBoost with GPU acceleration
- H2O AutoML with optional Sparkling Water integration
- Ensemble model combining the best individual models

The best performing model is used to generate the final submission.
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error
import shutil

# Set up logging
log_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_available_libraries():
    """Check which ML libraries are available in the environment"""
    libraries = {}
    
    # Check standard libraries
    libraries['numpy'] = check_library('numpy')
    libraries['pandas'] = check_library('pandas')
    libraries['sklearn'] = check_library('sklearn')
    
    # Check ML libraries
    libraries['lightgbm'] = check_library('lightgbm')
    libraries['xgboost'] = check_library('xgboost')
    libraries['h2o'] = check_library('h2o')
    libraries['pysparkling'] = check_library('pysparkling')
    libraries['torch'] = check_library('torch')
    libraries['tpot'] = check_library('tpot')
    
    # Check GPU libraries
    libraries['cuml'] = check_library('cuml')
    
    # Check Numerai API
    libraries['numerapi'] = check_library('numerapi')
    
    # Log available libraries
    logger.info("=== Available Libraries ===")
    for lib, available in libraries.items():
        logger.info(f"{lib}: {available}")
    
    return libraries

def check_library(lib_name):
    """Check if a library is available"""
    try:
        __import__(lib_name)
        return True
    except ImportError:
        return False

def check_gpu_availability():
    """Check if GPU is available"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'devices': []
    }
    
    # Check CUDA availability via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device_name
                })
    except:
        pass
    
    # Log GPU information
    if gpu_info['cuda_available']:
        logger.info(f"CUDA is available with {gpu_info['gpu_count']} devices:")
        for device in gpu_info['devices']:
            logger.info(f"  - GPU {device['id']}: {device['name']}")
    else:
        logger.info("No GPUs detected")
    
    return gpu_info

def download_data(script_dir):
    """Download Numerai and Yiedl data"""
    logger.info("Downloading Numerai and Yiedl data")
    
    download_script = os.path.join(script_dir, "download_numerai_yiedl_data.py")
    
    try:
        result = subprocess.run(
            [sys.executable, download_script],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Data download complete")
        
        # Try to extract file paths from the log
        output_lines = result.stdout.split("\n") + result.stderr.split("\n")
        
        train_file = None
        live_file = None
        current_round = None
        
        for line in output_lines:
            if "Merged train data:" in line:
                train_file = line.split("Merged train data:")[-1].strip()
            elif "Merged live data:" in line:
                live_file = line.split("Merged live data:")[-1].strip()
            elif "Numerai current round:" in line:
                try:
                    current_round = int(line.split("Numerai current round:")[-1].strip())
                except:
                    pass
        
        return {
            'train_file': train_file,
            'live_file': live_file,
            'current_round': current_round
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None

def run_lgbm_xgb_model(script_dir, data_files, use_gpu=False, gpu_id=0, output_dir="submissions"):
    """Run LightGBM/XGBoost model training"""
    logger.info("Running LightGBM/XGBoost model training")
    
    script_path = os.path.join(script_dir, "train_and_predict.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--train-file", data_files['train_file'],
        "--live-file", data_files['live_file'],
        "--output-dir", output_dir
    ]
    
    if 'current_round' in data_files and data_files['current_round']:
        cmd.extend(["--round", str(data_files['current_round'])])
    
    if use_gpu:
        cmd.append("--gpu")
        cmd.extend(["--gpu-id", str(gpu_id)])
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Extract submission file paths and RMSE values from output
        output_lines = process.stdout.split("\n") + process.stderr.split("\n")
        
        submissions = {}
        rmse_values = {}
        
        for line in output_lines:
            if "Saved submission to" in line:
                file_path = line.split("Saved submission to")[-1].strip()
                if "lightgbm" in file_path.lower():
                    submissions['lightgbm'] = file_path
                elif "xgboost" in file_path.lower():
                    submissions['xgboost'] = file_path
                elif "ensemble" in file_path.lower():
                    submissions['ensemble'] = file_path
            elif "RMSE:" in line:
                for model in ['lightgbm', 'xgboost', 'ensemble']:
                    if model in line.lower():
                        try:
                            rmse = float(line.split("RMSE:")[-1].split(",")[0].strip())
                            rmse_values[model] = rmse
                        except:
                            pass
        
        return {
            'submissions': submissions,
            'rmse_values': rmse_values
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"LightGBM/XGBoost model training failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None

def run_h2o_automl(script_dir, data_files, use_sparkling=False, max_runtime=600, output_dir="submissions"):
    """Run H2O AutoML model training"""
    logger.info("Running H2O AutoML model training")
    
    script_path = os.path.join(script_dir, "h2o_automl_crypto.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--train-file", data_files['train_file'],
        "--live-file", data_files['live_file'],
        "--output-dir", output_dir,
        "--max-runtime", str(max_runtime)
    ]
    
    if 'current_round' in data_files and data_files['current_round']:
        cmd.extend(["--round", str(data_files['current_round'])])
    
    if use_sparkling:
        cmd.append("--use-sparkling")
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Extract submission file paths and RMSE values from output
        output_lines = process.stdout.split("\n") + process.stderr.split("\n")
        
        submission_file = None
        rmse_value = None
        
        for line in output_lines:
            if "Submission file:" in line:
                submission_file = line.split("Submission file:")[-1].strip()
            elif "Validation RMSE:" in line:
                try:
                    rmse_value = float(line.split("Validation RMSE:")[-1].strip())
                except:
                    pass
        
        return {
            'submission': submission_file,
            'rmse': rmse_value
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"H2O AutoML model training failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None

def create_ensemble_submission(submissions, output_dir="submissions", round_num=None):
    """Create an ensemble submission by averaging predictions from all models"""
    logger.info("Creating ensemble submission from all models")
    
    # Gather all submission files
    all_submissions = []
    for model_type, results in submissions.items():
        if model_type == 'lgbm_xgb':
            if 'submissions' in results:
                all_submissions.extend(list(results['submissions'].values()))
        elif model_type == 'h2o_automl':
            if 'submission' in results:
                all_submissions.append(results['submission'])
    
    if not all_submissions:
        logger.error("No submission files found to create ensemble")
        return None
    
    logger.info(f"Creating ensemble from {len(all_submissions)} submission files")
    
    # Read all submission files
    submission_dfs = []
    for file in all_submissions:
        try:
            df = pd.read_csv(file)
            submission_dfs.append(df)
            logger.info(f"Loaded submission file: {file}, shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading submission file {file}: {e}")
    
    if not submission_dfs:
        logger.error("Failed to load any submission files")
        return None
    
    # Create ensemble by averaging predictions
    ensemble_df = submission_dfs[0][['id']].copy()
    
    # Get predictions from each model
    for df in submission_dfs:
        df = df.set_index('id')
    
    # Average predictions
    ensemble_df['prediction'] = 0
    for df in submission_dfs:
        ensemble_df = ensemble_df.set_index('id')
        ensemble_df['prediction'] += df.set_index('id')['prediction'] / len(submission_dfs)
        ensemble_df = ensemble_df.reset_index()
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    round_str = f"_round{round_num}" if round_num else ""
    ensemble_file = os.path.join(output_dir, f"final_ensemble{round_str}_{timestamp}.csv")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(ensemble_file), exist_ok=True)
    
    # Save ensemble submission
    ensemble_df.to_csv(ensemble_file, index=False)
    logger.info(f"Saved ensemble submission to {ensemble_file}")
    
    return ensemble_file

def find_best_model(submissions):
    """Find the best model based on validation RMSE"""
    logger.info("Finding best model based on validation RMSE")
    
    best_model = None
    best_rmse = float('inf')
    best_submission = None
    
    # Check LightGBM/XGBoost models
    if 'lgbm_xgb' in submissions and submissions['lgbm_xgb']:
        if 'rmse_values' in submissions['lgbm_xgb']:
            for model, rmse in submissions['lgbm_xgb']['rmse_values'].items():
                logger.info(f"{model} RMSE: {rmse}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    if 'submissions' in submissions['lgbm_xgb'] and model in submissions['lgbm_xgb']['submissions']:
                        best_submission = submissions['lgbm_xgb']['submissions'][model]
    
    # Check H2O AutoML model
    if 'h2o_automl' in submissions and submissions['h2o_automl']:
        if 'rmse' in submissions['h2o_automl'] and submissions['h2o_automl']['rmse']:
            rmse = submissions['h2o_automl']['rmse']
            logger.info(f"h2o_automl RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = 'h2o_automl'
                if 'submission' in submissions['h2o_automl']:
                    best_submission = submissions['h2o_automl']['submission']
    
    # Check ensemble
    if 'ensemble' in submissions and submissions['ensemble']:
        if isinstance(submissions['ensemble'], str):
            # If ensemble is just a file path
            logger.info("Including final ensemble in comparison")
            best_model = 'ensemble'
            best_submission = submissions['ensemble']
    
    logger.info(f"Best model: {best_model}, RMSE: {best_rmse}")
    
    return {
        'model': best_model,
        'rmse': best_rmse,
        'submission': best_submission
    }

def copy_best_submission(best_model, output_dir="submissions", round_num=None):
    """Copy the best submission file to a final submission file"""
    if not best_model or 'submission' not in best_model or not best_model['submission']:
        logger.error("No best submission file to copy")
        return None
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    round_str = f"_round{round_num}" if round_num else ""
    model_name = best_model['model'] if 'model' in best_model else 'unknown'
    
    final_file = os.path.join(output_dir, f"final_submission_{model_name}{round_str}_{timestamp}.csv")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    
    # Copy the best submission
    shutil.copy2(best_model['submission'], final_file)
    logger.info(f"Copied best submission ({model_name}) to {final_file}")
    
    return final_file

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run multiple models for Numerai crypto prediction')
    
    parser.add_argument('--output-dir', type=str, default='submissions', help='Directory to save submission files')
    parser.add_argument('--skip-data-download', action='store_true', help='Skip downloading new data')
    parser.add_argument('--train-file', type=str, help='Path to merged training data parquet file (if skipping download)')
    parser.add_argument('--live-file', type=str, help='Path to merged live data parquet file (if skipping download)')
    parser.add_argument('--round', type=int, help='Current Numerai round number')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--skip-lgbm-xgb', action='store_true', help='Skip LightGBM/XGBoost models')
    parser.add_argument('--skip-h2o', action='store_true', help='Skip H2O AutoML models')
    parser.add_argument('--use-h2o-sparkling', action='store_true', help='Use H2O Sparkling Water instead of standalone H2O')
    parser.add_argument('--h2o-runtime', type=int, default=600, help='Maximum runtime in seconds for H2O AutoML')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join(Path.cwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check available libraries
    available_libs = check_available_libraries()
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    use_gpu = args.use_gpu and gpu_info['cuda_available']
    
    # Download data if needed
    data_files = None
    if not args.skip_data_download:
        data_files = download_data(script_dir)
    else:
        if args.train_file and args.live_file:
            data_files = {
                'train_file': args.train_file,
                'live_file': args.live_file,
                'current_round': args.round
            }
        else:
            logger.error("Skipping data download but no data files provided")
            return 1
    
    if not data_files:
        logger.error("No data files available. Exiting.")
        return 1
    
    # Run models and collect results
    submissions = {}
    
    # Run LightGBM/XGBoost models
    if not args.skip_lgbm_xgb and (available_libs['lightgbm'] or available_libs['xgboost']):
        logger.info("Running LightGBM/XGBoost models")
        lgbm_xgb_results = run_lgbm_xgb_model(
            script_dir, 
            data_files, 
            use_gpu=use_gpu,
            gpu_id=args.gpu_id,
            output_dir=output_dir
        )
        submissions['lgbm_xgb'] = lgbm_xgb_results
    else:
        if args.skip_lgbm_xgb:
            logger.info("Skipping LightGBM/XGBoost models as requested")
        else:
            logger.info("Skipping LightGBM/XGBoost models: libraries not available")
    
    # Run H2O AutoML
    if not args.skip_h2o and available_libs['h2o']:
        logger.info("Running H2O AutoML model")
        h2o_results = run_h2o_automl(
            script_dir, 
            data_files, 
            use_sparkling=args.use_h2o_sparkling and available_libs['pysparkling'],
            max_runtime=args.h2o_runtime,
            output_dir=output_dir
        )
        submissions['h2o_automl'] = h2o_results
    else:
        if args.skip_h2o:
            logger.info("Skipping H2O AutoML model as requested")
        else:
            logger.info("Skipping H2O AutoML model: library not available")
    
    # Create ensemble from all models
    if len(submissions) > 1:
        logger.info("Creating ensemble from all models")
        ensemble_file = create_ensemble_submission(
            submissions,
            output_dir=output_dir,
            round_num=data_files.get('current_round') or args.round
        )
        submissions['ensemble'] = ensemble_file
    
    # Find best model
    best_model = find_best_model(submissions)
    
    # Copy best submission to final file
    final_submission = copy_best_submission(
        best_model,
        output_dir=output_dir,
        round_num=data_files.get('current_round') or args.round
    )
    
    # Print summary
    logger.info("\n===== MODEL COMPARISON SUMMARY =====")
    
    if 'lgbm_xgb' in submissions and submissions['lgbm_xgb'] and 'rmse_values' in submissions['lgbm_xgb']:
        logger.info("LightGBM/XGBoost Models:")
        for model, rmse in submissions['lgbm_xgb']['rmse_values'].items():
            logger.info(f"  - {model}: RMSE = {rmse:.6f}")
    
    if 'h2o_automl' in submissions and submissions['h2o_automl'] and 'rmse' in submissions['h2o_automl']:
        logger.info(f"H2O AutoML: RMSE = {submissions['h2o_automl']['rmse']:.6f}")
    
    if best_model:
        logger.info(f"\nBest model: {best_model['model']}")
        if 'rmse' in best_model:
            logger.info(f"Best RMSE: {best_model['rmse']:.6f}")
        
        # Check if we achieved the target RMSE
        if 'rmse' in best_model and best_model['rmse'] <= 0.25:
            logger.info(f"SUCCESS! Target RMSE of 0.25 achieved with {best_model['model']} model.")
        elif 'rmse' in best_model:
            logger.info(f"Target RMSE of 0.25 not yet achieved. Best RMSE: {best_model['rmse']:.6f}")
    
    if final_submission:
        logger.info(f"\nFinal submission file: {final_submission}")
    
    logger.info("\nModel comparison complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())