#!/usr/bin/env python3
"""
create_submission.py - Create a submission file for Numerai Crypto

This script formats prediction data and creates a properly formatted submission
file in the configured submission directory.
"""
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories for files
PREDICTION_DIR = "/media/knight2/EDB/numer_crypto_temp/prediction"
SUBMISSIONS_DIR = "/media/knight2/EDB/numer_crypto_temp/submission"

def create_submission(output_dir=None, prediction_file=None, round_number=None):
    """
    Create a submission file from prediction data.
    
    Args:
        output_dir (str): Directory to save the submission file
        prediction_file (str): Path to the prediction file
        round_number (int): Tournament round number
        
    Returns:
        str: Path to the created submission file
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = SUBMISSIONS_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use latest prediction file if not specified
    if prediction_file is None:
        # Look for prediction files in the prediction directory
        prediction_files = [f for f in os.listdir(PREDICTION_DIR) if f.endswith(('.csv', '.parquet'))]
        if not prediction_files:
            logger.error(f"No prediction files found in {PREDICTION_DIR}")
            return None
        
        # Sort by modification time (most recent first)
        prediction_files.sort(key=lambda x: os.path.getmtime(os.path.join(PREDICTION_DIR, x)), reverse=True)
        prediction_file = os.path.join(PREDICTION_DIR, prediction_files[0])
        logger.info(f"Using latest prediction file: {prediction_file}")
    
    # Load prediction data
    try:
        logger.info(f"Loading prediction data from {prediction_file}")
        # Simple file loading without dependencies
        if prediction_file.endswith('.csv'):
            predictions = pd.read_csv(prediction_file)
        elif prediction_file.endswith('.parquet'):
            try:
                predictions = pd.read_parquet(prediction_file)
            except:
                logger.error("Could not read parquet file, pandas may be missing parquet dependencies")
                return None
        else:
            logger.error(f"Unsupported file format: {prediction_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading prediction file: {e}")
        return None
    
    # Check if predictions have the required columns (lowercase for Numerai)
    required_columns = {'symbol', 'prediction'}
    if not all(col in predictions.columns for col in required_columns):
        # Try to create the required columns from existing ones
        if 'Symbol' in predictions.columns and 'symbol' not in predictions.columns:
            predictions['symbol'] = predictions['Symbol']
        elif any(col.lower() == 'symbol' for col in predictions.columns):
            symbol_col = next(col for col in predictions.columns if col.lower() == 'symbol')
            predictions['symbol'] = predictions[symbol_col]
        
        if 'Prediction' in predictions.columns and 'prediction' not in predictions.columns:
            predictions['prediction'] = predictions['Prediction']
        elif any(col.lower() == 'prediction' for col in predictions.columns):
            pred_col = next(col for col in predictions.columns if col.lower() == 'prediction')
            predictions['prediction'] = predictions[pred_col]
        
        # Check again after attempting to create the columns
        if not all(col in predictions.columns for col in required_columns):
            logger.error(f"Prediction data is missing required columns: {required_columns}")
            logger.error(f"Available columns: {list(predictions.columns)}")
            return None
    
    # Select only the required columns in the correct order: symbol, prediction
    submission_data = predictions[['symbol', 'prediction']].copy()
    
    # Format the predictions (ensure predictions are between 0 and 1)
    submission_data['prediction'] = submission_data['prediction'].clip(0, 1)
    
    # Create the submission file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if round_number:
        file_name = f"numerai_crypto_submission_round_{round_number}_{timestamp}.csv"
    else:
        file_name = f"numerai_crypto_submission_{timestamp}.csv"
    
    submission_file = os.path.join(output_dir, file_name)
    
    # Save the submission file
    try:
        submission_data.to_csv(submission_file, index=False)
        logger.info(f"Submission file created: {submission_file}")
        
        # Print submission stats
        symbol_count = len(submission_data['symbol'].unique())
        logger.info(f"Submission contains predictions for {symbol_count} symbols")
        logger.info(f"Prediction range: {submission_data['prediction'].min():.4f} to {submission_data['prediction'].max():.4f}")
        logger.info(f"Prediction mean: {submission_data['prediction'].mean():.4f}")
        
        return submission_file
    except Exception as e:
        logger.error(f"Error saving submission file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create submission file for Numerai Crypto')
    parser.add_argument('--output-dir', help='Directory to save the submission file')
    parser.add_argument('--prediction-file', help='Path to the prediction file')
    parser.add_argument('--round', type=int, help='Tournament round number')
    
    args = parser.parse_args()
    
    logger.info("Starting create_submission.py")
    
    submission_file = create_submission(
        output_dir=args.output_dir,
        prediction_file=args.prediction_file,
        round_number=args.round
    )
    
    if submission_file:
        logger.info(f"Submission file created successfully: {submission_file}")
        return True
    else:
        logger.error("Failed to create submission file")
        return False

if __name__ == "__main__":
    main()