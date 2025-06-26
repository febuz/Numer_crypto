#!/usr/bin/env python3
"""
Simple submission script using Numerai's CryptoAPI
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_and_submit_predictions(output_path, public_id, secret_key):
    """Create predictions and submit to Numerai"""
    try:
        from numerapi import NumerAPI  # Use standard NumerAPI
    except ImportError:
        logger.error("numerapi package not installed. Run 'pip install numerapi'")
        return False
    
    try:
        # Generate prediction file
        logger.info("Creating prediction file...")
        
        # Top 20 cryptocurrencies by market cap
        symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'DOGE', 'LINK', 'DOT', 'MATIC', 
                  'AVAX', 'ATOM', 'UNI', 'LTC', 'BCH', 'FIL', 'ALGO', 'XTZ', 'EOS', 'AAVE']
        
        # Generate predictions (balanced around 0.5)
        np.random.seed(42)  # For reproducibility
        predictions = np.random.normal(0.5, 0.05, len(symbols))
        predictions = np.clip(predictions, 0.35, 0.65)  # Ensure values are within reasonable range
        
        # Create submission dataframe
        df = pd.DataFrame({'symbol': symbols, 'prediction': predictions})
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Created submission file with {len(df)} predictions at {output_path}")
        
        # Initialize NumerAPI
        logger.info("Initializing Numerai API...")
        napi = NumerAPI(public_id=public_id, secret_key=secret_key)
        
        # Get current round and print account info
        try:
            current_round = napi.get_current_round()
            logger.info(f"Current round: {current_round}")
            
            # Try to get user information
            user = napi.get_user()
            logger.info(f"User: {user.get('username')}")
        except Exception as e:
            logger.warning(f"Could not get user/round info: {e}")
        
        # Submit predictions
        logger.info(f"Submitting predictions from {output_path}...")
        
        # Try to list all models
        try:
            models = napi.get_models()
            logger.info(f"Available models: {models}")
            
            # If models found, try to submit to the first one
            if models:
                model_id = models[0]['id']
                logger.info(f"Using model ID: {model_id}")
                submission_id = napi.upload_predictions(file_path=output_path, model_id=model_id)
                logger.info(f"Submission successful with ID: {submission_id}")
            else:
                # Try submission without model ID
                logger.info("No models found, submitting without model ID...")
                submission_id = napi.upload_predictions(file_path=output_path)
                logger.info(f"Submission successful with ID: {submission_id}")
        except Exception as e:
            logger.error(f"Submission failed: {e}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create and submit predictions to Numerai')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save the prediction file')
    parser.add_argument('--public-id', type=str, 
                        default=os.environ.get('NUMERAI_PUBLIC_ID', "ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG"),
                        help='Numerai API public ID')
    parser.add_argument('--secret-key', type=str, 
                        default=os.environ.get('NUMERAI_SECRET_KEY', "LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY"),
                        help='Numerai API secret key')
    
    args = parser.parse_args()
    
    # Create and submit predictions
    if create_and_submit_predictions(args.output, args.public_id, args.secret_key):
        logger.info("Submission process completed successfully!")
        return 0
    else:
        logger.error("Submission process failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())