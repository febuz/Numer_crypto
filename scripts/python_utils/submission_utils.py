#!/usr/bin/env python3
"""
Submission utilities for Numerai Crypto Pipeline
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_dummy_submission(output_path: str, 
                           assets: Optional[List[str]] = None) -> None:
    """
    Create a dummy submission file with random predictions for given assets
    
    Args:
        output_path: Path to save the submission file
        assets: List of asset symbols to include (default to common crypto assets if None)
    """
    if assets is None:
        # Default list of common crypto assets
        assets = [
            'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'EOS', 'ADA', 'XLM', 'TRX', 'BNB',
            'DOT', 'LINK', 'DOGE', 'UNI', 'SOL', 'AVAX', 'MATIC', 'ATOM', 'ALGO', 'VET'
        ]
    
    # Create random predictions (between 0.1 and 0.9)
    predictions = np.random.uniform(0.1, 0.9, len(assets))
    
    # Create DataFrame
    df = pd.DataFrame({
        'symbol': assets,
        'prediction': predictions
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Created dummy submission file with {len(df)} rows at {output_path}")
    
    return True

def check_submission_format(submission_path: str) -> bool:
    """
    Check if a submission file has the correct format
    
    Args:
        submission_path: Path to the submission file to check
        
    Returns:
        Boolean indicating if the format is valid
    """
    try:
        # Read the submission file
        df = pd.read_csv(submission_path)
        
        # Check required columns
        if 'symbol' not in df.columns or 'prediction' not in df.columns:
            logger.error(f"Missing required columns in {submission_path}")
            return False
        
        # Check for missing values
        if df['symbol'].isna().any() or df['prediction'].isna().any():
            logger.error(f"Missing values found in {submission_path}")
            return False
        
        # Check prediction values range (should be between 0 and 1)
        if (df['prediction'] < 0).any() or (df['prediction'] > 1).any():
            logger.error(f"Prediction values out of range [0,1] in {submission_path}")
            return False
        
        # Check for duplicates
        if df['symbol'].duplicated().any():
            logger.error(f"Duplicate symbols found in {submission_path}")
            return False
        
        # All checks passed
        logger.info(f"Submission file {submission_path} has valid format")
        return True
        
    except Exception as e:
        logger.error(f"Error checking submission file {submission_path}: {e}")
        return False

if __name__ == "__main__":
    # Simple command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Submission utilities for Numerai Crypto Pipeline")
    parser.add_argument("--create-dummy", type=str, help="Create a dummy submission file at the specified path")
    parser.add_argument("--check-format", type=str, help="Check if a submission file has the correct format")
    
    args = parser.parse_args()
    
    if args.create_dummy:
        create_dummy_submission(args.create_dummy)
    
    if args.check_format:
        check_submission_format(args.check_format)