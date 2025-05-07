#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import log utils
from utils.log_utils import setup_logging

# Set up logging to external directory
logger = setup_logging(name=__name__, level=logging.INFO)

# List of real cryptocurrency symbols
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'XLM', 'DOGE',
                  'UNI', 'AAVE', 'SNX', 'SUSHI', 'YFI', 'COMP', 'MKR', '1INCH', 'BAL', 'CRV']

def generate_numerai_mock_data(output_dir, n_samples=1000, n_features=20, n_symbols=10):
    """Generate mock Numerai data for testing.
    
    Args:
        output_dir (str): Directory to save mock data
        n_samples (int): Number of samples per symbol
        n_features (int): Number of features
        n_symbols (int): Number of symbols to generate
    
    Returns:
        dict: Paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random symbols
    symbols = random.sample(CRYPTO_SYMBOLS, min(n_symbols, len(CRYPTO_SYMBOLS)))
    
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate train data
    train_data = []
    for symbol in symbols:
        for date in dates:
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol
            }
            
            # Add features
            for i in range(1, n_features + 1):
                row[f'feature_{i}'] = np.random.normal()
            
            # Add target
            row['target'] = np.random.choice([0, 1])
            
            train_data.append(row)
    
    train_df = pd.DataFrame(train_data)
    
    # Generate live universe data
    universe_data = []
    for symbol in symbols:
        universe_data.append({'Symbol': symbol})
    
    universe_df = pd.DataFrame(universe_data)
    
    # Save files
    train_file = os.path.join(output_dir, 'train_data.parquet')
    universe_file = os.path.join(output_dir, 'live_universe.parquet')
    
    # Save as parquet if pyarrow/fastparquet is available, otherwise as CSV
    try:
        train_df.to_parquet(train_file)
        universe_df.to_parquet(universe_file)
        logger.info(f"Saved parquet files to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not save as parquet: {e}")
        # Fall back to CSV
        train_file = os.path.join(output_dir, 'train_data.csv')
        universe_file = os.path.join(output_dir, 'live_universe.csv')
        train_df.to_csv(train_file, index=False)
        universe_df.to_csv(universe_file, index=False)
        logger.info(f"Saved CSV files to {output_dir}")
    
    return {
        'train_data': train_file,
        'live_universe': universe_file,
        'symbols': symbols,
        'train_df': train_df,
        'universe_df': universe_df
    }

def generate_yiedl_mock_data(output_dir, n_samples=1000, n_symbols=10, overlap_pct=0.7):
    """Generate mock Yiedl data for testing.
    
    Args:
        output_dir (str): Directory to save mock data
        n_samples (int): Number of samples per symbol
        n_symbols (int): Number of symbols to generate
        overlap_pct (float): Percentage of symbols to overlap with Numerai symbols
    
    Returns:
        dict: Paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random symbols, ensuring some overlap with Numerai
    n_overlap = int(n_symbols * overlap_pct)
    overlap_symbols = random.sample(CRYPTO_SYMBOLS, min(n_overlap, len(CRYPTO_SYMBOLS)))
    
    # Add some unique symbols
    remaining_symbols = list(set(CRYPTO_SYMBOLS) - set(overlap_symbols))
    unique_symbols = random.sample(remaining_symbols, min(n_symbols - n_overlap, len(remaining_symbols)))
    
    symbols = overlap_symbols + unique_symbols
    
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_samples)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate Yiedl data
    yiedl_data = []
    for symbol in symbols:
        for date in dates:
            row = {
                'date': date.strftime('%Y-%m-%d'),
                'asset': symbol,
                'price': random.uniform(1, 100000),
                'volume': random.uniform(1000, 10000000),
                'market_cap': random.uniform(1000000, 1000000000000),
                'volatility': random.uniform(0.01, 0.5)
            }
            yiedl_data.append(row)
    
    yiedl_df = pd.DataFrame(yiedl_data)
    
    # Save file
    yiedl_file = os.path.join(output_dir, 'yiedl_latest.csv')
    yiedl_df.to_csv(yiedl_file, index=False)
    logger.info(f"Saved Yiedl mock data to {yiedl_file}")
    
    return {
        'latest': yiedl_file,
        'symbols': symbols,
        'overlap_symbols': overlap_symbols,
        'yiedl_df': yiedl_df
    }

def generate_full_mock_dataset(base_dir):
    """Generate a complete mock dataset with Numerai and Yiedl data.
    
    Args:
        base_dir (str): Base directory to save mock data
    
    Returns:
        dict: Paths and metadata for the generated datasets
    """
    # Create directories
    numerai_dir = os.path.join(base_dir, 'numerai')
    yiedl_dir = os.path.join(base_dir, 'yiedl')
    
    os.makedirs(numerai_dir, exist_ok=True)
    os.makedirs(yiedl_dir, exist_ok=True)
    
    # Generate data
    logger.info("Generating Numerai mock data...")
    numerai_data = generate_numerai_mock_data(numerai_dir)
    
    logger.info("Generating Yiedl mock data with overlapping symbols...")
    yiedl_data = generate_yiedl_mock_data(yiedl_dir, 
                                         overlap_pct=0.7)
    
    # Calculate actual overlap
    numerai_symbols = set(numerai_data['symbols'])
    yiedl_symbols = set(yiedl_data['symbols'])
    overlap_symbols = numerai_symbols.intersection(yiedl_symbols)
    
    logger.info(f"Generated mock data with {len(overlap_symbols)} overlapping symbols:")
    logger.info(f"Overlapping symbols: {', '.join(sorted(overlap_symbols))}")
    
    return {
        'numerai_data': numerai_data,
        'yiedl_data': yiedl_data,
        'overlap_symbols': sorted(list(overlap_symbols)),
        'numerai_dir': numerai_dir,
        'yiedl_dir': yiedl_dir
    }

if __name__ == '__main__':
    # Generate a full mock dataset in /tmp
    mock_data_dir = '/tmp/numerai_crypto_mock_data'
    result = generate_full_mock_dataset(mock_data_dir)
    
    logger.info(f"\nMock dataset generated successfully:")
    logger.info(f"  Numerai data: {result['numerai_dir']}")
    logger.info(f"  Yiedl data: {result['yiedl_dir']}")
    logger.info(f"  Overlapping symbols: {', '.join(result['overlap_symbols'])}")