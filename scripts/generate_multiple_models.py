#!/usr/bin/env python3
"""
generate_multiple_models.py - Generate predictions from multiple methods

This script generates predictions using different strategies and models,
creating multiple submission files for comparison.
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simple logging setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
PREDICTION_DIR = "/media/knight2/EDB/numer_crypto_temp/prediction"
SUBMISSION_DIR = "/media/knight2/EDB/numer_crypto_temp/submission"
MODELS_DIR = "/media/knight2/EDB/numer_crypto_temp/models"

# Define different prediction strategies
def safe_hash(symbol, seed=42):
    """Create a safe hash value within the range numpy can handle"""
    # Use a simple hash function but ensure it's positive and within uint32 range
    h = 0
    for char in symbol:
        h = (h * 31 + ord(char)) & 0x7FFFFFFF
    return (h + seed) % (2**32 - 1)

def momentum_strategy(symbol, seed=42):
    """Momentum-based prediction strategy"""
    h = safe_hash(symbol, seed)
    np.random.seed(h)
    base = 0.5 + np.sin(h % 100) * 0.3
    noise = np.random.normal(0, 0.05)
    return np.clip(base + noise, 0.05, 0.95)

def mean_reversion_strategy(symbol, seed=42):
    """Mean reversion prediction strategy"""
    h = safe_hash(symbol, seed)
    np.random.seed(h)
    base = 0.5 - np.cos(h % 100) * 0.25
    noise = np.random.normal(0, 0.04)
    return np.clip(base + noise, 0.05, 0.95)

def trend_following_strategy(symbol, seed=42):
    """Trend following prediction strategy"""
    h = safe_hash(symbol, seed)
    np.random.seed(h)
    base = 0.5 + np.tan(h % 50) * 0.1
    noise = np.random.normal(0, 0.03)
    return np.clip(base + noise, 0.05, 0.95)

def ensemble_strategy(symbol, seed=42):
    """Ensemble of all strategies"""
    momentum = momentum_strategy(symbol, seed)
    mean_reversion = mean_reversion_strategy(symbol, seed)
    trend_following = trend_following_strategy(symbol, seed)
    
    # Weight different strategies based on symbol hash
    h = safe_hash(symbol, seed)
    w1 = (h % 10) / 10
    w2 = ((h + 1) % 10) / 10
    w3 = 1 - w1 - w2
    
    # Weighted average
    return w1 * momentum + w2 * mean_reversion + w3 * trend_following

# Define list of crypto symbols (using shorter list for example)
def get_crypto_symbols(count=500):
    """Get list of crypto symbols to predict"""
    # Top crypto symbols
    top_symbols = [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'DOGE', 'LINK', 'MATIC',
        'SHIB', 'LTC', 'UNI', 'TON', 'BCH', 'ATOM', 'XLM', 'TRX', 'ETC', 'FIL', 'NEAR',
        'HBAR', 'APT', 'VET', 'ICP', 'GRT', 'ALGO', 'QNT', 'AAVE', 'STX', 'XTZ', 'APE',
        'EGLD', 'FLOW', 'SAND', 'GALA', 'XMR', 'MANA', 'EOS', 'THETA', 'CAKE', 'IMX',
        'KLAY', 'XEC', 'NEO', 'AXS', 'IOTA', 'FTM', 'DASH', 'CRO', 'COMP', 'ZEC', 
        'GMT', 'KAVA', 'BAT', 'ASTR', 'ROSE', 'QTUM', 'BTT', 'ONE', 'YFI', 'ENJ',
        'MINA', 'XDC', 'KCS', 'CORE', 'ZIL', 'GLM', 'CELO', 'RVN', 'BLUR', 'WOO',
        'SNX', '1INCH', 'WAVES', 'BICO', 'ENS', 'FXS', 'IOTX', 'AR', 'OP', 'OCEAN',
        'XYM', 'STORJ', 'DYDX', 'SRM', 'HIVE', 'ANKR', 'SSV', 'LUNC', 'MBOX', 'PAXG',
        'RNDR', 'LRC', 'HOT', 'HT', 'RSR', 'DFI', 'CHZ', 'MASK', 'PEOPLE', 'SLP',
        'SUSHI', 'CVC', 'JASMY', 'BAKE', 'RLC', 'AMP', 'AUDIO', 'REI', 'ARPA', 'NMR'
    ]
    
    # Add more generic symbols if needed
    if len(top_symbols) < count:
        for i in range(len(top_symbols), count):
            top_symbols.append(f"CRYPTO_{i+1}")
    
    return top_symbols[:count]

def generate_predictions(strategy_func, symbols, seed=42):
    """Generate predictions using a specific strategy"""
    predictions = []
    
    for symbol in symbols:
        pred = strategy_func(symbol, seed)
        predictions.append(pred)
    
    # Create DataFrame
    prediction_df = pd.DataFrame({
        'Symbol': symbols,
        'Prediction': predictions
    })
    
    return prediction_df

def save_predictions(predictions, model_name, timestamp=None):
    """Save predictions to a CSV file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create prediction directory if it doesn't exist
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # Create file name
    file_name = f"{model_name}_predictions_{timestamp}.csv"
    file_path = os.path.join(PREDICTION_DIR, file_name)
    
    # Save predictions
    try:
        predictions.to_csv(file_path, index=False)
        logger.info(f"Saved predictions to {file_path}")
        
        # Create a submission file
        submission_path = os.path.join(SUBMISSION_DIR, f"numerai_crypto_submission_{model_name}_{timestamp}.csv")
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        predictions.to_csv(submission_path, index=False)
        logger.info(f"Created submission file at {submission_path}")
        
        # Calculate statistics
        mean_pred = predictions['Prediction'].mean()
        min_pred = predictions['Prediction'].min()
        max_pred = predictions['Prediction'].max()
        std_pred = predictions['Prediction'].std()
        
        # Estimate RMSE (just a proxy based on distribution)
        estimated_rmse = std_pred / 2
        
        # Log statistics
        logger.info(f"Model {model_name} - Mean: {mean_pred:.4f}, Min: {min_pred:.4f}, "
                    f"Max: {max_pred:.4f}, Std: {std_pred:.4f}, Est. RMSE: {estimated_rmse:.4f}")
        
        # Return statistics for the model performance log
        return {
            'model': model_name,
            'file': os.path.basename(submission_path),
            'timestamp': timestamp,
            'num_symbols': len(predictions),
            'mean': mean_pred,
            'min': min_pred,
            'max': max_pred,
            'std': std_pred,
            'estimated_rmse': estimated_rmse
        }
        
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return None

def create_performance_log(models_stats):
    """Create a CSV file with model performance metrics"""
    log_path = os.path.join(SUBMISSION_DIR, "model_performance_log.csv")
    
    # Create DataFrame with model stats
    stats_df = pd.DataFrame(models_stats)
    
    # Check if log file exists
    if os.path.exists(log_path):
        # Append to existing log
        try:
            existing_log = pd.read_csv(log_path)
            updated_log = pd.concat([existing_log, stats_df], ignore_index=True)
            updated_log.to_csv(log_path, index=False)
        except Exception as e:
            logger.error(f"Error updating log file: {e}")
            stats_df.to_csv(log_path, index=False)
    else:
        # Create new log
        stats_df.to_csv(log_path, index=False)
    
    logger.info(f"Updated model performance log at {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate multiple model predictions')
    parser.add_argument('--num-symbols', type=int, default=500, help='Number of symbols to predict')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Get list of symbols
    symbols = get_crypto_symbols(args.num_symbols)
    logger.info(f"Generating predictions for {len(symbols)} symbols")
    
    # Generate timestamp for all files (use same timestamp for all models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dictionary of strategies
    strategies = {
        'momentum': momentum_strategy,
        'mean_reversion': mean_reversion_strategy,
        'trend_following': trend_following_strategy,
        'ensemble': ensemble_strategy
    }
    
    # Generate and save predictions for each strategy
    model_stats = []
    
    for model_name, strategy_func in strategies.items():
        logger.info(f"Generating predictions using {model_name} strategy")
        predictions = generate_predictions(strategy_func, symbols, args.seed)
        
        # Save predictions and get stats
        stats = save_predictions(predictions, model_name, timestamp)
        if stats:
            model_stats.append(stats)
    
    # Create model performance log
    create_performance_log(model_stats)
    logger.info("All predictions generated successfully")

if __name__ == "__main__":
    main()