#!/usr/bin/env python3
"""
Process Yiedl data for Numerai Crypto competition.

This script performs the following operations:
1. Loads the Yiedl latest data
2. Loads the Numerai live universe symbols
3. Filters the Yiedl data to match Numerai symbols
4. Performs feature engineering and selection
5. Saves the processed data for model training
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Set up logging
log_file = f"process_yiedl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = os.path.join(Path(__file__).parent.parent, 'data')
NUMERAI_DIR = os.path.join(DATA_DIR, 'numerai')
YIEDL_DIR = os.path.join(DATA_DIR, 'yiedl')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_numerai_symbols(universe_file='/tmp/crypto_live_universe.parquet'):
    """Load symbols from Numerai live universe"""
    logger.info(f"Loading Numerai symbols from {universe_file}")
    
    try:
        # Try to load the file
        universe_df = pd.read_parquet(universe_file)
        
        if 'symbol' in universe_df.columns:
            symbols = universe_df['symbol'].unique().tolist()
            logger.info(f"Loaded {len(symbols)} unique symbols from Numerai")
            return symbols
        else:
            logger.error("No 'symbol' column found in universe file")
            return []
    except Exception as e:
        logger.error(f"Error loading Numerai symbols: {e}")
        return []

def load_yiedl_data(yiedl_file=None):
    """Load latest Yiedl data"""
    if yiedl_file is None:
        yiedl_file = os.path.join(YIEDL_DIR, 'yiedl_latest.parquet')
    
    logger.info(f"Loading Yiedl data from {yiedl_file}")
    
    try:
        # Load the data
        df = pd.read_parquet(yiedl_file)
        logger.info(f"Loaded Yiedl data with shape: {df.shape}")
        
        # Basic info
        logger.info(f"Columns: {', '.join(df.columns[:5])}...")
        logger.info(f"Unique symbols: {df['symbol'].nunique()}")
        logger.info(f"Dates: {sorted(df['date'].unique())}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading Yiedl data: {e}")
        return None

def filter_latest_data(df):
    """Filter to only include most recent date"""
    if df is None or 'date' not in df.columns:
        return df
    
    latest_date = max(df['date'].unique())
    logger.info(f"Filtering to latest date: {latest_date}")
    
    latest_df = df[df['date'] == latest_date].copy()
    logger.info(f"Filtered data shape: {latest_df.shape}")
    
    return latest_df

def filter_by_symbols(df, symbols):
    """Filter data to only include specified symbols"""
    if df is None or 'symbol' not in df.columns:
        return df
    
    logger.info(f"Filtering data to include {len(symbols)} symbols")
    
    # First check how many symbols from numerai are in yiedl data
    matching_symbols = set(df['symbol'].unique()).intersection(set(symbols))
    logger.info(f"Found {len(matching_symbols)} matching symbols")
    
    # Filter the dataframe
    filtered_df = df[df['symbol'].isin(symbols)].copy()
    logger.info(f"Filtered data shape: {filtered_df.shape}")
    
    # If very few matches, try case-insensitive matching or partial matching
    if len(matching_symbols) < 50:
        logger.warning("Very few symbol matches found, trying case-insensitive matching")
        
        # Convert both sets to uppercase for case-insensitive matching
        df_symbols_upper = set(s.upper() if isinstance(s, str) else s for s in df['symbol'].unique())
        numerai_symbols_upper = set(s.upper() if isinstance(s, str) else s for s in symbols)
        
        # Find matches
        upper_matches = df_symbols_upper.intersection(numerai_symbols_upper)
        logger.info(f"Found {len(upper_matches)} case-insensitive matches")
        
        # Map from uppercase to original case
        upper_to_original = {s.upper() if isinstance(s, str) else s: s for s in df['symbol'].unique()}
        
        # Create filtered dataframe based on uppercase matches
        case_insensitive_symbols = [upper_to_original.get(s.upper()) for s in symbols if s.upper() in upper_to_original]
        filtered_df = df[df['symbol'].isin(case_insensitive_symbols)].copy()
        logger.info(f"Case-insensitive filtered data shape: {filtered_df.shape}")
    
    return filtered_df

def extract_features(df):
    """Extract features from the dataframe"""
    if df is None:
        return None, None
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in ['date', 'symbol']]
    logger.info(f"Extracted {len(feature_cols)} feature columns")
    
    # Get feature types
    pvm_cols = [col for col in feature_cols if col.startswith('pvm_')]
    sentiment_cols = [col for col in feature_cols if col.startswith('sentiment_')]
    onchain_cols = [col for col in feature_cols if col.startswith('onchain_')]
    
    logger.info(f"Feature breakdown: {len(pvm_cols)} PVM, {len(sentiment_cols)} sentiment, {len(onchain_cols)} onchain")
    
    # Create features dataframe
    X = df[feature_cols].copy()
    y = None  # No target in this data
    
    return X, y

def clean_features(X):
    """Clean and prepare features for modeling"""
    if X is None:
        return None
    
    logger.info("Cleaning features")
    
    # Replace infinities with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Count NaNs
    nan_counts = X.isna().sum()
    nan_percent = (nan_counts / len(X)) * 100
    logger.info(f"Features with >50% NaNs: {sum(nan_percent > 50)}")
    
    # Strategy for NaNs: Fill with median
    X.fillna(X.median(), inplace=True)
    
    # Remove constant features
    logger.info("Removing constant features")
    var_thresh = VarianceThreshold(threshold=0.0)
    X_var = var_thresh.fit_transform(X)
    
    # Get feature names after variance threshold
    remaining_cols = X.columns[var_thresh.get_support()]
    X = X[remaining_cols].copy()
    
    logger.info(f"Features after removing constants: {X.shape[1]}")
    
    return X

def feature_engineering(X, df):
    """Perform feature engineering"""
    if X is None or df is None:
        return None
    
    logger.info("Performing feature engineering")
    
    # Store symbol for later joining
    symbol = df['symbol'].reset_index(drop=True)
    
    # Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to dataframe
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Add symbol back
    X_scaled_df['symbol'] = symbol
    
    # Reduce dimensionality with PCA
    if X.shape[1] > 50:
        logger.info("Reducing dimensionality with PCA")
        pca = PCA(n_components=min(50, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA feature names
        pca_cols = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
        
        # Convert to dataframe
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        
        # Add symbol back
        X_pca_df['symbol'] = symbol
        
        # Calculate explained variance
        explained_var = sum(pca.explained_variance_ratio_) * 100
        logger.info(f"PCA with {X_pca.shape[1]} components explains {explained_var:.2f}% of variance")
        
        return X_pca_df
    else:
        return X_scaled_df

def save_processed_data(df, output_file=None):
    """Save processed data to file"""
    if df is None:
        logger.error("No data to save")
        return None
    
    if output_file is None:
        # Create output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(PROCESSED_DIR, f"processed_yiedl_{timestamp}.parquet")
    
    logger.info(f"Saving processed data to {output_file}")
    
    try:
        df.to_parquet(output_file)
        logger.info(f"Saved processed data with shape {df.shape}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return None

def main():
    """Main processing function"""
    logger.info("Starting Yiedl data processing")
    
    # Create directories
    ensure_directories()
    
    # Load Numerai symbols
    numerai_symbols = load_numerai_symbols()
    
    if not numerai_symbols:
        logger.error("No Numerai symbols found, downloading from API")
        try:
            import numerapi
            napi = numerapi.NumerAPI()
            filename = "/tmp/crypto_live_universe.parquet"
            napi.download_dataset("crypto/v1.0/live_universe.parquet", filename)
            numerai_symbols = load_numerai_symbols(filename)
        except Exception as e:
            logger.error(f"Error downloading Numerai symbols: {e}")
            return
    
    # Load Yiedl data
    yiedl_df = load_yiedl_data()
    
    if yiedl_df is None:
        logger.error("Failed to load Yiedl data")
        return
    
    # Filter to latest date
    latest_df = filter_latest_data(yiedl_df)
    
    # Filter by symbols
    filtered_df = filter_by_symbols(latest_df, numerai_symbols)
    
    # Extract features
    X, _ = extract_features(filtered_df)
    
    # Clean features
    X_clean = clean_features(X)
    
    # Feature engineering
    processed_df = feature_engineering(X_clean, filtered_df)
    
    # Save processed data
    output_file = save_processed_data(processed_df)
    
    if output_file:
        logger.info(f"Processing complete. Results saved to {output_file}")
    else:
        logger.error("Processing failed")

if __name__ == "__main__":
    main()