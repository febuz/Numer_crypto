#!/usr/bin/env python3
"""
Process Yiedl data for Numerai Crypto.

This script fetches cryptocurrency market data and prepares it for model training.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from data.retrieval import NumeraiDataRetriever
from utils.log_utils import setup_logging

# Set up logging to external directory
logger = setup_logging(name=__name__, level=logging.INFO)

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    logger.info(f"Data directories created/verified")

def process_data():
    """Process the Yiedl data for Numerai Crypto."""
    logger.info("Starting Yiedl data processing")
    
    # Initialize data retriever
    data_retriever = NumeraiDataRetriever()
    
    try:
        # Download current datasets
        datasets = data_retriever.download_current_dataset()
        logger.info(f"Downloaded datasets for round {datasets.get('round')}")
        
        # Process training data
        train_data = data_retriever.load_dataset('training')
        logger.info(f"Loaded training data with shape {getattr(train_data, 'shape', '')}")
        
        # Process validation data  
        validation_data = data_retriever.load_dataset('validation')
        logger.info(f"Loaded validation data with shape {getattr(validation_data, 'shape', '')}")
        
        # Process tournament data
        tournament_data = data_retriever.load_dataset('tournament')
        logger.info(f"Loaded tournament data with shape {getattr(tournament_data, 'shape', '')}")
        
        # Get feature metadata
        features = data_retriever.load_features()
        logger.info(f"Loaded feature metadata with {len(features.get('feature_names', []))} features")
        
        # Save timestamp for processed files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Simple transformations to ensure data is ready for modeling
        try:
            if hasattr(train_data, 'fill_null'):
                # Handle Polars DataFrame
                logger.info("Processing using Polars")
                # Polars doesn't have inplace, so reassign
                train_data = train_data.fill_null(0)
                validation_data = validation_data.fill_null(0)
                tournament_data = tournament_data.fill_null(0)
            elif hasattr(train_data, 'fillna'):
                # Handle pandas DataFrame
                logger.info("Processing using pandas")
                train_data.fillna(0, inplace=True)
                validation_data.fillna(0, inplace=True)
                tournament_data.fillna(0, inplace=True)
            else:
                # Unknown DataFrame type, try different approaches
                logger.info("Unknown DataFrame type, trying different approaches")
                
                # Method 1: Try converting to pandas
                try:
                    import pandas as pd
                    if hasattr(train_data, 'to_pandas'):
                        train_data = train_data.to_pandas()
                        validation_data = validation_data.to_pandas()
                        tournament_data = tournament_data.to_pandas()
                        
                        train_data.fillna(0, inplace=True)
                        validation_data.fillna(0, inplace=True)
                        tournament_data.fillna(0, inplace=True)
                    else:
                        # Method 2: Just leave as is, assume no nulls in sample data
                        pass
                except ImportError:
                    # Method 3: Just leave as is, assume no nulls in sample data
                    pass
        except Exception as e:
            logger.warning(f"Error filling nulls: {e}")
            logger.info("Continuing without null handling")
        
        # Save processed data
        try:
            # Try using parquet format first
            processed_train_path = os.path.join(PROCESSED_DATA_DIR, f"train_processed_{timestamp}.parquet")
            processed_val_path = os.path.join(PROCESSED_DATA_DIR, f"validation_processed_{timestamp}.parquet")
            processed_tournament_path = os.path.join(PROCESSED_DATA_DIR, f"tournament_processed_{timestamp}.parquet")
            
            try:
                if hasattr(train_data, 'write_parquet'):
                    # Handle Polars DataFrame
                    train_data.write_parquet(processed_train_path)
                    validation_data.write_parquet(processed_val_path)
                    tournament_data.write_parquet(processed_tournament_path)
                elif hasattr(train_data, 'to_parquet'):
                    # Handle Pandas DataFrame
                    train_data.to_parquet(processed_train_path)
                    validation_data.to_parquet(processed_val_path)
                    tournament_data.to_parquet(processed_tournament_path)
                else:
                    # Unknown DataFrame type, fallback to CSV
                    raise ValueError("Unknown DataFrame type, falling back to CSV")
            except Exception as e:
                logger.warning(f"Parquet save failed: {e}. Falling back to CSV format")
                # Use CSV as a fallback
                processed_train_path = os.path.join(PROCESSED_DATA_DIR, f"train_processed_{timestamp}.csv")
                processed_val_path = os.path.join(PROCESSED_DATA_DIR, f"validation_processed_{timestamp}.csv")
                processed_tournament_path = os.path.join(PROCESSED_DATA_DIR, f"tournament_processed_{timestamp}.csv")
                
                if hasattr(train_data, 'write_csv'):
                    # Handle Polars DataFrame
                    train_data.write_csv(processed_train_path)
                    validation_data.write_csv(processed_val_path)
                    tournament_data.write_csv(processed_tournament_path)
                elif hasattr(train_data, 'to_csv'):
                    # Handle Pandas DataFrame
                    train_data.to_csv(processed_train_path, index=False)
                    validation_data.to_csv(processed_val_path, index=False)
                    tournament_data.to_csv(processed_tournament_path, index=False)
                else:
                    # Last resort: save as numpy arrays
                    import numpy as np
                    np.save(processed_train_path.replace('.csv', '.npy'), train_data)
                    np.save(processed_val_path.replace('.csv', '.npy'), validation_data)
                    np.save(processed_tournament_path.replace('.csv', '.npy'), tournament_data)
                    
                    # Update paths
                    processed_train_path = processed_train_path.replace('.csv', '.npy')
                    processed_val_path = processed_val_path.replace('.csv', '.npy')
                    processed_tournament_path = processed_tournament_path.replace('.csv', '.npy')
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
        
        logger.info(f"Saved processed data to {PROCESSED_DATA_DIR}")
        logger.info("Data processing completed successfully")
        
        return {
            'train': processed_train_path,
            'validation': processed_val_path,
            'tournament': processed_tournament_path,
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise

def main():
    """Main function to run the data processing pipeline."""
    try:
        ensure_directories()
        results = process_data()
        logger.info(f"Process completed successfully. Files created at {results['timestamp']}")
        return 0
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())