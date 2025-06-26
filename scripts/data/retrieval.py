"""
Data retrieval and preparation for Numerai Crypto predictions.

This module handles downloading and preparing data from Numerai and Yiedl,
focusing on the cryptocurrencies that are eligible for Numerai predictions.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from utils.data.download_numerai import download_numerai_crypto_data
from utils.data.download_yiedl import download_yiedl_data
from utils.data.load_numerai import load_numerai_data, get_eligible_crypto_symbols
from utils.data.load_yiedl import load_yiedl_data, get_yiedl_crypto_symbols
from utils.data.create_merged_dataset import create_merged_datasets, get_overlapping_symbols
from utils.data.report_merge_summary import report_data_summary

class NumeraiDataRetriever:
    """
    Class for retrieving and preparing data for Numerai Crypto predictions.
    
    This class downloads data from Numerai and Yiedl, identifies the overlap
    in cryptocurrency symbols, and prepares merged datasets for model training
    and prediction.
    """
    
    def __init__(self, tournament="crypto", base_dir=None, api_key=None, api_secret=None):
        """
        Initialize the data retriever.
        
        Args:
            tournament: Tournament name (default: crypto)
            base_dir: Base directory for data storage
            api_key: Optional Numerai API key
            api_secret: Optional Numerai API secret
        """
        self.tournament = tournament
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Set up base directory
        if base_dir is None:
            base_dir = os.path.join(os.path.expanduser("~"), "numer_crypto_data")
        self.base_dir = base_dir
        
        # Set up data directories
        self.numerai_dir = os.path.join(self.base_dir, "numerai")
        self.yiedl_dir = os.path.join(self.base_dir, "yiedl")
        self.merged_dir = os.path.join(self.base_dir, "merged")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        
        # Create necessary directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.numerai_dir, exist_ok=True)
        os.makedirs(self.yiedl_dir, exist_ok=True)
        os.makedirs(self.merged_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize data containers
        self.numerai_files = {}
        self.yiedl_files = {}
        self.numerai_data = {}
        self.yiedl_data = {}
        self.merged_data = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def download_current_datasets(self):
        """
        Download the current datasets from Numerai and Yiedl.
        
        Returns:
            dict: Information about downloaded files
        """
        self.logger.info("Downloading current datasets")
        
        # Download Numerai data
        self.numerai_files = download_numerai_crypto_data(
            self.numerai_dir, 
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        # Download Yiedl data
        self.yiedl_files = download_yiedl_data(self.yiedl_dir)
        
        return {
            'numerai': self.numerai_files,
            'yiedl': self.yiedl_files
        }
    
    def load_datasets(self):
        """
        Load datasets from downloaded files.
        
        Returns:
            dict: Loaded datasets
        """
        self.logger.info("Loading datasets")
        
        # Load Numerai data
        self.numerai_data = load_numerai_data(self.numerai_files)
        
        # Load Yiedl data
        self.yiedl_data = load_yiedl_data(self.yiedl_files)
        
        return {
            'numerai': self.numerai_data,
            'yiedl': self.yiedl_data
        }
    
    def prepare_merged_datasets(self):
        """
        Create merged datasets from loaded data.
        
        Returns:
            dict: Information about merged datasets
        """
        self.logger.info("Preparing merged datasets")
        
        # Create merged datasets
        self.merged_data = create_merged_datasets(
            self.numerai_data,
            self.yiedl_data,
            self.merged_dir,
            current_round=self.numerai_data.get('current_round')
        )
        
        # Generate summary report
        report_data_summary(
            self.numerai_data,
            self.yiedl_data,
            self.merged_data,
            output_dir=self.reports_dir
        )
        
        return self.merged_data
    
    def get_eligible_symbols(self):
        """
        Get list of eligible cryptocurrency symbols for Numerai predictions.
        
        Returns:
            list: Eligible cryptocurrency symbols
        """
        # Get symbols from Numerai data
        numerai_symbols = get_eligible_crypto_symbols(self.numerai_data)
        
        # Get symbols from Yiedl data
        yiedl_symbols = get_yiedl_crypto_symbols(self.yiedl_data)
        
        # Get overlapping symbols
        overlapping_symbols = get_overlapping_symbols(self.numerai_data, self.yiedl_data)
        
        # Store in merged_data for future reference
        self.merged_data['numerai_symbols'] = numerai_symbols
        self.merged_data['yiedl_symbols'] = yiedl_symbols
        self.merged_data['overlapping_symbols'] = overlapping_symbols
        
        # Return overlapping symbols as eligible
        return overlapping_symbols
    
    def get_training_data(self):
        """
        Get training data for model training.
        
        Returns:
            DataFrame: Training data
        """
        # Check if we already have merged data
        if 'train' in self.merged_data:
            return self.merged_data['train']
        
        # If we have a file path, load it
        if 'train_file' in self.merged_data and os.path.exists(self.merged_data['train_file']):
            import pandas as pd
            self.merged_data['train'] = pd.read_parquet(self.merged_data['train_file'])
            return self.merged_data['train']
        
        # Otherwise, prepare data
        if not self.numerai_files:
            self.download_current_datasets()
        
        if not self.numerai_data:
            self.load_datasets()
        
        self.prepare_merged_datasets()
        
        if 'train' in self.merged_data:
            return self.merged_data['train']
        
        raise ValueError("Failed to retrieve training data")
    
    def get_validation_data(self):
        """
        Get validation data for model evaluation.
        
        Returns:
            DataFrame: Validation data
        """
        # For simplicity, we're using the same training data but will split it
        train_data = self.get_training_data()
        
        # In a real implementation, you would have a proper validation set
        # For now, we'll just return the training data
        return train_data
    
    def get_tournament_data(self):
        """
        Get tournament data for predictions.
        
        Returns:
            DataFrame: Tournament data
        """
        # Check if we already have merged data
        if 'live' in self.merged_data:
            return self.merged_data['live']
        
        # If we have a file path, load it
        if 'live_file' in self.merged_data and os.path.exists(self.merged_data['live_file']):
            import pandas as pd
            self.merged_data['live'] = pd.read_parquet(self.merged_data['live_file'])
            return self.merged_data['live']
        
        # Otherwise, prepare data
        if not self.numerai_files:
            self.download_current_datasets()
        
        if not self.numerai_data:
            self.load_datasets()
        
        self.prepare_merged_datasets()
        
        if 'live' in self.merged_data:
            return self.merged_data['live']
        
        raise ValueError("Failed to retrieve tournament data")
    
    def submit_predictions(self, file_path=None, predictions_df=None, submission_id=None):
        """
        Submit predictions to Numerai.
        
        Args:
            file_path: Path to prediction file (CSV)
            predictions_df: DataFrame with predictions
            submission_id: Optional submission ID
            
        Returns:
            dict: Submission result
        """
        self.logger.info("Submitting predictions to Numerai")
        
        try:
            import numerapi
            napi = numerapi.NumerAPI(self.api_key, self.api_secret)
            
            if file_path and os.path.exists(file_path):
                submission_id = napi.upload_predictions(file_path, tournament=self.tournament)
                return {"submission_id": submission_id}
            elif predictions_df is not None:
                import pandas as pd
                import tempfile
                
                # Ensure correct column names for Numerai submission
                if 'Symbol' in predictions_df.columns and 'Prediction' in predictions_df.columns:
                    # Already in correct format
                    pass
                elif 'id' in predictions_df.columns and 'prediction' in predictions_df.columns:
                    # Convert to Symbol, Prediction format
                    predictions_df = pd.DataFrame({
                        'Symbol': predictions_df['id'].str.split('_').str[0],
                        'Prediction': predictions_df['prediction']
                    })
                else:
                    raise ValueError("Predictions must have either ['Symbol', 'Prediction'] or ['id', 'prediction'] columns")
                
                # Create a temporary file for submission
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                    file_path = temp_file.name
                    predictions_df.to_csv(file_path, index=False)
                
                # Submit predictions
                submission_id = napi.upload_predictions(file_path, tournament=self.tournament)
                
                # Clean up the temporary file
                os.unlink(file_path)
                
                return {"submission_id": submission_id}
            else:
                raise ValueError("Either file_path or predictions_df must be provided")
                
        except ImportError:
            self.logger.error("numerapi package not installed")
            return {"error": "numerapi package not installed"}
        except Exception as e:
            self.logger.error(f"Error submitting predictions: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Initialize data retriever
    retriever = NumeraiDataRetriever()
    
    # Download data
    retriever.download_current_datasets()
    
    # Load data
    retriever.load_datasets()
    
    # Prepare merged datasets
    retriever.prepare_merged_datasets()
    
    # Get eligible symbols
    symbols = retriever.get_eligible_symbols()
    
    # Print first 10 symbols
    print(f"First 10 eligible symbols: {symbols[:10]}")