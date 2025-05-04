"""
Data retrieval functions for the Numerai Crypto project.
"""
from numerapi import NumerAPI
import pandas as pd
import os
import time
from numer_crypto.config.settings import NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY, DATA_DIR
from numer_crypto.utils.data_utils import ensure_data_dir


class NumeraiDataRetriever:
    """
    Class for retrieving data from the Numerai API.
    """
    
    def __init__(self, public_id=None, secret_key=None):
        """
        Initialize the NumeraiDataRetriever.
        
        Args:
            public_id (str): Numerai API public ID
            secret_key (str): Numerai API secret key
        """
        self.public_id = public_id or NUMERAI_PUBLIC_ID
        self.secret_key = secret_key or NUMERAI_SECRET_KEY
        self.napi = NumerAPI(self.public_id, self.secret_key)
        self.data_dir = ensure_data_dir()
        
    def download_current_dataset(self, tournament='crypto'):
        """
        Download the current dataset for a tournament.
        
        Args:
            tournament (str): Tournament name
            
        Returns:
            tuple: Paths to the downloaded files
        """
        print(f"Downloading {tournament} dataset...")
        
        # Get latest round
        current_round = self.napi.get_current_round(tournament=tournament)
        print(f"Current round: {current_round}")
        
        # Download datasets
        training_data_path = self.napi.download_dataset(
            filename="train_data.parquet",
            dest_path=self.data_dir
        )
        
        validation_data_path = self.napi.download_dataset(
            filename="validation_data.parquet",
            dest_path=self.data_dir
        )
        
        tournament_data_path = self.napi.download_dataset(
            filename="tournament_data.parquet",
            dest_path=self.data_dir
        )
        
        features_path = self.napi.download_dataset(
            filename="features.json",
            dest_path=self.data_dir
        )
        
        return {
            'training': training_data_path,
            'validation': validation_data_path,
            'tournament': tournament_data_path,
            'features': features_path,
            'round': current_round
        }
    
    def load_dataset(self, dataset_type='training'):
        """
        Load a previously downloaded dataset.
        
        Args:
            dataset_type (str): Type of dataset ('training', 'validation', 'tournament')
            
        Returns:
            DataFrame: The loaded dataset
        """
        filename_map = {
            'training': 'train_data.parquet',
            'validation': 'validation_data.parquet',
            'tournament': 'tournament_data.parquet',
        }
        
        if dataset_type not in filename_map:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
            
        file_path = os.path.join(self.data_dir, filename_map[dataset_type])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}. Please download it first.")
            
        df = pd.read_parquet(file_path)
        print(f"Loaded {dataset_type} dataset with shape {df.shape}")
        
        return df
    
    def load_features(self):
        """
        Load the features.json file with feature metadata.
        
        Returns:
            dict: Feature metadata
        """
        import json
        
        features_path = os.path.join(self.data_dir, 'features.json')
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}. Please download it first.")
            
        with open(features_path, 'r') as f:
            features = json.load(f)
            
        return features
    
    def get_feature_names(self):
        """
        Get the feature names from the features.json file.
        
        Returns:
            list: Feature names
        """
        features = self.load_features()
        return features.get('feature_names', [])
    
    def get_feature_groups(self):
        """
        Get the feature groups from the features.json file.
        
        Returns:
            dict: Feature groups
        """
        features = self.load_features()
        return features.get('feature_groups', {})
    
    def submit_predictions(self, predictions_df, tournament='crypto'):
        """
        Submit predictions to Numerai.
        
        Args:
            predictions_df (DataFrame): DataFrame with predictions
            tournament (str): Tournament name
            
        Returns:
            dict: Submission result
        """
        # Verify dataframe format
        required_columns = ['id', 'prediction']
        for col in required_columns:
            if col not in predictions_df.columns:
                raise ValueError(f"Missing required column in predictions DataFrame: {col}")
        
        # Save predictions to CSV
        csv_path = os.path.join(self.data_dir, f"{tournament}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)
        
        # Submit predictions
        submission_id = self.napi.upload_predictions(
            file_path=csv_path,
            tournament=tournament
        )
        
        print(f"Submitted predictions with ID: {submission_id}")
        
        return {"submission_id": submission_id}