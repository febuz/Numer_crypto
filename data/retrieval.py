"""
Data retrieval functions for the Numerai Crypto project.
"""
from numerapi import NumerAPI
import polars as pl
from pyspark.sql import SparkSession
import os
import time
from numer_crypto.config.settings import NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY, DATA_DIR, SPARK_CONFIG
from numer_crypto.utils.data_utils import ensure_data_dir


class NumeraiDataRetriever:
    """
    Class for retrieving data from the Numerai API.
    """
    
    def __init__(self, public_id=None, secret_key=None, use_spark=False):
        """
        Initialize the NumeraiDataRetriever.
        
        Args:
            public_id (str): Numerai API public ID
            secret_key (str): Numerai API secret key
            use_spark (bool): Whether to use Spark for data processing
        """
        self.public_id = public_id or NUMERAI_PUBLIC_ID
        self.secret_key = secret_key or NUMERAI_SECRET_KEY
        self.napi = NumerAPI(self.public_id, self.secret_key)
        self.data_dir = ensure_data_dir()
        self.use_spark = use_spark
        self.spark = None
        
        # Initialize Spark if needed
        if self.use_spark:
            self.spark = SparkSession.builder \
                .appName(SPARK_CONFIG.get('app_name', 'NumeraiSpark')) \
                .config("spark.executor.memory", SPARK_CONFIG.get('executor_memory', '4g')) \
                .config("spark.driver.memory", SPARK_CONFIG.get('driver_memory', '4g')) \
                .config("spark.executor.cores", SPARK_CONFIG.get('executor_cores', '2')) \
                .getOrCreate()
        
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
            DataFrame: The loaded dataset (Polars DataFrame or Spark DataFrame depending on use_spark)
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
        
        if self.use_spark:
            # Use Spark to read parquet
            df = self.spark.read.parquet(file_path)
            print(f"Loaded {dataset_type} dataset with shape ({df.count()}, {len(df.columns)})")
        else:
            # Use Polars to read parquet
            df = pl.read_parquet(file_path)
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
            predictions_df (DataFrame): DataFrame with predictions (Polars, Spark, or pandas DataFrame)
            tournament (str): Tournament name
            
        Returns:
            dict: Submission result
        """
        # Verify dataframe format and convert if necessary
        required_columns = ['id', 'prediction']
        
        if self.use_spark:
            # Handle Spark DataFrame
            if not all(col in predictions_df.columns for col in required_columns):
                raise ValueError(f"Missing required column in predictions DataFrame: {col}")
            
            # Convert Spark DataFrame to local pandas or polars for submission
            csv_path = os.path.join(self.data_dir, f"{tournament}_predictions.csv")
            predictions_df.select("id", "prediction").write.option("header", "true").csv(csv_path, mode="overwrite")
            
            # Find the actual CSV file (Spark creates a directory with part files)
            import glob
            part_files = glob.glob(os.path.join(csv_path, "part-*.csv"))
            if part_files:
                # Rename the first part file to be the actual predictions file
                actual_csv_path = os.path.join(self.data_dir, f"{tournament}_predictions_actual.csv")
                os.rename(part_files[0], actual_csv_path)
            else:
                raise FileNotFoundError("No part files found in Spark CSV output")
        else:
            # Handle Polars DataFrame
            if isinstance(predictions_df, pl.DataFrame):
                if not all(col in predictions_df.columns for col in required_columns):
                    raise ValueError(f"Missing required column in predictions DataFrame")
                
                # Save predictions to CSV using Polars
                csv_path = os.path.join(self.data_dir, f"{tournament}_predictions.csv")
                predictions_df.write_csv(csv_path, include_header=True)
                actual_csv_path = csv_path
            else:
                # Handle other DataFrame types (e.g., pandas as fallback)
                if not all(col in predictions_df.columns for col in required_columns):
                    raise ValueError(f"Missing required column in predictions DataFrame")
                
                # Save predictions to CSV
                csv_path = os.path.join(self.data_dir, f"{tournament}_predictions.csv")
                try:
                    # Try to use to_csv for pandas compatibility
                    predictions_df.to_csv(csv_path, index=False)
                except AttributeError:
                    raise TypeError("Unsupported DataFrame type. Please use Polars, Spark, or pandas DataFrame")
                actual_csv_path = csv_path
        
        # Submit predictions
        submission_id = self.napi.upload_predictions(
            file_path=actual_csv_path,
            tournament=tournament
        )
        
        print(f"Submitted predictions with ID: {submission_id}")
        
        return {"submission_id": submission_id}