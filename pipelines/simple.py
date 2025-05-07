"""
Simple pipeline for quick submissions in Numerai Crypto competition.
This implementation focuses on speed, with a runtime target of 15-30 minutes.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
    SUBMISSIONS_DIR, CHECKPOINTS_DIR, HARDWARE_CONFIG
)
from data.retrieval import NumeraiDataRetriever
from models.lightgbm_model import LightGBMModel
from utils.data import save_dataframe, load_dataframe
from utils.gpu import setup_gpu_training
from utils.log_utils import setup_logging

# Configure logging
logger = setup_logging(name=__name__, level=logging.INFO)
logger = logging.getLogger('SimplePipeline')

class SimplePipeline:
    """
    Implementation of a simple pipeline for quick Numerai Crypto submissions.
    Targets 15-30 minute runtime with RMSE < 0.020.
    """
    
    def __init__(self, tournament="crypto", time_budget_minutes=30):
        """
        Initialize the simple pipeline.
        
        Args:
            tournament: Tournament name (default: crypto)
            time_budget_minutes: Maximum time budget in minutes
        """
        self.tournament = tournament
        self.time_budget_seconds = time_budget_minutes * 60
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_retriever = NumeraiDataRetriever(tournament=tournament)
        
        # Ensure directories exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # Setup GPU
        self.gpu_id = 0  # Use first GPU for simple pipeline
        setup_gpu_training(self.gpu_id)
        
        logger.info(f"Initialized Simple Pipeline with GPU {self.gpu_id} and time budget of {time_budget_minutes} minutes")
    
    def time_remaining(self):
        """Check how much time is left in the budget"""
        elapsed = time.time() - self.start_time
        return max(0, self.time_budget_seconds - elapsed)
    
    def create_basic_features(self, df):
        """
        Create a small set of basic features quickly.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with basic features
        """
        # Start with all features
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        df_features = df[feature_cols + ['target', 'id']].copy()
        
        # Basic feature groups (10 groups of 5 features)
        n_groups = min(10, len(feature_cols) // 5)
        for i in range(n_groups):
            group_cols = feature_cols[i*5:(i+1)*5]
            if len(group_cols) > 1:
                # Create mean and std features for each group
                df_features[f'group_{i}_mean'] = df_features[group_cols].mean(axis=1)
                df_features[f'group_{i}_std'] = df_features[group_cols].std(axis=1)
                
                # Create simple ratios between first and other features in group
                base_col = group_cols[0]
                for col in group_cols[1:]:
                    # Avoid division by zero
                    df_features[f'ratio_{base_col}_{col}'] = df_features[base_col] / (df_features[col] + 1e-8)
        
        # Create 5 simple rolling statistics (fast to compute)
        windows = [3, 5]
        for window in windows:
            for col in feature_cols[:5]:  # Only use first 5 features for speed
                df_features[f'{col}_rolling_mean_{window}'] = df_features[col].rolling(window=window, min_periods=1).mean()
                df_features[f'{col}_rolling_std_{window}'] = df_features[col].rolling(window=window, min_periods=1).std()
        
        # Fill any NaN values
        df_features = df_features.fillna(0)
        
        logger.info(f"Created {df_features.shape[1] - len(feature_cols) - 2} new features")
        return df_features
    
    def run(self):
        """
        Execute the complete simple pipeline.
        
        Returns:
            dict: Results with metrics and paths to output files
        """
        results = {}
        
        try:
            # Step 1: Data Retrieval (20% of time budget)
            step_time_budget = self.time_budget_seconds * 0.2
            logger.info(f"Starting data retrieval with {step_time_budget/60:.1f} minute budget")
            
            step_start = time.time()
            train_data = self.data_retriever.get_training_data()
            validation_data = self.data_retriever.get_validation_data()
            tournament_data = self.data_retriever.get_tournament_data()
            
            # Save raw data
            train_path = os.path.join(RAW_DATA_DIR, f"train_simple_{self.timestamp}.parquet")
            val_path = os.path.join(RAW_DATA_DIR, f"validation_simple_{self.timestamp}.parquet")
            tournament_path = os.path.join(RAW_DATA_DIR, f"tournament_simple_{self.timestamp}.parquet")
            
            save_dataframe(train_data, train_path)
            save_dataframe(validation_data, val_path)
            save_dataframe(tournament_data, tournament_path)
            
            step_duration = time.time() - step_start
            logger.info(f"Data retrieval completed in {step_duration/60:.1f} minutes")
            results['data_paths'] = {'train': train_path, 'validation': val_path, 'tournament': tournament_path}
            
            # Step 2: Basic Feature Creation (30% of time budget)
            step_time_budget = self.time_budget_seconds * 0.3
            logger.info(f"Starting basic feature creation with {step_time_budget/60:.1f} minute budget")
            
            step_start = time.time()
            train_features = self.create_basic_features(train_data)
            validation_features = self.create_basic_features(validation_data)
            tournament_features = self.create_basic_features(tournament_data)
            
            # Save feature data
            train_features_path = os.path.join(FEATURES_DIR, f"train_features_simple_{self.timestamp}.parquet")
            val_features_path = os.path.join(FEATURES_DIR, f"validation_features_simple_{self.timestamp}.parquet")
            tournament_features_path = os.path.join(FEATURES_DIR, f"tournament_features_simple_{self.timestamp}.parquet")
            
            save_dataframe(train_features, train_features_path)
            save_dataframe(validation_features, val_features_path)
            save_dataframe(tournament_features, tournament_features_path)
            
            step_duration = time.time() - step_start
            logger.info(f"Feature creation completed in {step_duration/60:.1f} minutes, generated {train_features.shape[1]} total columns")
            results['feature_paths'] = {'train': train_features_path, 'validation': val_features_path, 'tournament': tournament_features_path}
            
            # Step 3: Model Training (30% of time budget)
            step_time_budget = self.time_budget_seconds * 0.3
            logger.info(f"Starting model training with {step_time_budget/60:.1f} minute budget")
            
            # Prepare data
            target_col = 'target'
            feature_cols = [col for col in train_features.columns if col not in ['id', 'target']]
            
            X_train = train_features[feature_cols]
            y_train = train_features[target_col]
            X_val = validation_features[feature_cols]
            y_val = validation_features[target_col]
            X_tournament = tournament_features[feature_cols]
            
            # Train a single LightGBM model (faster than XGBoost)
            step_start = time.time()
            
            # Use simplified parameters for faster training
            lightgbm_model = LightGBMModel(
                name=f"lightgbm_simple_{self.timestamp}", 
                gpu_id=self.gpu_id,
                params={
                    'num_leaves': 63,              # Fewer leaves
                    'learning_rate': 0.1,          # Higher learning rate
                    'max_depth': 6,                # Shallower trees
                    'min_data_in_leaf': 20,        # Smaller leaf size
                    'feature_fraction': 0.8,       # Use 80% of features
                    'bagging_fraction': 0.8,       # Use 80% of data
                    'bagging_freq': 1,             # More frequent bagging
                    'device_type': 'gpu',          # Use GPU
                    'max_bin': 63,                 # Fewer bins
                    'verbosity': -1,               # Less output
                    'boosting_type': 'gbdt',       # Standard gradient boosting
                    'objective': 'regression',     # For regression tasks
                    'metric': 'rmse',              # Evaluation metric
                    'n_estimators': 200,           # Fewer trees
                }
            )
            
            # Train model with validation
            lightgbm_model.train(X_train, y_train, X_val, y_val)
            lightgbm_model.save(CHECKPOINTS_DIR)
            
            # Predict on validation data
            val_pred = lightgbm_model.predict(X_val)
            val_rmse = np.sqrt(np.mean((val_pred - y_val)**2))
            logger.info(f"Validation RMSE: {val_rmse:.6f}")
            
            step_duration = time.time() - step_start
            logger.info(f"Model training completed in {step_duration/60:.1f} minutes")
            
            results['validation_rmse'] = val_rmse
            
            # Step 4: Generate Predictions (20% of time budget)
            step_time_budget = self.time_budget_seconds * 0.2
            logger.info(f"Generating final predictions with {step_time_budget/60:.1f} minute budget")
            
            step_start = time.time()
            tournament_ids = tournament_data['id'].values
            
            # Generate predictions
            tournament_pred = lightgbm_model.predict(X_tournament)
            
            # Create submission dataframe
            submission = pd.DataFrame({'id': tournament_ids, 'prediction': tournament_pred})
            
            # Save submission
            submission_path = os.path.join(SUBMISSIONS_DIR, f"lightgbm_simple_submission_{self.timestamp}.csv")
            submission.to_csv(submission_path, index=False)
            
            step_duration = time.time() - step_start
            logger.info(f"Prediction generation completed in {step_duration/60:.1f} minutes")
            
            results['submission_path'] = submission_path
            
            # Create summary report
            total_duration = time.time() - self.start_time
            logger.info(f"Simple pipeline completed in {total_duration/60:.2f} minutes")
            logger.info(f"Validation RMSE: {val_rmse:.6f}")
            
            results['success'] = True
            results['duration_minutes'] = total_duration / 60
            results['rmse'] = val_rmse
            
            return results
            
        except Exception as e:
            logger.error(f"Error in simple pipeline: {str(e)}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
            return results

# For direct execution
if __name__ == "__main__":
    pipeline = SimplePipeline()
    results = pipeline.run()
    print(f"Pipeline complete with RMSE: {results.get('rmse', 'N/A')}")
    if results.get('success', False):
        print(f"Submission path: {results['submission_path']}")
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")