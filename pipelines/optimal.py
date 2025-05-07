"""
Optimal pipeline for lowest RMSE in Numerai Crypto competition.
This implementation focuses on high memory feature engineering and GPU-accelerated models.
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
from features.high_memory import HighMemoryFeatureGenerator
from features.selector import FeatureSelector
from models.lightgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.ensemble import ensemble_predictions
from utils.data import save_dataframe, load_dataframe
from utils.gpu import setup_gpu_training, monitor_gpu_usage
from utils.log_utils import setup_logging

# Configure logging
logger = setup_logging(name=__name__, level=logging.INFO)
logger = logging.getLogger('OptimalPipeline')

class OptimalPipeline:
    """
    Implementation of the optimal pipeline for achieving lowest RMSE on Numerai Crypto.
    Uses high-memory (600GB) feature engineering and GPU acceleration.
    """
    
    def __init__(self, tournament="crypto", time_budget_hours=8):
        """
        Initialize the optimal pipeline.
        
        Args:
            tournament: Tournament name (default: crypto)
            time_budget_hours: Maximum time budget in hours
        """
        self.tournament = tournament
        self.time_budget_seconds = time_budget_hours * 3600
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_retriever = NumeraiDataRetriever(tournament=tournament)
        
        # Ensure directories exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # Setup GPU tracking
        self.num_gpus = HARDWARE_CONFIG['gpu_count']
        self.gpu_ids = list(range(self.num_gpus))
        setup_gpu_training(self.gpu_ids[0])  # Use first GPU for initial setup
        
        logger.info(f"Initialized Optimal Pipeline with {self.num_gpus} GPUs and time budget of {time_budget_hours} hours")
    
    def time_remaining(self):
        """Check how much time is left in the budget"""
        elapsed = time.time() - self.start_time
        return max(0, self.time_budget_seconds - elapsed)
    
    def run(self):
        """
        Execute the complete optimal pipeline.
        
        Returns:
            dict: Results with metrics and paths to output files
        """
        results = {}
        
        try:
            # Step 1: Data Retrieval (15% of time budget)
            step_time_budget = self.time_budget_seconds * 0.15
            logger.info(f"Starting data retrieval with {step_time_budget/60:.1f} minute budget")
            
            step_start = time.time()
            train_data = self.data_retriever.get_training_data()
            validation_data = self.data_retriever.get_validation_data()
            tournament_data = self.data_retriever.get_tournament_data()
            
            # Save raw data
            train_path = os.path.join(RAW_DATA_DIR, f"train_{self.timestamp}.parquet")
            val_path = os.path.join(RAW_DATA_DIR, f"validation_{self.timestamp}.parquet")
            tournament_path = os.path.join(RAW_DATA_DIR, f"tournament_{self.timestamp}.parquet")
            
            save_dataframe(train_data, train_path)
            save_dataframe(validation_data, val_path)
            save_dataframe(tournament_data, tournament_path)
            
            step_duration = time.time() - step_start
            logger.info(f"Data retrieval completed in {step_duration/60:.1f} minutes")
            results['data_paths'] = {'train': train_path, 'validation': val_path, 'tournament': tournament_path}
            
            # Step 2: Feature Engineering (40% of time budget)
            step_time_budget = self.time_budget_seconds * 0.4
            logger.info(f"Starting high-memory feature engineering with {step_time_budget/60:.1f} minute budget")
            
            # Initialize feature generator
            feature_generator = HighMemoryFeatureGenerator(
                output_dir=FEATURES_DIR,
                memory_limit_gb=600,
                time_budget_seconds=step_time_budget
            )
            
            # Generate features
            step_start = time.time()
            train_features = feature_generator.generate_features(train_data, is_train=True)
            validation_features = feature_generator.generate_features(validation_data, is_train=False)
            tournament_features = feature_generator.generate_features(tournament_data, is_train=False)
            
            # Save feature data
            train_features_path = os.path.join(FEATURES_DIR, f"train_features_{self.timestamp}.parquet")
            val_features_path = os.path.join(FEATURES_DIR, f"validation_features_{self.timestamp}.parquet")
            tournament_features_path = os.path.join(FEATURES_DIR, f"tournament_features_{self.timestamp}.parquet")
            
            save_dataframe(train_features, train_features_path)
            save_dataframe(validation_features, val_features_path)
            save_dataframe(tournament_features, tournament_features_path)
            
            step_duration = time.time() - step_start
            logger.info(f"Feature engineering completed in {step_duration/60:.1f} minutes, generated {train_features.shape[1]} features")
            results['feature_paths'] = {'train': train_features_path, 'validation': val_features_path, 'tournament': tournament_features_path}
            
            # Step 3: Feature Selection (15% of time budget)
            step_time_budget = self.time_budget_seconds * 0.15
            logger.info(f"Starting feature selection with {step_time_budget/60:.1f} minute budget")
            
            # Initialize feature selector
            feature_selector = FeatureSelector(
                output_dir=FEATURES_DIR,
                time_budget_seconds=step_time_budget,
                n_features=2000  # Target 2000 features for final model
            )
            
            # Select features
            step_start = time.time()
            target_col = 'target'
            selected_features = feature_selector.select_features(
                train_features, validation_features, target_col=target_col
            )
            
            # Apply feature selection
            X_train = train_features[selected_features]
            y_train = train_features[target_col]
            X_val = validation_features[selected_features]
            y_val = validation_features[target_col]
            X_tournament = tournament_features[selected_features]
            
            # Save selected feature datasets
            selected_train_path = os.path.join(FEATURES_DIR, f"selected_train_{self.timestamp}.parquet")
            selected_val_path = os.path.join(FEATURES_DIR, f"selected_validation_{self.timestamp}.parquet")
            selected_tournament_path = os.path.join(FEATURES_DIR, f"selected_tournament_{self.timestamp}.parquet")
            
            save_dataframe(X_train, selected_train_path)
            save_dataframe(X_val, selected_val_path)
            save_dataframe(X_tournament, selected_tournament_path)
            
            step_duration = time.time() - step_start
            logger.info(f"Feature selection completed in {step_duration/60:.1f} minutes, selected {len(selected_features)} features")
            results['selected_feature_paths'] = {'train': selected_train_path, 'validation': selected_val_path, 'tournament': selected_tournament_path}
            
            # Step 4: Model Training (20% of time budget)
            step_time_budget = self.time_budget_seconds * 0.2
            logger.info(f"Starting model training with {step_time_budget/60:.1f} minute budget")
            
            # Train multiple models with GPU acceleration
            models = []
            predictions = []
            
            # LightGBM Model
            gpu_id = self.gpu_ids[0]
            step_start = time.time()
            lightgbm_model = LightGBMModel(name=f"lightgbm_{self.timestamp}", gpu_id=gpu_id)
            lightgbm_model.train(X_train, y_train, X_val, y_val)
            lightgbm_model.save(CHECKPOINTS_DIR)
            
            # Predict on validation data
            val_pred_lgbm = lightgbm_model.predict(X_val)
            val_rmse_lgbm = np.sqrt(np.mean((val_pred_lgbm - y_val)**2))
            logger.info(f"LightGBM validation RMSE: {val_rmse_lgbm:.6f}")
            models.append(lightgbm_model)
            predictions.append(val_pred_lgbm)
            
            # XGBoost Model
            gpu_id = self.gpu_ids[1 % self.num_gpus]  # Use second GPU if available
            xgboost_model = XGBoostModel(model_id=f"xgboost_{self.timestamp}", gpu_id=gpu_id)
            xgboost_model.train(X_train, y_train, X_val, y_val)
            xgboost_model.save(CHECKPOINTS_DIR)
            
            # Predict on validation data
            val_pred_xgb = xgboost_model.predict(X_val)
            val_rmse_xgb = np.sqrt(np.mean((val_pred_xgb - y_val)**2))
            logger.info(f"XGBoost validation RMSE: {val_rmse_xgb:.6f}")
            models.append(xgboost_model)
            predictions.append(val_pred_xgb)
            
            # Ensemble predictions (simple average for now)
            val_pred_ensemble = ensemble_predictions(predictions)
            val_rmse_ensemble = np.sqrt(np.mean((val_pred_ensemble - y_val)**2))
            logger.info(f"Ensemble validation RMSE: {val_rmse_ensemble:.6f}")
            
            step_duration = time.time() - step_start
            logger.info(f"Model training completed in {step_duration/60:.1f} minutes")
            
            results['validation_rmse'] = {
                'lightgbm': val_rmse_lgbm,
                'xgboost': val_rmse_xgb,
                'ensemble': val_rmse_ensemble
            }
            
            # Step 5: Generate Predictions (10% of time budget)
            step_time_budget = self.time_budget_seconds * 0.1
            logger.info(f"Generating final predictions with {step_time_budget/60:.1f} minute budget")
            
            step_start = time.time()
            tournament_ids = tournament_data['id'].values
            
            # Generate predictions for each model
            tournament_pred_lgbm = lightgbm_model.predict(X_tournament)
            tournament_pred_xgb = xgboost_model.predict(X_tournament)
            
            # Create ensemble prediction
            tournament_preds = [tournament_pred_lgbm, tournament_pred_xgb]
            tournament_pred_ensemble = ensemble_predictions(tournament_preds)
            
            # Create submission dataframes
            submission_lgbm = pd.DataFrame({'id': tournament_ids, 'prediction': tournament_pred_lgbm})
            submission_xgb = pd.DataFrame({'id': tournament_ids, 'prediction': tournament_pred_xgb})
            submission_ensemble = pd.DataFrame({'id': tournament_ids, 'prediction': tournament_pred_ensemble})
            
            # Save submissions
            submission_lgbm_path = os.path.join(SUBMISSIONS_DIR, f"lightgbm_submission_{self.timestamp}.csv")
            submission_xgb_path = os.path.join(SUBMISSIONS_DIR, f"xgboost_submission_{self.timestamp}.csv")
            submission_ensemble_path = os.path.join(SUBMISSIONS_DIR, f"ensemble_submission_{self.timestamp}.csv")
            
            submission_lgbm.to_csv(submission_lgbm_path, index=False)
            submission_xgb.to_csv(submission_xgb_path, index=False)
            submission_ensemble.to_csv(submission_ensemble_path, index=False)
            
            step_duration = time.time() - step_start
            logger.info(f"Prediction generation completed in {step_duration/60:.1f} minutes")
            
            results['submission_paths'] = {
                'lightgbm': submission_lgbm_path,
                'xgboost': submission_xgb_path,
                'ensemble': submission_ensemble_path
            }
            
            # Create summary report
            total_duration = time.time() - self.start_time
            logger.info(f"Optimal pipeline completed in {total_duration/3600:.2f} hours")
            logger.info(f"Best validation RMSE: {val_rmse_ensemble:.6f}")
            
            results['success'] = True
            results['duration_hours'] = total_duration / 3600
            results['best_rmse'] = val_rmse_ensemble
            
            # Return the ensemble submission as the main result
            return results
            
        except Exception as e:
            logger.error(f"Error in optimal pipeline: {str(e)}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
            return results

# For direct execution
if __name__ == "__main__":
    pipeline = OptimalPipeline()
    results = pipeline.run()
    print(f"Pipeline complete with best RMSE: {results.get('best_rmse', 'N/A')}")
    if results.get('success', False):
        print(f"Submission path: {results['submission_paths']['ensemble']}")
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")