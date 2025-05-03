"""
XGBoost model implementation for the Numerai Crypto project.
"""
import os
import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator
import cloudpickle
import time
from numer_crypto.config.settings import MODELS_DIR, XGBOOST_PARAMS
from numer_crypto.utils.data_utils import convert_pandas_to_h2o


class H2OXGBoostModel:
    """
    H2O XGBoost model for Numerai Crypto predictions.
    """
    
    def __init__(self, h2o_instance=None, params=None):
        """
        Initialize the H2O XGBoost model.
        
        Args:
            h2o_instance: H2O instance
            params (dict): Model parameters
        """
        self.h2o = h2o_instance or h2o
        self.params = params or XGBOOST_PARAMS
        self.model = None
        self.feature_names = None
        self.target_column = None
        self.model_id = None
        
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def train(self, train_df, valid_df=None, feature_cols=None, target_col='target', model_id=None):
        """
        Train the XGBoost model.
        
        Args:
            train_df (DataFrame): Training data
            valid_df (DataFrame): Validation data (optional)
            feature_cols (list): Feature column names
            target_col (str): Target column name
            model_id (str): Model ID
            
        Returns:
            H2OXGBoostModel: Self for chaining
        """
        print("Training XGBoost model...")
        start_time = time.time()
        
        # Set model ID
        self.model_id = model_id or f"xgb_model_{int(time.time())}"
        
        # Convert pandas dataframes to H2O frames
        h2o_train = convert_pandas_to_h2o(train_df, self.h2o)
        h2o_valid = convert_pandas_to_h2o(valid_df, self.h2o) if valid_df is not None else None
        
        # Set feature and target columns
        self.feature_names = feature_cols or list(set(train_df.columns) - {target_col})
        self.target_column = target_col
        
        # Create and train XGBoost model
        model = H2OXGBoostEstimator(
            model_id=self.model_id,
            ntrees=self.params.get('ntrees', 500),
            max_depth=self.params.get('max_depth', 6),
            learn_rate=self.params.get('learn_rate', 0.1),
            sample_rate=self.params.get('sample_rate', 0.8),
            col_sample_rate=self.params.get('col_sample_rate', 0.8),
            score_each_iteration=True,
            seed=42
        )
        
        model.train(
            x=self.feature_names,
            y=self.target_column,
            training_frame=h2o_train,
            validation_frame=h2o_valid
        )
        
        self.model = model
        
        # Print training info
        elapsed_time = time.time() - start_time
        print(f"Model training completed in {elapsed_time:.2f} seconds")
        print(f"Training AUC: {model.auc()}")
        if h2o_valid is not None:
            print(f"Validation AUC: {model.auc(valid=True)}")
        
        return self
    
    def predict(self, data_df):
        """
        Generate predictions using the trained model.
        
        Args:
            data_df (DataFrame): Data to predict on
            
        Returns:
            DataFrame: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to H2O frame
        h2o_data = convert_pandas_to_h2o(data_df, self.h2o)
        
        # Generate predictions
        preds = self.model.predict(h2o_data)
        
        # Convert to pandas
        preds_df = preds.as_data_frame()
        
        return preds_df
    
    def save_model(self, filepath=None):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            str: Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"{self.model_id}.pickle")
        
        # Extract model info
        model_info = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'model_id': self.model_id,
            'params': self.params
        }
        
        # Save with cloudpickle
        with open(filepath, 'wb') as f:
            cloudpickle.dump(model_info, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath, h2o_instance=None):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            h2o_instance: H2O instance
            
        Returns:
            H2OXGBoostModel: Loaded model
        """
        # Create a new instance
        instance = cls(h2o_instance=h2o_instance)
        
        # Load model info
        with open(filepath, 'rb') as f:
            model_info = cloudpickle.load(f)
        
        # Set model attributes
        instance.model = model_info['model']
        instance.feature_names = model_info['feature_names']
        instance.target_column = model_info['target_column']
        instance.model_id = model_info['model_id']
        instance.params = model_info['params']
        
        print(f"Model loaded from {filepath}")
        return instance
    
    def get_feature_importance(self):
        """
        Get feature importances from the trained model.
        
        Returns:
            DataFrame: Feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        varimp = self.model.varimp(use_pandas=True)
        return varimp