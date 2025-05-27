"""
PyCaret model implementation for Numerai Crypto with multi-GPU support.
Utilizes PyCaret's AutoML capabilities with 3-GPU configuration for optimal performance.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyCaret
try:
    from pycaret.regression import (
        setup as regression_setup, 
        compare_models, 
        create_model, 
        tune_model, 
        ensemble_model, 
        blend_models, 
        stack_models,
        finalize_model,
        predict_model,
        pull,
        save_model,
        load_model
    )
    from pycaret.classification import (
        setup as classification_setup,
        compare_models as compare_models_clf,
        create_model as create_model_clf,
        tune_model as tune_model_clf,
        ensemble_model as ensemble_model_clf,
        blend_models as blend_models_clf,
        stack_models as stack_models_clf,
        finalize_model as finalize_model_clf,
        predict_model as predict_model_clf,
        pull as pull_clf,
        save_model as save_model_clf,
        load_model as load_model_clf
    )
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logger.warning("PyCaret is not installed. Please install it first: pip install pycaret")

# Import project settings
try:
    from config.settings import HARDWARE_CONFIG
except ImportError:
    HARDWARE_CONFIG = {'gpu_count': 0}
    logger.warning("Could not import hardware config. Using defaults.")

class PyCaretModel:
    """PyCaret model implementation with multi-GPU support"""
    
    def __init__(self, 
                 task_type: str = "regression",
                 use_gpu: bool = True,
                 gpu_count: int = 3,
                 seed: int = 42,
                 model_id: str = "pycaret_model",
                 session_id: int = 42):
        """
        Initialize PyCaret model
        
        Args:
            task_type: Type of ML task ('regression' or 'classification')
            use_gpu: Whether to use GPU acceleration
            gpu_count: Number of GPUs to utilize (default: 3)
            seed: Random seed
            model_id: Model ID for saving/loading
            session_id: PyCaret session ID
        """
        self.model_id = model_id
        self.task_type = task_type.lower()
        self.seed = seed
        self.session_id = session_id
        self.model = None
        self.best_models = []
        self.ensemble_model = None
        self.setup_complete = False
        self.feature_names = None
        
        # Check if PyCaret is available
        if not PYCARET_AVAILABLE:
            logger.warning("PyCaret is not installed. Using fallback model.")
            self.pycaret_available = False
            return
        else:
            self.pycaret_available = True
        
        # GPU configuration
        self.use_gpu = use_gpu
        self.gpu_count = min(gpu_count, HARDWARE_CONFIG.get('gpu_count', 0))
        
        if self.gpu_count == 0 or not self.use_gpu:
            self.use_gpu = False
            logger.info("Using CPU for PyCaret.")
        else:
            logger.info(f"Using {self.gpu_count} GPUs for PyCaret.")
        
        # Validate task type
        if self.task_type not in ['regression', 'classification']:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be 'regression' or 'classification'")
        
        logger.info(f"PyCaret {self.task_type} model initialized with session_id={self.session_id}")
    
    def setup_experiment(self, 
                        X_train: Union[pd.DataFrame, np.ndarray],
                        y_train: Union[pd.Series, np.ndarray],
                        test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                        train_size: float = 0.8,
                        fold_strategy: str = "timeseries",
                        fold: int = 5,
                        normalize: bool = True,
                        feature_selection: bool = True,
                        remove_multicollinearity: bool = True,
                        multicollinearity_threshold: float = 0.9,
                        remove_outliers: bool = True,
                        outliers_threshold: float = 0.05,
                        transformation: bool = True,
                        transform_target: bool = False,
                        handle_unknown_categorical: bool = True,
                        unknown_categorical_method: str = 'least_frequent',
                        pca: bool = False,
                        pca_components: Optional[int] = None,
                        ignore_low_variance: bool = True,
                        combine_rare_levels: bool = True,
                        rare_level_threshold: float = 0.10,
                        bin_numeric_features: Optional[List[str]] = None) -> bool:
        """
        Setup PyCaret experiment with multi-GPU configuration
        
        Args:
            X_train: Training features
            y_train: Training targets
            test_data: Test data (optional)
            train_size: Proportion of training data
            fold_strategy: Cross-validation strategy
            fold: Number of folds
            normalize: Whether to normalize features
            feature_selection: Whether to perform feature selection
            remove_multicollinearity: Whether to remove multicollinear features
            multicollinearity_threshold: Threshold for multicollinearity
            remove_outliers: Whether to remove outliers
            outliers_threshold: Threshold for outlier detection
            transformation: Whether to apply transformations
            transform_target: Whether to transform target variable
            handle_unknown_categorical: Whether to handle unknown categories
            unknown_categorical_method: Method for handling unknown categories
            pca: Whether to apply PCA
            pca_components: Number of PCA components
            ignore_low_variance: Whether to ignore low variance features
            combine_rare_levels: Whether to combine rare levels
            rare_level_threshold: Threshold for rare levels
            bin_numeric_features: List of features to bin
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        if not self.pycaret_available:
            logger.warning("PyCaret not available. Skipping setup.")
            return False
        
        try:
            # Prepare data
            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            
            if isinstance(y_train, np.ndarray):
                y_train = pd.Series(y_train, name='target')
            elif isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]
            
            # Combine features and target
            train_data = X_train.copy()
            train_data['target'] = y_train
            
            # Save feature names
            self.feature_names = X_train.columns.tolist()
            
            # Configure GPU settings
            gpu_config = {}
            if self.use_gpu:
                # Set environment variables for multi-GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.gpu_count)])
                
                # Configure GPU parameters for models
                gpu_config = {
                    'catboost_gpu_id': list(range(self.gpu_count)),
                    'lightgbm_device': 'gpu',
                    'xgboost_tree_method': 'gpu_hist',
                    'xgboost_gpu_id': 0  # XGBoost uses single GPU but can be parallelized
                }
            
            # Setup experiment based on task type
            if self.task_type == 'regression':
                self.experiment = regression_setup(
                    data=train_data,
                    test_data=test_data,
                    target='target',
                    session_id=self.session_id,
                    train_size=train_size,
                    fold_strategy=fold_strategy,
                    fold=fold,
                    normalize=normalize,
                    feature_selection=feature_selection,
                    remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold=multicollinearity_threshold,
                    remove_outliers=remove_outliers,
                    outliers_threshold=outliers_threshold,
                    transformation=transformation,
                    transform_target=transform_target,
                    handle_unknown_categorical=handle_unknown_categorical,
                    unknown_categorical_method=unknown_categorical_method,
                    pca=pca,
                    pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,
                    combine_rare_levels=combine_rare_levels,
                    rare_level_threshold=rare_level_threshold,
                    bin_numeric_features=bin_numeric_features,
                    silent=True,
                    use_gpu=self.use_gpu
                )
            else:  # classification
                self.experiment = classification_setup(
                    data=train_data,
                    test_data=test_data,
                    target='target',
                    session_id=self.session_id,
                    train_size=train_size,
                    fold_strategy=fold_strategy,
                    fold=fold,
                    normalize=normalize,
                    feature_selection=feature_selection,
                    remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold=multicollinearity_threshold,
                    remove_outliers=remove_outliers,
                    outliers_threshold=outliers_threshold,
                    transformation=transformation,
                    transform_target=transform_target,
                    handle_unknown_categorical=handle_unknown_categorical,
                    unknown_categorical_method=unknown_categorical_method,
                    pca=pca,
                    pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,
                    combine_rare_levels=combine_rare_levels,
                    rare_level_threshold=rare_level_threshold,
                    bin_numeric_features=bin_numeric_features,
                    silent=True,
                    use_gpu=self.use_gpu
                )
            
            self.setup_complete = True
            logger.info(f"PyCaret {self.task_type} experiment setup completed successfully")
            logger.info(f"Dataset shape: {train_data.shape}")
            logger.info(f"Feature engineering applied: normalize={normalize}, feature_selection={feature_selection}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up PyCaret experiment: {str(e)}")
            self.setup_complete = False
            return False
    
    def compare_models_gpu(self, 
                          include: Optional[List[str]] = None,
                          exclude: Optional[List[str]] = None,
                          fold: Optional[int] = None,
                          round_digits: int = 4,
                          cross_validation: bool = True,
                          sort_metric: str = 'RMSE',
                          n_select: int = 5) -> pd.DataFrame:
        """
        Compare multiple models with GPU acceleration
        
        Args:
            include: List of models to include
            exclude: List of models to exclude
            fold: Number of CV folds
            round_digits: Number of decimal places
            cross_validation: Whether to use cross-validation
            sort_metric: Metric to sort by
            n_select: Number of top models to select
            
        Returns:
            DataFrame with model comparison results
        """
        if not self.setup_complete:
            raise ValueError("Experiment not setup. Call setup_experiment() first.")
        
        try:
            # Define GPU-optimized models for comparison
            if include is None:
                if self.use_gpu:
                    # GPU-optimized models
                    include = ['lightgbm', 'xgboost', 'catboost', 'rf', 'et', 'gbr']
                else:
                    # CPU models
                    include = ['lr', 'ridge', 'rf', 'et', 'gbr', 'ada', 'lightgbm']
            
            if exclude is None:
                # Exclude slow models for GPU training
                exclude = ['lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn']
            
            logger.info(f"Comparing models: {include}")
            logger.info(f"Using {self.gpu_count} GPUs" if self.use_gpu else "Using CPU")
            
            # Compare models based on task type
            if self.task_type == 'regression':
                results = compare_models(
                    include=include,
                    exclude=exclude,
                    fold=fold,
                    round=round_digits,
                    cross_validation=cross_validation,
                    sort=sort_metric
                )
            else:  # classification
                results = compare_models_clf(
                    include=include,
                    exclude=exclude,
                    fold=fold,
                    round=round_digits,
                    cross_validation=cross_validation,
                    sort=sort_metric
                )
            
            # Get comparison results
            if self.task_type == 'regression':
                comparison_df = pull()
            else:
                comparison_df = pull_clf()
            
            # Select top models
            if isinstance(results, list):
                self.best_models = results[:n_select]
            else:
                self.best_models = [results]
            
            logger.info(f"Model comparison completed. Selected top {len(self.best_models)} models.")
            logger.info(f"Best model: {type(self.best_models[0]).__name__}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def create_and_tune_models(self, 
                              model_list: Optional[List[str]] = None,
                              optimize_metric: str = 'RMSE',
                              search_library: str = 'optuna',
                              search_algorithm: str = 'tpe',
                              n_iter: int = 100,
                              early_stopping: Union[bool, str, int] = 'asha',
                              early_stopping_max_iters: int = 100) -> List:
        """
        Create and tune models with GPU optimization
        
        Args:
            model_list: List of model names to create and tune
            optimize_metric: Metric to optimize
            search_library: Hyperparameter search library
            search_algorithm: Search algorithm
            n_iter: Number of iterations
            early_stopping: Early stopping strategy
            early_stopping_max_iters: Maximum iterations for early stopping
            
        Returns:
            List of tuned models
        """
        if not self.setup_complete:
            raise ValueError("Experiment not setup. Call setup_experiment() first.")
        
        try:
            if model_list is None:
                if self.use_gpu:
                    model_list = ['lightgbm', 'xgboost', 'catboost', 'rf']
                else:
                    model_list = ['lightgbm', 'rf', 'et', 'gbr']
            
            tuned_models = []
            
            for model_name in model_list:
                logger.info(f"Creating and tuning {model_name}...")
                
                try:
                    # Create base model
                    if self.task_type == 'regression':
                        base_model = create_model(model_name)
                    else:
                        base_model = create_model_clf(model_name)
                    
                    # Configure GPU-specific parameters
                    custom_grid = {}
                    if model_name == 'lightgbm' and self.use_gpu:
                        custom_grid = {
                            'device': ['gpu'],
                            'gpu_platform_id': [0],
                            'gpu_device_id': [0],
                            'num_leaves': [31, 50, 100],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [100, 200, 500]
                        }
                    elif model_name == 'xgboost' and self.use_gpu:
                        custom_grid = {
                            'tree_method': ['gpu_hist'],
                            'gpu_id': [0],
                            'max_depth': [3, 6, 9],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [100, 200, 500]
                        }
                    elif model_name == 'catboost' and self.use_gpu:
                        custom_grid = {
                            'task_type': ['GPU'],
                            'devices': [f'0:{self.gpu_count-1}'],
                            'depth': [4, 6, 8],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'iterations': [100, 200, 500]
                        }
                    
                    # Tune model
                    if self.task_type == 'regression':
                        tuned_model = tune_model(
                            base_model,
                            optimize=optimize_metric,
                            search_library=search_library,
                            search_algorithm=search_algorithm,
                            n_iter=n_iter,
                            early_stopping=early_stopping,
                            early_stopping_max_iters=early_stopping_max_iters,
                            custom_grid=custom_grid if custom_grid else None
                        )
                    else:
                        tuned_model = tune_model_clf(
                            base_model,
                            optimize=optimize_metric,
                            search_library=search_library,
                            search_algorithm=search_algorithm,
                            n_iter=n_iter,
                            early_stopping=early_stopping,
                            early_stopping_max_iters=early_stopping_max_iters,
                            custom_grid=custom_grid if custom_grid else None
                        )
                    
                    tuned_models.append(tuned_model)
                    logger.info(f"Successfully tuned {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to tune {model_name}: {str(e)}")
                    continue
            
            self.best_models = tuned_models
            logger.info(f"Created and tuned {len(tuned_models)} models")
            
            return tuned_models
            
        except Exception as e:
            logger.error(f"Error creating and tuning models: {str(e)}")
            raise
    
    def create_ensemble(self, 
                       models: Optional[List] = None,
                       ensemble_method: str = 'Blending',
                       optimize_metric: str = 'RMSE',
                       method: str = 'auto') -> Any:
        """
        Create ensemble model from best models
        
        Args:
            models: List of models to ensemble
            ensemble_method: Type of ensemble ('Blending', 'Stacking', 'Voting')
            optimize_metric: Metric to optimize
            method: Ensemble method ('auto', 'Blending', 'Stacking')
            
        Returns:
            Ensemble model
        """
        if not self.setup_complete:
            raise ValueError("Experiment not setup. Call setup_experiment() first.")
        
        if models is None:
            models = self.best_models
        
        if len(models) < 2:
            logger.warning("Need at least 2 models for ensemble. Returning single model.")
            return models[0] if models else None
        
        try:
            logger.info(f"Creating {ensemble_method} ensemble with {len(models)} models")
            
            if ensemble_method.lower() == 'blending':
                if self.task_type == 'regression':
                    ensemble = blend_models(models, optimize=optimize_metric)
                else:
                    ensemble = blend_models_clf(models, optimize=optimize_metric)
            elif ensemble_method.lower() == 'stacking':
                if self.task_type == 'regression':
                    ensemble = stack_models(models, optimize=optimize_metric)
                else:
                    ensemble = stack_models_clf(models, optimize=optimize_metric)
            else:
                logger.warning(f"Unknown ensemble method: {ensemble_method}. Using blending.")
                if self.task_type == 'regression':
                    ensemble = blend_models(models, optimize=optimize_metric)
                else:
                    ensemble = blend_models_clf(models, optimize=optimize_metric)
            
            self.ensemble_model = ensemble
            self.model = ensemble  # Set as primary model
            
            logger.info(f"{ensemble_method} ensemble created successfully")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            # Fallback to best single model
            if models:
                self.model = models[0]
                logger.info("Fallback to best single model")
                return models[0]
            raise
    
    def finalize_and_train(self, model=None) -> Any:
        """
        Finalize model training on full dataset
        
        Args:
            model: Model to finalize (uses ensemble_model if None)
            
        Returns:
            Finalized model
        """
        if not self.setup_complete:
            raise ValueError("Experiment not setup. Call setup_experiment() first.")
        
        if model is None:
            model = self.ensemble_model or (self.best_models[0] if self.best_models else None)
        
        if model is None:
            raise ValueError("No model available to finalize")
        
        try:
            logger.info("Finalizing model training on full dataset...")
            
            if self.task_type == 'regression':
                finalized_model = finalize_model(model)
            else:
                finalized_model = finalize_model_clf(model)
            
            self.model = finalized_model
            logger.info("Model finalization completed")
            
            return finalized_model
            
        except Exception as e:
            logger.error(f"Error finalizing model: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Features
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call training methods first.")
        
        try:
            # Prepare data
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            # Generate predictions
            if self.task_type == 'regression':
                predictions = predict_model(self.model, data=X)
            else:
                predictions = predict_model_clf(self.model, data=X)
            
            # Extract prediction column
            pred_col = 'prediction_label' if self.task_type == 'classification' else 'prediction_score'
            if pred_col not in predictions.columns:
                pred_col = [col for col in predictions.columns if 'prediction' in col.lower()][0]
            
            return predictions[pred_col].values
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call training methods first.")
        
        try:
            # This is a simplified implementation
            # Feature importance extraction depends on the specific model type
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            else:
                # For ensemble models, try to get importance from base models
                logger.warning("Feature importance not directly available for this model type")
                return {name: 1.0 for name in self.feature_names}
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {name: 1.0 for name in self.feature_names}
    
    def save_model(self, directory: str) -> str:
        """
        Save model to disk
        
        Args:
            directory: Directory to save model
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call training methods first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save model
            model_path = os.path.join(directory, f"{self.model_id}")
            
            if self.task_type == 'regression':
                save_model(self.model, model_path)
            else:
                save_model_clf(self.model, model_path)
            
            # Save metadata
            metadata = {
                'model_id': self.model_id,
                'task_type': self.task_type,
                'feature_names': self.feature_names,
                'use_gpu': self.use_gpu,
                'gpu_count': self.gpu_count,
                'session_id': self.session_id
            }
            
            metadata_path = os.path.join(directory, f"{self.model_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from disk
        
        Args:
            model_path: Path to saved model
        """
        try:
            # Load model
            if self.task_type == 'regression':
                self.model = load_model(model_path)
            else:
                self.model = load_model_clf(model_path)
            
            # Load metadata if available
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.session_id = metadata.get('session_id', self.session_id)
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def train_complete_pipeline(self,
                               X_train: Union[pd.DataFrame, np.ndarray],
                               y_train: Union[pd.Series, np.ndarray],
                               X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                               y_val: Optional[Union[pd.Series, np.ndarray]] = None,
                               compare_models_first: bool = True,
                               tune_top_models: bool = True,
                               create_ensemble: bool = True,
                               ensemble_method: str = 'Blending') -> Dict[str, Any]:
        """
        Complete training pipeline with PyCaret
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            compare_models_first: Whether to compare models first
            tune_top_models: Whether to tune top models
            create_ensemble: Whether to create ensemble
            ensemble_method: Ensemble method to use
            
        Returns:
            Dictionary with training results
        """
        try:
            # Setup experiment
            logger.info("Setting up PyCaret experiment...")
            test_data = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, np.ndarray):
                    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
                if isinstance(y_val, np.ndarray):
                    y_val = pd.Series(y_val, name='target')
                elif isinstance(y_val, pd.DataFrame):
                    y_val = y_val.iloc[:, 0]
                
                test_data = X_val.copy()
                test_data['target'] = y_val
            
            setup_success = self.setup_experiment(X_train, y_train, test_data=test_data)
            
            if not setup_success:
                raise ValueError("Failed to setup PyCaret experiment")
            
            results = {
                'setup_complete': True,
                'models_compared': False,
                'models_tuned': False,
                'ensemble_created': False
            }
            
            # Compare models
            if compare_models_first:
                logger.info("Comparing models...")
                comparison_df = self.compare_models_gpu(n_select=5)
                results['comparison_results'] = comparison_df
                results['models_compared'] = True
            
            # Tune top models
            if tune_top_models:
                logger.info("Tuning top models...")
                tuned_models = self.create_and_tune_models(n_iter=50)
                results['tuned_models_count'] = len(tuned_models)
                results['models_tuned'] = True
            
            # Create ensemble
            if create_ensemble and len(self.best_models) > 1:
                logger.info("Creating ensemble...")
                ensemble = self.create_ensemble(ensemble_method=ensemble_method)
                results['ensemble_created'] = True
                results['ensemble_type'] = ensemble_method
            
            # Finalize model
            logger.info("Finalizing model...")
            final_model = self.finalize_and_train()
            results['model_finalized'] = True
            
            logger.info("PyCaret training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in PyCaret training pipeline: {str(e)}")
            raise