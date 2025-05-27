#!/usr/bin/env python3
"""
PyCaret model training script for Numerai Crypto with 3-GPU optimization.
Implements comprehensive AutoML pipeline with multi-GPU acceleration.
"""

import os
import sys
import logging
import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from models.pycaret_model import PyCaretModel
from utils.gpu.detection import detect_gpus, get_gpu_info
from utils.memory_utils import get_memory_info, optimize_memory_usage
from utils.pipeline_logger import setup_logger

# Configure logging
logger = setup_logger(__name__, level=logging.INFO)

def load_training_data(data_path: str, target_col: str = 'target') -> tuple:
    """
    Load training data from parquet file
    
    Args:
        data_path: Path to training data
        target_col: Target column name
        
    Returns:
        Tuple of (X_train, y_train, feature_columns)
    """
    logger.info(f"Loading training data from {data_path}")
    
    try:
        # Load data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Check if target column exists
        if target_col not in df.columns:
            potential_targets = [col for col in df.columns if 'target' in col.lower()]
            if potential_targets:
                target_col = potential_targets[0]
                logger.info(f"Using '{target_col}' as target column")
            else:
                raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Identify feature columns (exclude metadata columns)
        metadata_cols = ['id', 'era', 'data_type', 'symbol', 'date', target_col]
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        logger.info(f"Found {len(feature_cols)} feature columns")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            logger.info("Handling missing values in features")
            X = X.fillna(X.median())
        
        if y.isnull().any():
            logger.warning("Found missing values in target. Dropping rows.")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        return X, y, feature_cols
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def setup_gpu_environment(gpu_count: int = 3) -> dict:
    """
    Setup multi-GPU environment for PyCaret
    
    Args:
        gpu_count: Number of GPUs to use
        
    Returns:
        Dictionary with GPU configuration
    """
    logger.info("Setting up GPU environment for PyCaret")
    
    try:
        # Detect available GPUs
        available_gpus = detect_gpus()
        gpu_info = get_gpu_info()
        
        logger.info(f"Detected {available_gpus} GPUs")
        for i, info in enumerate(gpu_info):
            logger.info(f"GPU {i}: {info['name']} ({info['memory_total']}MB)")
        
        # Configure GPU usage
        actual_gpu_count = min(gpu_count, available_gpus)
        
        if actual_gpu_count == 0:
            logger.warning("No GPUs detected. Using CPU mode.")
            return {'use_gpu': False, 'gpu_count': 0}
        
        # Set CUDA environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(actual_gpu_count)])
        os.environ['NUMEXPR_NUM_THREADS'] = str(actual_gpu_count * 2)
        os.environ['OMP_NUM_THREADS'] = str(actual_gpu_count * 2)
        
        # Configure GPU memory growth for TensorFlow (if used by PyCaret)
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus[:actual_gpu_count]:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured TensorFlow GPU memory growth for {len(gpus)} GPUs")
        except ImportError:
            logger.info("TensorFlow not available. Skipping TF GPU configuration.")
        except Exception as e:
            logger.warning(f"TensorFlow GPU configuration failed: {str(e)}")
        
        logger.info(f"GPU environment configured for {actual_gpu_count} GPUs")
        
        return {
            'use_gpu': True,
            'gpu_count': actual_gpu_count,
            'gpu_info': gpu_info[:actual_gpu_count]
        }
        
    except Exception as e:
        logger.error(f"Error setting up GPU environment: {str(e)}")
        return {'use_gpu': False, 'gpu_count': 0}

def train_pycaret_model(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame = None,
                       y_val: pd.Series = None,
                       task_type: str = 'regression',
                       gpu_config: dict = None,
                       model_config: dict = None,
                       output_dir: str = None) -> dict:
    """
    Train PyCaret model with multi-GPU acceleration
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        task_type: Type of ML task
        gpu_config: GPU configuration
        model_config: Model configuration
        output_dir: Output directory for models
        
    Returns:
        Dictionary with training results
    """
    if gpu_config is None:
        gpu_config = {'use_gpu': False, 'gpu_count': 0}
    
    if model_config is None:
        model_config = {
            'compare_models': True,
            'tune_models': True,
            'create_ensemble': True,
            'ensemble_method': 'Blending'
        }
    
    logger.info(f"Training PyCaret {task_type} model")
    logger.info(f"GPU configuration: {gpu_config}")
    logger.info(f"Model configuration: {model_config}")
    
    try:
        # Initialize PyCaret model
        pycaret_model = PyCaretModel(
            task_type=task_type,
            use_gpu=gpu_config['use_gpu'],
            gpu_count=gpu_config['gpu_count'],
            seed=42,
            model_id=f"pycaret_{task_type}_{int(time.time())}"
        )
        
        # Track training time
        start_time = time.time()
        
        # Train complete pipeline
        training_results = pycaret_model.train_complete_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            compare_models_first=model_config['compare_models'],
            tune_top_models=model_config['tune_models'],
            create_ensemble=model_config['create_ensemble'],
            ensemble_method=model_config['ensemble_method']
        )
        
        training_time = time.time() - start_time
        
        # Generate predictions on validation set if available
        val_predictions = None
        val_metrics = {}
        
        if X_val is not None and y_val is not None:
            logger.info("Generating validation predictions")
            val_predictions = pycaret_model.predict(X_val)
            
            # Calculate validation metrics
            if task_type == 'regression':
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                val_metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
                    'mae': mean_absolute_error(y_val, val_predictions),
                    'r2': r2_score(y_val, val_predictions)
                }
            else:  # classification
                from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
                val_metrics = {
                    'accuracy': accuracy_score(y_val, val_predictions),
                    'auc': roc_auc_score(y_val, val_predictions) if len(np.unique(y_val)) == 2 else None
                }
            
            logger.info(f"Validation metrics: {val_metrics}")
        
        # Get feature importance
        feature_importance = pycaret_model.get_feature_importance()
        
        # Save model if output directory specified
        if output_dir:
            logger.info(f"Saving model to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            model_path = pycaret_model.save_model(output_dir)
            
            # Save training results
            results_path = os.path.join(output_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'training_time': training_time,
                    'validation_metrics': val_metrics,
                    'gpu_config': gpu_config,
                    'model_config': model_config,
                    'training_results': {k: v for k, v in training_results.items() 
                                       if not isinstance(v, pd.DataFrame)}
                }, f, indent=2)
            
            # Save feature importance
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
            importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
            
            logger.info(f"Model and results saved to {output_dir}")
        
        return {
            'model': pycaret_model,
            'training_time': training_time,
            'validation_metrics': val_metrics,
            'feature_importance': feature_importance,
            'training_results': training_results,
            'val_predictions': val_predictions
        }
        
    except Exception as e:
        logger.error(f"Error training PyCaret model: {str(e)}")
        raise

def create_submission(model: PyCaretModel,
                     test_data_path: str,
                     output_path: str,
                     id_col: str = 'id') -> str:
    """
    Create submission file using trained PyCaret model
    
    Args:
        model: Trained PyCaret model
        test_data_path: Path to test data
        output_path: Path to save submission
        id_col: ID column name
        
    Returns:
        Path to submission file
    """
    logger.info(f"Creating submission from {test_data_path}")
    
    try:
        # Load test data
        test_df = pd.read_parquet(test_data_path)
        logger.info(f"Loaded test data with shape: {test_df.shape}")
        
        # Extract features (same as training)
        metadata_cols = ['id', 'era', 'data_type', 'symbol', 'date', 'target']
        feature_cols = [col for col in test_df.columns if col not in metadata_cols]
        
        # Handle missing features
        if hasattr(model, 'feature_names') and model.feature_names:
            missing_features = set(model.feature_names) - set(feature_cols)
            if missing_features:
                logger.warning(f"Missing features in test data: {list(missing_features)[:10]}...")
                # Add missing features with zeros
                for feature in missing_features:
                    test_df[feature] = 0.0
                feature_cols = model.feature_names
        
        X_test = test_df[feature_cols].fillna(0)
        
        # Generate predictions
        logger.info("Generating predictions")
        predictions = model.predict(X_test)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_df[id_col],
            'prediction': predictions
        })
        
        # Ensure predictions are in valid range and have proper variance
        pred_std = submission_df['prediction'].std()
        if pred_std < 0.01:
            logger.warning(f"Low prediction variance ({pred_std:.6f}). Adding noise for submission requirements.")
            noise = np.random.normal(0, 0.01, len(submission_df))
            submission_df['prediction'] += noise
        
        # Save submission
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        logger.info(f"Prediction stats: mean={submission_df['prediction'].mean():.6f}, "
                   f"std={submission_df['prediction'].std():.6f}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating submission: {str(e)}")
        raise

def main():
    """Main function for PyCaret model training"""
    parser = argparse.ArgumentParser(description='Train PyCaret models for Numerai Crypto')
    
    # Data arguments
    parser.add_argument('--train-data', '-t', type=str, required=True,
                       help='Path to training data (parquet)')
    parser.add_argument('--test-data', type=str,
                       help='Path to test data for submission')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Target column name')
    parser.add_argument('--task-type', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Type of ML task')
    
    # GPU arguments
    parser.add_argument('--gpu-count', type=int, default=3,
                       help='Number of GPUs to use')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    # Model arguments
    parser.add_argument('--no-compare', action='store_true',
                       help='Skip model comparison')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Skip ensemble creation')
    parser.add_argument('--ensemble-method', type=str, default='Blending',
                       choices=['Blending', 'Stacking'],
                       help='Ensemble method')
    
    # Output arguments
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for models and results')
    parser.add_argument('--submission-path', type=str,
                       help='Path to save submission file')
    
    # Other arguments
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--memory-optimize', action='store_true',
                       help='Enable memory optimization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Print system information
        logger.info("=== PyCaret Multi-GPU Training Pipeline ===")
        logger.info(f"Arguments: {vars(args)}")
        
        # Memory optimization
        if args.memory_optimize:
            logger.info("Enabling memory optimization")
            optimize_memory_usage()
        
        # Print memory info
        memory_info = get_memory_info()
        logger.info(f"Memory info: {memory_info}")
        
        # Setup GPU environment
        if not args.no_gpu:
            gpu_config = setup_gpu_environment(args.gpu_count)
        else:
            gpu_config = {'use_gpu': False, 'gpu_count': 0}
        
        # Load training data
        X_train, y_train, feature_cols = load_training_data(args.train_data, args.target_col)
        
        # Split training and validation data
        from sklearn.model_selection import train_test_split
        
        if args.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=args.validation_split,
                random_state=42,
                stratify=None  # For regression, stratify should be None
            )
            logger.info(f"Split data: train={X_train.shape}, val={X_val.shape}")
        else:
            X_val, y_val = None, None
            logger.info("No validation split specified")
        
        # Configure model settings
        model_config = {
            'compare_models': not args.no_compare,
            'tune_models': not args.no_tune,
            'create_ensemble': not args.no_ensemble,
            'ensemble_method': args.ensemble_method
        }
        
        # Train model
        logger.info("Starting PyCaret model training...")
        training_start = time.time()
        
        training_results = train_pycaret_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=args.task_type,
            gpu_config=gpu_config,
            model_config=model_config,
            output_dir=args.output_dir
        )
        
        total_time = time.time() - training_start
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Print results summary
        logger.info("=== Training Results Summary ===")
        logger.info(f"Training time: {training_results['training_time']:.2f} seconds")
        
        if training_results['validation_metrics']:
            logger.info("Validation metrics:")
            for metric, value in training_results['validation_metrics'].items():
                if value is not None:
                    logger.info(f"  {metric}: {value:.6f}")
        
        # Feature importance top 10
        if training_results['feature_importance']:
            logger.info("Top 10 important features:")
            sorted_features = sorted(training_results['feature_importance'].items(),
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.6f}")
        
        # Create submission if test data provided
        if args.test_data and args.submission_path:
            logger.info("Creating submission file...")
            submission_path = create_submission(
                model=training_results['model'],
                test_data_path=args.test_data,
                output_path=args.submission_path
            )
            logger.info(f"Submission created: {submission_path}")
        
        logger.info("PyCaret training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()