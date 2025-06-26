#!/usr/bin/env python3
"""
Quick Demo of GPU Accelerated Feature Engineering and ML

This script provides a minimal example of:
1. GPU-accelerated feature engineering with batching
2. GPU-accelerated ML model training with LightGBM
3. Simple evaluation of model performance
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import GPU Math Accelerator
from scripts.features.gpu_math_accelerator import GPUMathAccelerator

class Timer:
    def __init__(self, name="Operation"):
        self.name = name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {self.duration:.2f} seconds")

def create_synthetic_data(rows=10000, cols=20):
    """Create synthetic data for testing"""
    logger.info(f"Creating synthetic data: {rows} rows, {cols} cols")
    
    # Set random seed
    np.random.seed(42)
    
    # Create features with some correlation structure
    X = np.random.randn(rows, cols).astype(np.float32)
    
    # Create target (linear combination of features with noise)
    weights = np.random.randn(cols)
    y = X.dot(weights) + np.random.randn(rows) * 0.5
    
    # Convert to binary for classification
    y_binary = (y > 0).astype(np.int32)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(cols)]
    
    return X, y_binary, feature_names

def demo_gpu_feature_engineering(X, feature_names, batch_size=5000):
    """Demonstrate GPU-accelerated feature engineering with batching"""
    logger.info(f"Starting GPU feature engineering with batching (batch size: {batch_size})")
    
    # Initialize GPU Math Accelerator
    os.environ["GPU_MEMORY_LIMIT"] = "12.0"  # Set conservative limit
    accelerator = GPUMathAccelerator()
    
    # Get data shape
    n_samples, n_features = X.shape
    
    # Process in batches
    transformed_batches = []
    transformed_names = None
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: rows {i}-{end_idx}")
        
        # Get batch
        batch_data = X[i:end_idx]
        
        # Transform batch
        with Timer(f"GPU transform batch {i//batch_size + 1}"):
            batch_result, batch_names = accelerator.generate_all_math_transforms(
                batch_data, feature_names,
                include_trig=True,
                include_poly=True,
                max_interactions=100,
                include_random_baselines=False
            )
        
        # Save results
        transformed_batches.append(batch_result)
        
        if transformed_names is None:
            transformed_names = batch_names
    
    # Combine batches
    with Timer("Combining batches"):
        transformed_X = np.vstack(transformed_batches)
    
    logger.info(f"Original shape: {X.shape}, Transformed shape: {transformed_X.shape}")
    logger.info(f"Added {len(transformed_names)} new features")
    
    return transformed_X, transformed_names

def demo_gpu_model_training(X, y, feature_names):
    """Demonstrate GPU-accelerated model training with LightGBM"""
    logger.info("Starting GPU model training with LightGBM")
    
    try:
        import lightgbm as lgb
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Set parameters for GPU training
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
        
        # Train model
        with Timer("Training LightGBM with GPU"):
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[test_data],
                callbacks=[lgb.log_evaluation(10)]
            )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        
        logger.info(f"Model performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        importance_dict = dict(zip(feature_names, importance))
        
        # Show top 10 features
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features:")
        for feature, score in top_features:
            logger.info(f"  {feature}: {score}")
        
        return model, importance_dict
        
    except ImportError:
        logger.warning("LightGBM not available, skipping model training")
        return None, {}

def main():
    # Step 1: Create synthetic data
    X, y, feature_names = create_synthetic_data(rows=10000, cols=20)
    
    # Step 2: GPU feature engineering with batching
    transformed_X, transformed_names = demo_gpu_feature_engineering(X, feature_names, batch_size=5000)
    
    # Step 3: GPU model training
    model, importance = demo_gpu_model_training(transformed_X, y, transformed_names)
    
    logger.info("Quick demo completed successfully")

if __name__ == "__main__":
    main()