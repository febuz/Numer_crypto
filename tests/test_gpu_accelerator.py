#!/usr/bin/env python3
"""
Test script for GPU Math Accelerator 

This script verifies that our GPU math accelerator can handle large datasets without
exceeding memory limits and tests H2O XGBoost and LGBM compatibility.
"""

import os
import sys
import time
import logging
import numpy as np
import argparse

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

def test_large_dataset(rows=3000000, cols=3000, max_interactions=150):
    """Test GPU Math Accelerator with a large dataset"""
    logger.info(f"Creating test dataset: {rows:,} rows × {cols:,} columns")
    
    # Create a smaller sample dataset first to avoid memory issues
    sample_rows = min(100000, rows)
    logger.info(f"First creating sample dataset with {sample_rows:,} rows")
    
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(sample_rows, cols).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(cols)]
    
    # Initialize accelerator
    logger.info("Initializing GPU Math Accelerator")
    os.environ["GPU_MEMORY_LIMIT"] = "20.0"  # Set memory limit to 20GB
    accelerator = GPUMathAccelerator()
    
    # Test interaction transforms with sample data
    logger.info(f"Testing GPU interactions with sample data")
    start_time = time.time()
    transformed, transform_names = accelerator.gpu_interaction_transforms(
        sample_data, feature_names, max_interactions=max_interactions
    )
    elapsed = time.time() - start_time
    logger.info(f"Generated {transformed.shape[1]} interaction features in {elapsed:.2f}s")
    
    # Now try with full dataset
    logger.info(f"Creating full dataset with {rows:,} rows")
    data = np.random.randn(rows, cols).astype(np.float32)
    
    # Use batch processing to handle large dataset
    batch_size = 1000000
    logger.info(f"Testing GPU interactions with full data using batch size {batch_size:,}")
    
    # Test interaction transforms with batched processing
    start_time = time.time()
    try:
        transformed, transform_names = accelerator.generate_all_math_transforms(
            data, feature_names,
            include_trig=False,
            include_poly=False, 
            max_interactions=max_interactions,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        logger.info(f"Generated {transformed.shape[1]} total features in {elapsed:.2f}s")
        logger.info("✅ GPU Math Accelerator test PASSED")
    except Exception as e:
        logger.error(f"❌ GPU Math Accelerator test FAILED: {e}")
        return False
    
    return True

def test_h2o_xgboost():
    """Test H2O Sparkling XGBoost compatibility"""
    try:
        import h2o
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        
        logger.info("Initializing H2O")
        h2o.init()
        
        logger.info("Creating test dataset")
        # Small dataset for quick testing
        rows, cols = 10000, 10
        X = np.random.randn(rows, cols).astype(np.float32)
        y = (np.random.randn(rows) > 0).astype(np.float32)
        
        # Convert to H2O frame
        train = h2o.H2OFrame(np.column_stack([X, y]))
        train.columns = [f"feature_{i}" for i in range(cols)] + ["target"]
        
        # Train model
        logger.info("Training H2O XGBoost model")
        model = H2OXGBoostEstimator(
            ntrees=10,
            max_depth=3,
            learn_rate=0.1,
            gpu_id=0  # Use first GPU
        )
        
        model.train(
            x=[f"feature_{i}" for i in range(cols)],
            y="target",
            training_frame=train
        )
        
        # Verify model
        logger.info(f"Model performance: {model.model_performance()}")
        logger.info("✅ H2O XGBoost test PASSED")
        
        # Shutdown H2O
        h2o.shutdown()
        return True
    except ImportError:
        logger.warning("H2O or XGBoost not available, skipping test")
        return None
    except Exception as e:
        logger.error(f"❌ H2O XGBoost test FAILED: {e}")
        return False

def test_lightgbm_gpu():
    """Test LightGBM with GPU acceleration"""
    try:
        import lightgbm as lgb
        
        logger.info("Creating test dataset for LightGBM")
        # Small dataset for quick testing
        rows, cols = 10000, 10
        X = np.random.randn(rows, cols).astype(np.float32)
        y = (np.random.randn(rows) > 0).astype(np.float32)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Set GPU parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
        
        # Train model
        logger.info("Training LightGBM model with GPU acceleration")
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=10
        )
        
        # Verify model
        preds = gbm.predict(X)
        acc = np.mean((preds > 0.5) == y)
        logger.info(f"LightGBM model accuracy: {acc:.4f}")
        logger.info("✅ LightGBM GPU test PASSED")
        return True
    except ImportError:
        logger.warning("LightGBM not available, skipping test")
        return None
    except Exception as e:
        logger.error(f"❌ LightGBM GPU test FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test GPU acceleration capabilities')
    parser.add_argument('--rows', type=int, default=3000000, help='Number of rows for large dataset test')
    parser.add_argument('--cols', type=int, default=3000, help='Number of columns for large dataset test')
    parser.add_argument('--interactions', type=int, default=100, help='Max feature interactions')
    parser.add_argument('--skip-large', action='store_true', help='Skip large dataset test')
    parser.add_argument('--skip-h2o', action='store_true', help='Skip H2O XGBoost test')
    parser.add_argument('--skip-lgbm', action='store_true', help='Skip LightGBM GPU test')
    
    args = parser.parse_args()
    
    logger.info("Starting GPU acceleration tests")
    
    # Test GPU detection
    try:
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        for i in range(n_gpus):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode('utf-8')
            mem = props['totalGlobalMem'] / (1024**3)
            logger.info(f"GPU {i}: {name} with {mem:.1f} GB memory")
    except ImportError:
        logger.warning("CuPy not available, using Torch for GPU detection")
        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                for i in range(n_gpus):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU {i}: {name} with {mem:.1f} GB memory")
            else:
                logger.warning("No CUDA-compatible GPUs detected!")
        except ImportError:
            logger.error("Neither CuPy nor PyTorch available, cannot detect GPUs")
    
    # Run tests based on args
    results = {}
    
    if not args.skip_large:
        logger.info("Running large dataset test")
        results['large_dataset'] = test_large_dataset(args.rows, args.cols, args.interactions)
    
    if not args.skip_h2o:
        logger.info("Running H2O XGBoost test")
        results['h2o_xgboost'] = test_h2o_xgboost()
    
    if not args.skip_lgbm:
        logger.info("Running LightGBM GPU test")
        results['lightgbm_gpu'] = test_lightgbm_gpu()
    
    # Summarize results
    logger.info("\n===== TEST RESULTS =====")
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED (dependencies missing)"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    main()