#!/usr/bin/env python3
"""
Quick Yiedl Submission for Numerai Crypto

This script creates a high-quality submission for the Numerai Crypto competition using Yiedl data:
- Optimized for a 30-minute time constraint
- Focuses on a strong LightGBM model with GPU acceleration
- Uses minimal feature engineering for speed
- Saves predictions to a submission file

Usage:
    python quick_yiedl_submission.py [--output OUTPUT_FILE]
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import logging
import zipfile

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Install required dependencies
required_packages = ['lightgbm', 'pyarrow', 'scikit-learn', 'matplotlib']
for package in required_packages:
    try:
        __import__(package)
        print(f"{package} already installed")
    except ImportError:
        print(f"{package} not available. Installing...")
        os.system(f"pip3 install {package}")
        try:
            __import__(package)
            print(f"Successfully installed {package}")
        except ImportError:
            print(f"Failed to install {package}")

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM still not available after installation attempt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Quick Yiedl Submission for Numerai Crypto')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for predictions (defaults to timestamp)')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU acceleration if available')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    return parser.parse_args()

def check_gpu_availability():
    """Check if GPU is available for LightGBM"""
    if not LIGHTGBM_AVAILABLE:
        return False
    
    try:
        # Try to import optional GPU detection module
        import subprocess
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], 
                                                  stderr=subprocess.PIPE,
                                                  encoding='utf-8')
        return True
    except:
        pass
    
    # Try using LightGBM's device_type parameter
    try:
        # Create a dummy dataset
        data = np.random.rand(100, 10)
        label = np.random.randint(0, 2, 100)
        lgb_data = lgb.Dataset(data, label=label)
        
        # Try to train a tiny model with GPU
        params = {'device': 'gpu', 'gpu_device_id': 0}
        bst = lgb.train(params, lgb_data, num_boost_round=1)
        return True
    except Exception as e:
        logger.warning(f"GPU not available for LightGBM: {e}")
        return False

def load_yiedl_data():
    """Load Yiedl data from parquet and zip files"""
    yiedl_dir = project_root / "data" / "yiedl"
    latest_file = yiedl_dir / "yiedl_latest.parquet"
    historical_zip = yiedl_dir / "yiedl_historical.zip"
    
    logger.info("Loading Yiedl data...")
    
    # Load latest data
    if latest_file.exists():
        latest_df = pd.read_parquet(latest_file)
        logger.info(f"Loaded latest data: {latest_df.shape}")
    else:
        latest_df = None
        logger.warning("Latest data file not found")
    
    # Extract historical data from zip if needed
    historical_df = None
    if historical_zip.exists():
        tmp_dir = yiedl_dir / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Check if historical parquet already extracted
        historical_parquet = tmp_dir / "yiedl_historical.parquet"
        if historical_parquet.exists():
            historical_df = pd.read_parquet(historical_parquet)
            logger.info(f"Loaded historical data from extracted file: {historical_df.shape}")
        else:
            logger.info(f"Extracting historical data from zip file...")
            try:
                with zipfile.ZipFile(historical_zip, 'r') as zip_ref:
                    parquet_files = [f for f in zip_ref.namelist() if f.endswith('.parquet')]
                    if parquet_files:
                        zip_ref.extract(parquet_files[0], path=tmp_dir)
                        # Rename to expected name
                        os.rename(tmp_dir / parquet_files[0], historical_parquet)
                        historical_df = pd.read_parquet(historical_parquet)
                        logger.info(f"Loaded historical data: {historical_df.shape}")
                    else:
                        logger.warning("No parquet files found in historical zip")
            except Exception as e:
                logger.error(f"Error extracting historical data: {e}")
    else:
        logger.warning("Historical data zip not found")
    
    return latest_df, historical_df

def preprocess_data(df, is_training=True):
    """Preprocess the data for modeling"""
    if df is None:
        return None
    
    logger.info(f"Preprocessing {'training' if is_training else 'prediction'} data...")
    
    # Copy to avoid modifying original
    df = df.copy()
    
    # Check and handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.info(f"Found {missing_values} missing values")
        # Fill missing values with median
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna("unknown")
    
    # Identify feature columns (exclude ID, target, and any other non-feature columns)
    feature_cols = [col for col in df.columns if col not in ['id', 'target', 'era', 'data_type']]
    logger.info(f"Using {len(feature_cols)} base features")
    
    # Add a few simple engineered features
    df = add_basic_features(df, feature_cols)
    
    # Update feature columns list
    feature_cols = [col for col in df.columns if col not in ['id', 'target', 'era', 'data_type']]
    logger.info(f"Using {len(feature_cols)} features after engineering")
    
    return df, feature_cols

def add_basic_features(df, feature_cols):
    """Add some basic engineered features for better performance"""
    # Get numeric columns only
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return df
    
    # Add some ratio features between pairs of columns (limit to avoid explosion)
    added_count = 0
    for i in range(min(10, len(numeric_cols))):
        for j in range(i+1, min(i+5, len(numeric_cols))):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            # Ratio feature (handle division by zero)
            ratio_name = f"{col1}_div_{col2}"
            df[ratio_name] = df[col1] / (df[col2] + 1e-8)
            added_count += 1
            
            # Difference feature
            diff_name = f"{col1}_minus_{col2}"
            df[diff_name] = df[col1] - df[col2]
            added_count += 1
            
            # Break if we've added enough features
            if added_count >= 20:
                break
        if added_count >= 20:
            break
    
    # Add squared terms for a few features
    for i in range(min(5, len(numeric_cols))):
        col = numeric_cols[i]
        df[f"{col}_squared"] = df[col] ** 2
    
    return df

def train_optimized_model(train_df, feature_cols, use_gpu=False, seed=42):
    """Train an optimized LightGBM model for the 30-minute time constraint"""
    if not LIGHTGBM_AVAILABLE:
        logger.error("LightGBM not available. Cannot train model.")
        return None
    
    # Extract features and target
    X = train_df[feature_cols]
    y = train_df['target']
    
    # Perform a simple train/validation split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Set optimized parameters for speed and performance
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_bin': 255,
        'verbose': -1,
        'seed': seed
    }
    
    # Add GPU parameters if available and requested
    if use_gpu:
        try:
            logger.info("Attempting to use GPU for training...")
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        except Exception as e:
            logger.warning(f"Error setting GPU parameters: {e}")
    
    # Train the model (with early stopping for safety within time constraint)
    logger.info("Training LightGBM model...")
    start_time = time.time()
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        early_stopping_rounds=50
    )
    training_time = time.time() - start_time
    
    logger.info(f"Model trained in {training_time:.2f} seconds, best iteration: {bst.best_iteration}")
    
    # Get validation score
    val_pred = bst.predict(X_val)
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, val_pred)
    logger.info(f"Validation AUC: {val_auc:.4f}")
    
    return bst

def predict_and_save(model, test_df, feature_cols, output_file):
    """Generate predictions and save to output file"""
    if model is None or test_df is None:
        logger.error("Model or test data is None. Cannot generate predictions.")
        return None
    
    # Extract features
    X_test = test_df[feature_cols]
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': predictions
    })
    
    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
    
    return submission_df

def main():
    """Main function to run the quick Yiedl submission pipeline"""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Check if GPU is available and requested
    use_gpu = args.gpu and check_gpu_availability()
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Create output path
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(project_root, "data", "submissions"), exist_ok=True)
        args.output = os.path.join(project_root, "data", "submissions", f"yiedl_submission_{timestamp}.csv")
    
    # Load data
    latest_df, historical_df = load_yiedl_data()
    
    # Process data
    latest_processed, latest_features = preprocess_data(latest_df, is_training=False)
    historical_processed, historical_features = preprocess_data(historical_df, is_training=True)
    
    # Ensure feature consistency 
    if historical_processed is not None and latest_processed is not None:
        # Use the intersection of features to ensure consistency
        common_features = list(set(historical_features).intersection(set(latest_features)))
        logger.info(f"Using {len(common_features)} common features")
        
        # Train model
        model = train_optimized_model(
            historical_processed, 
            common_features, 
            use_gpu=use_gpu, 
            seed=args.random_seed
        )
        
        # Generate predictions
        submission_df = predict_and_save(
            model, 
            latest_processed, 
            common_features, 
            args.output
        )
        
        # Create a second submission with slight randomization
        if submission_df is not None:
            second_df = submission_df.copy()
            np.random.seed(args.random_seed + 123)
            noise = np.random.normal(0, 0.01, size=len(second_df))  # 1% noise
            second_df['prediction'] = np.clip(second_df['prediction'] + noise, 0, 1)
            
            # Save second submission
            second_output = os.path.splitext(args.output)[0] + "_v2.csv"
            second_df.to_csv(second_output, index=False)
            logger.info(f"Second submission saved to {second_output}")
    else:
        logger.error("Missing data. Need both historical and latest datasets.")
        return 1
    
    logger.info("Quick Yiedl submission complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())