#!/usr/bin/env python3
"""
Comprehensive Test of Crypto Ensemble Pipeline

This test demonstrates:
1. Batched feature engineering with GPU acceleration
2. GPU-accelerated ML model training (LightGBM, XGBoost, H2O AutoML)
3. Model ensembling for crypto prediction
4. Performance comparison between models

The test uses a subset of historical crypto data for 5 cryptocurrencies.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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

# Create a simple timer context manager
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

def download_sample_crypto_data(data_dir: str, crypto_symbols: List[str], force: bool = False) -> str:
    """
    Download or create sample crypto data for testing
    
    Args:
        data_dir: Directory to store data
        crypto_symbols: List of crypto symbols to include
        force: Force download/regeneration
        
    Returns:
        Path to the saved CSV file
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # File path for saved data
    output_file = os.path.join(data_dir, "sample_crypto_data.csv")
    
    # Check if file already exists
    if os.path.exists(output_file) and not force:
        logger.info(f"Using existing sample data: {output_file}")
        return output_file
    
    logger.info(f"Generating sample data for {crypto_symbols}")
    
    try:
        # Try to use yfinance for real data if available
        import yfinance as yf
        
        # Create empty dataframe to hold all data
        all_data = []
        
        # Download data for each symbol
        for symbol in crypto_symbols:
            # Add -USD suffix for crypto
            ticker = f"{symbol}-USD"
            logger.info(f"Downloading data for {ticker}")
            
            # Get 2 years of daily data
            df = yf.download(
                ticker, 
                period="2y",
                interval="1d"
            )
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Append to list
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved real crypto data to {output_file}")
        
    except ImportError:
        logger.info("yfinance not available, generating synthetic data")
        
        # Generate synthetic data
        np.random.seed(42)
        
        # Number of days (2 years)
        days = 730
        
        # Create date range
        start_date = datetime(2023, 1, 1)
        dates = [start_date + pd.Timedelta(days=i) for i in range(days)]
        
        # Generate data for each symbol
        all_data = []
        
        for symbol in crypto_symbols:
            # Start price between $10 and $50,000
            start_price = np.random.uniform(10, 50000)
            
            # Generate price series with realistic volatility
            price_series = [start_price]
            for i in range(1, days):
                # Daily return with 3% standard deviation (high for crypto)
                daily_return = np.random.normal(0.0005, 0.03)  # Slight upward bias
                price_series.append(price_series[-1] * (1 + daily_return))
            
            # Create dataframe
            df = pd.DataFrame({
                'Date': dates,
                'Open': price_series,
                'High': [p * (1 + np.random.uniform(0, 0.05)) for p in price_series],
                'Low': [p * (1 - np.random.uniform(0, 0.05)) for p in price_series],
                'Close': [p * (1 + np.random.normal(0, 0.02)) for p in price_series],
                'Adj Close': [p * (1 + np.random.normal(0, 0.02)) for p in price_series],
                'Volume': [np.random.uniform(1000, 10000000) for _ in range(days)],
                'symbol': symbol
            })
            
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved synthetic crypto data to {output_file}")
    
    return output_file

def prepare_features(data_path: str, batch_size: int = 10000) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare features from crypto data with batched processing
    
    Args:
        data_path: Path to crypto data CSV
        batch_size: Size of batches for feature engineering
        
    Returns:
        Tuple of (feature_df, feature_names, target_names)
    """
    logger.info(f"Preparing features from {data_path} with batch_size={batch_size}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'Date'])
    
    # Create lag features and rolling statistics for each symbol
    symbols = df['symbol'].unique()
    
    all_features = []
    
    for symbol in symbols:
        logger.info(f"Processing features for {symbol}")
        
        # Get data for this symbol
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Create basic price and volume features
        symbol_df['returns_1d'] = symbol_df['Close'].pct_change(1)
        symbol_df['returns_5d'] = symbol_df['Close'].pct_change(5)
        symbol_df['returns_10d'] = symbol_df['Close'].pct_change(10)
        symbol_df['returns_20d'] = symbol_df['Close'].pct_change(20)
        
        symbol_df['log_returns_1d'] = np.log(symbol_df['Close'] / symbol_df['Close'].shift(1))
        symbol_df['log_volume'] = np.log(symbol_df['Volume'])
        symbol_df['log_volume_change'] = symbol_df['log_volume'].diff()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            # Price stats
            symbol_df[f'rolling_mean_{window}d'] = symbol_df['Close'].rolling(window).mean()
            symbol_df[f'rolling_std_{window}d'] = symbol_df['Close'].rolling(window).std()
            symbol_df[f'rolling_max_{window}d'] = symbol_df['Close'].rolling(window).max()
            symbol_df[f'rolling_min_{window}d'] = symbol_df['Close'].rolling(window).min()
            
            # Price relative to rolling stats
            symbol_df[f'close_minus_mean_{window}d'] = symbol_df['Close'] - symbol_df[f'rolling_mean_{window}d']
            symbol_df[f'close_to_mean_{window}d'] = symbol_df['Close'] / symbol_df[f'rolling_mean_{window}d']
            symbol_df[f'close_to_max_{window}d'] = symbol_df['Close'] / symbol_df[f'rolling_max_{window}d']
            symbol_df[f'close_to_min_{window}d'] = symbol_df['Close'] / symbol_df[f'rolling_min_{window}d']
            
            # Volume stats
            symbol_df[f'volume_rolling_mean_{window}d'] = symbol_df['Volume'].rolling(window).mean()
            symbol_df[f'volume_rolling_std_{window}d'] = symbol_df['Volume'].rolling(window).std()
            symbol_df[f'volume_to_mean_{window}d'] = symbol_df['Volume'] / symbol_df[f'volume_rolling_mean_{window}d']
            
            # Volatility (std of returns)
            symbol_df[f'volatility_{window}d'] = symbol_df['returns_1d'].rolling(window).std()
            
            # Momentum indicators
            symbol_df[f'momentum_{window}d'] = symbol_df['Close'] / symbol_df['Close'].shift(window) - 1
        
        # Relative Strength Index (RSI)
        delta = symbol_df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        for window in [14, 30]:
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            
            # Calculate RSI
            rs = avg_gain / avg_loss
            symbol_df[f'rsi_{window}d'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        symbol_df['ema_12d'] = symbol_df['Close'].ewm(span=12, adjust=False).mean()
        symbol_df['ema_26d'] = symbol_df['Close'].ewm(span=26, adjust=False).mean()
        symbol_df['macd'] = symbol_df['ema_12d'] - symbol_df['ema_26d']
        symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9, adjust=False).mean()
        symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
        
        # Create target: next day's return
        symbol_df['target_return_1d'] = symbol_df['returns_1d'].shift(-1)
        symbol_df['target_return_5d'] = symbol_df['returns_5d'].shift(-5)
        symbol_df['target_direction_1d'] = (symbol_df['target_return_1d'] > 0).astype(int)
        symbol_df['target_direction_5d'] = (symbol_df['target_return_5d'] > 0).astype(int)
        
        # Keep only rows with sufficient history and a target
        symbol_df = symbol_df.dropna()
        
        # Add to list
        all_features.append(symbol_df)
    
    # Combine all features
    feature_df = pd.concat(all_features, ignore_index=True)
    
    # Convert symbol to categorical
    feature_df['symbol'] = feature_df['symbol'].astype('category')
    
    # Get feature and target names
    categorical_cols = ['symbol']
    target_cols = ['target_return_1d', 'target_return_5d', 
                  'target_direction_1d', 'target_direction_5d']
    
    # Select features (exclude date and target)
    exclude_cols = ['Date'] + target_cols
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    
    logger.info(f"Created {len(feature_cols)} base features")
    
    # Now apply GPU accelerated feature transformations in batches
    logger.info("Applying GPU accelerated transformations in batches")
    
    # Initialize GPU Math Accelerator
    os.environ["GPU_MEMORY_LIMIT"] = "12.0"  # Set conservative limit
    accelerator = GPUMathAccelerator()
    
    # Process numeric features with GPU accelerator in batches
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # Convert to numpy array for GPU processing
    numeric_data = feature_df[numeric_cols].values.astype(np.float32)
    
    # Apply transformations in batches
    n_samples = numeric_data.shape[0]
    transformed_batches = []
    transformed_names = None
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        logger.info(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: rows {i}-{end_idx}")
        
        # Get batch
        batch_data = numeric_data[i:end_idx]
        
        # Apply transformations
        with Timer(f"GPU transform batch {i//batch_size + 1}"):
            batch_result, batch_names = accelerator.generate_all_math_transforms(
                batch_data, numeric_cols,
                include_trig=False,  # Disable trig for speed
                include_poly=False,  # Disable poly for speed
                max_interactions=200,
                include_random_baselines=False
            )
        
        # Save batch results
        transformed_batches.append(batch_result)
        
        if transformed_names is None:
            transformed_names = batch_names
        
        # Clean up memory
        del batch_data
        del batch_result
    
    # Combine all batches
    with Timer("Combining transformed batches"):
        all_transformed = np.vstack(transformed_batches)
    
    logger.info(f"Created {all_transformed.shape[1]} additional features through transformations")
    
    # Create dataframe from transformed features
    transformed_df = pd.DataFrame(all_transformed, columns=transformed_names)
    
    # Add categorical columns (one-hot encoded)
    categorical_dummies = pd.get_dummies(feature_df[categorical_cols], prefix=categorical_cols)
    
    # Combine all features
    with Timer("Combining all features"):
        complete_features = pd.concat([
            feature_df[['Date'] + target_cols],  # Keep date and targets
            feature_df[numeric_cols],            # Original numeric features
            transformed_df,                      # Transformed features
            categorical_dummies                  # One-hot encoded categorical features
        ], axis=1)
    
    # Final feature list
    final_feature_cols = numeric_cols + transformed_names + list(categorical_dummies.columns)
    
    logger.info(f"Final dataset: {complete_features.shape[0]} rows, {complete_features.shape[1]} columns")
    
    return complete_features, final_feature_cols, target_cols

def train_test_split_by_time(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (latest data in test set)
    
    Args:
        df: Dataframe with 'Date' column
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date
    df = df.sort_values('Date')
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train data: {train_df.shape[0]} rows, Test data: {test_df.shape[0]} rows")
    logger.info(f"Train date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    logger.info(f"Test date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return train_df, test_df

def train_lgbm_model(train_df: pd.DataFrame, feature_cols: List[str], target: str, 
                    use_gpu: bool = True) -> Tuple[Any, Dict[str, float]]:
    """
    Train a LightGBM model with GPU acceleration
    
    Args:
        train_df: Training data
        feature_cols: Feature columns
        target: Target column
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info(f"Training LightGBM model for target {target}")
    
    try:
        import lightgbm as lgb
        
        # Extract features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Set parameters
        params = {
            'objective': 'regression' if 'return' in target else 'binary',
            'metric': 'rmse' if 'return' in target else 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Add GPU parameters if enabled
        if use_gpu:
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        # Train model
        with Timer(f"Training LightGBM for {target}" + (" with GPU" if use_gpu else " with CPU")):
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100
            )
        
        # Return model and some metrics
        metrics = {
            'feature_importance': dict(zip(
                feature_cols, 
                model.feature_importance(importance_type='gain')
            ))
        }
        
        return model, metrics
    
    except ImportError:
        logger.warning("LightGBM not available, skipping model")
        return None, {}

def train_xgboost_model(train_df: pd.DataFrame, feature_cols: List[str], target: str,
                       use_gpu: bool = True) -> Tuple[Any, Dict[str, float]]:
    """
    Train an XGBoost model with GPU acceleration
    
    Args:
        train_df: Training data
        feature_cols: Feature columns
        target: Target column
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info(f"Training XGBoost model for target {target}")
    
    try:
        import xgboost as xgb
        
        # Extract features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror' if 'return' in target else 'binary:logistic',
            'eval_metric': 'rmse' if 'return' in target else 'logloss',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1
        }
        
        # Add GPU parameters if enabled
        if use_gpu:
            params['gpu_id'] = 0
            params['tree_method'] = 'gpu_hist'
        
        # Train model
        with Timer(f"Training XGBoost for {target}" + (" with GPU" if use_gpu else " with CPU")):
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100
            )
        
        # Calculate feature importance
        importance = model.get_score(importance_type='gain')
        
        # Map importance to all features (some may not be used)
        feature_importance = {feature: importance.get(feature, 0) for feature in feature_cols}
        
        # Return model and metrics
        metrics = {
            'feature_importance': feature_importance
        }
        
        return model, metrics
    
    except ImportError:
        logger.warning("XGBoost not available, skipping model")
        return None, {}

def train_h2o_automl(train_df: pd.DataFrame, feature_cols: List[str], target: str,
                    use_gpu: bool = True, max_models: int = 10) -> Tuple[Any, Dict[str, float]]:
    """
    Train H2O AutoML models with GPU acceleration
    
    Args:
        train_df: Training data
        feature_cols: Feature columns
        target: Target column
        use_gpu: Whether to use GPU acceleration
        max_models: Maximum number of models to train
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info(f"Training H2O AutoML for target {target}")
    
    try:
        import h2o
        from h2o.automl import H2OAutoML
        
        # Initialize H2O
        h2o.init()
        
        # Prepare data subset (for speed)
        subset_size = min(100000, train_df.shape[0])
        train_subset = train_df.sample(subset_size, random_state=42)
        
        # Convert to H2O frame
        train_h2o = h2o.H2OFrame(train_subset[feature_cols + [target]])
        
        # Set feature and target names
        x = feature_cols
        y = target
        
        # Set column types
        if 'return' in target:
            train_h2o[y] = train_h2o[y].asfactor() if 'direction' in target else train_h2o[y].asnumeric()
        else:
            train_h2o[y] = train_h2o[y].asfactor()
        
        # Train AutoML
        with Timer(f"Training H2O AutoML for {target}" + (" with GPU" if use_gpu else " with CPU")):
            aml = H2OAutoML(
                max_models=max_models,
                seed=42,
                sort_metric="RMSE" if 'return' in target and 'direction' not in target else "AUC",
                max_runtime_secs=300  # Limit to 5 minutes for testing
            )
            
            aml.train(x=x, y=y, training_frame=train_h2o)
        
        # Get leaderboard
        lb = aml.leaderboard
        logger.info(f"AutoML Leaderboard for {target}:\n{lb.head(5)}")
        
        # Get best model
        best_model = aml.leader
        
        # Extract metrics
        metrics = {
            'leaderboard': lb.as_data_frame().head(5).to_dict(),
            'training_metrics': best_model.training_metrics().as_data_frame().to_dict()
        }
        
        return best_model, metrics
    
    except ImportError:
        logger.warning("H2O not available, skipping AutoML")
        return None, {}
    except Exception as e:
        logger.error(f"Error in H2O AutoML: {e}")
        return None, {}
    finally:
        try:
            h2o.cluster().shutdown()
        except:
            pass

def evaluate_model(model: Any, model_type: str, test_df: pd.DataFrame, 
                  feature_cols: List[str], target: str) -> Dict[str, float]:
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
        model_type: Type of model ('lgbm', 'xgboost', 'h2o')
        test_df: Test data
        feature_cols: Feature columns
        target: Target column
        
    Returns:
        Dictionary of metrics
    """
    if model is None:
        return {}
    
    logger.info(f"Evaluating {model_type} model for target {target}")
    
    try:
        # Extract features and target
        X_test = test_df[feature_cols]
        y_test = test_df[target].values
        
        # Make predictions
        if model_type == 'lgbm':
            import lightgbm as lgb
            y_pred = model.predict(X_test)
        elif model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)
        elif model_type == 'h2o':
            import h2o
            test_h2o = h2o.H2OFrame(X_test)
            preds = model.predict(test_h2o)
            pred_df = preds.as_data_frame()
            
            if 'return' in target and 'direction' not in target:
                # Regression
                y_pred = pred_df['predict'].values
            else:
                # Classification
                y_pred = pred_df['p1'].values  # Probability of positive class
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate metrics
        metrics = {}
        
        if 'direction' in target:
            # Binary classification metrics
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            
            # Convert to binary predictions
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
            metrics['auc'] = roc_auc_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred_binary)
            metrics['recall'] = recall_score(y_test, y_pred_binary)
            metrics['f1'] = f1_score(y_test, y_pred_binary)
            
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            # Directional accuracy (sign of prediction matches sign of actual)
            sign_accuracy = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(int))
            metrics['directional_accuracy'] = sign_accuracy
        
        # Log results
        for metric, value in metrics.items():
            logger.info(f"{model_type} {metric}: {value:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating {model_type} model: {e}")
        return {}

def create_ensemble_predictions(models: Dict[str, Any], model_types: Dict[str, str], 
                              test_df: pd.DataFrame, feature_cols: List[str], 
                              target: str) -> np.ndarray:
    """
    Create ensemble predictions from multiple models
    
    Args:
        models: Dictionary of trained models
        model_types: Dictionary of model types
        test_df: Test data
        feature_cols: Feature columns
        target: Target column
        
    Returns:
        Array of ensemble predictions
    """
    logger.info(f"Creating ensemble predictions for {target}")
    
    # Extract features
    X_test = test_df[feature_cols]
    
    all_predictions = []
    
    # Get predictions from each model
    for model_name, model in models.items():
        if model is None:
            continue
        
        model_type = model_types[model_name]
        
        try:
            # Make predictions
            if model_type == 'lgbm':
                import lightgbm as lgb
                y_pred = model.predict(X_test)
            elif model_type == 'xgboost':
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest)
            elif model_type == 'h2o':
                import h2o
                test_h2o = h2o.H2OFrame(X_test)
                preds = model.predict(test_h2o)
                pred_df = preds.as_data_frame()
                
                if 'return' in target and 'direction' not in target:
                    # Regression
                    y_pred = pred_df['predict'].values
                else:
                    # Classification
                    y_pred = pred_df['p1'].values  # Probability of positive class
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            # Add to list
            all_predictions.append(y_pred)
            logger.info(f"Added predictions from {model_name}")
            
        except Exception as e:
            logger.error(f"Error getting predictions from {model_name}: {e}")
    
    # If no predictions, return empty array
    if not all_predictions:
        return np.array([])
    
    # Stack predictions
    stacked_preds = np.column_stack(all_predictions)
    
    # Simple average ensemble
    ensemble_preds = np.mean(stacked_preds, axis=1)
    
    return ensemble_preds

def main():
    parser = argparse.ArgumentParser(description='Test Crypto Ensemble Pipeline')
    parser.add_argument('--data-dir', type=str, default='./data/test', help='Directory for data')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for feature engineering')
    parser.add_argument('--force-download', action='store_true', help='Force data download/generation')
    parser.add_argument('--skip-lgbm', action='store_true', help='Skip LightGBM model')
    parser.add_argument('--skip-xgboost', action='store_true', help='Skip XGBoost model')
    parser.add_argument('--skip-h2o', action='store_true', help='Skip H2O AutoML')
    parser.add_argument('--cpu-only', action='store_true', help='Use CPU only (no GPU)')
    
    args = parser.parse_args()
    
    # Define crypto symbols to use
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
    
    logger.info("Starting Crypto Ensemble Pipeline Test")
    logger.info(f"Using crypto symbols: {crypto_symbols}")
    
    # Step 1: Download or generate sample data
    data_path = download_sample_crypto_data(args.data_dir, crypto_symbols, args.force_download)
    
    # Step 2: Prepare features with batched GPU processing
    with Timer("Feature preparation"):
        feature_df, feature_cols, target_cols = prepare_features(data_path, args.batch_size)
    
    # Step 3: Split data
    train_df, test_df = train_test_split_by_time(feature_df)
    
    # Step 4: Train models
    # We'll focus on predicting next day's direction (classification)
    target = 'target_direction_1d'
    
    models = {}
    model_types = {}
    
    if not args.skip_lgbm:
        # Train LightGBM
        lgbm_model, lgbm_metrics = train_lgbm_model(
            train_df, feature_cols, target, use_gpu=not args.cpu_only
        )
        models['lgbm'] = lgbm_model
        model_types['lgbm'] = 'lgbm'
    
    if not args.skip_xgboost:
        # Train XGBoost
        xgb_model, xgb_metrics = train_xgboost_model(
            train_df, feature_cols, target, use_gpu=not args.cpu_only
        )
        models['xgboost'] = xgb_model
        model_types['xgboost'] = 'xgboost'
    
    if not args.skip_h2o:
        # Train H2O AutoML
        h2o_model, h2o_metrics = train_h2o_automl(
            train_df, feature_cols, target, use_gpu=not args.cpu_only
        )
        models['h2o'] = h2o_model
        model_types['h2o'] = 'h2o'
    
    # Step 5: Evaluate individual models
    model_metrics = {}
    
    for model_name, model in models.items():
        if model is None:
            continue
        
        metrics = evaluate_model(
            model, model_types[model_name], test_df, feature_cols, target
        )
        model_metrics[model_name] = metrics
    
    # Step 6: Create and evaluate ensemble
    if len(models) > 1:
        ensemble_preds = create_ensemble_predictions(
            models, model_types, test_df, feature_cols, target
        )
        
        # Evaluate ensemble
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        y_test = test_df[target].values
        ensemble_metrics = {}
        
        # Calculate metrics
        if len(ensemble_preds) > 0:
            ensemble_metrics['accuracy'] = accuracy_score(y_test, (ensemble_preds > 0.5).astype(int))
            ensemble_metrics['auc'] = roc_auc_score(y_test, ensemble_preds)
            
            # Log results
            logger.info(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.4f}")
            logger.info(f"Ensemble AUC: {ensemble_metrics['auc']:.4f}")
            
            # Compare to individual models
            logger.info("\nModel Comparison:")
            logger.info(f"{'Model':<10} {'Accuracy':<10} {'AUC':<10}")
            logger.info("-" * 30)
            
            for model_name, metrics in model_metrics.items():
                logger.info(f"{model_name:<10} {metrics.get('accuracy', 0):.4f}     {metrics.get('auc', 0):.4f}")
            
            logger.info(f"{'Ensemble':<10} {ensemble_metrics.get('accuracy', 0):.4f}     {ensemble_metrics.get('auc', 0):.4f}")
    
    logger.info("Crypto Ensemble Pipeline Test completed successfully")

if __name__ == "__main__":
    main()