import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import gc
import os

warnings.filterwarnings('ignore')

# Define paths
data_dir = Path("../data/yiedl")
latest_path = data_dir / "yiedl_latest.parquet"
historical_path = data_dir / "yiedl_historical.parquet"

# Check if data exists
print("Checking for data files...")
if latest_path.exists():
    print(f"Found latest dataset: {latest_path}")
else:
    print(f"Latest dataset not found at {latest_path}")
    exit(1)

# Load the data
print("\nLoading latest dataset...")
try:
    df_latest = pl.read_parquet(latest_path)
    print(f"Latest dataset loaded successfully. Shape: {df_latest.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Print basic info
print(f"\nLatest dataset shape: {df_latest.shape}")
print(f"\nLatest dataset columns preview: {df_latest.columns[:10]}")
print(f"Total columns: {len(df_latest.columns)}")

# Identify column types
def identify_column_groups(columns):
    pvm_cols = [col for col in columns if col.startswith('pvm_')]
    sentiment_cols = [col for col in columns if col.startswith('sentiment_')]
    onchain_cols = [col for col in columns if col.startswith('onchain_')]
    date_symbol_cols = ['date', 'symbol']
    other_cols = [col for col in columns if col not in pvm_cols + sentiment_cols + onchain_cols + date_symbol_cols]
    
    return {
        'pvm': pvm_cols,
        'sentiment': sentiment_cols,
        'onchain': onchain_cols,
        'date_symbol': date_symbol_cols,
        'other': other_cols
    }

# Identify column groups
column_groups = identify_column_groups(df_latest.columns)

# Print summary of column groups
print("Column Group Summary:")
for group, cols in column_groups.items():
    print(f"{group}: {len(cols)} columns")

# Convert to pandas for further processing (using a subset of columns)
columns_to_use = [
    *column_groups['date_symbol'], 
    *column_groups['pvm'][:15], 
    *column_groups['sentiment'][:15], 
    *column_groups['onchain'][:15]
]

print(f"\nUsing {len(columns_to_use)} columns for analysis")

# Convert to pandas DataFrame
print("Converting to pandas...")
df_pandas = df_latest.select(columns_to_use).to_pandas()
print(f"Pandas DataFrame shape: {df_pandas.shape}")

# Create synthetic target
print("\nCreating synthetic target variable...")
pvm_cols = column_groups['pvm']
if pvm_cols:
    # Use the first PVM column as a basis
    pvm_col = pvm_cols[0]
    # Convert to pandas for easier manipulation
    pvm_series = df_pandas[pvm_col] if pvm_col in df_pandas.columns else df_latest.select(pvm_col).to_pandas()[pvm_col]
    # Create target: 1 if value > median, 0 otherwise
    target = (pvm_series > pvm_series.median()).astype(int)
    df_pandas['target'] = target
    print(f"Created target based on {pvm_col}")
else:
    print("Using random target as fallback")
    df_pandas['target'] = np.random.randint(0, 2, size=len(df_pandas))

# Feature Engineering
print("\nPerforming feature engineering...")

def create_polynomial_features(df, feature_cols, degree=2, interaction_only=False, max_features=100):
    # Limit the number of features to avoid memory issues
    if len(feature_cols) > max_features:
        print(f"Limiting features from {len(feature_cols)} to {max_features}")
        feature_cols = feature_cols[:max_features]
    
    # Fill NaN values with 0 for the feature columns
    feature_df = df[feature_cols].copy().fillna(0)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(feature_df)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(feature_cols)
    
    # Convert to DataFrame
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    return poly_df

# Get features by category
pvm_features = [col for col in df_pandas.columns if col.startswith('pvm_')]
sentiment_features = [col for col in df_pandas.columns if col.startswith('sentiment_')]
onchain_features = [col for col in df_pandas.columns if col.startswith('onchain_')]

# Create polynomial features for each category
print("Creating polynomial features for PVM...")
pvm_poly_df = create_polynomial_features(
    df_pandas, pvm_features, degree=2, interaction_only=True, max_features=10
)

print("Creating polynomial features for sentiment...")
sentiment_poly_df = create_polynomial_features(
    df_pandas, sentiment_features, degree=2, interaction_only=True, max_features=10
)

# Combine all feature sets
all_features = pd.concat([df_pandas.drop(['date', 'symbol', 'target'], axis=1), 
                          pvm_poly_df, sentiment_poly_df], axis=1)

print(f"Total features after engineering: {all_features.shape[1]}")

# Train-test split
print("\nSplitting data into train and test sets...")
X = all_features
y = df_pandas['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Train XGBoost model with GPU acceleration if available
print("\nTraining XGBoost model...")
try:
    # Convert data to numpy arrays for XGBoost
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    
    # Try with GPU
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'seed': 42
    }
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dtest = xgb.DMatrix(X_test_np, label=y_test.values)
    
    # Train model
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    xgb_model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100,  # Reduced for testing
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=25
    )
    print("XGBoost model trained with GPU acceleration")
    
except Exception as e:
    print(f"GPU training failed with error: {e}")
    print("Falling back to CPU training...")
    
    # Convert data to numpy arrays for XGBoost
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'tree_method': 'hist',
        'seed': 42
    }
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dtest = xgb.DMatrix(X_test_np, label=y_test.values)
    
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    xgb_model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=25
    )

# Evaluate XGBoost model
y_pred_proba = xgb_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nXGBoost Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Train LightGBM model with GPU if available
print("\nTraining LightGBM model...")
try:
    # Try with GPU
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'seed': 42
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_np, label=y_train_np)
    test_data = lgb.Dataset(X_test_np, label=y_test.values, reference=train_data)
    
    # Set callback for early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=10)]
    
    # Train model - print progress manually
    print("Starting LightGBM training with GPU...")
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,  # Reduced for testing
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=callbacks
    )
    print("LightGBM model trained with GPU acceleration")
    
except Exception as e:
    print(f"GPU training failed with error: {e}")
    print("Falling back to CPU training...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'device': 'cpu',
        'seed': 42
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_np, label=y_train_np)
    test_data = lgb.Dataset(X_test_np, label=y_test.values, reference=train_data)
    
    # Set callback for early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=10)]
    
    # Train model - print progress manually
    print("Starting LightGBM training with CPU...")
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=callbacks
    )

# Evaluate LightGBM model
y_pred_proba_lgb = lgb_model.predict(X_test_np, num_iteration=lgb_model.best_iteration)
y_pred_lgb = (y_pred_proba_lgb > 0.5).astype(int)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)

print(f"\nLightGBM Model Performance:")
print(f"Accuracy: {accuracy_lgb:.4f}")
print(f"AUC: {auc_lgb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgb))

# Compare models
print("\nModel Performance Comparison:")
comparison = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Accuracy': [accuracy, accuracy_lgb],
    'AUC': [auc, auc_lgb]
})
print(comparison)

# Save models
print("\nSaving models...")
models_dir = Path("../models/yiedl")
models_dir.mkdir(exist_ok=True, parents=True)

# Save XGBoost model
model_path = models_dir / "xgboost_model.json"
xgb_model.save_model(model_path)
print(f"XGBoost model saved to {model_path}")

# Save LightGBM model
model_path = models_dir / "lightgbm_model.txt"
lgb_model.save_model(str(model_path))
print(f"LightGBM model saved to {model_path}")

print("\nAnalysis complete!")