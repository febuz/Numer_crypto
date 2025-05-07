#!/usr/bin/env python3
"""
Feature Selection Module for Numerai Crypto

This module implements various feature selection methods to identify
the most relevant features for prediction models. Results are saved
in the external feature store directory.
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualizations will be disabled.")

# H2O imports
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("H2O not available. Some functionality may be limited.")

# Constants
FSTORE_DIR = "/media/knight2/EDB/fstore"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects the most relevant features using various methods"""
    
    def __init__(self, output_dir=FSTORE_DIR):
        """
        Initialize the feature selector
        
        Args:
            output_dir (str): Directory to store feature selection results
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        self.run_dir = os.path.join(self.output_dir, f"feature_selection_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        logger.info(f"Feature Selector initialized. Output dir: {self.run_dir}")
    
    def load_data(self, input_path):
        """
        Load data for feature selection
        
        Args:
            input_path (str): Path to input parquet file
            
        Returns:
            DataFrame: Pandas DataFrame with loaded data
        """
        logger.info(f"Loading data from {input_path}")
        
        try:
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def initialize_h2o(self, max_mem_size="8G"):
        """
        Initialize H2O for feature selection
        
        Args:
            max_mem_size (str): Maximum memory size for H2O
        """
        if not H2O_AVAILABLE:
            logger.warning("H2O not available. Skipping initialization.")
            return False
        
        try:
            logger.info(f"Initializing H2O with max_mem_size={max_mem_size}")
            h2o.init(max_mem_size=max_mem_size)
            logger.info(f"H2O initialized successfully. Version: {h2o.__version__}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {str(e)}")
            return False
    
    def shutdown_h2o(self):
        """Shutdown H2O cluster"""
        if H2O_AVAILABLE:
            try:
                h2o.cluster().shutdown()
                logger.info("H2O cluster shutdown completed")
            except Exception as e:
                logger.warning(f"Error during H2O shutdown: {str(e)}")
    
    def select_features_correlation(self, df, target_col, top_n=500, threshold=0.0):
        """
        Select features based on correlation with target
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            top_n (int): Number of top features to select
            threshold (float): Minimum absolute correlation threshold
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Selecting features based on correlation with '{target_col}'")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        try:
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col != target_col]
            
            logger.info(f"Computing correlations for {len(numeric_cols)} numeric columns")
            
            # Compute correlations
            correlations = df[numeric_cols].corrwith(df[target_col]).abs()
            
            # Apply threshold and get top features
            correlations = correlations[correlations >= threshold].sort_values(ascending=False)
            
            selected_features = correlations.head(top_n).index.tolist()
            
            logger.info(f"Selected {len(selected_features)} features based on correlation")
            
            # Create correlation plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(12, 8))
                correlations.head(50).plot(kind='bar')
                plt.title(f'Top 50 Features Correlated with {target_col}')
                plt.xlabel('Feature')
                plt.ylabel('Absolute Correlation')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "correlation_feature_selection.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping correlation plot.")
            
            # Save correlations to CSV
            correlations.to_csv(os.path.join(self.run_dir, "feature_correlations.csv"))
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in correlation-based feature selection: {str(e)}")
            raise
    
    def select_features_h2o_automl(self, df, target_col, top_n=500, 
                                  max_runtime_secs=300, other_cols=None):
        """
        Select features using H2O AutoML feature importance
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            top_n (int): Number of top features to select
            max_runtime_secs (int): Maximum runtime for AutoML in seconds
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names
        """
        if not H2O_AVAILABLE:
            logger.warning("H2O not available. Skipping H2O-based feature selection.")
            return []
        
        logger.info(f"Selecting features using H2O AutoML (runtime: {max_runtime_secs}s)")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Initialize H2O if not initialized
        if not h2o.is_running():
            if not self.initialize_h2o():
                return []
        
        try:
            # Prepare other columns to keep
            if other_cols is None:
                other_cols = []
            
            # Get all columns except target and other cols
            feature_cols = [col for col in df.columns 
                          if col != target_col and col not in other_cols]
            
            # Convert to H2O frame
            logger.info("Converting data to H2O frame")
            h2o_df = h2o.H2OFrame(df)
            
            # Set column types
            h2o_df[target_col] = h2o_df[target_col].asfactor() if df[target_col].nunique() <= 30 else h2o_df[target_col]
            
            # Run AutoML
            logger.info("Running H2O AutoML")
            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                seed=42,
                max_models=10  # Limit number of models for feature selection
            )
            
            aml.train(y=target_col, training_frame=h2o_df)
            
            # Get feature importance from leader model
            logger.info("Getting feature importance from leader model")
            if aml.leader is None:
                logger.warning("No leader model found. Falling back to correlation-based selection.")
                return self.select_features_correlation(df, target_col, top_n)
            
            try:
                # Get variable importance
                varimp = aml.leader.varimp(use_pandas=True)
                
                if varimp is None or len(varimp) == 0:
                    logger.warning("No variable importance available. Falling back to correlation-based selection.")
                    return self.select_features_correlation(df, target_col, top_n)
                
                # Select top features
                selected_features = varimp['variable'].tolist()
                
                # Limit to top_n
                if len(selected_features) > top_n:
                    selected_features = selected_features[:top_n]
                
                logger.info(f"Selected {len(selected_features)} features using H2O AutoML")
                
                # Save variable importance to CSV
                varimp.to_csv(os.path.join(self.run_dir, "h2o_feature_importance.csv"), index=False)
                
                # Create importance plot if matplotlib is available
                if len(varimp) > 0 and MATPLOTLIB_AVAILABLE:
                    plt.figure(figsize=(12, 8))
                    data = varimp.head(50)
                    plt.barh(data['variable'], data['relative_importance'])
                    plt.title('Top 50 Features by H2O AutoML Importance')
                    plt.xlabel('Relative Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.run_dir, "h2o_feature_importance.png"))
                    plt.close()
                elif not MATPLOTLIB_AVAILABLE:
                    logger.info("Matplotlib not available. Skipping H2O importance plot.")
                
                return selected_features
            except Exception as e:
                logger.warning(f"Error getting variable importance: {str(e)}")
                logger.warning("Falling back to correlation-based selection.")
                return self.select_features_correlation(df, target_col, top_n)
        except Exception as e:
            logger.error(f"Error in H2O AutoML feature selection: {str(e)}")
            logger.warning("Falling back to correlation-based selection.")
            return self.select_features_correlation(df, target_col, top_n)
    
    def select_features_with_permutation_importance(self, df, target_col, 
                                                  model_type='rf', top_n=500, n_repeats=5,
                                                  other_cols=None):
        """
        Select features using permutation importance
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            model_type (str): Model type ('rf' for RandomForest, 'lgb' for LightGBM)
            top_n (int): Number of top features to select
            n_repeats (int): Number of times to repeat permutation
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Selecting features using permutation importance with {model_type}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, accuracy_score
            from sklearn.feature_selection import RFE
            from sklearn.decomposition import PCA
            
            # Prepare other columns to keep
            if other_cols is None:
                other_cols = []
            
            # Get all columns except target and other cols
            feature_cols = [col for col in df.columns 
                          if col != target_col and col not in other_cols]
            
            # Limit number of features to prevent memory issues
            if len(feature_cols) > 1000:
                logger.info(f"Too many features ({len(feature_cols)}). Using correlation pre-filtering.")
                # Pre-filter using correlation
                feature_cols = self.select_features_correlation(df, target_col, top_n=1000)
            
            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # Determine if classification or regression
            is_classification = df[target_col].nunique() <= 30  # Heuristic
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train model
            logger.info(f"Training {'classification' if is_classification else 'regression'} model")
            if model_type == 'rf':
                # Random Forest
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'lgb':
                # LightGBM
                try:
                    import lightgbm as lgb
                    if is_classification:
                        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                except ImportError:
                    logger.warning("LightGBM not available. Using Random Forest instead.")
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            if is_classification:
                score = accuracy_score(y_test, y_pred)
                logger.info(f"Model accuracy: {score:.4f}")
            else:
                score = r2_score(y_test, y_pred)
                logger.info(f"Model R² score: {score:.4f}")
            
            # Compute permutation importance
            logger.info(f"Computing permutation importance (n_repeats={n_repeats})")
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = importance_df['feature'].head(top_n).tolist()
            
            logger.info(f"Selected {len(selected_features)} features using permutation importance")
            
            # Save importance to CSV
            importance_df.to_csv(os.path.join(self.run_dir, "permutation_importance.csv"), index=False)
            
            # Create importance plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(12, 8))
                data = importance_df.head(50)
                plt.barh(data['feature'], data['importance'])
                plt.title('Top 50 Features by Permutation Importance')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "permutation_importance.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping permutation importance plot.")
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in permutation importance feature selection: {str(e)}")
            logger.warning("Falling back to correlation-based selection.")
            return self.select_features_correlation(df, target_col, top_n)
    
    def remove_highly_correlated_features(self, df, features, threshold=0.95):
        """
        Remove highly correlated features
        
        Args:
            df (DataFrame): Pandas DataFrame with features
            features (list): List of feature names to check
            threshold (float): Correlation threshold (0.0 to 1.0)
            
        Returns:
            list: Filtered feature names
        """
        logger.info(f"Removing highly correlated features (threshold={threshold})")
        
        try:
            # Create correlation matrix
            corr_matrix = df[features].corr().abs()
            
            # Create upper triangle matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            # Remove highly correlated features
            filtered_features = [f for f in features if f not in to_drop]
            
            logger.info(f"Removed {len(to_drop)} highly correlated features. Kept {len(filtered_features)}.")
            
            # Save dropped features to file
            with open(os.path.join(self.run_dir, "dropped_correlated_features.json"), 'w') as f:
                json.dump({
                    'threshold': threshold,
                    'dropped_features': to_drop,
                    'dropped_feature_count': len(to_drop),
                    'remaining_feature_count': len(filtered_features)
                }, f, indent=4)
            
            return filtered_features
        except Exception as e:
            logger.error(f"Error removing highly correlated features: {str(e)}")
            return features
    
    def select_features_rfe(self, df, target_col, model_type='rf', top_n=500, step=0.1, other_cols=None):
        """
        Select features using Recursive Feature Elimination
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            model_type (str): Model type ('rf' for RandomForest, 'lgb' for LightGBM)
            top_n (int): Number of top features to select
            step (float): Percentage of features to remove at each step
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Selecting features using Recursive Feature Elimination with {model_type}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.feature_selection import RFE
            from sklearn.model_selection import train_test_split
            
            # Prepare other columns to keep
            if other_cols is None:
                other_cols = []
            
            # Get all columns except target and other cols
            feature_cols = [col for col in df.columns 
                          if col != target_col and col not in other_cols]
            
            # Limit number of features to prevent memory issues
            if len(feature_cols) > 1000:
                logger.info(f"Too many features ({len(feature_cols)}). Using correlation pre-filtering.")
                # Pre-filter using correlation
                feature_cols = self.select_features_correlation(df, target_col, top_n=1000)
            
            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # Determine if classification or regression
            is_classification = df[target_col].nunique() <= 30  # Heuristic
            
            # Calculate step (number of features to remove at each iteration)
            n_features_to_select = min(top_n, len(feature_cols))
            step_value = max(1, int(len(feature_cols) * step))
            
            # Train model
            logger.info(f"Training {'classification' if is_classification else 'regression'} model for RFE")
            if model_type == 'rf':
                # Random Forest
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'lgb':
                # LightGBM
                try:
                    import lightgbm as lgb
                    if is_classification:
                        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                except ImportError:
                    logger.warning("LightGBM not available. Using Random Forest instead.")
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Apply RFE
            logger.info(f"Applying RFE to select {n_features_to_select} features with step {step_value}")
            rfe = RFE(estimator=model, 
                      n_features_to_select=n_features_to_select,
                      step=step_value, 
                      verbose=1)
            
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selected features
            selected_features = [feature_cols[i] for i, selected in enumerate(rfe.support_) if selected]
            
            logger.info(f"Selected {len(selected_features)} features using RFE")
            
            # Create ranking plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(12, 8))
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'rank': rfe.ranking_
                }).sort_values('rank')
                
                # Plot top 50 features
                data = feature_importance.head(50)
                plt.barh(data['feature'], 1/data['rank'])  # Inverse rank for visualization
                plt.title('Top 50 Features by RFE Ranking')
                plt.xlabel('Importance (1/rank)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "rfe_feature_ranking.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping RFE ranking plot.")
                
            # Save rankings to CSV regardless of matplotlib availability
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'rank': rfe.ranking_
            }).sort_values('rank')
            
            # Save rankings to CSV
            feature_importance.to_csv(os.path.join(self.run_dir, "rfe_feature_ranking.csv"), index=False)
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in RFE feature selection: {str(e)}")
            logger.warning("Falling back to correlation-based selection.")
            return self.select_features_correlation(df, target_col, top_n)
    
    def select_features_pca(self, df, target_col, n_components=None, variance_threshold=0.95, other_cols=None):
        """
        Select features using Principal Component Analysis
        
        Args:
            df (DataFrame): Pandas DataFrame with features
            target_col (str): Target column name (not used in transformation, only for filtering)
            n_components (int): Number of components to keep (if None, use variance_threshold)
            variance_threshold (float): Minimum explained variance to achieve
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names or component loadings
        """
        logger.info(f"Selecting features using PCA")
        
        try:
            # Prepare other columns to keep
            if other_cols is None:
                other_cols = []
                
            # Add target to other_cols if it exists
            if target_col in df.columns and target_col not in other_cols:
                other_cols.append(target_col)
            
            # Get feature columns (exclude other cols)
            feature_cols = [col for col in df.columns if col not in other_cols]
            
            logger.info(f"Applying PCA to {len(feature_cols)} features")
            
            # Prepare data - scale features for PCA
            X = df[feature_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create PCA model
            if n_components is None:
                # If not specified, use all components
                pca = PCA(n_components=min(len(feature_cols), len(df) - 1))
            else:
                # Use specified number of components
                pca = PCA(n_components=min(n_components, len(feature_cols), len(df) - 1))
            
            # Fit PCA
            pca.fit(X_scaled)
            
            # Determine number of components to keep based on variance threshold
            if n_components is None:
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
                logger.info(f"Selected {n_components_threshold} components to explain {variance_threshold:.2%} of variance")
            else:
                n_components_threshold = n_components
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                logger.info(f"Selected {n_components_threshold} components explaining {cumulative_variance[n_components_threshold-1]:.2%} of variance")
            
            # Get feature importance from component loadings
            components = pca.components_[:n_components_threshold]
            n_components_used = components.shape[0]
            
            # Calculate feature importance based on loadings
            feature_importance = np.abs(components).sum(axis=0) / n_components_used
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Select top features based on PCA importance
            top_n = min(500, len(feature_cols))  # Default to 500 max features
            selected_features = importance_df['feature'].head(top_n).tolist()
            
            logger.info(f"Selected {len(selected_features)} features using PCA importance")
            
            # Save feature importance to CSV
            importance_df.to_csv(os.path.join(self.run_dir, "pca_feature_importance.csv"), index=False)
            
            # Create variance explained plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                         np.cumsum(pca.explained_variance_ratio_), marker='o')
                plt.axhline(y=variance_threshold, color='r', linestyle='--')
                plt.axvline(x=n_components_threshold, color='g', linestyle='--')
                plt.title('Explained Variance by Components')
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')
                plt.grid(True)
                plt.savefig(os.path.join(self.run_dir, "pca_variance_explained.png"))
                plt.close()
                
                # Create feature importance plot
                plt.figure(figsize=(12, 8))
                data = importance_df.head(50)
                plt.barh(data['feature'], data['importance'])
                plt.title('Top 50 Features by PCA Importance')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "pca_feature_importance.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping PCA plots.")
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in PCA feature selection: {str(e)}")
            logger.warning("Falling back to correlation-based selection.")
            if target_col in df.columns:
                return self.select_features_correlation(df, target_col, top_n=500)
            else:
                # If no target column, return first 500 features
                return [col for col in df.columns if col not in other_cols][:500]
    
    def select_features_stability(self, df, target_col, model_type='rf', top_n=500, 
                                n_subsamples=100, subsample_size=0.75, other_cols=None):
        """
        Select features using stability selection
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            model_type (str): Model type ('rf' for RandomForest, 'lgb' for LightGBM)
            top_n (int): Number of top features to select
            n_subsamples (int): Number of subsamples to generate
            subsample_size (float): Size of each subsample as a fraction of the original data
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names
        """
        logger.info(f"Selecting features using stability selection with {model_type}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Prepare other columns to keep
            if other_cols is None:
                other_cols = []
            
            # Get all columns except target and other cols
            feature_cols = [col for col in df.columns 
                          if col != target_col and col not in other_cols]
            
            # Limit number of features to prevent memory issues
            if len(feature_cols) > 1000:
                logger.info(f"Too many features ({len(feature_cols)}). Using correlation pre-filtering.")
                # Pre-filter using correlation
                feature_cols = self.select_features_correlation(df, target_col, top_n=1000)
            
            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # Determine if classification or regression
            is_classification = df[target_col].nunique() <= 30  # Heuristic
            
            # Initialize feature importance counter
            feature_importance = {feature: 0 for feature in feature_cols}
            
            # Stability selection loop
            logger.info(f"Running stability selection with {n_subsamples} subsamples")
            for i in range(n_subsamples):
                if i % 10 == 0:
                    logger.info(f"Processing subsample {i+1}/{n_subsamples}")
                
                # Create subsample
                subsample_idx = np.random.choice(
                    range(len(df)), 
                    size=int(len(df) * subsample_size), 
                    replace=False
                )
                X_subsample = X.iloc[subsample_idx]
                y_subsample = y.iloc[subsample_idx]
                
                # Train model on subsample
                if model_type == 'rf':
                    # Random Forest
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
                elif model_type == 'lgb':
                    # LightGBM
                    try:
                        import lightgbm as lgb
                        if is_classification:
                            model = lgb.LGBMClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                        else:
                            model = lgb.LGBMRegressor(n_estimators=50, random_state=i, n_jobs=-1)
                    except ImportError:
                        logger.warning("LightGBM not available. Using Random Forest instead.")
                        if is_classification:
                            model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                        else:
                            model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Fit model
                model.fit(X_subsample, y_subsample)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Get top features for this subsample
                    top_indices = np.argsort(importances)[-top_n:]
                    
                    # Increment importance counter for selected features
                    for idx in top_indices:
                        feature_importance[feature_cols[idx]] += 1
            
            # Calculate stability scores (frequency of selection)
            stability_scores = {feature: count / n_subsamples 
                              for feature, count in feature_importance.items()}
            
            # Sort features by stability score
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, score in sorted_features[:top_n]]
            
            logger.info(f"Selected {len(selected_features)} features using stability selection")
            
            # Create stability score DataFrame
            stability_df = pd.DataFrame([
                {'feature': feature, 'stability_score': score} 
                for feature, score in sorted_features
            ])
            
            # Save stability scores to CSV
            stability_df.to_csv(os.path.join(self.run_dir, "stability_selection_scores.csv"), index=False)
            
            # Create stability score plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(12, 8))
                data = stability_df.head(50)
                plt.barh(data['feature'], data['stability_score'])
                plt.title('Top 50 Features by Stability Score')
                plt.xlabel('Stability Score (Selection Frequency)')
                plt.xlim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "stability_selection_scores.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping stability score plot.")
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in stability selection: {str(e)}")
            logger.warning("Falling back to correlation-based selection.")
            return self.select_features_correlation(df, target_col, top_n)
    
    def select_features_multi_method(self, df, target_col, top_n=500, 
                                   h2o_runtime=300, correlation_threshold=0.0,
                                   remove_correlated=True, other_cols=None):
        """
        Select features using multiple methods and combine results
        
        Args:
            df (DataFrame): Pandas DataFrame with features and target
            target_col (str): Target column name
            top_n (int): Number of top features to select
            h2o_runtime (int): Maximum runtime for H2O AutoML in seconds
            correlation_threshold (float): Minimum correlation threshold
            remove_correlated (bool): Whether to remove highly correlated features
            other_cols (list): Other columns to include (e.g., id, date)
            
        Returns:
            list: Selected feature names
            dict: Feature importance scores
        """
        logger.info(f"Selecting features using multiple methods for '{target_col}'")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        try:
            # Method 1: Correlation-based selection
            correlation_features = self.select_features_correlation(
                df, target_col, top_n=top_n, threshold=correlation_threshold
            )
            
            # Method 2: H2O AutoML-based selection (if available)
            h2o_features = []
            if H2O_AVAILABLE:
                h2o_features = self.select_features_h2o_automl(
                    df, target_col, top_n=top_n, max_runtime_secs=h2o_runtime, other_cols=other_cols
                )
            
            # Method 3: Permutation importance-based selection
            perm_features = self.select_features_with_permutation_importance(
                df, target_col, model_type='rf', top_n=top_n, other_cols=other_cols
            )
            
            # Method 4: Recursive Feature Elimination
            rfe_features = self.select_features_rfe(
                df, target_col, model_type='rf', top_n=top_n, other_cols=other_cols
            )
            
            # Method 5: PCA-based selection
            pca_features = self.select_features_pca(
                df, target_col, n_components=None, variance_threshold=0.95, other_cols=other_cols
            )
            
            # Method 6: Stability selection
            stability_features = self.select_features_stability(
                df, target_col, model_type='rf', top_n=top_n, 
                n_subsamples=50, subsample_size=0.75, other_cols=other_cols
            )
            
            # Combine results with weights
            feature_scores = {}
            
            # Assign scores based on rankings
            for i, feature in enumerate(correlation_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
            
            for i, feature in enumerate(h2o_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
            
            for i, feature in enumerate(perm_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
                
            for i, feature in enumerate(rfe_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
                
            for i, feature in enumerate(pca_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
                
            for i, feature in enumerate(stability_features):
                feature_scores[feature] = feature_scores.get(feature, 0) + (top_n - i) / top_n
            
            # Sort by score
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [f[0] for f in sorted_features[:top_n]]
            
            # Remove highly correlated features if requested
            if remove_correlated:
                selected_features = self.remove_highly_correlated_features(df, selected_features)
            
            logger.info(f"Selected {len(selected_features)} features using multiple methods")
            
            # Save feature scores to CSV
            score_df = pd.DataFrame([
                {'feature': feature, 'score': score} 
                for feature, score in sorted_features
            ])
            score_df.to_csv(os.path.join(self.run_dir, "combined_feature_scores.csv"), index=False)
            
            # Create feature scores plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(12, 8))
                data = score_df.head(50)
                plt.barh(data['feature'], data['score'])
                plt.title('Top 50 Features by Combined Score')
                plt.xlabel('Feature Score')
                plt.tight_layout()
                plt.savefig(os.path.join(self.run_dir, "combined_feature_scores.png"))
                plt.close()
            else:
                logger.info("Matplotlib not available. Skipping combined feature scores plot.")
            
            # Save selected features to JSON
            with open(os.path.join(self.run_dir, "selected_features.json"), 'w') as f:
                json.dump({
                    'timestamp': self.timestamp,
                    'target_column': target_col,
                    'feature_count': len(selected_features),
                    'features': selected_features
                }, f, indent=4)
            
            return selected_features, feature_scores
        except Exception as e:
            logger.error(f"Error in multi-method feature selection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to correlation method only
            logger.warning("Falling back to correlation-based selection only.")
            selected_features = self.select_features_correlation(df, target_col, top_n=top_n)
            
            # Create feature scores dictionary
            feature_scores = {feature: 1.0 for feature in selected_features}
            
            return selected_features, feature_scores


def main():
    """Main function to select features for Numerai Crypto"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Select features for Numerai Crypto')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input parquet file')
    parser.add_argument('--target', '-t', type=str, default='target', help='Target column name')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--top-n', '-n', type=int, default=500, help='Number of top features to select')
    parser.add_argument('--h2o-runtime', '-r', type=int, default=300, 
                        help='Maximum runtime for H2O AutoML in seconds')
    parser.add_argument('--correlation-threshold', '-c', type=float, default=0.0,
                        help='Minimum correlation threshold')
    parser.add_argument('--no-remove-correlated', action='store_true',
                        help='Do not remove highly correlated features')
    parser.add_argument('--methods', '-m', type=str, default='all',
                        help='Comma-separated list of selection methods to use (correlation,h2o,permutation,rfe,pca,stability,all)')
    parser.add_argument('--max-memory', type=str, default='16G',
                        help='Maximum memory to use for H2O')
    
    args = parser.parse_args()
    
    # Create feature selector
    selector = FeatureSelector()
    
    try:
        # Load data
        df = selector.load_data(args.input)
        
        # Select features
        other_cols = ['id', 'date', 'era', 'data_type', 'symbol']
        other_cols = [col for col in other_cols if col in df.columns]
        
        # Parse methods
        methods = args.methods.lower().split(',')
        use_all = 'all' in methods
        
        # Check if target column exists
        if args.target not in df.columns:
            logger.warning(f"Target column '{args.target}' not found. Available columns: {df.columns[:10]}...")
            # Try to find a suitable target column
            potential_targets = [col for col in df.columns if 'target' in col.lower()]
            if potential_targets:
                target_col = potential_targets[0]
                logger.info(f"Using '{target_col}' as target column")
            else:
                logger.error(f"No suitable target column found. Using PCA-based selection instead.")
                # Use PCA-based selection which doesn't require a target
                selected_features = selector.select_features_pca(
                    df=df,
                    target_col=None,
                    n_components=None,
                    variance_threshold=0.95,
                    other_cols=other_cols
                )
                feature_scores = {feature: 1.0 for feature in selected_features}
        else:
            # Initialize H2O if needed and available
            if use_all or 'h2o' in methods:
                if H2O_AVAILABLE:
                    selector.initialize_h2o(max_mem_size=args.max_memory)
            
            selected_features, feature_scores = selector.select_features_multi_method(
                df=df,
                target_col=args.target,
                top_n=args.top_n,
                h2o_runtime=args.h2o_runtime,
                correlation_threshold=args.correlation_threshold,
                remove_correlated=not args.no_remove_correlated,
                other_cols=other_cols
            )
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(selector.run_dir, "selected_features.json")
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': selector.timestamp,
                'target_column': args.target,
                'feature_count': len(selected_features),
                'features': selected_features
            }, f, indent=4)
        
        logger.info(f"Selected {len(selected_features)} features. Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown H2O if initialized
        selector.shutdown_h2o()
        
        # Print completion summary
        if 'selected_features' in locals() and selected_features:
            logger.info(f"Feature selection completed successfully.")
            logger.info(f"Selected {len(selected_features)} features from {len(df.columns)} total columns.")


if __name__ == "__main__":
    main()