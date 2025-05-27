#!/usr/bin/env python3
"""
DataGravitator module for intelligent ensemble model selection and combination.

This module provides the DataGravitator class for automatically selecting
and combining the best performing models based on various metrics and strategies.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

class DataGravitator:
    """
    DataGravitator class for intelligent model ensemble selection and combination.
    
    This class analyzes multiple model predictions and selects the best performing
    models based on various metrics like Information Coefficient (IC), Sharpe ratio,
    and other performance indicators.
    """
    
    def __init__(self, 
                 output_dir: str,
                 log_level: int = logging.INFO,
                 min_ic_threshold: float = 0.005,
                 min_sharpe_threshold: float = 0.3,
                 neutralize: bool = False,
                 tournament: str = 'crypto'):
        """
        Initialize DataGravitator.
        
        Args:
            output_dir: Directory for output files
            log_level: Logging level
            min_ic_threshold: Minimum IC threshold for model selection
            min_sharpe_threshold: Minimum Sharpe ratio threshold
            neutralize: Whether to neutralize features
            tournament: Tournament type ('crypto', 'signals', etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_ic_threshold = min_ic_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
        self.neutralize = neutralize
        self.tournament = tournament
        
        # Initialize attributes
        self.selected_signals = []
        self.processed_models = []
        self.model_metrics = {}
        self.ensemble_predictions = None
        
        # Setup logging
        self.logger = logging.getLogger(f'DataGravitator_{tournament}')
        self.logger.setLevel(log_level)
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_model_predictions(self, models_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load model predictions from directory.
        
        Args:
            models_dir: Directory containing model prediction files
            
        Returns:
            Dictionary mapping model names to prediction DataFrames
        """
        models_dir = Path(models_dir)
        model_predictions = {}
        
        # Look for prediction files
        prediction_files = list(models_dir.glob("*.csv")) + list(models_dir.glob("*.parquet"))
        
        for file_path in prediction_files:
            try:
                model_name = file_path.stem
                
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)
                
                # Ensure required columns exist
                if 'id' in df.columns and 'prediction' in df.columns:
                    model_predictions[model_name] = df
                    self.processed_models.append(model_name)
                    self.logger.info(f"Loaded model: {model_name} with {len(df)} predictions")
                else:
                    self.logger.warning(f"Skipping {file_path}: missing 'id' or 'prediction' columns")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(model_predictions)} model predictions")
        return model_predictions
    
    def calculate_ic(self, predictions: pd.Series, targets: pd.Series) -> float:
        """Calculate Information Coefficient (Spearman correlation)."""
        try:
            valid_mask = ~(predictions.isna() | targets.isna())
            if valid_mask.sum() < 10:  # Need at least 10 valid samples
                return 0.0
            
            corr, _ = spearmanr(predictions[valid_mask], targets[valid_mask])
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def calculate_sharpe(self, predictions: pd.Series, targets: pd.Series) -> float:
        """Calculate Sharpe ratio of predictions."""
        try:
            valid_mask = ~(predictions.isna() | targets.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            returns = predictions[valid_mask] * targets[valid_mask]
            if returns.std() == 0:
                return 0.0
            
            sharpe = returns.mean() / returns.std()
            return sharpe if not np.isnan(sharpe) else 0.0
        except:
            return 0.0
    
    def evaluate_models(self, 
                       model_predictions: Dict[str, pd.DataFrame],
                       target_col: str = 'target') -> Dict[str, Dict]:
        """
        Evaluate model performance using various metrics.
        
        Args:
            model_predictions: Dictionary of model predictions
            target_col: Column name for target values
            
        Returns:
            Dictionary of model metrics
        """
        metrics = {}
        
        # For crypto tournament, we might not have targets in live predictions
        # So we'll use proxy metrics based on prediction quality
        
        for model_name, df in model_predictions.items():
            try:
                predictions = df['prediction']
                
                # Basic prediction statistics
                model_metrics = {
                    'mean_prediction': predictions.mean(),
                    'std_prediction': predictions.std(),
                    'min_prediction': predictions.min(),
                    'max_prediction': predictions.max(),
                    'prediction_count': len(predictions),
                    'valid_predictions': predictions.notna().sum(),
                    'prediction_range': predictions.max() - predictions.min(),
                    'prediction_skew': predictions.skew() if len(predictions) > 3 else 0.0
                }
                
                # If targets are available, calculate correlation metrics
                if target_col in df.columns:
                    targets = df[target_col]
                    model_metrics.update({
                        'ic': self.calculate_ic(predictions, targets),
                        'sharpe': self.calculate_sharpe(predictions, targets),
                        'mse': mean_squared_error(targets[targets.notna()], 
                                                predictions[targets.notna()]) if targets.notna().sum() > 0 else 1.0
                    })
                else:
                    # For live predictions without targets, use proxy metrics
                    # Higher entropy/variance often indicates more informative predictions
                    model_metrics.update({
                        'ic': min(model_metrics['std_prediction'], 0.1),  # Proxy: higher std = better
                        'sharpe': model_metrics['prediction_range'] * 2,  # Proxy: wider range = better
                        'mse': 1.0 - model_metrics['std_prediction']  # Proxy: lower for higher std
                    })
                
                metrics[model_name] = model_metrics
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate model {model_name}: {e}")
                metrics[model_name] = {'ic': 0.0, 'sharpe': 0.0, 'mse': 1.0}
        
        self.model_metrics = metrics
        return metrics
    
    def select_models(self, 
                     model_metrics: Dict[str, Dict],
                     selection_method: str = 'combined_rank',
                     top_n: int = 5) -> List[str]:
        """
        Select best models based on metrics and selection method.
        
        Args:
            model_metrics: Model performance metrics
            selection_method: Method for selection ('combined_rank', 'threshold', 'pareto')
            top_n: Number of top models to select
            
        Returns:
            List of selected model names
        """
        if not model_metrics:
            return []
        
        if selection_method == 'threshold':
            # Select models above threshold
            selected = []
            for model_name, metrics in model_metrics.items():
                if (metrics.get('ic', 0) >= self.min_ic_threshold and 
                    metrics.get('sharpe', 0) >= self.min_sharpe_threshold):
                    selected.append(model_name)
            
            # If no models meet threshold, take top performers
            if not selected:
                selected = sorted(model_metrics.keys(), 
                                key=lambda x: model_metrics[x].get('ic', 0), 
                                reverse=True)[:top_n]
        
        elif selection_method == 'pareto':
            # Simple Pareto front selection based on IC and Sharpe
            models_list = list(model_metrics.keys())
            selected = []
            
            for model in models_list:
                ic = model_metrics[model].get('ic', 0)
                sharpe = model_metrics[model].get('sharpe', 0)
                
                is_dominated = False
                for other_model in models_list:
                    if model == other_model:
                        continue
                    
                    other_ic = model_metrics[other_model].get('ic', 0)
                    other_sharpe = model_metrics[other_model].get('sharpe', 0)
                    
                    # Check if current model is dominated
                    if other_ic >= ic and other_sharpe >= sharpe and (other_ic > ic or other_sharpe > sharpe):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    selected.append(model)
            
            # Limit to top_n
            selected = selected[:top_n]
        
        else:  # combined_rank (default)
            # Rank models by combined IC and Sharpe scores
            model_scores = {}
            for model_name, metrics in model_metrics.items():
                ic_score = metrics.get('ic', 0)
                sharpe_score = metrics.get('sharpe', 0)
                # Combined score with equal weighting
                combined_score = 0.5 * ic_score + 0.5 * sharpe_score
                model_scores[model_name] = combined_score
            
            # Select top models
            selected = sorted(model_scores.keys(), 
                            key=lambda x: model_scores[x], 
                            reverse=True)[:top_n]
        
        self.selected_signals = selected
        self.logger.info(f"Selected {len(selected)} models: {selected}")
        return selected
    
    def create_ensemble(self,
                       model_predictions: Dict[str, pd.DataFrame],
                       selected_models: List[str],
                       ensemble_method: str = 'mean_rank') -> pd.DataFrame:
        """
        Create ensemble predictions from selected models.
        
        Args:
            model_predictions: Dictionary of model predictions
            selected_models: List of selected model names
            ensemble_method: Method for combining predictions
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not selected_models:
            self.logger.warning("No models selected for ensemble")
            return pd.DataFrame()
        
        # Get predictions from selected models
        selected_predictions = {}
        for model_name in selected_models:
            if model_name in model_predictions:
                df = model_predictions[model_name]
                selected_predictions[model_name] = df.set_index('id')['prediction']
        
        if not selected_predictions:
            self.logger.warning("No valid predictions found for selected models")
            return pd.DataFrame()
        
        # Combine predictions
        pred_df = pd.DataFrame(selected_predictions)
        
        if ensemble_method == 'mean':
            ensemble_pred = pred_df.mean(axis=1)
        elif ensemble_method == 'median':
            ensemble_pred = pred_df.median(axis=1)
        elif ensemble_method == 'mean_weighted':
            # Weight by model performance (IC score)
            weights = []
            for model in selected_models:
                weight = max(self.model_metrics.get(model, {}).get('ic', 0.01), 0.01)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            ensemble_pred = (pred_df * weights).sum(axis=1)
        else:  # mean_rank (default)
            # Rank-based ensemble
            ranked_df = pred_df.rank(axis=0, method='average')
            ensemble_pred = ranked_df.mean(axis=1)
            # Normalize to [0, 1] range
            ensemble_pred = (ensemble_pred - ensemble_pred.min()) / (ensemble_pred.max() - ensemble_pred.min())
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'id': pred_df.index,
            'prediction': ensemble_pred.values
        }).reset_index(drop=True)
        
        self.ensemble_predictions = result_df
        self.logger.info(f"Created ensemble with {len(result_df)} predictions using {ensemble_method}")
        return result_df
    
    def save_submission(self, 
                       ensemble_df: pd.DataFrame,
                       output_path: Optional[str] = None) -> str:
        """
        Save ensemble predictions as submission file.
        
        Args:
            ensemble_df: DataFrame with ensemble predictions
            output_path: Optional custom output path
            
        Returns:
            Path to saved submission file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = self.output_dir / f"gravitator_submission_{timestamp}.csv"
        
        ensemble_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved submission to {output_path}")
        return str(output_path)
    
    def save_metrics_report(self) -> str:
        """
        Save detailed metrics report.
        
        Returns:
            Path to saved metrics report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.output_dir / f"metrics_report_{timestamp}.csv"
        
        # Convert metrics to DataFrame
        metrics_list = []
        for model_name, metrics in self.model_metrics.items():
            metrics_dict = {'model': model_name, **metrics}
            metrics_dict['selected'] = model_name in self.selected_signals
            metrics_list.append(metrics_dict)
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(metrics_path, index=False)
        
        self.logger.info(f"Saved metrics report to {metrics_path}")
        return str(metrics_path)
    
    def run_full_pipeline(self,
                         models_dir: str,
                         target_col: str = 'target',
                         ensemble_method: str = 'mean_rank',
                         selection_method: str = 'combined_rank',
                         top_n: int = 5,
                         auto_submit: bool = False,
                         today_only: bool = True,
                         include_live_universe: bool = True) -> str:
        """
        Run the complete DataGravitator pipeline.
        
        Args:
            models_dir: Directory with model predictions
            target_col: Target column name
            ensemble_method: Method for combining predictions
            selection_method: Method for selecting models
            top_n: Number of models to select
            auto_submit: Whether to auto-submit (not implemented)
            today_only: Whether to use only today's data
            include_live_universe: Whether to include all live universe symbols
            
        Returns:
            Path to generated submission file
        """
        self.logger.info("Starting DataGravitator full pipeline")
        
        # Load model predictions
        model_predictions = self.load_model_predictions(models_dir)
        
        if not model_predictions:
            raise ValueError(f"No valid model predictions found in {models_dir}")
        
        # Evaluate models
        model_metrics = self.evaluate_models(model_predictions, target_col)
        
        # Select best models
        selected_models = self.select_models(model_metrics, selection_method, top_n)
        
        if not selected_models:
            raise ValueError("No models selected based on criteria")
        
        # Create ensemble
        ensemble_df = self.create_ensemble(model_predictions, selected_models, ensemble_method)
        
        if ensemble_df.empty:
            raise ValueError("Failed to create ensemble predictions")
        
        # Save submission
        submission_path = self.save_submission(ensemble_df)
        
        self.logger.info(f"DataGravitator pipeline completed successfully")
        self.logger.info(f"Selected models: {selected_models}")
        self.logger.info(f"Submission saved to: {submission_path}")
        
        return submission_path