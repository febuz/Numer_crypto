"""
Metrics Database Module

This module provides a simple metrics database for tracking model and pipeline performance over time.
It allows storage, retrieval, and analysis of metrics to monitor long-term performance trends.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MetricsDB:
    """
    Metrics database for tracking model and pipeline performance.
    
    This class provides a simple SQLite-based database for storing and retrieving
    performance metrics. It supports various metric types including model performance,
    feature stability, and pipeline statistics.
    """
    
    def __init__(self, db_path: str = "/media/knight2/EDB/numer_crypto_temp/metrics/metrics.db"):
        """
        Initialize the metrics database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_dir = self.db_path.parent
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize database connection and tables
        self._init_db()
        
        logger.info(f"Metrics database initialized at {db_path}")
    
    def _init_db(self) -> None:
        """Initialize database connection and create tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            
            # Runs table for tracking pipeline runs
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                round_id TEXT,
                description TEXT,
                params TEXT,
                metrics TEXT
            )
            ''')
            
            # Models table for tracking model metrics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                run_id TEXT,
                model_name TEXT,
                model_type TEXT,
                timestamp TEXT,
                params TEXT,
                metrics TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            ''')
            
            # Features table for tracking feature metrics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp TEXT,
                feature_count INTEGER,
                stage TEXT,
                metrics TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            ''')
            
            # Feature importance table for tracking individual feature importance
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                importance_id TEXT PRIMARY KEY,
                model_id TEXT,
                feature_name TEXT,
                importance REAL,
                stability REAL,
                metrics TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
            ''')
            
            # Submissions table for tracking tournament submissions
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp TEXT,
                round_id TEXT,
                file_path TEXT,
                model_ids TEXT,
                metrics TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def record_run(self, 
                  run_id: Optional[str] = None,
                  round_id: Optional[str] = None,
                  description: str = "",
                  params: Dict[str, Any] = None,
                  metrics: Dict[str, Any] = None) -> str:
        """
        Record a pipeline run.
        
        Args:
            run_id: Unique identifier for the run (optional, will be generated if not provided)
            round_id: Tournament round ID (optional)
            description: Description of the run
            params: Dictionary of pipeline parameters
            metrics: Dictionary of pipeline metrics
            
        Returns:
            Run ID
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        timestamp = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO runs (run_id, timestamp, round_id, description, params, metrics) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    timestamp,
                    round_id,
                    description,
                    json.dumps(params or {}),
                    json.dumps(metrics or {})
                )
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded run {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Error recording run: {e}")
            return ""
    
    def record_model(self,
                    model_id: Optional[str] = None,
                    run_id: str = "",
                    model_name: str = "",
                    model_type: str = "",
                    params: Dict[str, Any] = None,
                    metrics: Dict[str, Any] = None) -> str:
        """
        Record model metrics.
        
        Args:
            model_id: Unique identifier for the model (optional, will be generated if not provided)
            run_id: Run ID this model is associated with
            model_name: Name of the model
            model_type: Type of model (e.g., 'lightgbm', 'xgboost')
            params: Dictionary of model parameters
            metrics: Dictionary of model metrics
            
        Returns:
            Model ID
        """
        if model_id is None:
            model_id = f"model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        timestamp = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO models (model_id, run_id, model_name, model_type, timestamp, params, metrics) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    run_id,
                    model_name,
                    model_type,
                    timestamp,
                    json.dumps(params or {}),
                    json.dumps(metrics or {})
                )
            )
            
            conn.commit()
            
            # If feature importance is provided, record it
            if metrics and 'feature_importance' in metrics:
                feature_importances = metrics['feature_importance']
                for feature_name, importance in feature_importances.items():
                    importance_id = f"imp_{model_id}_{feature_name}"
                    
                    # Get stability if available
                    stability = metrics.get('feature_stability', {}).get(feature_name, 0.0)
                    
                    # Additional metrics for this feature
                    feature_metrics = {
                        'model_id': model_id,
                        'feature_name': feature_name,
                        'importance': importance,
                        'stability': stability
                    }
                    
                    cursor.execute(
                        "INSERT INTO feature_importance (importance_id, model_id, feature_name, importance, stability, metrics) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            importance_id,
                            model_id,
                            feature_name,
                            float(importance),
                            float(stability),
                            json.dumps(feature_metrics)
                        )
                    )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded model {model_id} for run {run_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error recording model: {e}")
            return ""
    
    def record_features(self,
                       feature_id: Optional[str] = None,
                       run_id: str = "",
                       feature_count: int = 0,
                       stage: str = "",
                       metrics: Dict[str, Any] = None) -> str:
        """
        Record feature metrics.
        
        Args:
            feature_id: Unique identifier for the feature set (optional, will be generated if not provided)
            run_id: Run ID this feature set is associated with
            feature_count: Number of features in the set
            stage: Processing stage (e.g., 'initial', 'reduced')
            metrics: Dictionary of feature metrics
            
        Returns:
            Feature ID
        """
        if feature_id is None:
            feature_id = f"feat_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        timestamp = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO features (feature_id, run_id, timestamp, feature_count, stage, metrics) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    feature_id,
                    run_id,
                    timestamp,
                    feature_count,
                    stage,
                    json.dumps(metrics or {})
                )
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded features {feature_id} for run {run_id}")
            return feature_id
            
        except Exception as e:
            logger.error(f"Error recording features: {e}")
            return ""
    
    def record_submission(self,
                         submission_id: Optional[str] = None,
                         run_id: str = "",
                         round_id: Optional[str] = None,
                         file_path: str = "",
                         model_ids: List[str] = None,
                         metrics: Dict[str, Any] = None) -> str:
        """
        Record a tournament submission.
        
        Args:
            submission_id: Unique identifier for the submission (optional, will be generated if not provided)
            run_id: Run ID this submission is associated with
            round_id: Tournament round ID (optional)
            file_path: Path to submission file
            model_ids: List of model IDs used in the submission
            metrics: Dictionary of submission metrics
            
        Returns:
            Submission ID
        """
        if submission_id is None:
            submission_id = f"sub_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        timestamp = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO submissions (submission_id, run_id, timestamp, round_id, file_path, model_ids, metrics) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    run_id,
                    timestamp,
                    round_id,
                    file_path,
                    json.dumps(model_ids or []),
                    json.dumps(metrics or {})
                )
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded submission {submission_id} for run {run_id}")
            return submission_id
            
        except Exception as e:
            logger.error(f"Error recording submission: {e}")
            return ""
    
    def get_runs(self, 
                limit: int = 10, 
                offset: int = 0,
                round_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent pipeline runs.
        
        Args:
            limit: Maximum number of runs to return
            offset: Offset for pagination
            round_id: Filter by tournament round ID (optional)
            
        Returns:
            List of run dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if round_id:
                cursor.execute(
                    "SELECT * FROM runs WHERE round_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (round_id, limit, offset)
                )
            else:
                cursor.execute(
                    "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert rows to dictionaries
            runs = []
            for row in rows:
                run_dict = dict(row)
                
                # Parse JSON fields
                run_dict['params'] = json.loads(run_dict['params'])
                run_dict['metrics'] = json.loads(run_dict['metrics'])
                
                runs.append(run_dict)
            
            return runs
            
        except Exception as e:
            logger.error(f"Error getting runs: {e}")
            return []
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID.
        
        Args:
            run_id: Run ID to retrieve
            
        Returns:
            Run dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dictionary
            run_dict = dict(row)
            
            # Parse JSON fields
            run_dict['params'] = json.loads(run_dict['params'])
            run_dict['metrics'] = json.loads(run_dict['metrics'])
            
            # Get associated models
            cursor.execute("SELECT * FROM models WHERE run_id = ?", (run_id,))
            model_rows = cursor.fetchall()
            
            models = []
            for model_row in model_rows:
                model_dict = dict(model_row)
                model_dict['params'] = json.loads(model_dict['params'])
                model_dict['metrics'] = json.loads(model_dict['metrics'])
                models.append(model_dict)
            
            run_dict['models'] = models
            
            # Get associated features
            cursor.execute("SELECT * FROM features WHERE run_id = ?", (run_id,))
            feature_rows = cursor.fetchall()
            
            features = []
            for feature_row in feature_rows:
                feature_dict = dict(feature_row)
                feature_dict['metrics'] = json.loads(feature_dict['metrics'])
                features.append(feature_dict)
            
            run_dict['features'] = features
            
            # Get associated submissions
            cursor.execute("SELECT * FROM submissions WHERE run_id = ?", (run_id,))
            submission_rows = cursor.fetchall()
            
            submissions = []
            for submission_row in submission_rows:
                submission_dict = dict(submission_row)
                submission_dict['model_ids'] = json.loads(submission_dict['model_ids'])
                submission_dict['metrics'] = json.loads(submission_dict['metrics'])
                submissions.append(submission_dict)
            
            run_dict['submissions'] = submissions
            
            conn.close()
            return run_dict
            
        except Exception as e:
            logger.error(f"Error getting run {run_id}: {e}")
            return None
    
    def get_models(self, 
                  run_id: Optional[str] = None, 
                  model_type: Optional[str] = None,
                  limit: int = 10, 
                  offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get models with filtering options.
        
        Args:
            run_id: Filter by run ID (optional)
            model_type: Filter by model type (optional)
            limit: Maximum number of models to return
            offset: Offset for pagination
            
        Returns:
            List of model dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM models"
            params = []
            
            # Build query based on filters
            conditions = []
            if run_id:
                conditions.append("run_id = ?")
                params.append(run_id)
            
            if model_type:
                conditions.append("model_type = ?")
                params.append(model_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert rows to dictionaries
            models = []
            for row in rows:
                model_dict = dict(row)
                
                # Parse JSON fields
                model_dict['params'] = json.loads(model_dict['params'])
                model_dict['metrics'] = json.loads(model_dict['metrics'])
                
                models.append(model_dict)
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model by ID.
        
        Args:
            model_id: Model ID to retrieve
            
        Returns:
            Model dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dictionary
            model_dict = dict(row)
            
            # Parse JSON fields
            model_dict['params'] = json.loads(model_dict['params'])
            model_dict['metrics'] = json.loads(model_dict['metrics'])
            
            # Get feature importances
            cursor.execute("SELECT * FROM feature_importance WHERE model_id = ?", (model_id,))
            importance_rows = cursor.fetchall()
            
            importances = {}
            for imp_row in importance_rows:
                feature_name = imp_row['feature_name']
                importance = imp_row['importance']
                importances[feature_name] = importance
            
            model_dict['feature_importances'] = importances
            
            conn.close()
            return model_dict
            
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_feature_history(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature count history.
        
        Args:
            run_id: Filter by run ID (optional)
            
        Returns:
            DataFrame with feature history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM features"
            params = []
            
            if run_id:
                query += " WHERE run_id = ?"
                params.append(run_id)
                
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Parse metrics JSON
            df['metrics_parsed'] = df['metrics'].apply(json.loads)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting feature history: {e}")
            return pd.DataFrame()
    
    def get_model_performance_history(self, 
                                     model_type: Optional[str] = None,
                                     metric_name: str = 'rmse') -> pd.DataFrame:
        """
        Get model performance history.
        
        Args:
            model_type: Filter by model type (optional)
            metric_name: Name of the metric to track
            
        Returns:
            DataFrame with model performance history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT model_id, model_name, model_type, timestamp, metrics FROM models"
            params = []
            
            if model_type:
                query += " WHERE model_type = ?"
                params.append(model_type)
                
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Extract metric from JSON
            df['metrics_parsed'] = df['metrics'].apply(json.loads)
            df[metric_name] = df['metrics_parsed'].apply(
                lambda x: x.get(metric_name, x.get('metrics', {}).get(metric_name, None))
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting model performance history: {e}")
            return pd.DataFrame()
    
    def plot_model_performance(self, 
                              model_types: List[str] = None, 
                              metric_name: str = 'rmse',
                              output_path: Optional[str] = None) -> str:
        """
        Plot model performance history.
        
        Args:
            model_types: List of model types to include (optional)
            metric_name: Name of the metric to plot
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if model_types is None:
            model_types = ['lightgbm', 'xgboost', 'catboost', 'tabnet']
        
        try:
            # Get data for each model type
            dfs = []
            for model_type in model_types:
                df = self.get_model_performance_history(model_type, metric_name)
                if not df.empty:
                    dfs.append(df)
            
            if not dfs:
                logger.warning("No data available for plotting")
                return ""
                
            # Create plot
            plt.figure(figsize=(12, 6))
            
            for df in dfs:
                model_type = df['model_type'].iloc[0]
                plt.plot(df['timestamp'], df[metric_name], 'o-', label=model_type)
            
            plt.title(f'Model {metric_name.upper()} Over Time')
            plt.xlabel('Date')
            plt.ylabel(metric_name.upper())
            plt.legend()
            plt.grid(True)
            
            # Save plot
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(self.db_dir / f"model_performance_{metric_name}_{timestamp}.png")
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Model performance plot saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error plotting model performance: {e}")
            return ""
    
    def plot_feature_importance(self, 
                               model_id: str,
                               top_n: int = 20,
                               output_path: Optional[str] = None) -> str:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_id: Model ID to plot feature importance for
            top_n: Number of top features to include
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get model info
            cursor.execute("SELECT model_name, model_type FROM models WHERE model_id = ?", (model_id,))
            model_row = cursor.fetchone()
            
            if not model_row:
                logger.warning(f"Model {model_id} not found")
                return ""
                
            model_name, model_type = model_row
            
            # Get feature importances
            cursor.execute(
                "SELECT feature_name, importance, stability FROM feature_importance WHERE model_id = ? ORDER BY importance DESC LIMIT ?",
                (model_id, top_n)
            )
            importance_rows = cursor.fetchall()
            
            if not importance_rows:
                logger.warning(f"No feature importance data for model {model_id}")
                return ""
                
            conn.close()
            
            # Prepare data for plotting
            features = [row[0] for row in importance_rows]
            importances = [row[1] for row in importance_rows]
            stabilities = [row[2] for row in importance_rows]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Importance plot
            y_pos = np.arange(len(features))
            ax1.barh(y_pos, importances)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_title(f'Feature Importance for {model_name} ({model_type})')
            ax1.set_xlabel('Importance')
            
            # Stability plot (if available)
            if any(s > 0 for s in stabilities):
                ax2.barh(y_pos, stabilities)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(features)
                ax2.set_title(f'Feature Stability for {model_name} ({model_type})')
                ax2.set_xlabel('Stability (higher is better)')
            else:
                ax2.set_visible(False)
            
            # Save plot
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(self.db_dir / f"feature_importance_{model_id}_{timestamp}.png")
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Feature importance plot saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return ""
    
    def generate_performance_report(self, 
                                   run_id: Optional[str] = None,
                                   output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            run_id: Run ID to report on (optional, uses latest if not provided)
            output_path: Path to save report (optional)
            
        Returns:
            Path to saved report
        """
        try:
            # Get run data
            if run_id:
                run = self.get_run(run_id)
                if not run:
                    logger.warning(f"Run {run_id} not found")
                    return ""
            else:
                # Get latest run
                runs = self.get_runs(limit=1)
                if not runs:
                    logger.warning("No runs found")
                    return ""
                run = runs[0]
                run_id = run['run_id']
            
            # Get model data
            models = self.get_models(run_id=run_id)
            
            # Get feature data
            feature_history = self.get_feature_history(run_id=run_id)
            
            # Define report path
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(self.db_dir / f"performance_report_{run_id}_{timestamp}.html")
            
            # Generate HTML report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Report - Run {run_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 10px; margin: 10px; min-width: 200px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Performance Report</h1>
                <p>Run ID: {run_id}</p>
                <p>Timestamp: {run['timestamp']}</p>
                <p>Round ID: {run['round_id'] or 'N/A'}</p>
                <p>Description: {run['description']}</p>
                
                <h2>Pipeline Metrics</h2>
                <div class="metrics">
            """
            
            # Add pipeline metrics
            for metric_name, metric_value in run['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-name">{metric_name}</div>
                        <div class="metric-value">{metric_value:.4f}</div>
                    </div>
                    """
            
            html += """
                </div>
                
                <h2>Models</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Type</th>
                        <th>RMSE</th>
                        <th>R²</th>
                        <th>Features</th>
                    </tr>
            """
            
            # Add model rows
            for model in models:
                metrics = model['metrics']
                rmse = metrics.get('rmse', metrics.get('metrics', {}).get('rmse', 'N/A'))
                r2 = metrics.get('r2', metrics.get('metrics', {}).get('r2', 'N/A'))
                n_features = metrics.get('n_features', 'N/A')
                
                html += f"""
                <tr>
                    <td>{model['model_name']}</td>
                    <td>{model['model_type']}</td>
                    <td>{rmse if isinstance(rmse, str) else f'{rmse:.4f}'}</td>
                    <td>{r2 if isinstance(r2, str) else f'{r2:.4f}'}</td>
                    <td>{n_features}</td>
                </tr>
                """
            
            html += """
                </table>
                
                <h2>Feature Evolution</h2>
            """
            
            # Add feature evolution if available
            if not feature_history.empty:
                # Create feature count plot
                plt.figure(figsize=(10, 6))
                plt.plot(feature_history['timestamp'], feature_history['feature_count'], 'o-')
                plt.title('Feature Count Evolution')
                plt.xlabel('Time')
                plt.ylabel('Feature Count')
                plt.grid(True)
                
                # Save plot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                feature_plot = str(self.db_dir / f"feature_evolution_{run_id}_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(feature_plot)
                plt.close()
                
                # Add plot to report
                feature_plot_filename = os.path.basename(feature_plot)
                html += f'<img src="{feature_plot_filename}" alt="Feature Evolution">\n'
                
                # Feature table
                html += """
                <table>
                    <tr>
                        <th>Stage</th>
                        <th>Feature Count</th>
                        <th>Timestamp</th>
                    </tr>
                """
                
                for _, row in feature_history.iterrows():
                    html += f"""
                    <tr>
                        <td>{row['stage']}</td>
                        <td>{row['feature_count']}</td>
                        <td>{row['timestamp']}</td>
                    </tr>
                    """
                
                html += "</table>"
            else:
                html += "<p>No feature evolution data available.</p>"
            
            # Feature importance plots for each model
            html += "<h2>Feature Importance</h2>"
            
            for model in models:
                model_id = model['model_id']
                # Create feature importance plot
                importance_plot = self.plot_feature_importance(model_id)
                
                if importance_plot:
                    importance_plot_filename = os.path.basename(importance_plot)
                    html += f"""
                    <h3>Feature Importance - {model['model_name']} ({model['model_type']})</h3>
                    <img src="{importance_plot_filename}" alt="Feature Importance">
                    """
            
            # Performance comparison with previous runs
            html += """
                <h2>Performance Comparison</h2>
            """
            
            # Get model performance history
            model_types = list(set(model['model_type'] for model in models))
            performance_plot = self.plot_model_performance(model_types=model_types)
            
            if performance_plot:
                performance_plot_filename = os.path.basename(performance_plot)
                html += f'<img src="{performance_plot_filename}" alt="Model Performance">\n'
            else:
                html += "<p>No performance history available for comparison.</p>"
            
            html += """
            </body>
            </html>
            """
            
            # Write HTML report
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Performance report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return ""
    
    def get_feature_importance_trend(self, 
                                    feature_name: str,
                                    model_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get trend of a feature's importance over time.
        
        Args:
            feature_name: Name of the feature to track
            model_type: Filter by model type (optional)
            
        Returns:
            DataFrame with feature importance trend
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = """
            SELECT m.model_id, m.model_name, m.model_type, m.timestamp, 
                   fi.feature_name, fi.importance, fi.stability
            FROM feature_importance fi
            JOIN models m ON fi.model_id = m.model_id
            WHERE fi.feature_name = ?
            """
            params = [feature_name]
            
            if model_type:
                query += " AND m.model_type = ?"
                params.append(model_type)
                
            query += " ORDER BY m.timestamp"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append(dict(row))
            
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting feature importance trend: {e}")
            return pd.DataFrame()
    
    def plot_feature_importance_trend(self, 
                                     feature_names: List[str],
                                     model_type: Optional[str] = None,
                                     output_path: Optional[str] = None) -> str:
        """
        Plot trend of feature importance over time.
        
        Args:
            feature_names: List of feature names to track
            model_type: Filter by model type (optional)
            output_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            for feature_name in feature_names:
                df = self.get_feature_importance_trend(feature_name, model_type)
                
                if not df.empty:
                    plt.plot(df['timestamp'], df['importance'], 'o-', label=feature_name)
            
            plt.title(f'Feature Importance Trend {f"for {model_type}" if model_type else ""}')
            plt.xlabel('Date')
            plt.ylabel('Importance')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(self.db_dir / f"feature_trend_{timestamp}.png")
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Feature importance trend plot saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error plotting feature importance trend: {e}")
            return ""


class PipelineMetricsCollector:
    """
    Metrics collector that integrates with the pipeline to track performance.
    
    This class provides a simple interface for collecting and recording metrics
    during pipeline execution, with methods to track each stage of the pipeline.
    """
    
    def __init__(self, 
                db_path: str = "/media/knight2/EDB/numer_crypto_temp/metrics/metrics.db",
                run_id: Optional[str] = None,
                round_id: Optional[str] = None,
                description: str = ""):
        """
        Initialize the metrics collector.
        
        Args:
            db_path: Path to metrics database
            run_id: Unique identifier for the run (optional, will be generated if not provided)
            round_id: Tournament round ID (optional)
            description: Description of the run
        """
        self.db = MetricsDB(db_path)
        
        # Initialize or generate run ID
        if run_id is None:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        else:
            self.run_id = run_id
            
        self.round_id = round_id
        self.description = description
        
        # Initialize metrics containers
        self.pipeline_metrics = {}
        self.pipeline_params = {}
        
        # Record initial run
        self.db.record_run(
            run_id=self.run_id,
            round_id=self.round_id,
            description=self.description
        )
        
        logger.info(f"PipelineMetricsCollector initialized with run ID: {self.run_id}")
    
    def record_pipeline_params(self, params: Dict[str, Any]) -> None:
        """
        Record pipeline parameters.
        
        Args:
            params: Dictionary of pipeline parameters
        """
        self.pipeline_params.update(params)
        
        # Update run record
        self.db.record_run(
            run_id=self.run_id,
            round_id=self.round_id,
            description=self.description,
            params=self.pipeline_params,
            metrics=self.pipeline_metrics
        )
        
        logger.info(f"Recorded pipeline parameters for run {self.run_id}")
    
    def record_pipeline_metric(self, metric_name: str, metric_value: Any) -> None:
        """
        Record a pipeline metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        self.pipeline_metrics[metric_name] = metric_value
        
        # Update run record
        self.db.record_run(
            run_id=self.run_id,
            round_id=self.round_id,
            description=self.description,
            params=self.pipeline_params,
            metrics=self.pipeline_metrics
        )
        
        logger.info(f"Recorded pipeline metric {metric_name}={metric_value} for run {self.run_id}")
    
    def record_feature_metrics(self, 
                             feature_count: int,
                             stage: str,
                             metrics: Dict[str, Any] = None) -> str:
        """
        Record feature metrics.
        
        Args:
            feature_count: Number of features
            stage: Processing stage (e.g., 'initial', 'reduced')
            metrics: Dictionary of feature metrics (optional)
            
        Returns:
            Feature ID
        """
        if metrics is None:
            metrics = {}
        
        # Add feature count to pipeline metrics
        self.pipeline_metrics[f'feature_count_{stage}'] = feature_count
        
        # Update run record
        self.db.record_run(
            run_id=self.run_id,
            round_id=self.round_id,
            description=self.description,
            params=self.pipeline_params,
            metrics=self.pipeline_metrics
        )
        
        # Record feature metrics
        feature_id = self.db.record_features(
            run_id=self.run_id,
            feature_count=feature_count,
            stage=stage,
            metrics=metrics
        )
        
        logger.info(f"Recorded feature metrics for stage {stage} with {feature_count} features")
        return feature_id
    
    def record_model_metrics(self,
                           model_name: str,
                           model_type: str,
                           params: Dict[str, Any] = None,
                           metrics: Dict[str, Any] = None) -> str:
        """
        Record model metrics.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'lightgbm', 'xgboost')
            params: Dictionary of model parameters (optional)
            metrics: Dictionary of model metrics (optional)
            
        Returns:
            Model ID
        """
        if params is None:
            params = {}
        
        if metrics is None:
            metrics = {}
        
        # Add key metrics to pipeline metrics
        for key in ['rmse', 'r2', 'mse']:
            if key in metrics:
                self.pipeline_metrics[f'{model_type}_{key}'] = metrics[key]
        
        # Update run record
        self.db.record_run(
            run_id=self.run_id,
            round_id=self.round_id,
            description=self.description,
            params=self.pipeline_params,
            metrics=self.pipeline_metrics
        )
        
        # Record model metrics
        model_id = self.db.record_model(
            run_id=self.run_id,
            model_name=model_name,
            model_type=model_type,
            params=params,
            metrics=metrics
        )
        
        logger.info(f"Recorded metrics for model {model_name} ({model_type})")
        return model_id
    
    def record_submission(self,
                         file_path: str,
                         model_ids: List[str] = None,
                         metrics: Dict[str, Any] = None) -> str:
        """
        Record a submission.
        
        Args:
            file_path: Path to submission file
            model_ids: List of model IDs used in the submission (optional)
            metrics: Dictionary of submission metrics (optional)
            
        Returns:
            Submission ID
        """
        if metrics is None:
            metrics = {}
        
        # Record submission
        submission_id = self.db.record_submission(
            run_id=self.run_id,
            round_id=self.round_id,
            file_path=file_path,
            model_ids=model_ids or [],
            metrics=metrics
        )
        
        logger.info(f"Recorded submission {submission_id} for run {self.run_id}")
        return submission_id
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a performance report for the current run.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Path to saved report
        """
        return self.db.generate_performance_report(
            run_id=self.run_id,
            output_path=output_path
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Metrics Database')
    parser.add_argument('--db-path', type=str, 
                      default='/media/knight2/EDB/numer_crypto_temp/metrics/metrics.db',
                      help='Path to metrics database')
    parser.add_argument('--report', action='store_true',
                      help='Generate performance report')
    parser.add_argument('--run-id', type=str, default=None,
                      help='Run ID for report (optional)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path for report (optional)')
    
    args = parser.parse_args()
    
    # Initialize database
    db = MetricsDB(args.db_path)
    
    # Generate report if requested
    if args.report:
        report_path = db.generate_performance_report(
            run_id=args.run_id,
            output_path=args.output
        )
        if report_path:
            print(f"Report generated: {report_path}")
        else:
            print("Failed to generate report")