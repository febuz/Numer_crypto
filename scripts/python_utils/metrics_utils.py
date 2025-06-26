#!/usr/bin/env python3
"""
Utilities for tracking and displaying metrics for Numerai Crypto Pipeline
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def display_recent_runs(metrics_db_path: str, limit: int = 5) -> None:
    """Display recent pipeline runs from the metrics database"""
    try:
        # Add repository root to path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.append(repo_root)
        
        from utils.metrics.metrics_db import MetricsDB
        
        if not os.path.exists(metrics_db_path):
            logger.warning(f"Metrics database not found at {metrics_db_path}")
            print("No metrics database found")
            return
            
        db = MetricsDB(metrics_db_path)
        runs = db.get_runs(limit=limit)
        
        if runs:
            print(f'Latest run: {runs[0]["run_id"]}')
            for run in runs:
                print(f'  {run["timestamp"]}: {run["description"]}')
        else:
            print('No runs found in metrics database')
            
    except ImportError as e:
        logger.error(f"Failed to import metrics database module: {e}")
        print("Metrics database module not available")
    except Exception as e:
        logger.error(f"Error accessing metrics database: {e}")
        print(f"Error accessing metrics database: {e}")

def add_pipeline_run(metrics_db_path: str, 
                   description: str,
                   execution_time_seconds: int,
                   num_models: int,
                   num_submissions: int) -> None:
    """Add a new pipeline run to the metrics database"""
    try:
        # Add repository root to path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.append(repo_root)
        
        from utils.metrics.metrics_db import MetricsDB
        
        # Create metrics directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_db_path), exist_ok=True)
            
        db = MetricsDB(metrics_db_path)
        
        # Add the run to the database
        run_id = db.add_run(
            description=description,
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time_seconds,
            num_models=num_models,
            num_submissions=num_submissions
        )
        
        logger.info(f"Added pipeline run to metrics database with ID: {run_id}")
        print(f"Added pipeline run to metrics database with ID: {run_id}")
            
    except ImportError as e:
        logger.error(f"Failed to import metrics database module: {e}")
        print("Metrics database module not available")
    except Exception as e:
        logger.error(f"Error adding run to metrics database: {e}")
        print(f"Error adding run to metrics database: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metrics utilities for Numerai Crypto Pipeline")
    parser.add_argument("--display-runs", action="store_true", help="Display recent pipeline runs")
    parser.add_argument("--add-run", action="store_true", help="Add a new pipeline run")
    parser.add_argument("--db-path", type=str, help="Path to the metrics database")
    parser.add_argument("--description", type=str, help="Description of the pipeline run")
    parser.add_argument("--execution-time", type=int, help="Execution time in seconds")
    parser.add_argument("--num-models", type=int, help="Number of models created")
    parser.add_argument("--num-submissions", type=int, help="Number of submissions created")
    parser.add_argument("--limit", type=int, default=5, help="Limit the number of runs to display")
    
    args = parser.parse_args()
    
    if args.display_runs and args.db_path:
        display_recent_runs(args.db_path, args.limit)
    elif args.add_run and args.db_path and args.description:
        add_pipeline_run(
            args.db_path,
            args.description,
            args.execution_time or 0,
            args.num_models or 0,
            args.num_submissions or 0
        )
    else:
        parser.print_help()