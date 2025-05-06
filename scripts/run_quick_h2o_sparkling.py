#!/usr/bin/env python3
"""
Quick demo of H2O Sparkling Water AutoML for Numerai Crypto.
This script runs H2O Sparkling Water AutoML for a short time (5 minutes by default)
to demonstrate model training and produce summary metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from pathlib import Path

# Set up logging
log_file = f"quick_h2o_sparkling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def find_latest_processed_data():
    """Find the most recently processed data file"""
    processed_dir = os.path.join(Path(__file__).parent.parent, 'data', 'processed')
    processed_files = [f for f in os.listdir(processed_dir) 
                      if f.startswith('processed_yiedl_') and f.endswith('.parquet')]
    
    if not processed_files:
        logger.error("No processed data files found")
        return None
    
    # Sort by timestamp
    processed_files.sort(reverse=True)
    latest_file = os.path.join(processed_dir, processed_files[0])
    
    logger.info(f"Using latest processed data: {latest_file}")
    return latest_file

def load_data(data_file):
    """Load processed data for modeling"""
    if data_file is None or not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return None
    
    logger.info(f"Loading data from {data_file}")
    
    try:
        # Load the data
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def create_synthetic_target(df):
    """Create synthetic target for training"""
    logger.info("Creating synthetic target")
    
    # Copy data without symbol
    X = df.drop('symbol', axis=1).copy()
    
    # Create target
    np.random.seed(42)
    weights = np.random.normal(0, 1, X.shape[1])
    weights = weights / np.sqrt(np.sum(weights**2))
    
    y = X.values @ weights
    noise = np.random.normal(0, 0.1, len(y))
    y = y + noise
    
    # Scale to [0, 1] range
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Add target to dataframe
    df_with_target = df.copy()
    df_with_target['target'] = y
    
    logger.info(f"Created dataset with target, shape: {df_with_target.shape}")
    
    return df_with_target

def run_h2o_sparkling(data, max_runtime_secs=300):
    """Run H2O Sparkling Water AutoML for a short time"""
    logger.info(f"Setting up Spark and H2O Sparkling Water (max_runtime: {max_runtime_secs}s)")
    
    try:
        # Create Spark session
        from pyspark.sql import SparkSession
        
        logger.info("Creating Spark session")
        spark = SparkSession.builder \
            .appName("QuickH2OSparkling") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()
        
        logger.info(f"Created Spark session: {spark.version}")
        
        # Initialize H2O Sparkling
        from pysparkling import H2OContext, H2OConf
        
        logger.info("Initializing H2O Sparkling Water")
        h2o_conf = H2OConf(spark)
        h2o_conf.set_internal_cluster_mode()
        h2o_context = H2OContext.getOrCreate(spark, conf=h2o_conf)
        
        logger.info(f"H2O Sparkling initialized: {h2o_context.get_flow_url()}")
        
        # Import h2o
        import h2o
        from h2o.automl import H2OAutoML
        
        # Convert to Spark DataFrame
        logger.info("Converting data to Spark DataFrame")
        spark_df = spark.createDataFrame(data)
        
        # Convert to H2O Frame
        logger.info("Converting to H2O Frame")
        h2o_df = h2o_context.asH2OFrame(spark_df)
        
        # Split data
        logger.info("Splitting data into train/validation sets")
        train_h2o, valid_h2o = h2o_df.split_frame(ratios=[0.8], seed=42)
        
        # Identify features
        feature_cols = [col for col in data.columns if col not in ['symbol', 'target']]
        logger.info(f"Using {len(feature_cols)} features")
        
        # Run AutoML
        logger.info(f"Running H2O AutoML for {max_runtime_secs} seconds")
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=20,
            seed=42,
            sort_metric="RMSE",
            exclude_algos=["DeepLearning"],  # Exclude DL for faster runtime
            nfolds=5
        )
        
        start_time = time.time()
        aml.train(x=feature_cols, y="target", training_frame=train_h2o, validation_frame=valid_h2o)
        runtime = time.time() - start_time
        
        # Get leaderboard
        logger.info("AutoML training complete, retrieving results")
        lb = aml.leaderboard.as_data_frame()
        logger.info(f"AutoML Leaderboard (top 10):\n{lb.head(10)}")
        
        # Extract model performance
        models_info = []
        lb_rows = min(10, lb.shape[0])
        
        for i in range(lb_rows):
            model_id = lb.iloc[i, 0]
            model = h2o.get_model(model_id)
            
            # Get performance on train and validation
            train_perf = model.model_performance(train_h2o)
            valid_perf = model.model_performance(valid_h2o)
            
            models_info.append({
                'model_id': model_id,
                'rank': i + 1,
                'train_rmse': train_perf.rmse(),
                'valid_rmse': valid_perf.rmse(),
                'model_type': model_id.split('_')[0],
                'r2': valid_perf.r2()
            })
            
            logger.info(f"Model {i+1}: {model_id}")
            logger.info(f"  - Train RMSE: {train_perf.rmse():.4f}")
            logger.info(f"  - Valid RMSE: {valid_perf.rmse():.4f}")
            logger.info(f"  - R²: {valid_perf.r2():.4f}")
        
        # Create summary table
        summary_table = pd.DataFrame(models_info)
        
        # Calculate overfitting ratio
        summary_table['overfitting_ratio'] = summary_table['train_rmse'] / summary_table['valid_rmse']
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(Path(__file__).parent.parent, 'models', 'h2o_sparkling')
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, f"h2o_sparkling_summary_{timestamp}.csv")
        summary_table.to_csv(summary_file, index=False)
        
        # Save markdown table too
        md_file = os.path.join(output_dir, f"h2o_sparkling_summary_{timestamp}.md")
        with open(md_file, 'w') as f:
            f.write("# H2O Sparkling Water AutoML Results\n\n")
            f.write("## Model Performance\n\n")
            f.write(summary_table.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            f.write(f"Total runtime: {runtime:.2f} seconds")
        
        logger.info(f"Saved summary to {summary_file} and {md_file}")
        
        # Create prediction for demonstration
        logger.info("Generating predictions for demonstration")
        preds = aml.leader.predict(h2o_df)
        pred_df = h2o.as_list(preds)
        
        prediction_file = os.path.join(output_dir, f"h2o_sparkling_predictions_{timestamp}.csv")
        symbol_df = h2o.as_list(h2o_df['symbol'])
        
        # Combine symbol and prediction
        demo_preds = pd.DataFrame({
            'symbol': symbol_df.iloc[:, 0],
            'prediction': pred_df.iloc[:, 0]
        })
        
        demo_preds.to_csv(prediction_file, index=False)
        
        logger.info(f"Saved demonstration predictions to {prediction_file}")
        
        # Cleanup
        logger.info("Shutting down H2O")
        h2o.cluster().shutdown()
        
        logger.info("Stopping Spark")
        spark.stop()
        
        return summary_table, models_info
    
    except Exception as e:
        logger.error(f"Error running H2O Sparkling AutoML: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def print_summary_table(summary_table):
    """Print formatted summary table to console"""
    if summary_table is None:
        return
    
    print("\n" + "="*80)
    print("                      H2O SPARKLING WATER AUTOML RESULTS")
    print("="*80)
    print("\nModel Performance:\n")
    
    # Format for console
    from tabulate import tabulate
    headers = ['Rank', 'Model Type', 'Train RMSE', 'Valid RMSE', 'R²', 'Overfitting Ratio']
    table_data = []
    
    for _, row in summary_table.iterrows():
        table_data.append([
            int(row['rank']),
            row['model_type'],
            f"{row['train_rmse']:.4f}",
            f"{row['valid_rmse']:.4f}",
            f"{row['r2']:.4f}",
            f"{row['overfitting_ratio']:.4f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("\n" + "="*80)

def main():
    """Main function"""
    # Find the latest processed data
    data_file = find_latest_processed_data()
    
    if data_file is None:
        logger.error("No processed data file found. Please run process_yiedl_data.py first.")
        return 1
    
    # Load data
    df = load_data(data_file)
    
    if df is None:
        logger.error("Failed to load data.")
        return 1
    
    # Create synthetic target
    df_with_target = create_synthetic_target(df)
    
    # Get runtime from command line argument
    max_runtime_secs = 300  # Default 5 minutes
    if len(sys.argv) > 1:
        try:
            max_runtime_secs = int(sys.argv[1])
            logger.info(f"Using command line runtime: {max_runtime_secs} seconds")
        except ValueError:
            logger.warning(f"Invalid runtime '{sys.argv[1]}', using default: {max_runtime_secs} seconds")
    
    # Run H2O Sparkling Water AutoML
    summary_table, models_info = run_h2o_sparkling(df_with_target, max_runtime_secs)
    
    # Print summary table
    print_summary_table(summary_table)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())