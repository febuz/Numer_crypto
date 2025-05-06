#!/usr/bin/env python3
"""
H2O AutoML model for Numerai crypto prediction with Yiedl data.

This script uses H2O's AutoML capabilities to train models on combined
Numerai crypto and Yiedl data, with support for both regular H2O and
Sparkling Water (H2O + Spark) implementations.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
log_file = f"h2o_automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def check_h2o_availability():
    """Check if H2O and/or H2O Sparkling are available"""
    h2o_available = False
    sparkling_available = False
    h2o_version = None
    java_version = None
    
    # Check H2O
    try:
        import h2o
        h2o_available = True
        h2o_version = h2o.__version__
        logger.info(f"H2O is available, version: {h2o_version}")
    except ImportError:
        logger.warning("H2O is not installed")
    
    # Check Java version
    try:
        import subprocess
        java_process = subprocess.Popen(['java', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, java_version_str = java_process.communicate()
        if b"version" in java_version_str:
            if b"1.8" in java_version_str or b"8." in java_version_str:
                java_version = "8"
            elif b"11." in java_version_str:
                java_version = "11"
            elif b"17." in java_version_str:
                java_version = "17"
            else:
                java_version = "unknown"
        logger.info(f"Java version detected: {java_version}")
    except:
        logger.warning("Could not detect Java version")
    
    # Check H2O Sparkling
    try:
        from pysparkling import H2OContext, H2OConf
        sparkling_available = True
        logger.info("H2O Sparkling is available")
    except ImportError:
        logger.warning("H2O Sparkling is not installed")
    
    return {
        'h2o_available': h2o_available,
        'sparkling_available': sparkling_available,
        'h2o_version': h2o_version,
        'java_version': java_version
    }

def initialize_h2o(use_sparkling=False, spark_session=None, h2o_port=54321):
    """Initialize H2O or H2O Sparkling"""
    if use_sparkling and spark_session is None:
        logger.error("Spark session is required for H2O Sparkling")
        return None
    
    if use_sparkling:
        try:
            from pysparkling import H2OContext, H2OConf
            
            logger.info("Initializing H2O Sparkling Water")
            
            # Configure H2O connection
            h2o_conf = H2OConf(spark_session)
            h2o_conf.set_internal_cluster_mode()
            h2o_conf.set_port(h2o_port)
            
            # Initialize H2O context
            h2o_context = H2OContext.getOrCreate(spark_session, conf=h2o_conf)
            
            logger.info(f"H2O Sparkling initialized: {h2o_context.get_flow_url()}")
            
            import h2o
            return h2o_context, h2o
        except Exception as e:
            logger.error(f"Failed to initialize H2O Sparkling: {e}")
            return None
    else:
        try:
            import h2o
            
            logger.info("Initializing standalone H2O")
            h2o.init(port=h2o_port)
            
            logger.info(f"H2O initialized: {h2o.cluster().cluster_summary()}")
            return None, h2o
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {e}")
            return None, None

def create_spark_session(app_name="NumeraiCryptoH2O", executor_memory="4g", driver_memory="4g"):
    """Create a Spark session for H2O Sparkling"""
    try:
        logger.info("Creating Spark session")
        
        from pyspark.sql import SparkSession
        
        # Create Spark session with appropriate configuration
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.kryoserializer.buffer.max", "2000M") \
            .getOrCreate()
        
        logger.info(f"Spark session created: {spark.version}")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}")
        return None

def load_data(train_file, live_file):
    """Load training and live data from parquet files"""
    logger.info(f"Loading train data from {train_file}")
    train_df = pd.read_parquet(train_file)
    logger.info(f"Train data shape: {train_df.shape}")
    
    logger.info(f"Loading live data from {live_file}")
    live_df = pd.read_parquet(live_file)
    logger.info(f"Live data shape: {live_df.shape}")
    
    return train_df, live_df

def clean_data(df):
    """Clean data by handling missing values, infinities, etc."""
    logger.info(f"Cleaning data of shape {df.shape}")
    
    # Make a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Replace infinities with NaN
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    
    # Count missing values before filling
    missing_counts = cleaned_df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        logger.info(f"Columns with missing values: {len(cols_with_missing)}")
        for col, count in cols_with_missing.items():
            logger.info(f"  - {col}: {count} missing values ({count/len(cleaned_df)*100:.2f}%)")
    
    # Fill missing values with mean for each column
    for col in cleaned_df.columns:
        if col not in ['id', 'target', 'era', 'date', 'asset', 'data_type']:
            if cleaned_df[col].isna().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill any remaining NaNs with 0
    cleaned_df = cleaned_df.fillna(0)
    
    # Log data cleaning results
    logger.info(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    
    return cleaned_df

def prepare_h2o_frames(train_df, live_df, h2o, h2o_context=None, use_sparkling=False):
    """Prepare H2O frames for model training"""
    logger.info("Converting data to H2O frames")
    
    # Identify feature columns
    metadata_cols = ['id', 'target', 'era', 'date', 'asset', 'data_type']
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]
    
    logger.info(f"Number of features: {len(feature_cols)}")
    
    if use_sparkling and h2o_context is not None:
        # For Sparkling, first convert to Spark dataframes
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        # Convert pandas to spark
        train_spark = spark.createDataFrame(train_df)
        live_spark = spark.createDataFrame(live_df)
        
        # Convert spark to h2o
        train_h2o = h2o_context.asH2OFrame(train_spark)
        live_h2o = h2o_context.asH2OFrame(live_spark)
    else:
        # For standalone H2O, convert directly
        train_h2o = h2o.H2OFrame(train_df)
        live_h2o = h2o.H2OFrame(live_df)
    
    # Convert specified columns to categorical if needed
    for col in ['asset', 'era', 'date']:
        if col in train_h2o.columns:
            train_h2o[col] = train_h2o[col].asfactor()
        if col in live_h2o.columns:
            live_h2o[col] = live_h2o[col].asfactor()
    
    # Split training data into train and validation
    if 'era' in train_h2o.columns:
        # Split by era (time-based)
        train_h2o['era_num'] = train_h2o['era'].asnumeric()
        
        # Get unique eras
        eras = train_h2o['era_num'].unique()
        eras.sort()
        
        # Use last 20% of eras for validation
        split_point = int(len(eras) * 0.8)
        train_eras = eras[:split_point]
        val_eras = eras[split_point:]
        
        # Split data based on eras
        train_data = train_h2o[train_h2o['era_num'].isin(train_eras)]
        val_data = train_h2o[train_h2o['era_num'].isin(val_eras)]
    else:
        # Random split
        train_data, val_data = train_h2o.split_frame(ratios=[0.8], seed=RANDOM_SEED)
    
    logger.info(f"Training data: {train_data.shape}")
    logger.info(f"Validation data: {val_data.shape}")
    logger.info(f"Live data: {live_h2o.shape}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'live': live_h2o,
        'features': feature_cols
    }

def train_h2o_automl(h2o_frames, h2o, max_runtime_secs=600, max_models=25):
    """Train H2O AutoML models"""
    logger.info(f"Training H2O AutoML (max_runtime_secs={max_runtime_secs}, max_models={max_models})")
    
    train_data = h2o_frames['train']
    val_data = h2o_frames['validation']
    features = h2o_frames['features']
    
    # Initialize H2O AutoML
    try:
        from h2o.automl import H2OAutoML
        
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            seed=RANDOM_SEED,
            sort_metric="RMSE",
            stopping_metric="RMSE"
        )
        
        # Train AutoML model
        aml.train(x=features, y="target", training_frame=train_data, validation_frame=val_data)
        
        # Get leaderboard
        lb = aml.leaderboard
        logger.info("AutoML Leaderboard (top 10 models):")
        logger.info(lb.head(10).as_data_frame().to_string(index=False))
        
        # Get best model
        best_model = aml.leader
        model_type = best_model.__class__.__name__
        logger.info(f"Best model: {model_type}")
        
        # Evaluate on validation data
        perf = best_model.model_performance(val_data)
        rmse = perf.rmse()
        mae = perf.mae()
        
        logger.info(f"Validation RMSE: {rmse}")
        logger.info(f"Validation MAE: {mae}")
        
        # Check if we've achieved target RMSE of 0.25
        if rmse <= 0.25:
            logger.info(f"TARGET ACHIEVED! H2O AutoML - RMSE: {rmse:.6f}")
        
        return aml
    except Exception as e:
        logger.error(f"Error training AutoML model: {e}")
        return None

def generate_predictions(aml, h2o_frames, h2o, h2o_context=None, use_sparkling=False):
    """Generate predictions using trained model"""
    logger.info("Generating predictions")
    
    live_h2o = h2o_frames['live']
    
    # Generate predictions
    predictions = aml.leader.predict(live_h2o)
    logger.info(f"Generated predictions shape: {predictions.shape}")
    
    # Convert to pandas
    if use_sparkling and h2o_context is not None:
        # For Sparkling Water, convert through Spark
        predictions_spark = h2o_context.asSparkFrame(predictions)
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        # Get ID column from live data
        id_col = h2o_context.asSparkFrame(live_h2o.cols(['id']))
        
        # Join predictions with IDs
        pred_with_id = id_col.join(
            predictions_spark, 
            id_col.row_id == predictions_spark.row_id
        ).select('id', 'predict')
        
        # Convert to pandas
        predictions_pd = pred_with_id.toPandas()
        predictions_pd.columns = ['id', 'prediction']
    else:
        # For standalone H2O, convert directly
        ids = live_h2o['id'].as_data_frame()
        preds = predictions['predict'].as_data_frame()
        
        # Create DataFrame with id and prediction
        predictions_pd = pd.DataFrame({
            'id': ids.values.flatten(),
            'prediction': preds.values.flatten()
        })
    
    logger.info(f"Final predictions shape: {predictions_pd.shape}")
    
    return predictions_pd

def save_submission(predictions_df, output_dir, prefix="h2o_automl", round_num=None):
    """Save submission file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Add round number if provided
    round_str = f"_round{round_num}" if round_num else ""
    
    filename = os.path.join(output_dir, f"{prefix}{round_str}_{timestamp}.csv")
    
    # Verify predictions before saving
    logger.info(f"Prediction stats: min={predictions_df['prediction'].min()}, max={predictions_df['prediction'].max()}, mean={predictions_df['prediction'].mean()}")
    
    # Check for NaN or infinity values
    if predictions_df['prediction'].isna().any():
        logger.warning(f"Predictions contain {predictions_df['prediction'].isna().sum()} NaN values. Filling with 0.")
        predictions_df['prediction'] = predictions_df['prediction'].fillna(0)
    
    if np.isinf(predictions_df['prediction']).any():
        logger.warning(f"Predictions contain infinity values. Replacing with 0.")
        predictions_df['prediction'] = predictions_df['prediction'].replace([np.inf, -np.inf], 0)
    
    # Save to CSV
    predictions_df.to_csv(filename, index=False)
    logger.info(f"Saved submission to {filename}")
    
    return filename

def save_model(aml, output_dir, prefix="h2o_automl_model"):
    """Save the H2O model"""
    # Create output directory
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f"{prefix}_{timestamp}")
    
    try:
        # Save model
        model_id = aml.leader.model_id
        saved_path = h2o.save_model(model=aml.leader, path=model_dir, force=True)
        logger.info(f"Saved H2O model to {saved_path}")
        
        # Also try to save as MOJO if possible
        try:
            mojo_path = aml.leader.download_mojo(path=model_dir, get_genmodel_jar=True)
            logger.info(f"Saved MOJO to {mojo_path}")
        except Exception as e:
            logger.warning(f"Could not save model as MOJO: {e}")
        
        return saved_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def shutdown_h2o(h2o, h2o_context=None):
    """Shutdown H2O and/or Spark"""
    # Shutdown H2O
    if h2o is not None:
        logger.info("Shutting down H2O")
        h2o.shutdown()
    
    # If Sparkling, also stop Spark
    if h2o_context is not None:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            logger.info("Stopping Spark")
            spark.stop()
        except Exception as e:
            logger.warning(f"Error stopping Spark: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train H2O AutoML models for Numerai crypto prediction')
    
    parser.add_argument('--train-file', type=str, help='Path to merged training data parquet file')
    parser.add_argument('--live-file', type=str, help='Path to merged live data parquet file')
    parser.add_argument('--output-dir', type=str, default='submissions', help='Directory to save submission files')
    parser.add_argument('--round', type=int, help='Current Numerai round number')
    parser.add_argument('--use-sparkling', action='store_true', help='Use H2O Sparkling Water instead of standalone H2O')
    parser.add_argument('--max-runtime', type=int, default=600, help='Maximum runtime in seconds for AutoML')
    parser.add_argument('--max-models', type=int, default=25, help='Maximum number of models to train in AutoML')
    parser.add_argument('--h2o-port', type=int, default=54321, help='Port for H2O server')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Check if we need to download data first
    if not args.train_file or not args.live_file:
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            import download_numerai_yiedl_data
            
            logger.info("Downloading Numerai and Yiedl data...")
            data_info = download_numerai_yiedl_data.main()
            
            args.train_file = data_info.get('train_file')
            args.live_file = data_info.get('live_file')
            
            if not args.round and 'current_round' in data_info:
                args.round = data_info['current_round']
            
            logger.info(f"Using downloaded data: train={args.train_file}, live={args.live_file}, round={args.round}")
        except (ImportError, Exception) as e:
            logger.error(f"Failed to download data: {e}")
            if not args.train_file or not args.live_file:
                logger.error("No data files specified. Exiting.")
                return 1
    
    # Check H2O availability
    h2o_info = check_h2o_availability()
    
    if not h2o_info['h2o_available']:
        logger.error("H2O is not available. Please install h2o package.")
        return 1
    
    if args.use_sparkling and not h2o_info['sparkling_available']:
        logger.error("H2O Sparkling is not available but was requested. Using standalone H2O instead.")
        args.use_sparkling = False
    
    # Initialize Spark if needed
    spark = None
    if args.use_sparkling:
        spark = create_spark_session()
        if spark is None:
            logger.error("Failed to create Spark session. Using standalone H2O instead.")
            args.use_sparkling = False
    
    # Initialize H2O or H2O Sparkling
    h2o_context, h2o = initialize_h2o(use_sparkling=args.use_sparkling, spark_session=spark, h2o_port=args.h2o_port)
    
    if h2o is None:
        logger.error("Failed to initialize H2O. Exiting.")
        return 1
    
    # Load and prepare data
    try:
        # Load data
        train_df, live_df = load_data(args.train_file, args.live_file)
        
        # Clean data
        train_df = clean_data(train_df)
        live_df = clean_data(live_df)
        
        # Convert to H2O frames
        h2o_frames = prepare_h2o_frames(train_df, live_df, h2o, h2o_context, args.use_sparkling)
        
        # Train AutoML model
        aml = train_h2o_automl(h2o_frames, h2o, max_runtime_secs=args.max_runtime, max_models=args.max_models)
        
        if aml is None:
            logger.error("Failed to train AutoML model. Exiting.")
            shutdown_h2o(h2o, h2o_context)
            return 1
        
        # Generate predictions
        predictions_df = generate_predictions(aml, h2o_frames, h2o, h2o_context, args.use_sparkling)
        
        # Create output directory
        output_dir = os.path.join(Path.cwd(), args.output_dir)
        
        # Save submission
        submission_file = save_submission(predictions_df, output_dir, round_num=args.round)
        
        # Save model
        model_file = save_model(aml, output_dir)
        
        # Print summary
        logger.info("\n===== H2O AUTOML RESULTS =====")
        logger.info(f"Best model: {aml.leader.model_id}")
        logger.info(f"Validation RMSE: {aml.leader.model_performance(h2o_frames['validation']).rmse()}")
        logger.info(f"Submission file: {submission_file}")
        if model_file:
            logger.info(f"Model file: {model_file}")
        
        # Shutdown H2O
        shutdown_h2o(h2o, h2o_context)
        
        return 0
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Ensure H2O is shutdown
        shutdown_h2o(h2o, h2o_context)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())