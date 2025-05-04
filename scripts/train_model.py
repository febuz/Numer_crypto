"""
Script to train a model for the Numerai Crypto competition.
"""
import argparse
import os
import time
import h2o
import polars as pl
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.models.xgboost_model import H2OXGBoostModel
from numer_crypto.utils.spark_utils import init_h2o
from numer_crypto.utils.data_utils import plot_feature_importance, convert_polars_to_h2o, convert_spark_to_h2o
from numer_crypto.config.settings import MODELS_DIR, SPARK_CONFIG


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for the Numerai Crypto competition')
    parser.add_argument('--download', action='store_true', help='Download the latest data')
    parser.add_argument('--tournament', type=str, default='crypto', help='Tournament name')
    parser.add_argument('--model-id', type=str, default=None, help='Model ID')
    parser.add_argument('--trees', type=int, default=500, help='Number of trees')
    parser.add_argument('--depth', type=int, default=6, help='Maximum tree depth')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    parser.add_argument('--use-spark', action='store_true', help='Use Spark for data processing')
    parser.add_argument('--use-h2o-context', action='store_true', help='Use H2O-Sparkling Water context for Spark integration')
    
    return parser.parse_args()


def main():
    """
    Main function to train a model.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize H2O and Spark/H2O context if needed
    print("Initializing H2O...")
    h2o_instance = init_h2o()
    h2o_context = None
    
    if args.use_spark and args.use_h2o_context:
        print("Initializing Spark and H2O-Sparkling Water context...")
        from pysparkling import H2OContext
        
        # Create Spark session
        spark = SparkSession.builder \
            .appName(SPARK_CONFIG.get('app_name', 'NumeraiSparklingWater')) \
            .config("spark.executor.memory", SPARK_CONFIG.get('executor_memory', '4g')) \
            .config("spark.driver.memory", SPARK_CONFIG.get('driver_memory', '4g')) \
            .config("spark.executor.cores", SPARK_CONFIG.get('executor_cores', '2')) \
            .getOrCreate()
            
        # Initialize H2O context with Spark
        h2o_context = H2OContext.getOrCreate(spark)
        print(f"Spark version: {spark.version}")
        print(f"H2O context version: {h2o_context.getSparklingWaterVersion()}")
    
    # Create data retriever with Spark option
    data_retriever = NumeraiDataRetriever(use_spark=args.use_spark)
    
    # Download data if requested
    if args.download:
        data_retriever.download_current_dataset(tournament=args.tournament)
    
    # Load datasets
    print("Loading datasets...")
    train_df = data_retriever.load_dataset('training')
    valid_df = data_retriever.load_dataset('validation')
    
    # Get feature names
    feature_names = data_retriever.get_feature_names()
    print(f"Number of features: {len(feature_names)}")
    
    # Create model ID if not provided
    model_id = args.model_id or f"xgb_model_{args.tournament}_{int(time.time())}"
    
    # Set model parameters
    model_params = {
        'ntrees': args.trees,
        'max_depth': args.depth,
        'learn_rate': args.lr,
        'sample_rate': 0.8,
        'col_sample_rate': 0.8,
    }
    
    # Convert data to H2O frames if needed
    if args.use_spark:
        print("Converting Spark DataFrames to H2O frames...")
        if args.use_h2o_context:
            # Use H2O Sparkling Water context for optimized conversion
            h2o_train = convert_spark_to_h2o(train_df, h2o_instance, h2o_context)
            h2o_valid = convert_spark_to_h2o(valid_df, h2o_instance, h2o_context)
        else:
            # Standard conversion path
            h2o_train = convert_spark_to_h2o(train_df, h2o_instance)
            h2o_valid = convert_spark_to_h2o(valid_df, h2o_instance)
    elif isinstance(train_df, pl.DataFrame):
        print("Converting Polars DataFrames to H2O frames...")
        h2o_train = convert_polars_to_h2o(train_df, h2o_instance)
        h2o_valid = convert_polars_to_h2o(valid_df, h2o_instance)
    else:
        # Assuming data is already compatible with H2O or needs no conversion
        h2o_train = train_df
        h2o_valid = valid_df
    
    # Create and train model
    model = H2OXGBoostModel(h2o_instance=h2o_instance, params=model_params)
    model.train(
        train_df=h2o_train,
        valid_df=h2o_valid,
        feature_cols=feature_names,
        target_col='target',
        model_id=model_id
    )
    
    # Save model if requested
    if not args.no_save:
        model_path = model.save_model()
        print(f"Model saved to {model_path}")
    
    # Plot feature importance
    feature_imp = model.get_feature_importance()
    fig = plot_feature_importance(model.model, feature_names, n_features=20)
    
    # Save feature importance plot
    fig_path = os.path.join(MODELS_DIR, f"{model_id}_feature_importance.png")
    fig.savefig(fig_path)
    print(f"Feature importance plot saved to {fig_path}")
    
    # Shutdown H2O and Spark
    h2o.shutdown(prompt=False)
    if args.use_spark and args.use_h2o_context:
        data_retriever.spark.stop()
    print("Done!")


if __name__ == "__main__":
    main()