"""
Script to generate predictions for the Numerai Crypto competition.
"""
import argparse
import os
import h2o
import polars as pl
from pyspark.sql import SparkSession

from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.models.xgboost_model import H2OXGBoostModel
from numer_crypto.utils.spark_utils import init_h2o
from numer_crypto.config.settings import MODELS_DIR, SPARK_CONFIG


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate predictions for the Numerai Crypto competition')
    parser.add_argument('--download', action='store_true', help='Download the latest data')
    parser.add_argument('--tournament', type=str, default='crypto', help='Tournament name')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--submit', action='store_true', help='Submit predictions to Numerai')
    parser.add_argument('--use-spark', action='store_true', help='Use Spark for data processing')
    
    return parser.parse_args()


def main():
    """
    Main function to generate predictions.
    """
    # Parse arguments
    args = parse_args()
    
    # Initialize H2O
    print("Initializing H2O...")
    h2o_instance = init_h2o()
    
    # Create data retriever with Spark option
    data_retriever = NumeraiDataRetriever(use_spark=args.use_spark)
    
    # Download data if requested
    if args.download:
        data_retriever.download_current_dataset(tournament=args.tournament)
    
    # Load tournament data
    print("Loading tournament data...")
    tournament_df = data_retriever.load_dataset('tournament')
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = H2OXGBoostModel.load_model(args.model_path, h2o_instance=h2o_instance)
    
    # Generate predictions
    print("Generating predictions...")
    preds = model.predict(tournament_df)
    
    # Create submission dataframe
    if args.use_spark:
        # Handle Spark DataFrame
        from pyspark.sql.functions import col
        spark = data_retriever.spark
        
        # Extract id column from tournament_df
        if hasattr(tournament_df, "select"):
            # Spark DataFrame
            ids = tournament_df.select("id")
            
            # Convert H2O predictions to Spark DataFrame
            if isinstance(preds, dict) and 'predict' in preds:
                # If preds is coming as a dict, convert to dataframe
                preds_spark = spark.createDataFrame(preds)
            else:
                # If preds is already a dataframe-like object, convert to Spark
                preds_spark = spark.createDataFrame(preds.as_data_frame())
            
            # Create submission dataframe by joining
            submission_df = ids.join(
                preds_spark.select(col("predict")),
                monotonically_increasing_id=True  # Join by row position
            )
        else:
            # Handle unexpected dataframe type
            raise TypeError("Unexpected DataFrame type. Check H2OXGBoostModel.predict() output format.")
    else:
        # Use Polars for non-Spark workflow
        if hasattr(tournament_df, "select"):
            # Handle if tournament_df is still a Spark DataFrame
            ids = tournament_df.select("id").toPandas()
            submission_df = pl.DataFrame({
                'id': ids['id'],
                'prediction': preds['predict']
            })
        elif isinstance(tournament_df, pl.DataFrame):
            # Handle Polars DataFrame
            submission_df = pl.DataFrame({
                'id': tournament_df['id'],
                'prediction': preds['predict']
            })
        else:
            # Handle unexpected DataFrame type (e.g., H2O frame)
            ids = tournament_df['id'] if hasattr(tournament_df, '__getitem__') else None
            if ids is None:
                raise TypeError("Cannot extract ids from tournament_df.")
            submission_df = pl.DataFrame({
                'id': ids,
                'prediction': preds['predict']
            })
    
    # Save predictions
    output_path = os.path.join(MODELS_DIR, f"{args.tournament}_predictions.csv")
    if args.use_spark and isinstance(submission_df, SparkSession):
        submission_df.write.csv(output_path, header=True, mode="overwrite")
    elif isinstance(submission_df, pl.DataFrame):
        submission_df.write_csv(output_path)
    else:
        # Fallback to pandas-like interface
        submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Submit if requested
    if args.submit:
        print("Submitting predictions to Numerai...")
        result = data_retriever.submit_predictions(submission_df, tournament=args.tournament)
        print(f"Submission ID: {result['submission_id']}")
    
    # Shutdown H2O
    h2o.shutdown(prompt=False)
    print("Done!")


if __name__ == "__main__":
    main()