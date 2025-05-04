"""
Script to generate predictions for the Numerai Crypto competition.
"""
import argparse
import os
import h2o
import pandas as pd

from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.models.xgboost_model import H2OXGBoostModel
from numer_crypto.utils.spark_utils import init_h2o
from numer_crypto.config.settings import MODELS_DIR


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
    
    # Create data retriever
    data_retriever = NumeraiDataRetriever()
    
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
    submission_df = pd.DataFrame({
        'id': tournament_df['id'],
        'prediction': preds['predict']
    })
    
    # Save predictions
    output_path = os.path.join(MODELS_DIR, f"{args.tournament}_predictions.csv")
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