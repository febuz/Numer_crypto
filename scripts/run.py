"""
Main script to run the Numerai Crypto prediction pipeline.
"""
import argparse
import os
import time
import h2o

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
    parser = argparse.ArgumentParser(description='Run the Numerai Crypto prediction pipeline')
    parser.add_argument('--download', action='store_true', help='Download the latest data')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--submit', action='store_true', help='Submit predictions to Numerai')
    parser.add_argument('--tournament', type=str, default='crypto', help='Tournament name')
    parser.add_argument('--model-id', type=str, default=None, help='Model ID for training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model for prediction')
    
    return parser.parse_args()


def main():
    """
    Main function to run the prediction pipeline.
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
    
    # Train model if requested
    if args.train:
        print("Training model...")
        
        # Load datasets
        train_df = data_retriever.load_dataset('training')
        valid_df = data_retriever.load_dataset('validation')
        
        # Get feature names
        feature_names = data_retriever.get_feature_names()
        
        # Create model ID if not provided
        model_id = args.model_id or f"xgb_model_{args.tournament}_{int(time.time())}"
        
        # Create and train model
        model = H2OXGBoostModel(h2o_instance=h2o_instance)
        model.train(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_names,
            target_col='target',
            model_id=model_id
        )
        
        # Save model
        model_path = model.save_model()
        print(f"Model saved to {model_path}")
    
    # Generate predictions if requested
    if args.predict:
        print("Generating predictions...")
        
        # Load model
        if args.model_path is None:
            # Use the latest model if path not provided
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pickle')]
            if not model_files:
                raise ValueError("No models found. Train a model first or provide --model-path.")
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
            model_path = os.path.join(MODELS_DIR, model_files[0])
            print(f"Using latest model: {model_path}")
        else:
            model_path = args.model_path
        
        # Load model
        model = H2OXGBoostModel.load_model(model_path, h2o_instance=h2o_instance)
        
        # Load tournament data
        tournament_df = data_retriever.load_dataset('tournament')
        
        # Generate predictions
        preds = model.predict(tournament_df)
        
        # Create submission dataframe
        import pandas as pd
        submission_df = pd.DataFrame({
            'id': tournament_df['id'],
            'prediction': preds['predict']
        })
        
        # Save predictions
        output_path = os.path.join(MODELS_DIR, f"{args.tournament}_predictions.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Submit predictions if requested
        if args.submit:
            print("Submitting predictions to Numerai...")
            result = data_retriever.submit_predictions(submission_df, tournament=args.tournament)
            print(f"Submission ID: {result['submission_id']}")
    
    # Shutdown H2O
    h2o.shutdown(prompt=False)
    print("Done!")


if __name__ == "__main__":
    main()