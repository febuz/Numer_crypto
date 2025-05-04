"""
Script to train a model for the Numerai Crypto competition.
"""
import argparse
import os
import time
import h2o
import pandas as pd
import matplotlib.pyplot as plt

from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.models.xgboost_model import H2OXGBoostModel
from numer_crypto.utils.spark_utils import init_h2o
from numer_crypto.utils.data_utils import plot_feature_importance
from numer_crypto.config.settings import MODELS_DIR


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
    
    return parser.parse_args()


def main():
    """
    Main function to train a model.
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
    
    # Create and train model
    model = H2OXGBoostModel(h2o_instance=h2o_instance, params=model_params)
    model.train(
        train_df=train_df,
        valid_df=valid_df,
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
    
    # Shutdown H2O
    h2o.shutdown(prompt=False)
    print("Done!")


if __name__ == "__main__":
    main()