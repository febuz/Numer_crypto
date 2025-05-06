#!/usr/bin/env python3
"""
Analyze Crypto Predictions

This script analyzes the predictions from the crypto ensemble model, providing:
- Visualization of prediction distribution
- Autocorrelation analysis
- Feature importance analysis
- Performance metrics based on validation data
- Comparison between multiple submission versions

Usage:
    python analyze_crypto_predictions.py --predictions PATH_TO_PREDICTIONS 
                                        [--validation PATH_TO_VALIDATION]
                                        [--compare PATH_TO_SECOND_PREDICTIONS]
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from numer_crypto.data.retrieval import NumeraiDataRetriever
from numer_crypto.config.settings import DATA_DIR, MODELS_DIR

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze Crypto Predictions')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--validation', type=str, default=None,
                        help='Path to validation data (if available)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to second predictions file for comparison')
    parser.add_argument('--tournament', type=str, default='crypto',
                        help='Tournament name')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: reports/analysis_TIMESTAMP)')
    return parser.parse_args()

def load_data(predictions_path, validation_path=None, tournament='crypto'):
    """Load prediction and validation data"""
    # Load predictions
    if os.path.exists(predictions_path):
        predictions_df = pd.read_csv(predictions_path)
        print(f"Loaded predictions: {predictions_df.shape}")
    else:
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    # Load validation data if provided
    validation_df = None
    if validation_path:
        if os.path.exists(validation_path):
            validation_df = pd.read_csv(validation_path)
            print(f"Loaded validation data: {validation_df.shape}")
        else:
            print(f"Validation file not found: {validation_path}")
    
    # Try to load validation data from default location if not provided
    if validation_df is None:
        try:
            # Initialize data retriever
            retriever = NumeraiDataRetriever()
            validation_df = retriever.load_dataset('validation')
            print(f"Loaded validation data from Numerai: {validation_df.shape}")
        except Exception as e:
            print(f"Could not load validation data: {e}")
    
    return predictions_df, validation_df

def analyze_prediction_distribution(predictions_df, output_dir):
    """Analyze and visualize the distribution of predictions"""
    print("Analyzing prediction distribution...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    preds = predictions_df['prediction']
    stats_data = {
        'count': len(preds),
        'mean': preds.mean(),
        'std': preds.std(),
        'min': preds.min(),
        'max': preds.max(),
        'median': preds.median(),
        'skew': stats.skew(preds),
        'kurtosis': stats.kurtosis(preds)
    }
    
    # Print statistics
    print("\nPrediction Statistics:")
    for key, value in stats_data.items():
        print(f"  {key}: {value:.6f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(preds, kde=True, bins=50)
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.axvline(preds.mean(), color='r', linestyle='--', label=f'Mean: {preds.mean():.4f}')
    plt.axvline(preds.median(), color='g', linestyle='--', label=f'Median: {preds.median():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    
    # Plot Q-Q plot to check for normality
    plt.figure(figsize=(10, 6))
    stats.probplot(preds, plot=plt)
    plt.title('Q-Q Plot of Predictions')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_qq_plot.png'))
    
    # Plot autocorrelation
    plt.figure(figsize=(12, 6))
    plot_acf(preds, lags=50)
    plt.title('Autocorrelation of Predictions')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_autocorrelation.png'))
    
    # Save statistics to file
    with open(os.path.join(output_dir, 'prediction_statistics.json'), 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"Prediction distribution analysis saved to {output_dir}")
    return stats_data

def compare_predictions(pred1_df, pred2_df, output_dir):
    """Compare two sets of predictions"""
    print("Comparing prediction sets...")
    
    # Ensure IDs match and sort by ID
    pred1_df = pred1_df.sort_values('id')
    pred2_df = pred2_df.sort_values('id')
    
    # Check if IDs match
    if not (pred1_df['id'] == pred2_df['id']).all():
        print("Warning: IDs do not match between prediction sets")
        # Join on ID
        comparison_df = pd.merge(
            pred1_df, pred2_df, on='id', suffixes=('_1', '_2')
        )
    else:
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'id': pred1_df['id'],
            'prediction_1': pred1_df['prediction'],
            'prediction_2': pred2_df['prediction'],
        })
    
    # Calculate difference
    comparison_df['difference'] = comparison_df['prediction_1'] - comparison_df['prediction_2']
    
    # Calculate correlation
    correlation = np.corrcoef(comparison_df['prediction_1'], comparison_df['prediction_2'])[0, 1]
    mean_diff = comparison_df['difference'].mean()
    std_diff = comparison_df['difference'].std()
    max_diff = comparison_df['difference'].abs().max()
    
    # Print statistics
    print("\nPrediction Comparison Statistics:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Mean Difference: {mean_diff:.6f}")
    print(f"  Std Deviation of Difference: {std_diff:.6f}")
    print(f"  Max Absolute Difference: {max_diff:.6f}")
    
    # Plot scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_df['prediction_1'], comparison_df['prediction_2'], alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'Prediction Comparison (r = {correlation:.4f})')
    plt.xlabel('Prediction Set 1')
    plt.ylabel('Prediction Set 2')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_comparison_scatter.png'))
    
    # Plot difference histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df['difference'], kde=True, bins=50)
    plt.title('Distribution of Prediction Differences')
    plt.xlabel('Difference (Set 1 - Set 2)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_difference_histogram.png'))
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(
        comparison_df['prediction_1'], 
        comparison_df['prediction_2'], 
        bins=50, 
        cmap=LinearSegmentedColormap.from_list('', ['#FFFFFF', '#3182bd']),
        normed=True
    )
    plt.colorbar(h[3], label='Density')
    plt.title(f'Prediction Comparison Heatmap (r = {correlation:.4f})')
    plt.xlabel('Prediction Set 1')
    plt.ylabel('Prediction Set 2')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'prediction_comparison_heatmap.png'))
    
    # Save statistics
    comparison_stats = {
        'correlation': correlation,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'max_abs_difference': max_diff
    }
    
    with open(os.path.join(output_dir, 'prediction_comparison_statistics.json'), 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    print(f"Prediction comparison analysis saved to {output_dir}")
    return comparison_stats

def evaluate_predictions(predictions_df, validation_df, output_dir):
    """Evaluate predictions against validation data"""
    print("Evaluating predictions against validation data...")
    
    if validation_df is None:
        print("No validation data available for evaluation")
        return None
    
    # Merge predictions with validation data
    merged_df = pd.merge(
        predictions_df, 
        validation_df[['id', 'target']], 
        on='id', 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("No matching IDs between predictions and validation data")
        return None
    
    print(f"Found {len(merged_df)} matching examples for evaluation")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, precision_recall_curve, auc
    
    metrics = {}
    
    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(merged_df['target'], merged_df['prediction'])
    except:
        metrics['roc_auc'] = None
    
    # Log loss
    try:
        metrics['log_loss'] = log_loss(merged_df['target'], merged_df['prediction'])
    except:
        metrics['log_loss'] = None
    
    # Brier score
    try:
        metrics['brier_score'] = brier_score_loss(merged_df['target'], merged_df['prediction'])
    except:
        metrics['brier_score'] = None
    
    # Precision-Recall AUC
    try:
        precision, recall, _ = precision_recall_curve(merged_df['target'], merged_df['prediction'])
        metrics['pr_auc'] = auc(recall, precision)
    except:
        metrics['pr_auc'] = None
    
    # Spearman correlation
    try:
        metrics['spearman_corr'] = stats.spearmanr(merged_df['target'], merged_df['prediction'])[0]
    except:
        metrics['spearman_corr'] = None
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: N/A")
    
    # Plot ROC curve
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(merged_df['target'], merged_df['prediction'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
    
    # Plot precision-recall curve
    try:
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUC = {metrics["pr_auc"]:.4f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    except Exception as e:
        print(f"Could not plot precision-recall curve: {e}")
    
    # Plot prediction vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_df['prediction'], merged_df['target'], alpha=0.1)
    plt.xlabel('Prediction')
    plt.ylabel('Actual Target')
    plt.title('Prediction vs Actual')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {k: float(v) if v is not None else None for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    print(f"Evaluation metrics saved to {output_dir}")
    return metrics

def main():
    """Main function"""
    args = parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, 'reports', f'analysis_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Load data
    predictions_df, validation_df = load_data(
        args.predictions, 
        args.validation, 
        args.tournament
    )
    
    # Analyze prediction distribution
    stats_data = analyze_prediction_distribution(predictions_df, output_dir)
    
    # Compare with second prediction set if provided
    if args.compare:
        try:
            compare_df = pd.read_csv(args.compare)
            print(f"Loaded comparison predictions: {compare_df.shape}")
            compare_predictions(predictions_df, compare_df, output_dir)
        except Exception as e:
            print(f"Error comparing prediction sets: {e}")
    
    # Evaluate against validation data if available
    if validation_df is not None:
        evaluate_predictions(predictions_df, validation_df, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()