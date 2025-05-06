#!/usr/bin/env python3
"""
Exploratory Data Analysis for Numerai Crypto Data

This module provides tools for exploring and visualizing Numerai crypto datasets,
analyzing feature distributions, and identifying patterns in the data.
Results are saved in the external EDA directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime

# Constants
EDA_OUTPUT_DIR = "/media/knight2/EDB/cryptos/EDA"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """Performs exploratory data analysis on crypto datasets"""
    
    def __init__(self, output_dir=EDA_OUTPUT_DIR):
        """Initialize the EDA class with output directory"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a run-specific output folder
        self.run_dir = os.path.join(self.output_dir, f"eda_run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        logger.info(f"EDA initialized. Output will be saved to {self.run_dir}")
    
    def analyze_dataset(self, df, dataset_name="dataset"):
        """
        Run a comprehensive analysis on the dataset
        
        Args:
            df (DataFrame): Pandas DataFrame containing the dataset
            dataset_name (str): Name of the dataset for labeling
        """
        logger.info(f"Starting analysis of {dataset_name} with shape {df.shape}")
        
        # Basic dataset stats
        self._analyze_basic_stats(df, dataset_name)
        
        # Feature type analysis
        self._analyze_feature_types(df, dataset_name)
        
        # Feature distribution analysis
        self._analyze_distributions(df, dataset_name)
        
        # Time series analysis (if date column exists)
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            self._analyze_time_series(df, date_cols[0], dataset_name)
        
        # Feature correlations
        self._analyze_correlations(df, dataset_name)
        
        # Generate summary report
        self._generate_summary_report(df, dataset_name)
        
        logger.info(f"Analysis of {dataset_name} completed")
    
    def _analyze_basic_stats(self, df, dataset_name):
        """Analyze basic statistics of the dataset"""
        logger.info(f"Analyzing basic statistics for {dataset_name}")
        
        # Create output directory
        output_dir = os.path.join(self.run_dir, "basic_stats")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get basic info
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "missing_values": df.isna().sum().sum(),
            "missing_percentage": df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        }
        
        # Save basic info
        with open(os.path.join(output_dir, f"{dataset_name}_info.json"), 'w') as f:
            json.dump(info, f, indent=4)
        
        # Generate missing values heatmap
        missing_values = df.isna().sum().sort_values(ascending=False)
        if missing_values.sum() > 0:
            top_missing = missing_values[missing_values > 0].head(30)
            plt.figure(figsize=(12, 8))
            top_missing.plot(kind='bar')
            plt.title('Top Columns with Missing Values')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_missing_values.png"))
            plt.close()
        
        # Generate basic stats
        stats = df.describe(include='all').transpose()
        stats.to_csv(os.path.join(output_dir, f"{dataset_name}_stats.csv"))
        
        logger.info(f"Basic statistics saved to {output_dir}")
    
    def _analyze_feature_types(self, df, dataset_name):
        """Analyze feature types and column patterns"""
        logger.info(f"Analyzing feature types for {dataset_name}")
        
        # Create output directory
        output_dir = os.path.join(self.run_dir, "feature_types")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get column types
        dtypes = df.dtypes.value_counts()
        
        # Get column prefixes (e.g., pvm_*, sentiment_*)
        prefixes = {}
        for col in df.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        # Plot column type distribution
        plt.figure(figsize=(10, 6))
        dtypes.plot(kind='bar')
        plt.title('Column Data Types')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_dtypes.png"))
        plt.close()
        
        # Plot column prefix distribution
        if prefixes:
            plt.figure(figsize=(12, 8))
            pd.Series(prefixes).sort_values(ascending=False).plot(kind='bar')
            plt.title('Column Prefixes')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_prefixes.png"))
            plt.close()
        
        # Save info to JSON
        with open(os.path.join(output_dir, f"{dataset_name}_feature_types.json"), 'w') as f:
            json.dump({
                "dtypes": dtypes.to_dict(),
                "prefixes": prefixes
            }, f, indent=4)
        
        logger.info(f"Feature type analysis saved to {output_dir}")
    
    def _analyze_distributions(self, df, dataset_name):
        """Analyze feature distributions"""
        logger.info(f"Analyzing feature distributions for {dataset_name}")
        
        # Create output directory
        output_dir = os.path.join(self.run_dir, "distributions")
        os.makedirs(output_dir, exist_ok=True)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Limit to 50 columns for visualization
        if len(numeric_cols) > 50:
            # Prefer columns that might be important (like target, price, etc.)
            priority_keywords = ['target', 'price', 'return', 'pvm', 'sentiment']
            priority_cols = []
            
            for keyword in priority_keywords:
                priority_cols.extend([col for col in numeric_cols if keyword in col.lower()])
            
            remaining_cols = [col for col in numeric_cols if col not in priority_cols]
            
            # Sample remaining columns if needed
            if len(priority_cols) < 50:
                remaining_sample = np.random.choice(
                    remaining_cols, 
                    size=min(50 - len(priority_cols), len(remaining_cols)), 
                    replace=False
                )
                plot_cols = list(priority_cols) + list(remaining_sample)
            else:
                plot_cols = priority_cols[:50]
        else:
            plot_cols = numeric_cols
        
        # Generate histograms
        for i, col in enumerate(plot_cols):
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_{col}_hist.png"))
                plt.close()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(plot_cols)} histograms")
            except Exception as e:
                logger.warning(f"Failed to plot histogram for {col}: {str(e)}")
        
        # Generate box plots for features grouped by column prefix
        prefixes = {}
        for col in plot_cols:
            if '_' in col:
                prefix = col.split('_')[0]
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(col)
        
        for prefix, cols in prefixes.items():
            if len(cols) <= 1:
                continue
            
            try:
                # Limit to 20 columns per prefix
                sample_cols = cols[:20] if len(cols) > 20 else cols
                
                # Get data for boxplot
                data = df[sample_cols].melt()
                
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='variable', y='value', data=data)
                plt.xticks(rotation=90)
                plt.title(f'Box Plot of {prefix} Features')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_{prefix}_boxplot.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to create box plot for {prefix}: {str(e)}")
        
        logger.info(f"Distribution analysis saved to {output_dir}")
    
    def _analyze_time_series(self, df, date_col, dataset_name):
        """Analyze time series patterns"""
        logger.info(f"Analyzing time series data for {dataset_name}")
        
        # Create output directory
        output_dir = os.path.join(self.run_dir, "time_series")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert date column if needed
        if df[date_col].dtype != 'datetime64[ns]':
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
                return
        
        # Check if symbol column exists for grouping
        symbol_col = None
        for col in df.columns:
            if col.lower() in ['symbol', 'asset', 'crypto', 'coin']:
                symbol_col = col
                break
        
        # Temporal distribution analysis
        try:
            # Resample by day
            date_counts = df[date_col].value_counts().sort_index()
            plt.figure(figsize=(14, 8))
            date_counts.plot(kind='line')
            plt.title('Data Points Over Time')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_time_distribution.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot time distribution: {str(e)}")
        
        # If symbol column exists, plot time series by symbol
        if symbol_col:
            try:
                symbols = df[symbol_col].unique()
                
                # Limit to 10 symbols for clarity
                plot_symbols = symbols[:10] if len(symbols) > 10 else symbols
                
                # Find columns that might be price-related
                price_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['price', 'close', 'pvm'])]
                
                if price_cols:
                    price_col = price_cols[0]  # Use the first price column
                    
                    plt.figure(figsize=(14, 8))
                    for symbol in plot_symbols:
                        symbol_data = df[df[symbol_col] == symbol]
                        symbol_data = symbol_data.sort_values(date_col)
                        plt.plot(symbol_data[date_col], symbol_data[price_col], label=symbol)
                    
                    plt.title(f'Price Series by Symbol ({price_col})')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{dataset_name}_price_by_symbol.png"))
                    plt.close()
            except Exception as e:
                logger.warning(f"Failed to plot time series by symbol: {str(e)}")
        
        logger.info(f"Time series analysis saved to {output_dir}")
    
    def _analyze_correlations(self, df, dataset_name):
        """Analyze feature correlations"""
        logger.info(f"Analyzing feature correlations for {dataset_name}")
        
        # Create output directory
        output_dir = os.path.join(self.run_dir, "correlations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Check if target column exists
        target_col = None
        for col in df.columns:
            if col.lower() == 'target':
                target_col = col
                break
        
        # If we have too many features, select a subset
        if len(numeric_cols) > 100:
            # If target exists, get highest correlated features
            if target_col:
                correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                top_correlated = correlations.head(99).index.tolist()  # Top 99 + target
                if target_col not in top_correlated:
                    top_correlated.append(target_col)
                selected_cols = top_correlated
            else:
                # Randomly select features
                selected_cols = np.random.choice(numeric_cols, size=100, replace=False)
        else:
            selected_cols = numeric_cols
        
        # Generate correlation matrix
        try:
            corr_matrix = df[selected_cols].corr()
            
            # Plot heatmap
            plt.figure(figsize=(20, 16))
            sns.heatmap(
                corr_matrix, 
                cmap='coolwarm', 
                annot=False, 
                linewidths=0.5,
                vmin=-1, 
                vmax=1
            )
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_correlation_matrix.png"))
            plt.close()
            
            # Save correlation matrix to CSV
            corr_matrix.to_csv(os.path.join(output_dir, f"{dataset_name}_correlation_matrix.csv"))
            
            # If target exists, show top correlated features
            if target_col:
                target_corr = corr_matrix[target_col].abs().sort_values(ascending=False).drop(target_col)
                top_target_corr = target_corr.head(30)
                
                plt.figure(figsize=(12, 8))
                top_target_corr.plot(kind='bar')
                plt.title(f'Top 30 Features Correlated with {target_col}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{dataset_name}_target_correlations.png"))
                plt.close()
                
                # Save target correlations to CSV
                target_corr.to_csv(os.path.join(output_dir, f"{dataset_name}_target_correlations.csv"))
        except Exception as e:
            logger.warning(f"Failed to generate correlation analysis: {str(e)}")
        
        logger.info(f"Correlation analysis saved to {output_dir}")
    
    def _generate_summary_report(self, df, dataset_name):
        """Generate a summary report of the EDA process"""
        logger.info(f"Generating summary report for {dataset_name}")
        
        # Create report content
        report = f"""
        # Exploratory Data Analysis Report: {dataset_name}
        
        Analysis timestamp: {self.timestamp}
        
        ## Dataset Overview
        
        - Rows: {len(df):,}
        - Columns: {len(df.columns):,}
        - Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB
        - Missing Values: {df.isna().sum().sum():,} ({df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)
        
        ## Column Type Distribution
        
        ```
        {df.dtypes.value_counts().to_string()}
        ```
        
        ## Column Prefixes
        
        {self._get_column_prefix_summary(df)}
        
        ## Missing Values
        
        {self._get_missing_values_summary(df)}
        
        ## Target Analysis
        
        {self._get_target_analysis(df)}
        
        ## Key Observations
        
        - Analysis directories have been created in: {self.run_dir}
        - Histograms and other visualizations are available in their respective folders
        - For more detailed analysis, refer to the generated CSV files
        """
        
        # Save report to file
        with open(os.path.join(self.run_dir, f"{dataset_name}_eda_report.md"), 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {os.path.join(self.run_dir, f'{dataset_name}_eda_report.md')}")
    
    def _get_column_prefix_summary(self, df):
        """Get summary of column prefixes"""
        prefixes = {}
        for col in df.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        if not prefixes:
            return "No column prefixes found."
        
        prefix_summary = "Most common column prefixes:\n\n```\n"
        for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
            prefix_summary += f"{prefix}: {count}\n"
        prefix_summary += "```"
        
        return prefix_summary
    
    def _get_missing_values_summary(self, df):
        """Get summary of missing values"""
        missing_values = df.isna().sum()
        missing_pct = (df.isna().sum() / len(df)) * 100
        
        if missing_values.sum() == 0:
            return "No missing values found in the dataset."
        
        missing_summary = "Columns with highest missing value percentages:\n\n```\n"
        missing_df = pd.DataFrame({'count': missing_values, 'percentage': missing_pct})
        missing_df = missing_df[missing_df['count'] > 0].sort_values('percentage', ascending=False)
        
        for col, row in missing_df.head(10).iterrows():
            missing_summary += f"{col}: {row['count']:,} ({row['percentage']:.2f}%)\n"
        missing_summary += "```"
        
        return missing_summary
    
    def _get_target_analysis(self, df):
        """Get analysis of target variable if it exists"""
        target_col = None
        for col in df.columns:
            if col.lower() == 'target':
                target_col = col
                break
        
        if not target_col:
            return "No target column found in the dataset."
        
        if df[target_col].dtype not in ['int64', 'float64']:
            return f"Target column '{target_col}' is not numeric."
        
        target_summary = f"Target column: {target_col}\n\n"
        target_summary += "```\n"
        target_summary += f"Min: {df[target_col].min()}\n"
        target_summary += f"Max: {df[target_col].max()}\n"
        target_summary += f"Mean: {df[target_col].mean()}\n"
        target_summary += f"Median: {df[target_col].median()}\n"
        target_summary += f"Std Dev: {df[target_col].std()}\n"
        target_summary += "```\n\n"
        
        # Get high correlation features
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if numeric_cols:
            correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            
            target_summary += "Top 5 features correlated with target:\n\n```\n"
            for col, corr in correlations.head(5).items():
                target_summary += f"{col}: {corr:.4f}\n"
            target_summary += "```"
        
        return target_summary


def main():
    """Main function to run the EDA process on a sample dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run exploratory data analysis on Numerai Crypto data')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input dataset (parquet format)')
    parser.add_argument('--name', '-n', type=str, default='dataset', help='Name of the dataset')
    parser.add_argument('--output', '-o', type=str, default=EDA_OUTPUT_DIR, help='Output directory for EDA results')
    
    args = parser.parse_args()
    
    logger.info(f"Starting EDA process for {args.input}")
    
    try:
        # Load data
        df = pd.read_parquet(args.input)
        
        # Run EDA
        eda = ExploratoryAnalysis(output_dir=args.output)
        eda.analyze_dataset(df, dataset_name=args.name)
        
        logger.info(f"EDA process completed. Results saved to {eda.run_dir}")
    except Exception as e:
        logger.error(f"Error during EDA process: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()