#!/usr/bin/env python3
"""
Generate features with proper symbol alignment for training and live prediction.
"""

import os
import sys
import argparse
import logging
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.symbol_manager import SymbolManager
from features.polars_generator import PolarsFeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignedFeatureGenerator:
    """
    Feature generator that ensures symbol alignment between training and live data.
    """
    
    def __init__(self, base_dir: str = "/media/knight2/EDB/numer_crypto_temp"):
        self.base_dir = Path(base_dir)
        self.symbol_manager = SymbolManager(base_dir)
        self.polars_generator = PolarsFeatureGenerator(
            output_dir=str(self.base_dir / "data" / "features"),
            max_features=100000  # Default, can be overridden
        )
        
    def generate_features(self, max_features: int = 100000):
        """
        Generate features with proper symbol alignment.
        """
        logger.info(f"Starting aligned feature generation with max_features={max_features}")
        
        # Get valid symbols for validation and submission template - only calculate this once
        # This is where the infinite loop was happening before
        start_time = datetime.now()
        logger.info(f"Calculating valid symbols at {start_time}")
        valid_symbols = self.symbol_manager.get_valid_symbols_for_features()
        end_time = datetime.now()
        logger.info(f"Found {len(valid_symbols)} valid symbols for features (calculation took {end_time - start_time})")
        
        # Get all symbols with targets
        # Use direct file reading instead of get_training_symbols to avoid another potential recursive call
        latest_train = self._find_latest_training_file()
        logger.info(f"Loading training data from: {latest_train}")
        
        # Load data
        train_df = pl.read_parquet(latest_train)
        
        # Get all symbols directly from the dataframe
        all_symbols_with_targets = set(train_df['symbol'].unique().to_list())
        logger.info(f"Found {len(all_symbols_with_targets)} total symbols with targets")
        
        # Get feature generation config - pass the calculated valid_symbols to avoid recursion
        config = self.symbol_manager.create_feature_generation_config(valid_symbols=valid_symbols)
        
        # Update polars generator settings
        self.polars_generator.max_features = max_features
        
        # We already loaded the training data, so reuse it
        logger.info(f"Loaded training data shape: {train_df.shape}")
        
        # Generate features using polars generator for ALL symbols
        feature_file = self._generate_polars_features(train_df, config)
        
        # Verify feature alignment - use the already calculated valid_symbols
        self._verify_feature_alignment(feature_file, valid_symbols)
        
        # Create submission template - will also use the already calculated valid_symbols
        self._create_submission_template(feature_file)
        
        return feature_file
    
    def _generate_polars_features(self, train_df: pl.DataFrame, config: dict) -> str:
        """
        Generate features using the polars generator.
        """
        logger.info("Generating features with polars...")
        
        # Save training data to temporary file for polars generator
        temp_file = self.base_dir / "data" / "temp_train_aligned.parquet"
        train_df.write_parquet(temp_file)
        
        # Generate features
        output_file = self.polars_generator.generate_features(
            input_file=str(temp_file),
            feature_modules=config['feature_params']
        )
        
        # Clean up temp file
        temp_file.unlink()
        
        return output_file
    
    def _verify_feature_alignment(self, feature_file: str, valid_symbols: set):
        """
        Verify that features are properly aligned with valid symbols.
        """
        logger.info("Verifying feature alignment...")
        
        # Load features
        feature_df = pl.read_parquet(feature_file)
        
        # Get unique symbols in features
        feature_symbols = set(feature_df['symbol'].unique().to_list())
        
        # Check alignment
        missing_symbols = valid_symbols - feature_symbols
        extra_symbols = feature_symbols - valid_symbols
        
        logger.info(f"Feature symbols: {len(feature_symbols)}")
        logger.info(f"Valid symbols: {len(valid_symbols)}")
        logger.info(f"Missing symbols: {len(missing_symbols)}")
        logger.info(f"Extra symbols: {len(extra_symbols)}")
        
        if missing_symbols:
            logger.warning(f"Missing symbols in features: {list(missing_symbols)[:10]}...")
            
        # Check latest date coverage
        latest_date = feature_df['date'].max()
        latest_symbols = set(
            feature_df.filter(pl.col('date') == latest_date)['symbol'].unique().to_list()
        )
        
        logger.info(f"Symbols on latest date ({latest_date}): {len(latest_symbols)}")
        
        # Verify overlap with live
        live_symbols = self.symbol_manager.get_live_symbols()
        live_overlap = latest_symbols & live_symbols
        
        logger.info(f"Overlap with live symbols: {len(live_overlap)}")
        
        if len(live_overlap) < 100:
            logger.warning(f"Insufficient overlap with live symbols: {len(live_overlap)}")
    
    def _create_submission_template(self, feature_file: str):
        """
        Create a submission template with proper symbols.
        """
        logger.info("Creating submission template...")
        
        # Load features
        feature_df = pl.read_parquet(feature_file)
        
        # Get latest date
        latest_date = feature_df['date'].max()
        
        # Get symbols on latest date
        latest_df = feature_df.filter(pl.col('date') == latest_date)
        
        # Get live symbols
        live_symbols = self.symbol_manager.get_live_symbols()
        
        # Get all feature symbols for enhanced visibility
        feature_symbols = set(feature_df['symbol'].unique().to_list())
        latest_date_symbols = set(latest_df['symbol'].unique().to_list())
        
        logger.info(f"Total feature symbols: {len(feature_symbols)}")
        logger.info(f"Symbols on latest date: {len(latest_date_symbols)}")
        logger.info(f"Live universe symbols: {len(live_symbols)}")
        
        # Get valid symbols once to avoid repeated calculations (causing infinite loop)
        valid_symbols = self.symbol_manager.get_valid_symbols_for_features()
        logger.info(f"Precalculated valid symbols: {len(valid_symbols)}")
        
        # Create template for all live symbols
        template_data = []
        
        for symbol in live_symbols:
            if symbol in latest_df['symbol'].to_list():
                # Symbol has features
                template_data.append({
                    'symbol': symbol,
                    'has_features': True,
                    'feature_date': latest_date,
                    'high_quality': symbol in valid_symbols  # Use precalculated valid symbols
                })
            else:
                # Symbol missing features on latest date - check if it exists in any features
                if symbol in feature_symbols:
                    # Has some features but not on the latest date
                    max_date = feature_df.filter(pl.col('symbol') == symbol)['date'].max()
                    template_data.append({
                        'symbol': symbol,
                        'has_features': False,
                        'feature_date': max_date,
                        'high_quality': False
                    })
                else:
                    # No features at all - will need default prediction
                    template_data.append({
                        'symbol': symbol,
                        'has_features': False,
                        'feature_date': None,
                        'high_quality': False
                    })
        
        # Save template
        template_df = pd.DataFrame(template_data)
        template_file = self.base_dir / "data" / "features" / "submission_template.csv"
        template_df.to_csv(template_file, index=False)
        
        # Categorize quality
        high_quality_symbols = template_df[template_df['high_quality'] == True]
        has_features_symbols = template_df[template_df['has_features'] == True]
        missing_features_symbols = template_df[template_df['has_features'] == False]
        
        logger.info(f"Submission template saved to: {template_file}")
        logger.info(f"High quality symbols: {len(high_quality_symbols)} ({len(high_quality_symbols)/len(template_df)*100:.1f}%)")
        logger.info(f"Symbols with features on latest date: {len(has_features_symbols)} ({len(has_features_symbols)/len(template_df)*100:.1f}%)")
        logger.info(f"Symbols without features on latest date: {len(missing_features_symbols)} ({len(missing_features_symbols)/len(template_df)*100:.1f}%)")
    
    def _find_latest_training_file(self) -> Path:
        """
        Find the latest training file.
        """
        pattern = "train_targets_r*.parquet"
        files = list(self.base_dir.rglob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No training files found matching {pattern}")
            
        # Sort by modification time
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return latest


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate aligned features')
    parser.add_argument('--max-features', type=int, default=100000, 
                       help='Maximum number of features to generate')
    parser.add_argument('--output-dir', type=str, 
                       default='/media/knight2/EDB/numer_crypto_temp/data/features',
                       help='Output directory for features')
    parser.add_argument('--save-feature-info', action='store_true',
                       help='Save feature information file in JSON format')
    
    args = parser.parse_args()
    
    # Create generator
    generator = AlignedFeatureGenerator()
    
    # Generate features
    feature_file = generator.generate_features(max_features=args.max_features)
    
    logger.info(f"Feature generation complete: {feature_file}")
    
    # Gather statistics
    feature_df = pl.read_parquet(feature_file)
    feature_symbols = set(feature_df['symbol'].unique().to_list())
    latest_date = feature_df['date'].max()
    latest_symbols = set(feature_df.filter(pl.col('date') == latest_date)['symbol'].unique().to_list())
    
    # Get different symbol sets for comparison
    training_symbols = generator.symbol_manager.get_training_symbols()
    live_symbols = generator.symbol_manager.get_live_symbols()
    filtered_symbols = generator.symbol_manager.get_valid_symbols_for_features()
    
    # Log comprehensive statistics
    logger.info(f"Feature Statistics Summary:")
    logger.info(f"  - Total training symbols: {len(training_symbols['all'])}")
    logger.info(f"  - Total live symbols: {len(live_symbols)}")
    logger.info(f"  - Quality-filtered symbols: {len(filtered_symbols)}")
    logger.info(f"  - Symbols in feature file: {len(feature_symbols)}")
    logger.info(f"  - Symbols on latest date: {len(latest_symbols)}")
    logger.info(f"  - Live symbols with features: {len(latest_symbols & live_symbols)}")
    
    # Save enhanced symbol mapping
    mapping_file = Path(args.output_dir) / f"symbol_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create enhanced mapping
    import json
    mapping = {
        'training_symbols': {
            'all': list(training_symbols['all']),
            'recent': list(training_symbols['recent']),
            'count': {
                'all': len(training_symbols['all']),
                'recent': len(training_symbols['recent'])
            }
        },
        'live_symbols': list(live_symbols),
        'live_count': len(live_symbols),
        'filtered_symbols': list(filtered_symbols),
        'filtered_count': len(filtered_symbols),
        'feature_symbols': list(feature_symbols),
        'feature_count': len(feature_symbols),
        'latest_date_symbols': list(latest_symbols),
        'latest_date_count': len(latest_symbols),
        'live_coverage': {
            'count': len(latest_symbols & live_symbols),
            'percentage': len(latest_symbols & live_symbols) / len(live_symbols) * 100 if live_symbols else 0
        },
        'timestamp': datetime.now().isoformat(),
        'feature_file': str(feature_file)
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2, default=str)
    
    logger.info(f"Enhanced symbol mapping saved to: {mapping_file}")
    
    # Save feature info file if requested
    if args.save_feature_info:
        feature_info_file = Path(args.output_dir) / f"polars_feature_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get column information
        exclude_cols = ['symbol', 'date', 'era', 'id', 'target']
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Get feature types
        feature_types = {
            'rolling': len([col for col in feature_cols if 'roll_' in col]),
            'lag': len([col for col in feature_cols if 'lag_' in col]),
            'ewm': len([col for col in feature_cols if 'ewm_' in col]),
            'interaction': len([col for col in feature_cols if '_X_' in col or '_DIV_' in col]),
            'base': len([col for col in feature_cols if not ('roll_' in col or 'lag_' in col or 
                                                             'ewm_' in col or '_X_' in col or '_DIV_' in col)])
        }
        
        feature_info = {
            'feature_file': str(feature_file),
            'total_features': len(feature_cols),
            'feature_types': feature_types,
            'row_count': feature_df.shape[0],
            'symbol_count': len(feature_symbols),
            'latest_date': str(latest_date),
            'latest_symbols_count': len(latest_symbols),
            'live_coverage': mapping['live_coverage'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(feature_info_file, 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)
            
        logger.info(f"Feature information saved to: {feature_info_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())