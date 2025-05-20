#!/usr/bin/env python3
"""
Symbol management for ensuring consistent symbol selection across training and live predictions.
"""

import os
import logging
import pandas as pd
import polars as pl
from typing import Set, List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SymbolManager:
    """
    Manages symbol consistency between training data and live predictions.
    """
    
    def __init__(self, base_dir: str = "/media/knight2/EDB/numer_crypto_temp"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.numerai_dir = self.data_dir / "numerai"
        
    def get_training_symbols(self, start_date: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Get all symbols available in training data with targets.
        
        Returns:
            Dict with keys 'all', 'recent', 'by_date' containing symbol sets
        """
        logger.info("Getting training symbols from Numerai data...")
        
        # Find the latest training data
        latest_train = self._find_latest_file("train_targets_r*.parquet")
        if not latest_train:
            raise FileNotFoundError("No training data found")
            
        logger.info(f"Using training data: {latest_train}")
        
        # Load training data
        train_df = pl.read_parquet(latest_train)
        
        # Get all unique symbols with targets
        all_symbols = set(train_df['symbol'].unique().to_list())
        logger.info(f"Total unique symbols in training data: {len(all_symbols)}")
        
        # Get recent symbols (last 30 days)
        if start_date is None:
            # Convert date string to datetime if needed
            max_date = train_df['date'].max()
            if isinstance(max_date, str):
                max_date = pd.to_datetime(max_date)
            recent_date = max_date - pd.Timedelta(days=30)
        else:
            recent_date = pd.to_datetime(start_date)
            
        recent_symbols = set(
            train_df.filter(pl.col('date') >= str(recent_date))['symbol'].unique().to_list()
        )
        logger.info(f"Symbols in recent data (since {recent_date}): {len(recent_symbols)}")
        
        # Get symbols by date
        date_symbols = {}
        for date in train_df['date'].unique().sort(descending=True).head(10).to_list():
            symbols = set(train_df.filter(pl.col('date') == date)['symbol'].unique().to_list())
            date_symbols[str(date)] = symbols
            
        return {
            'all': all_symbols,
            'recent': recent_symbols,
            'by_date': date_symbols
        }
    
    def get_live_symbols(self) -> Set[str]:
        """
        Get symbols required for live predictions.
        """
        logger.info("Getting live symbols...")
        
        # Find the latest live data
        latest_live = self._find_latest_file("live_universe_r*.parquet")
        if not latest_live:
            raise FileNotFoundError("No live data found")
            
        logger.info(f"Using live data: {latest_live}")
        
        # Load live data
        live_df = pl.read_parquet(latest_live)
        
        # Get live symbols
        live_symbols = set(live_df['symbol'].unique().to_list())
        logger.info(f"Total unique symbols in live data: {len(live_symbols)}")
        
        return live_symbols
    
    def get_valid_symbols_for_features(self, 
                                     min_history_days: int = 30,
                                     require_recent: bool = True,
                                     _cache: dict = {}) -> Set[str]:
        """
        Get symbols that have sufficient history for feature generation.
        
        Args:
            min_history_days: Minimum days of history required
            require_recent: Whether to require data in recent period
            _cache: Internal cache dictionary to prevent repeated calculations
            
        Returns:
            Set of valid symbols
        """
        # Use cached results if available with the same parameters
        cache_key = f"{min_history_days}_{require_recent}"
        if cache_key in _cache:
            logger.info(f"Using cached valid symbols (cache key: {cache_key})")
            return _cache[cache_key]
            
        logger.info(f"Getting valid symbols for features (min_history={min_history_days} days)...")
        
        # Find the latest training file
        latest_train = self._find_latest_file("train_targets_r*.parquet")
        if not latest_train:
            raise FileNotFoundError("No training data found")
            
        logger.info(f"Using training data: {latest_train}")
        
        # Load training data
        train_df = pl.read_parquet(latest_train)
        
        # Get all unique symbols with targets
        all_symbols = set(train_df['symbol'].unique().to_list())
        logger.info(f"Total unique symbols in training data: {len(all_symbols)}")
        
        # Get recent symbols (last 30 days)
        max_date = train_df['date'].max()
        if isinstance(max_date, str):
            max_date = pd.to_datetime(max_date)
        recent_date = max_date - pd.Timedelta(days=30)
            
        recent_symbols = set(
            train_df.filter(pl.col('date') >= str(recent_date))['symbol'].unique().to_list()
        )
        logger.info(f"Symbols in recent data (since {recent_date}): {len(recent_symbols)}")
        
        # Find the latest live file
        latest_live = self._find_latest_file("live_universe_r*.parquet")
        if not latest_live:
            raise FileNotFoundError("No live data found")
            
        logger.info(f"Using live data: {latest_live}")
        
        # Load live data
        live_df = pl.read_parquet(latest_live)
        
        # Get live symbols
        live_symbols = set(live_df['symbol'].unique().to_list())
        logger.info(f"Total unique symbols in live data: {len(live_symbols)}")
        
        # Find overlap
        overlap_symbols = all_symbols & live_symbols
        logger.info(f"Overlap between training and live: {len(overlap_symbols)} symbols")
        
        # If requiring recent data, further filter
        if require_recent:
            valid_symbols = overlap_symbols & recent_symbols
            logger.info(f"Valid symbols with recent data: {len(valid_symbols)}")
        else:
            valid_symbols = overlap_symbols
            
        # Additional filtering based on data quality
        valid_symbols = self._filter_by_data_quality(valid_symbols, min_history_days)
        
        # Cache the result
        _cache[cache_key] = valid_symbols
        
        return valid_symbols
    
    def _filter_by_data_quality(self, symbols: Set[str], min_days: int) -> Set[str]:
        """
        Filter symbols based on data quality criteria.
        """
        logger.info("Filtering symbols by data quality...")
        
        # Load training data
        latest_train = self._find_latest_file("train_targets_r*.parquet")
        train_df = pl.read_parquet(latest_train)
        
        valid_symbols = set()
        
        for symbol in symbols:
            symbol_data = train_df.filter(pl.col('symbol') == symbol)
            
            # Check history length
            date_count = symbol_data['date'].n_unique()
            if date_count < min_days:
                continue
                
            # Check for data gaps
            # First ensure dates are datetime objects, not strings
            dates = symbol_data['date'].unique().sort()
            
            # Convert to datetime if needed
            if dates.dtype == pl.Utf8:
                # Dates are strings, parse them to dates first
                dates = dates.str.to_date()
            
            # Now calculate differences
            date_diff = dates.diff().drop_nulls()
            max_gap = date_diff.max()
            
            # Skip if too many gaps (convert to days for comparison)
            if max_gap:
                # Handle different types of gap representation
                if hasattr(max_gap, 'total_seconds'):
                    # It's a timedelta or equivalent
                    max_gap_days = max_gap.total_seconds() / (24 * 60 * 60)
                elif isinstance(max_gap, pl.Duration):
                    # It's a polars duration
                    max_gap_days = max_gap.total_seconds() / (24 * 60 * 60)
                elif isinstance(max_gap, pd.Timedelta):
                    # It's a pandas timedelta
                    max_gap_days = max_gap.total_seconds() / (24 * 60 * 60)
                elif isinstance(max_gap, (int, float)):
                    # It's already a number (could be days or millis)
                    max_gap_days = max_gap / (24 * 60 * 60 * 1000) if max_gap > 1000 else max_gap
                else:
                    # Try to parse from string
                    try:
                        max_gap_days = float(str(max_gap).split()[0])
                    except (ValueError, IndexError):
                        max_gap_days = 0
                
                if max_gap_days > 7:
                    continue
                
            valid_symbols.add(symbol)
        
        logger.info(f"Symbols after quality filtering: {len(valid_symbols)}")
        return valid_symbols
    
    def _find_latest_file(self, pattern: str) -> Optional[Path]:
        """
        Find the latest file matching a pattern.
        """
        files = list(self.numerai_dir.rglob(pattern))
        if not files:
            return None
            
        # Sort by modification time
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return latest
    
    def create_feature_generation_config(self, valid_symbols=None) -> Dict:
        """
        Create configuration for feature generation with proper symbol handling.
        
        Args:
            valid_symbols: Optional pre-calculated valid symbols to avoid recursive calls
        """
        logger.info("Creating feature generation config...")
        
        # Get valid symbols if not provided
        if valid_symbols is None:
            logger.info("No valid symbols provided, calculating them...")
            valid_symbols = self.get_valid_symbols_for_features()
        
        # Get date ranges
        latest_train = self._find_latest_file("train_targets_r*.parquet")
        train_df = pl.read_parquet(latest_train)
        
        date_range = {
            'start': train_df['date'].min(),
            'end': train_df['date'].max()
        }
        
        config = {
            'valid_symbols': list(valid_symbols),
            'symbol_count': len(valid_symbols),
            'date_range': date_range,
            'min_history_days': 30,
            'required_columns': ['symbol', 'date', 'target'],
            'feature_params': [
                'rolling',
                'lag',
                'ewm',
                'interaction'
            ],
            'rolling_windows': [7, 14, 30, 60],
            'lag_periods': [1, 2, 3, 5, 7, 14],
            'ewm_spans': [7, 14, 21, 30],
            'include_interactions': True
        }
        
        logger.info(f"Config created with {len(valid_symbols)} valid symbols")
        return config
    
    def save_symbol_mapping(self, output_file: str):
        """
        Save symbol mapping for reference.
        """
        mapping = {
            'training_symbols': self.get_training_symbols(),
            'live_symbols': list(self.get_live_symbols()),
            'valid_feature_symbols': list(self.get_valid_symbols_for_features()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save as JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2, default=str)
            
        logger.info(f"Symbol mapping saved to {output_file}")


def get_symbol_manager(base_dir: Optional[str] = None) -> SymbolManager:
    """
    Get a configured SymbolManager instance.
    """
    if base_dir is None:
        base_dir = "/media/knight2/EDB/numer_crypto_temp"
    return SymbolManager(base_dir)