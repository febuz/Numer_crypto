#!/usr/bin/env python3
"""
Time series feature engineering utilities.

This module provides functions for creating time series features
for cryptocurrency data, including lag features, rolling statistics,
and technical indicators.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Callable

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.log_utils import setup_logging
from utils.memory_utils import optimize_dataframe_memory, log_memory_usage

# Set up logging
logger = setup_logging(name=__name__, level=logging.INFO)

def create_lag_features(df: pd.DataFrame,
                       group_col: str,
                       target_cols: List[str],
                       lag_periods: List[int],
                       date_col: str = 'date') -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        target_cols (List[str]): Columns to create lags for
        lag_periods (List[int]): List of lag periods
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with lag features
    """
    log_memory_usage("Before creating lag features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Sort the DataFrame by group and date
    if date_col in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
            try:
                # Convert date column to datetime if it's not already
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime. Using original ordering.")
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
    
    # Create lag features for each target column and lag period
    for col in target_cols:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping lag creation.")
            continue
        
        for lag in lag_periods:
            lag_name = f"{col}_lag_{lag}"
            result_df[lag_name] = result_df.groupby(group_col)[col].shift(lag)
            
            # Log progress
            if lag == lag_periods[-1] and col == target_cols[-1]:
                logger.info(f"Created {len(lag_periods) * len(target_cols)} lag features")
    
    log_memory_usage("After creating lag features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_rolling_features(df: pd.DataFrame,
                           group_col: str,
                           target_cols: List[str],
                           windows: List[int],
                           functions: Dict[str, Callable] = None,
                           date_col: str = 'date') -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        target_cols (List[str]): Columns to create rolling features for
        windows (List[int]): List of rolling window sizes
        functions (Dict[str, Callable]): Dictionary of function names and functions to apply
                           Default: {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with rolling features
    """
    log_memory_usage("Before creating rolling features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Default functions if not provided
    if functions is None:
        functions = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max
        }
    
    # Sort the DataFrame by group and date
    if date_col in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
            try:
                # Convert date column to datetime if it's not already
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime. Using original ordering.")
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
    
    # Create rolling features for each target column, window, and function
    feature_count = 0
    for col in target_cols:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping rolling feature creation.")
            continue
        
        for window in windows:
            for func_name, func in functions.items():
                feature_name = f"{col}_roll_{window}_{func_name}"
                result_df[feature_name] = result_df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).apply(func, raw=True)
                )
                feature_count += 1
    
    logger.info(f"Created {feature_count} rolling features")
    log_memory_usage("After creating rolling features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_ewm_features(df: pd.DataFrame,
                       group_col: str,
                       target_cols: List[str],
                       alphas: List[float],
                       functions: Dict[str, Callable] = None,
                       date_col: str = 'date') -> pd.DataFrame:
    """
    Create exponentially weighted moving average features for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        target_cols (List[str]): Columns to create EWM features for
        alphas (List[float]): List of smoothing factors (higher alpha gives more weight to recent observations)
        functions (Dict[str, Callable]): Dictionary of function names and functions to apply
                           Default: {'mean': lambda x: x.mean(), 'std': lambda x: x.std()}
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with EWM features
    """
    log_memory_usage("Before creating EWM features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Default functions if not provided
    if functions is None:
        functions = {
            'mean': lambda x: x.mean(),
            'std': lambda x: x.std()
        }
    
    # Sort the DataFrame by group and date
    if date_col in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
            try:
                # Convert date column to datetime if it's not already
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime. Using original ordering.")
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
    
    # Create EWM features for each target column, alpha, and function
    feature_count = 0
    for col in target_cols:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping EWM feature creation.")
            continue
        
        for alpha in alphas:
            for func_name, func in functions.items():
                feature_name = f"{col}_ewm_{alpha:.2f}_{func_name}"
                # Replace . with _ in feature name for better compatibility
                feature_name = feature_name.replace('.', '_')
                
                result_df[feature_name] = result_df.groupby(group_col)[col].transform(
                    lambda x: x.ewm(alpha=alpha, min_periods=1).agg(func)
                )
                feature_count += 1
    
    logger.info(f"Created {feature_count} EWM features")
    log_memory_usage("After creating EWM features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_diff_features(df: pd.DataFrame,
                        group_col: str,
                        target_cols: List[str],
                        periods: List[int],
                        date_col: str = 'date') -> pd.DataFrame:
    """
    Create differencing features for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        target_cols (List[str]): Columns to create difference features for
        periods (List[int]): List of periods for differencing
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with difference features
    """
    log_memory_usage("Before creating diff features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Sort the DataFrame by group and date
    if date_col in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
            try:
                # Convert date column to datetime if it's not already
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime. Using original ordering.")
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
    
    # Create differencing features for each target column and period
    feature_count = 0
    for col in target_cols:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping diff feature creation.")
            continue
        
        for period in periods:
            # Absolute difference
            feature_name = f"{col}_diff_{period}"
            result_df[feature_name] = result_df.groupby(group_col)[col].diff(period)
            
            # Percentage difference
            pct_feature_name = f"{col}_pct_diff_{period}"
            result_df[pct_feature_name] = result_df.groupby(group_col)[col].pct_change(period)
            
            feature_count += 2
    
    logger.info(f"Created {feature_count} differencing features")
    log_memory_usage("After creating diff features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_technical_indicators(df: pd.DataFrame,
                               group_col: str,
                               price_col: str = 'price',
                               volume_col: Optional[str] = 'volume',
                               high_col: Optional[str] = None,
                               low_col: Optional[str] = None,
                               date_col: str = 'date') -> pd.DataFrame:
    """
    Create common technical indicators for financial time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        price_col (str): Column containing price data
        volume_col (str, optional): Column containing volume data
        high_col (str, optional): Column containing high price data
        low_col (str, optional): Column containing low price data
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    try:
        import talib
        HAS_TALIB = True
    except ImportError:
        logger.warning("TA-Lib not installed. Using simple implementations instead.")
        HAS_TALIB = False
    
    log_memory_usage("Before creating technical indicators:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Sort the DataFrame by group and date
    if date_col in result_df.columns:
        if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
            try:
                # Convert date column to datetime if it's not already
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime. Using original ordering.")
        
        # Sort by group and date
        result_df = result_df.sort_values([group_col, date_col])
    
    # Check if required columns exist
    if price_col not in result_df.columns:
        logger.error(f"Price column '{price_col}' not found in DataFrame. Cannot create technical indicators.")
        return result_df
    
    # If high/low columns not provided, use price column
    if high_col is None:
        high_col = price_col
    if low_col is None:
        low_col = price_col
    
    # Create technical indicators
    feature_count = 0
    for asset, group in result_df.groupby(group_col):
        # Get price and volume data
        close_data = group[price_col].values
        
        # Initialize indicators for this group
        group_idx = group.index
        
        # --- Moving Average Indicators ---
        # Simple Moving Averages (SMA)
        sma_periods = [5, 10, 20, 50, 100]
        for period in sma_periods:
            feature_name = f"SMA_{period}"
            if HAS_TALIB:
                result_df.loc[group_idx, feature_name] = talib.SMA(close_data, timeperiod=period)
            else:
                result_df.loc[group_idx, feature_name] = pd.Series(close_data).rolling(window=period, min_periods=1).mean().values
            feature_count += 1
        
        # Exponential Moving Averages (EMA)
        ema_periods = [5, 10, 20, 50, 100]
        for period in ema_periods:
            feature_name = f"EMA_{period}"
            if HAS_TALIB:
                result_df.loc[group_idx, feature_name] = talib.EMA(close_data, timeperiod=period)
            else:
                result_df.loc[group_idx, feature_name] = pd.Series(close_data).ewm(span=period, min_periods=1, adjust=False).mean().values
            feature_count += 1
        
        # --- Oscillator Indicators ---
        # Relative Strength Index (RSI)
        rsi_periods = [14, 21]
        for period in rsi_periods:
            feature_name = f"RSI_{period}"
            if HAS_TALIB:
                result_df.loc[group_idx, feature_name] = talib.RSI(close_data, timeperiod=period)
            else:
                # Simple RSI implementation
                delta = pd.Series(close_data).diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                result_df.loc[group_idx, feature_name] = rsi.values
            feature_count += 1
        
        # MACD
        if HAS_TALIB:
            macd, macdsignal, macdhist = talib.MACD(close_data, fastperiod=12, slowperiod=26, signalperiod=9)
            result_df.loc[group_idx, "MACD"] = macd
            result_df.loc[group_idx, "MACD_signal"] = macdsignal
            result_df.loc[group_idx, "MACD_hist"] = macdhist
            feature_count += 3
        else:
            # Simple MACD implementation
            ema_12 = pd.Series(close_data).ewm(span=12, min_periods=1, adjust=False).mean()
            ema_26 = pd.Series(close_data).ewm(span=26, min_periods=1, adjust=False).mean()
            macd = ema_12 - ema_26
            macdsignal = macd.ewm(span=9, min_periods=1, adjust=False).mean()
            macdhist = macd - macdsignal
            result_df.loc[group_idx, "MACD"] = macd.values
            result_df.loc[group_idx, "MACD_signal"] = macdsignal.values
            result_df.loc[group_idx, "MACD_hist"] = macdhist.values
            feature_count += 3
        
        # --- Volatility Indicators ---
        # Bollinger Bands
        if HAS_TALIB:
            upperband, middleband, lowerband = talib.BBANDS(close_data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            result_df.loc[group_idx, "BB_upper"] = upperband
            result_df.loc[group_idx, "BB_middle"] = middleband
            result_df.loc[group_idx, "BB_lower"] = lowerband
            # Calculate BB Width
            result_df.loc[group_idx, "BB_width"] = (upperband - lowerband) / middleband
            feature_count += 4
        else:
            # Simple Bollinger Bands implementation
            sma_20 = pd.Series(close_data).rolling(window=20, min_periods=1).mean()
            std_20 = pd.Series(close_data).rolling(window=20, min_periods=1).std()
            upperband = sma_20 + 2 * std_20
            lowerband = sma_20 - 2 * std_20
            result_df.loc[group_idx, "BB_upper"] = upperband.values
            result_df.loc[group_idx, "BB_middle"] = sma_20.values
            result_df.loc[group_idx, "BB_lower"] = lowerband.values
            result_df.loc[group_idx, "BB_width"] = ((upperband - lowerband) / sma_20).values
            feature_count += 4
        
        # --- Volume-based Indicators ---
        if volume_col in result_df.columns:
            volume_data = group[volume_col].values
            
            # On-Balance Volume (OBV)
            if HAS_TALIB:
                result_df.loc[group_idx, "OBV"] = talib.OBV(close_data, volume_data)
                feature_count += 1
            else:
                # Simple OBV implementation
                close_diff = pd.Series(close_data).diff()
                obv = pd.Series(0, index=range(len(close_data)))
                for i in range(1, len(close_data)):
                    if close_diff.iloc[i] > 0:
                        obv.iloc[i] = obv.iloc[i-1] + volume_data[i]
                    elif close_diff.iloc[i] < 0:
                        obv.iloc[i] = obv.iloc[i-1] - volume_data[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                result_df.loc[group_idx, "OBV"] = obv.values
                feature_count += 1
    
    logger.info(f"Created {feature_count} technical indicators")
    log_memory_usage("After creating technical indicators:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_date_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create features from date column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Date column name
        
    Returns:
        pd.DataFrame: DataFrame with date features
    """
    log_memory_usage("Before creating date features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Check if date column exists
    if date_col not in result_df.columns:
        logger.error(f"Date column '{date_col}' not found in DataFrame. Cannot create date features.")
        return result_df
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(result_df[date_col]):
        try:
            result_df[date_col] = pd.to_datetime(result_df[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime. Cannot create date features.")
            return result_df
    
    # Create date features
    date_data = result_df[date_col]
    
    # Basic time components
    result_df['year'] = date_data.dt.year
    result_df['month'] = date_data.dt.month
    result_df['day'] = date_data.dt.day
    result_df['dayofweek'] = date_data.dt.dayofweek
    result_df['quarter'] = date_data.dt.quarter
    
    # Cyclical features for day of week (converts categorical to continuous)
    result_df['dayofweek_sin'] = np.sin(2 * np.pi * date_data.dt.dayofweek / 7)
    result_df['dayofweek_cos'] = np.cos(2 * np.pi * date_data.dt.dayofweek / 7)
    
    # Cyclical features for month
    result_df['month_sin'] = np.sin(2 * np.pi * date_data.dt.month / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * date_data.dt.month / 12)
    
    # Is weekend/holiday/month end features
    result_df['is_weekend'] = (date_data.dt.dayofweek >= 5).astype(int)
    result_df['is_month_end'] = date_data.dt.is_month_end.astype(int)
    result_df['is_quarter_end'] = date_data.dt.is_quarter_end.astype(int)
    
    # Days since start of data
    min_date = date_data.min()
    result_df['days_since_start'] = (date_data - min_date).dt.days
    
    logger.info("Created 13 date features")
    log_memory_usage("After creating date features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

def create_all_timeseries_features(df: pd.DataFrame,
                                  group_col: str,
                                  price_col: str = 'price',
                                  volume_col: Optional[str] = 'volume',
                                  target_cols: Optional[List[str]] = None,
                                  date_col: str = 'date') -> pd.DataFrame:
    """
    Create all time series features for financial data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (e.g., 'symbol' or 'asset')
        price_col (str): Column containing price data
        volume_col (str, optional): Column containing volume data
        target_cols (List[str], optional): Columns to create time series features for
                   (if None, uses price_col and volume_col if available)
        date_col (str): Date column for sorting
        
    Returns:
        pd.DataFrame: DataFrame with all time series features
    """
    log_memory_usage("Before creating all time series features:")
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # If target_cols not provided, use price_col and volume_col if available
    if target_cols is None:
        target_cols = [price_col]
        if volume_col in result_df.columns:
            target_cols.append(volume_col)
    
    # Create date features
    logger.info("Creating date features...")
    result_df = create_date_features(result_df, date_col=date_col)
    
    # Create lag features
    logger.info("Creating lag features...")
    lag_periods = [1, 2, 3, 5, 7, 14, 21, 28]
    result_df = create_lag_features(result_df, group_col, target_cols, lag_periods, date_col=date_col)
    
    # Create rolling features
    logger.info("Creating rolling features...")
    windows = [3, 5, 7, 14, 21, 28]
    functions = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max
    }
    result_df = create_rolling_features(result_df, group_col, target_cols, windows, functions, date_col=date_col)
    
    # Create EWM features
    logger.info("Creating EWM features...")
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    ewm_functions = {
        'mean': lambda x: x.mean()
    }
    result_df = create_ewm_features(result_df, group_col, target_cols, alphas, ewm_functions, date_col=date_col)
    
    # Create diff features
    logger.info("Creating differencing features...")
    diff_periods = [1, 2, 3, 5, 7]
    result_df = create_diff_features(result_df, group_col, target_cols, diff_periods, date_col=date_col)
    
    # Create technical indicators
    logger.info("Creating technical indicators...")
    result_df = create_technical_indicators(result_df, group_col, price_col, volume_col, date_col=date_col)
    
    logger.info("Completed creating all time series features")
    log_memory_usage("After creating all time series features:")
    
    # Optimize memory usage
    return optimize_dataframe_memory(result_df)

if __name__ == "__main__":
    # Test time series feature engineering
    import random
    
    # Create test data
    n_assets = 3
    n_days = 100
    assets = [f"ASSET_{i}" for i in range(1, n_assets + 1)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    data = []
    for asset in assets:
        price = 100
        volume = 10000
        for date in dates:
            # Random walk for price
            price *= (1 + random.uniform(-0.03, 0.03))
            # Random volume
            volume *= (1 + random.uniform(-0.2, 0.2))
            data.append({
                'date': date,
                'asset': asset,
                'price': price,
                'volume': volume
            })
    
    test_df = pd.DataFrame(data)
    logger.info(f"Created test data with shape {test_df.shape}")
    
    # Test creating all time series features
    result_df = create_all_timeseries_features(test_df, 'asset')
    
    logger.info(f"Result DataFrame shape: {result_df.shape}")
    logger.info(f"Added {result_df.shape[1] - test_df.shape[1]} new features")
    
    # Print feature names
    logger.info(f"New features: {', '.join(sorted(set(result_df.columns) - set(test_df.columns)))}")