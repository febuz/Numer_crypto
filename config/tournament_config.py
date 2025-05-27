"""
Tournament configuration for Numerai Crypto.
This file defines tournament-specific settings, including API endpoints.
"""

# Tournament name constant
TOURNAMENT_NAME = "crypto"

def get_tournament_name():
    """
    Returns the current tournament name.
    
    Returns:
        str: The name of the tournament
    """
    return TOURNAMENT_NAME

def get_tournament_endpoint(key):
    """
    Returns the API endpoint for the specified resource.
    
    Args:
        key (str): The resource key to get the endpoint for
            Supported keys:
            - train_targets: Training target data
            - live_universe: Live universe for prediction
            - train_data: Standard training data
            - train_data_alt: Alternative training data path format
            
    Returns:
        str: The endpoint URL path
    """
    endpoints = {
        "train_targets": f"{TOURNAMENT_NAME}/v1.0/train_targets.parquet", 
        "live_universe": f"{TOURNAMENT_NAME}/v1.0/live_universe.parquet",
        "train_data": f"{TOURNAMENT_NAME}/v1.0/train.parquet",
        # Alternative endpoint for train data when the standard format is unavailable
        "train_data_alt": f"{TOURNAMENT_NAME}/v1.0/features.parquet"
    }
    return endpoints.get(key)