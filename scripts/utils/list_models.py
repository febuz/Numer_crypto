#!/usr/bin/env python3
"""
List Numerai Crypto models for the authenticated user
"""
import os
import sys
from numerapi import CryptoAPI

# Set API credentials
public_id = "ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG"
secret_key = "LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY"

try:
    # Initialize API client
    napi = CryptoAPI(public_id, secret_key)
    
    # Get user info to see models
    user = napi.get_user()
    print("User info:", user)
    
    # Get models
    try:
        models = napi.get_models()
        print("\nModels:", models)
    except Exception as e:
        print(f"Error getting models: {e}")
    
    # Try to get submissions
    try:
        # Get available rounds
        current_round = napi.get_current_round()
        print(f"\nCurrent round: {current_round}")
        
        # Get leaderboard
        leaderboard = napi.get_leaderboard()
        print(f"\nLeaderboard entries: {len(leaderboard)}")
        if leaderboard:
            print("First 5 entries:", leaderboard[:5])
    except Exception as e:
        print(f"Error getting additional info: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)