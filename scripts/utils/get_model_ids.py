#!/usr/bin/env python3
"""
Get Numerai Crypto model IDs for submission
"""
import os
import sys
import json
from numerapi import CryptoAPI

def get_crypto_models(public_id, secret_key):
    """Get crypto models from Numerai API"""
    try:
        # Initialize API client
        napi = CryptoAPI(public_id, secret_key)
        
        # Try to get available models using a direct GraphQL query
        # Note: CryptoAPI doesn't have a direct get_models() method
        query = """
        query {
          account {
            id
            username
            email
            models {
              id
              name
              tournament
            }
          }
        }
        """
        
        result = napi.raw_query(query)
        print(json.dumps(result, indent=2))
        
        models = []
        if result.get('data', {}).get('account', {}).get('models'):
            models = result['data']['account']['models']
            
        # Filter for crypto models
        crypto_models = [m for m in models if m.get('tournament') == 'crypto']
        
        print(f"\nFound {len(crypto_models)} crypto models:")
        for model in crypto_models:
            print(f"  - {model.get('name')} (ID: {model.get('id')})")
            
        return crypto_models
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def main():
    # Get API credentials from environment or default
    public_id = os.environ.get('NUMERAI_PUBLIC_ID', "ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG")
    secret_key = os.environ.get('NUMERAI_SECRET_KEY', "LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY")
    
    # Get crypto models
    models = get_crypto_models(public_id, secret_key)
    
    # Try to get current round info
    try:
        napi = CryptoAPI(public_id, secret_key)
        current_round = napi.get_current_round()
        print(f"\nCurrent round: {current_round}")
    except Exception as e:
        print(f"Error getting current round: {e}")

if __name__ == "__main__":
    main()