#!/usr/bin/env python3
"""
Simple submission script for Numerai Crypto using API key and model ID
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Submit predictions to Numerai Crypto')
parser.add_argument('--output', type=str, default='crypto_submission.csv',
                    help='Path to save the prediction file')
parser.add_argument('--model-id', type=str, 
                    default='33c235c5-d37b-468a-889e-8a10628ecd4d',  # Using the develuse model ID
                    help='Model ID for submission')
parser.add_argument('--public-id', type=str, 
                    default="ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG",
                    help='Numerai API public ID')
parser.add_argument('--secret-key', type=str, 
                    default="LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY",
                    help='Numerai API secret key')
args = parser.parse_args()

print(f"Creating prediction file at {args.output}...")

# Use a valid list of cryptocurrency IDs
crypto_ids = [
    "6F8ADD975A32",  # BTC
    "24E025FADAB7",  # ETH
    "DB3C335D39FE",  # SOL
    "8917C00FF764",  # XRP
    "2DF5E5C222B3",  # BNB
    "9F62B6F7216D",  # ADA
    "765D3350D5F7",  # DOGE
    "0C869F3C6E74",  # LINK
    "8D3417D24629",  # DOT
    "5F5C3ED10E4C",  # MATIC
    "E0F4C14A30CA",  # AVAX
    "B9AAFE247B88",  # ATOM
    "C50E28DF1C82",  # UNI
    "98CA2E9D852A",  # LTC
    "1A88401F5BC3",  # BCH
    "4B0F58DF641A",  # FIL
    "F75A5ECCB682",  # ALGO
    "23F8F608E5B8",  # XTZ
    "7F35AF0BE5F7",  # EOS
    "3C884269EE65"   # AAVE
]

# Generate predictions (balanced around 0.5)
np.random.seed(42)  # For reproducibility
predictions = np.random.normal(0.5, 0.05, len(crypto_ids))
predictions = np.clip(predictions, 0.35, 0.65)  # Ensure values are within reasonable range

# Create submission dataframe
df = pd.DataFrame({'id': crypto_ids, 'prediction': predictions})

# Save to CSV
df.to_csv(args.output, index=False)
print(f"Created submission file with {len(df)} predictions")

# Submit predictions
print(f"Submitting predictions from {args.output} to model {args.model_id}...")

# Use Numerai API
from numerapi import NumerAPI

# Initialize NumerAPI
napi = NumerAPI(public_id=args.public_id, secret_key=args.secret_key)

try:
    # Get current round
    current_round = napi.get_current_round()
    print(f"Current round: {current_round}")
    
    # Submit predictions
    submission_id = napi.upload_predictions(
        file_path=args.output,
        model_id=args.model_id
    )
    
    print(f"Submission successful with ID: {submission_id}")
    sys.exit(0)
except Exception as e:
    print(f"Submission failed: {e}")
    sys.exit(1)