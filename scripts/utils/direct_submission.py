#!/usr/bin/env python3
"""
Direct submission to Numerai Crypto using raw HTTP requests
"""
import os
import sys
import argparse
import logging
import requests
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def direct_submission(submission_path, public_id, secret_key):
    """Submit predictions directly to Numerai API using HTTP requests"""
    try:
        # Check if file exists
        if not os.path.exists(submission_path):
            logger.error(f"Submission file not found: {submission_path}")
            return {"success": False, "error": "Submission file not found"}
        
        # Read submission file
        with open(submission_path, 'rb') as f:
            submission_data = f.read()
        
        # Create multipart form data
        url = 'https://api-tournament.numer.ai/graphql'  # Using main Numerai API endpoint
        
        # First, try to get current round
        query = """
        query {
          latestRound {
            number
          }
        }
        """
        
        headers = {
            'Authorization': f'Token {public_id}${secret_key}'
        }
        
        response = requests.post(url, json={'query': query}, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to get current round: {response.text}")
            return {"success": False, "error": f"Failed to get current round: {response.text}"}
        
        try:
            current_round = response.json()['data']['latestRound']['number']
            logger.info(f"Current round: {current_round}")
        except:
            logger.warning("Could not determine current round, using direct submission")
            current_round = None
        
        # Now upload the predictions
        operations = {
            'query': """
            mutation($file: Upload!) {
              createCryptoSubmission(file: $file) {
                id
              }
            }
            """,
            'variables': {
                'file': None
            }
        }
        
        # Convert operations to JSON string
        operations_str = json.dumps(operations)
        
        # Create the map
        map_data = {
            '0': ['variables.file']
        }
        map_str = json.dumps(map_data)
        
        # Create the form data
        files = {
            'operations': (None, operations_str),
            'map': (None, map_str),
            '0': ('submission.csv', submission_data, 'text/csv')
        }
        
        # Send the request
        logger.info("Uploading predictions...")
        response = requests.post(url, files=files, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Upload failed: {response.text}")
            return {"success": False, "error": f"Upload failed: {response.text}"}
        
        # Parse response
        result = response.json()
        logger.info(f"Upload response: {json.dumps(result, indent=2)}")
        
        if 'errors' in result:
            logger.error(f"Upload failed: {result['errors']}")
            return {"success": False, "error": str(result['errors'])}
        
        logger.info("Upload successful!")
        return {"success": True, "result": result}
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Direct submission to Numerai Crypto')
    parser.add_argument('--file', type=str, required=True, help='Path to prediction file')
    parser.add_argument('--public-id', type=str, 
                        default=os.environ.get('NUMERAI_PUBLIC_ID', "ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG"),
                        help='Numerai API public ID')
    parser.add_argument('--secret-key', type=str, 
                        default=os.environ.get('NUMERAI_SECRET_KEY', "LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY"),
                        help='Numerai API secret key')
    
    args = parser.parse_args()
    
    # Submit predictions
    result = direct_submission(args.file, args.public_id, args.secret_key)
    
    if result.get("success"):
        logger.info("Submission successful!")
        return 0
    else:
        logger.error(f"Submission failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())