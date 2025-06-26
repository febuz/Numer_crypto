#!/usr/bin/env python3
"""
Create a new Numerai Crypto model and upload predictions
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

def create_model_and_submit(submission_path, model_name, public_id, secret_key):
    """
    Create a new model and submit predictions using direct API calls
    
    Args:
        submission_path: Path to submission CSV file
        model_name: Name for the new model
        public_id: Numerai API public ID
        secret_key: Numerai API secret key
        
    Returns:
        dict: Response from API
    """
    try:
        from numerapi import CryptoAPI
    except ImportError:
        logger.error("numerapi package not installed. Run 'pip install numerapi'")
        return {"success": False, "error": "numerapi package not installed"}
    
    # Check if file exists
    if not os.path.exists(submission_path):
        logger.error(f"Submission file not found: {submission_path}")
        return {"success": False, "error": "Submission file not found"}
    
    try:
        # Initialize API client
        napi = CryptoAPI(public_id, secret_key)
        
        # 1. Check if model already exists
        logger.info(f"Checking if model '{model_name}' already exists...")
        
        # Use raw GraphQL query to search for models
        query = """
        query {
          account {
            models {
              id
              name
              tournament
            }
          }
        }
        """
        
        try:
            result = napi.raw_query(query)
            models = result.get('data', {}).get('account', {}).get('models', [])
            
            # Filter for crypto models with matching name
            existing_models = [m for m in models 
                               if m.get('tournament') == 'crypto' 
                               and m.get('name') == model_name]
            
            if existing_models:
                model_id = existing_models[0].get('id')
                logger.info(f"Found existing model: {model_name} with ID: {model_id}")
            else:
                # 2. Create new model
                logger.info(f"Creating new model: {model_name}...")
                
                # Direct API call to create model
                response = requests.post(
                    "https://api-crypto.numer.ai/graphql",
                    json={
                        "query": """
                        mutation($name: String!) {
                          createModel(name: $name, tournament: "crypto") {
                            id
                            name
                            tournament
                          }
                        }
                        """,
                        "variables": {
                            "name": model_name
                        }
                    },
                    headers={
                        "Authorization": f"Token {public_id}${secret_key}"
                    }
                )
                
                response_data = response.json()
                print("Create model response:", json.dumps(response_data, indent=2))
                
                if 'errors' in response_data:
                    logger.error(f"Failed to create model: {response_data['errors']}")
                    return {"success": False, "error": str(response_data['errors'])}
                
                model_id = response_data.get('data', {}).get('createModel', {}).get('id')
                if not model_id:
                    logger.error("Failed to get model ID from response")
                    return {"success": False, "error": "Failed to get model ID"}
                
                logger.info(f"Created new model: {model_name} with ID: {model_id}")
        
        except Exception as e:
            logger.error(f"Error querying/creating model: {e}")
            logger.info("Trying to upload without model ID...")
            
            # 3. Submit predictions with the model ID
            try:
                result = napi.upload_predictions(submission_path)
                logger.info(f"Submission successful: {result}")
                return {"success": True, "result": result, "model_id": None}
            except Exception as e:
                logger.error(f"Submission failed: {e}")
                return {"success": False, "error": str(e)}
            
        # 3. Submit predictions with the model ID
        logger.info(f"Uploading predictions for model {model_name} (ID: {model_id})...")
        result = napi.upload_predictions(submission_path, model_id=model_id)
        
        logger.info(f"Submission successful: {result}")
        return {"success": True, "result": result, "model_id": model_id}
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Create a Numerai Crypto model and submit predictions')
    parser.add_argument('--file', type=str, required=True, help='Path to prediction file')
    parser.add_argument('--model-name', type=str, default="ENSEMBLE_MODEL", 
                        help='Name for the model (default: ENSEMBLE_MODEL)')
    parser.add_argument('--public-id', type=str, 
                        default=os.environ.get('NUMERAI_PUBLIC_ID', "ZYUPMDSALDNBEA67XOUJZIV7UVBY4DHG"),
                        help='Numerai API public ID')
    parser.add_argument('--secret-key', type=str, 
                        default=os.environ.get('NUMERAI_SECRET_KEY', "LY6ZWGL7JOEYB3WGU5MSVT3URX5P6BRQJVQWLME46KSDRG4PRIZD6Z44FM6HT3WY"),
                        help='Numerai API secret key')
    
    args = parser.parse_args()
    
    # Create model and submit predictions
    result = create_model_and_submit(
        args.file,
        args.model_name,
        args.public_id,
        args.secret_key
    )
    
    if result.get("success"):
        logger.info(f"Process completed successfully with model ID: {result.get('model_id')}")
        return 0
    else:
        logger.error(f"Process failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())