#!/usr/bin/env python3
"""
Submit predictions to the Numerai API.
This script handles submission of prediction files to the Numerai tournament.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.retrieval import NumeraiDataRetriever
from config.settings import SUBMISSIONS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('submit')

def main():
    parser = argparse.ArgumentParser(description='Submit predictions to Numerai API')
    parser.add_argument('file_path', type=str, help='Path to submission CSV file')
    parser.add_argument('--tournament', type=str, default='crypto', 
                        help='Tournament name (default: crypto)')
    parser.add_argument('--submission-id', type=str, default=None,
                        help='Custom submission ID (default: timestamp)')
    parser.add_argument('--track-performance', action='store_true',
                        help='Track submission performance over time')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        logger.error(f"Submission file not found: {args.file_path}")
        return 1
    
    # Use timestamp for submission ID if not provided
    submission_id = args.submission_id
    if submission_id is None:
        submission_id = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Submit predictions
    logger.info(f"Submitting predictions to tournament '{args.tournament}' with ID '{submission_id}'")
    retriever = NumeraiDataRetriever(tournament=args.tournament)
    
    try:
        result = retriever.submit_predictions(
            args.file_path,
            submission_id=submission_id
        )
        
        logger.info(f"Submission successful!")
        logger.info(f"Result: {result}")
        
        # Save submission result
        if args.track_performance:
            import json
            from config.settings import SUBMISSIONS_DIR
            
            os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
            result_path = os.path.join(SUBMISSIONS_DIR, f"submission_result_{submission_id}.json")
            
            with open(result_path, 'w') as f:
                json.dump({
                    'submission_id': submission_id,
                    'file_path': args.file_path,
                    'tournament': args.tournament,
                    'timestamp': datetime.now().isoformat(),
                    'result': result
                }, f, indent=2)
            
            logger.info(f"Saved submission result to {result_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Submission failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())