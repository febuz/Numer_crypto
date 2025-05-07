#!/usr/bin/env python3
"""
Run the simple pipeline for Numerai Crypto.
This script executes the quick, simplified pipeline for rapid submissions
with a 15-30 minute runtime target (RMSE < 0.020).
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.simple import SimplePipeline
from config.settings import SUBMISSIONS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_simple')

def main():
    parser = argparse.ArgumentParser(description='Run simple pipeline for Numerai Crypto')
    parser.add_argument('--tournament', type=str, default='crypto', 
                        help='Tournament name (default: crypto)')
    parser.add_argument('--time-budget', type=float, default=30.0,
                        help='Time budget in minutes (default: 30.0)')
    parser.add_argument('--submission-id', type=str, default=None,
                        help='Custom submission ID (default: timestamp)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit results to Numerai API')
    args = parser.parse_args()
    
    # Use timestamp for submission ID if not provided
    submission_id = args.submission_id
    if submission_id is None:
        submission_id = f"simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run the pipeline
    logger.info(f"Starting simple pipeline for tournament '{args.tournament}' with time budget of {args.time_budget} minutes")
    pipeline = SimplePipeline(tournament=args.tournament, time_budget_minutes=args.time_budget)
    results = pipeline.run()
    
    if results['success']:
        logger.info(f"Pipeline completed successfully with RMSE: {results['rmse']:.6f}")
        logger.info(f"Duration: {results['duration_minutes']:.2f} minutes")
        logger.info(f"Submission path: {results['submission_path']}")
        
        # Submit to Numerai if requested
        if args.submit:
            from data.retrieval import NumeraiDataRetriever
            retriever = NumeraiDataRetriever(tournament=args.tournament)
            
            submission_path = results['submission_path']
            logger.info(f"Submitting predictions to Numerai: {submission_path}")
            
            submission_id = f"{submission_id}_{results['rmse']:.6f}"
            submission_result = retriever.submit_predictions(
                submission_path,
                submission_id=submission_id
            )
            
            logger.info(f"Submission result: {submission_result}")
    else:
        logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())