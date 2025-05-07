#!/usr/bin/env python3
"""
Run the optimal pipeline for Numerai Crypto.
This script executes the high-memory, GPU-accelerated optimal pipeline
for achieving the lowest possible RMSE (target < 0.018).
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipelines.optimal import OptimalPipeline
from config.settings import SUBMISSIONS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_optimal')

def main():
    parser = argparse.ArgumentParser(description='Run optimal pipeline for Numerai Crypto')
    parser.add_argument('--tournament', type=str, default='crypto', 
                        help='Tournament name (default: crypto)')
    parser.add_argument('--time-budget', type=float, default=8.0,
                        help='Time budget in hours (default: 8.0)')
    parser.add_argument('--submission-id', type=str, default=None,
                        help='Custom submission ID (default: timestamp)')
    parser.add_argument('--submit', action='store_true',
                        help='Submit results to Numerai API')
    args = parser.parse_args()
    
    # Use timestamp for submission ID if not provided
    submission_id = args.submission_id
    if submission_id is None:
        submission_id = f"optimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run the pipeline
    logger.info(f"Starting optimal pipeline for tournament '{args.tournament}' with time budget of {args.time_budget} hours")
    pipeline = OptimalPipeline(tournament=args.tournament, time_budget_hours=args.time_budget)
    results = pipeline.run()
    
    if results['success']:
        logger.info(f"Pipeline completed successfully with best RMSE: {results['best_rmse']:.6f}")
        logger.info(f"Duration: {results['duration_hours']:.2f} hours")
        
        # Show paths to submissions
        for model_name, submission_path in results['submission_paths'].items():
            logger.info(f"{model_name} submission: {submission_path}")
        
        # Submit to Numerai if requested
        if args.submit:
            from data.retrieval import NumeraiDataRetriever
            retriever = NumeraiDataRetriever(tournament=args.tournament)
            
            # Use ensemble submission by default
            submission_path = results['submission_paths']['ensemble']
            logger.info(f"Submitting ensemble predictions to Numerai: {submission_path}")
            
            submission_id = f"{submission_id}_{results['best_rmse']:.6f}"
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