#!/usr/bin/env python3
"""
Launcher script for Numerai Crypto models.

This script serves as a simple entry point to run models from the scripts directory.
"""

import os
import sys
import argparse

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def main():
    """Main function to parse arguments and launch the appropriate script"""
    parser = argparse.ArgumentParser(description='Numerai Crypto model launcher')
    parser.add_argument('--model', type=str, default='quick', 
                        choices=['quick', 'advanced', 'ensemble', 'h2o', 'yiedl'],
                        help='Model type to run')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for predictions')
    
    args, unknown_args = parser.parse_known_args()
    
    # Build command with all arguments
    if args.model == 'quick':
        from scripts.models.run_quick_model import main as run_model
    elif args.model == 'advanced':
        from scripts.models.run_advanced_model import main as run_model
    else:
        print(f"Model type '{args.model}' should be run using its runner script in scripts/runners/")
        print(f"For example: ./scripts/runners/run_{args.model}_model.sh")
        return 1
    
    # Pass all arguments to the model's main function
    sys.argv = sys.argv[:1] + unknown_args
    return run_model()

if __name__ == "__main__":
    sys.exit(main())