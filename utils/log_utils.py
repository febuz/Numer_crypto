#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import datetime
from pathlib import Path

# Add parent directory to path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import LOG_DIR

def setup_logging(name=None, level=logging.INFO, create_file=True):
    """
    Set up logging configuration with external log directory.
    
    Args:
        name (str, optional): Logger name. Defaults to None (root logger).
        level (int, optional): Logging level. Defaults to logging.INFO.
        create_file (bool, optional): Whether to create a file handler. Defaults to True.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create external log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handlers if they don't exist yet
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if create_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            script_name = name if name else "root"
            log_file = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Log file created at: {log_file}")
    
    return logger

def clean_repository_logs(repo_dir):
    """
    Remove all .log files from the repository directory.
    
    Args:
        repo_dir (str): Repository root directory
    
    Returns:
        int: Number of files removed
    """
    removed_count = 0
    repo_path = Path(repo_dir)
    
    for log_file in repo_path.glob('**/*.log'):
        try:
            log_file.unlink()
            print(f"Removed: {log_file}")
            removed_count += 1
        except Exception as e:
            print(f"Failed to remove {log_file}: {e}")
    
    return removed_count

if __name__ == "__main__":
    # When run as a script, clean log files from the repository
    import sys
    
    if len(sys.argv) > 1:
        repo_dir = sys.argv[1]
    else:
        # Default to the parent directory of this script
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    count = clean_repository_logs(repo_dir)
    print(f"Removed {count} log files from {repo_dir}")