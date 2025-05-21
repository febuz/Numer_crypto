#!/usr/bin/env python3
"""
pipeline_error_handler.py - Error handling utilities for the Numerai Crypto pipeline

This script provides error handling, reporting and recovery mechanisms for the pipeline.
"""

import os
import sys
import logging
import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineErrorHandler:
    """Handles errors in the Numerai Crypto pipeline"""
    
    def __init__(self, base_dir=None, error_dir=None, log_file=None, report_file=None):
        """
        Initialize the error handler
        
        Args:
            base_dir (str): Base directory for the pipeline
            error_dir (str): Directory to store error reports
            log_file (str): Path to the log file
            report_file (str): Path to the error report file
        """
        self.base_dir = base_dir or '/media/knight2/EDB/numer_crypto_temp'
        self.error_dir = error_dir or os.path.join(self.base_dir, 'errors')
        
        # Create error directory if it doesn't exist
        os.makedirs(self.error_dir, exist_ok=True)
        
        # Set log and report files
        self.log_file = log_file or os.path.join(self.error_dir, 'pipeline_error.log')
        self.report_file = report_file or os.path.join(self.base_dir, 'error_report.md')
        
        # Configure file handler for logging
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)
    
    def handle_error(self, stage, error, details=None, recovery_options=None):
        """
        Handle an error in the pipeline
        
        Args:
            stage (str): The pipeline stage where the error occurred
            error (Exception): The error that occurred
            details (dict): Additional details about the error
            recovery_options (list): List of recovery options
            
        Returns:
            bool: True if error handling was successful
        """
        # Log the error
        logger.error(f"Error in stage: {stage}")
        logger.error(f"Error message: {str(error)}")
        if details:
            logger.error(f"Details: {json.dumps(details, indent=2)}")
        
        # Get traceback
        tb_str = traceback.format_exc()
        logger.error(f"Traceback: {tb_str}")
        
        # Create error report
        self.create_error_report(stage, error, tb_str, details, recovery_options)
        
        # Send notification (if configured)
        # self.send_notification(stage, error, details)
        
        return True
    
    def create_error_report(self, stage, error, traceback_str, details=None, recovery_options=None):
        """
        Create a markdown error report
        
        Args:
            stage (str): The pipeline stage where the error occurred
            error (Exception): The error that occurred
            traceback_str (str): The traceback string
            details (dict): Additional details about the error
            recovery_options (list): List of recovery options
            
        Returns:
            str: Path to the error report file
        """
        # Create report content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"# Pipeline Error Report\n\n"
        report += f"## Error Information\n\n"
        report += f"- **Timestamp:** {timestamp}\n"
        report += f"- **Stage:** {stage}\n"
        report += f"- **Error:** {str(error)}\n\n"
        
        if details:
            report += f"## Details\n\n"
            report += f"```json\n{json.dumps(details, indent=2)}\n```\n\n"
        
        report += f"## Traceback\n\n"
        report += f"```\n{traceback_str}\n```\n\n"
        
        if recovery_options:
            report += f"## Recovery Options\n\n"
            for i, option in enumerate(recovery_options, 1):
                report += f"{i}. {option}\n"
        
        # Write report to file
        with open(self.report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Error report created: {self.report_file}")
        return self.report_file
    
    def send_notification(self, stage, error, details=None):
        """
        Send an error notification email (placeholder function)
        
        Args:
            stage (str): The pipeline stage where the error occurred
            error (Exception): The error that occurred
            details (dict): Additional details about the error
            
        Returns:
            bool: True if notification was sent successfully
        """
        # This is a placeholder for email notification
        # In a real implementation, you would configure SMTP settings
        # and send an actual email
        
        logger.info("Email notification would be sent here")
        return True

def check_prerequisites():
    """
    Check if all prerequisites for the pipeline are met
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    prerequisites_met = True
    
    # Check for required directories
    required_dirs = [
        '/media/knight2/EDB/numer_crypto_temp',
        '/media/knight2/EDB/numer_crypto_temp/data',
        '/media/knight2/EDB/numer_crypto_temp/models'
    ]
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            logger.error(f"Required directory not found: {directory}")
            prerequisites_met = False
    
    # Check for required Python packages
    try:
        import pandas
        import numpy
        import sklearn
    except ImportError as e:
        logger.error(f"Required Python package not found: {e}")
        prerequisites_met = False
    
    # Check for required files (modify as needed)
    required_files = [
        '/media/knight2/EDB/repos/Numer_crypto/go_pipeline.sh'
    ]
    
    for file_path in required_files:
        if not os.path.isfile(file_path):
            logger.error(f"Required file not found: {file_path}")
            prerequisites_met = False
    
    return prerequisites_met

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pipeline Error Handler for Numerai Crypto')
    parser.add_argument('--check-prerequisites', action='store_true', 
                        help='Check prerequisites for the pipeline')
    parser.add_argument('--base-dir', type=str, 
                        default='/media/knight2/EDB/numer_crypto_temp',
                        help='Base directory for the pipeline')
    parser.add_argument('--error-dir', type=str, default=None,
                        help='Directory to store error reports')
    parser.add_argument('--create-error-report', action='store_true',
                        help='Create an error report')
    parser.add_argument('--error-stage', type=str, default='unknown',
                        help='Pipeline stage where the error occurred')
    parser.add_argument('--error-message', type=str, default='Unknown error',
                        help='Error message')
    
    args = parser.parse_args()
    
    # Initialize error handler
    error_handler = PipelineErrorHandler(base_dir=args.base_dir, error_dir=args.error_dir)
    
    # Check prerequisites if requested
    if args.check_prerequisites:
        if check_prerequisites():
            logger.info("All prerequisites met")
            return 0
        else:
            error = Exception("Prerequisites not met")
            recovery_options = [
                "Ensure all required directories exist",
                "Install missing Python packages",
                "Check for required files"
            ]
            error_handler.handle_error("prerequisites", error, recovery_options=recovery_options)
            return 1
    
    # Create error report if requested
    if args.create_error_report:
        error = Exception(args.error_message)
        error_handler.handle_error(args.error_stage, error)
        return 0
    
    # Default: just return success
    return 0

if __name__ == "__main__":
    sys.exit(main())