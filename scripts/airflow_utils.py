#!/usr/bin/env python3
"""
Airflow utilities for Numerai Crypto pipeline.

This module provides utility functions for Airflow integration with the Numerai Crypto pipeline,
including initialization, starting/stopping services, and DAG management.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
AIRFLOW_HOME = "/media/knight2/EDB/numer_crypto_temp/airflow"
AIRFLOW_VENV_DIR = f"{os.environ.get('HOME')}/airflow_env/venv"
DAG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "airflow_dags", "numerai_crypto_pipeline_v3.py")
AIRFLOW_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "config", "airflow")
AIRFLOW_CONFIG_FILE = os.path.join(AIRFLOW_CONFIG_DIR, "airflow.cfg")

class AirflowUtils:
    """Utilities for Airflow operations with Numerai Crypto pipeline"""
    
    @staticmethod
    def setup_environment():
        """
        Set up Airflow environment variables
        
        Returns:
            dict: Environment variables
        """
        env = os.environ.copy()
        env["AIRFLOW_HOME"] = AIRFLOW_HOME
        
        # Create Airflow home directory if it doesn't exist
        os.makedirs(AIRFLOW_HOME, exist_ok=True)
        
        # Create all necessary subdirectories
        os.makedirs(os.path.join(AIRFLOW_HOME, "logs"), exist_ok=True)
        os.makedirs(os.path.join(AIRFLOW_HOME, "dags"), exist_ok=True)
        os.makedirs(os.path.join(AIRFLOW_HOME, "plugins"), exist_ok=True)
        
        return env
    
    @staticmethod
    def verify_airflow_installation():
        """
        Verify Airflow is installed and create virtual environment if needed
        
        Returns:
            bool: True if Airflow is available, False otherwise
        """
        # Check if Airflow virtual environment exists
        if not os.path.exists(AIRFLOW_VENV_DIR):
            logger.info(f"Airflow virtual environment not found at {AIRFLOW_VENV_DIR}. Creating...")
            try:
                # Create virtual environment
                subprocess.run([sys.executable, "-m", "venv", AIRFLOW_VENV_DIR], check=True)
                
                # Activate and install Airflow
                activate_script = os.path.join(AIRFLOW_VENV_DIR, "bin", "activate")
                AIRFLOW_VERSION="3.0.1"
                PYTHON_VERSION="3.12"
                CONSTRAINT_URL=f"https://raw.githubusercontent.com/apache/airflow/constraints-{AIRFLOW_VERSION}/constraints-{PYTHON_VERSION}.txt"
                
                subprocess.run([
                    "/bin/bash", "-c", 
                    f"source {activate_script} && " +
                    "pip install --upgrade pip wheel setuptools && " +
                    f"pip install 'apache-airflow=={AIRFLOW_VERSION}' --constraint '{CONSTRAINT_URL}' && " +
                    "pip install apache-airflow-providers-slack pyarrow fastparquet"
                ], check=True)
                
                logger.info("Airflow installation complete")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to set up Airflow environment: {e}")
                return False
        
        # Check if airflow command is available in the virtual environment
        try:
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && which airflow"
            ], capture_output=True, text=True)
            
            if not result.stdout:
                logger.error("Airflow command not found in virtual environment")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check for airflow command: {e}")
            return False
            
        return True
    
    @staticmethod
    def initialize_airflow():
        """
        Initialize Airflow database and copy DAG files
        
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(AIRFLOW_HOME, "dags"), exist_ok=True)
        
        # Copy custom airflow.cfg if it exists
        if os.path.exists(AIRFLOW_CONFIG_FILE):
            logger.info(f"Copying custom Airflow config from {AIRFLOW_CONFIG_FILE}")
            os.makedirs(os.path.dirname(os.path.join(AIRFLOW_HOME, "airflow.cfg")), exist_ok=True)
            with open(AIRFLOW_CONFIG_FILE, 'r') as src:
                with open(os.path.join(AIRFLOW_HOME, "airflow.cfg"), 'w') as dst:
                    dst.write(src.read())
        
        # Create symlink to DAG directory
        src_dag_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "airflow_dags")
        dst_dag_dir = os.path.join(AIRFLOW_HOME, "dags")
        
        # Create DAG directory if it doesn't exist
        os.makedirs(dst_dag_dir, exist_ok=True)
        
        # Create symbolic link for each DAG file
        for dag_file in os.listdir(src_dag_dir):
            if dag_file.endswith('.py'):
                src_file = os.path.join(src_dag_dir, dag_file)
                dst_file = os.path.join(dst_dag_dir, dag_file)
                
                # Remove existing symlink if it exists
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                    
                # Create symlink
                os.symlink(src_file, dst_file)
                logger.info(f"Created symlink for DAG file: {dag_file}")
        
        # Initialize Airflow database (updated for Airflow 3.0+)
        try:
            logger.info("Initializing Airflow database...")
            
            # First run migrate to create the database (Airflow 3.0 uses same command)
            subprocess.run([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow db migrate"
            ], env=env, check=True)
            
            # Install Flask-AppBuilder for user creation
            subprocess.run([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"pip install Flask-AppBuilder"
            ], env=env, check=True)
            
            # Start Airflow standalone briefly to create initial user
            standalone_process = subprocess.Popen([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow standalone"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a short time for the user to be created
            time.sleep(5)
            
            # Then kill the process
            standalone_process.terminate()
            time.sleep(1)
            standalone_process.kill()
            
            logger.info("Airflow database initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Airflow database: {e}")
            return False
    
    @staticmethod
    def create_airflow_user(username="admin", password=None, 
                           firstname="Admin", lastname="User", 
                           email="admin@example.com", role="Admin"):
        """
        Create Airflow admin user
        
        Args:
            username: Username
            password: Password (if None, will be automatically generated)
            firstname: First name
            lastname: Last name
            email: Email address
            role: Role (Admin or User)
            
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        try:
            logger.info(f"Creating Airflow user: {username}")
            subprocess.run([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow users create " +
                f"--username {username} --password {password} " +
                f"--firstname {firstname} --lastname {lastname} " +
                f"--role {role} --email {email}"
            ], env=env, check=True)
            
            logger.info(f"Airflow user '{username}' created successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Airflow user: {e}")
            return False
    
    @staticmethod
    def start_scheduler():
        """
        Start Airflow scheduler
        
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        # Check if scheduler is already running
        try:
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow scheduler' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow scheduler is already running")
                return True
        except subprocess.CalledProcessError:
            pass
        
        # Start scheduler in background
        try:
            logger.info("Starting Airflow scheduler...")
            subprocess.Popen([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow scheduler > {AIRFLOW_HOME}/logs/scheduler.log 2>&1 &"
            ], env=env, shell=True)
            
            # Wait for scheduler to start
            time.sleep(5)
            
            # Check if scheduler is running
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow scheduler' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow scheduler started successfully")
                return True
            else:
                logger.error("Failed to start Airflow scheduler")
                return False
        except Exception as e:
            logger.error(f"Error starting Airflow scheduler: {e}")
            return False
    
    @staticmethod
    def start_webserver():
        """
        Start Airflow webserver
        
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        # Check if webserver is already running
        try:
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow webserver' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow webserver is already running")
                return True
        except subprocess.CalledProcessError:
            pass
        
        # Start webserver in background
        try:
            logger.info("Starting Airflow webserver...")
            subprocess.Popen([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow webserver -p 8080 > {AIRFLOW_HOME}/logs/webserver.log 2>&1 &"
            ], env=env, shell=True)
            
            # Wait for webserver to start
            time.sleep(5)
            
            # Check if webserver is running
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow webserver' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow webserver started successfully")
                return True
            else:
                logger.error("Failed to start Airflow webserver")
                return False
        except Exception as e:
            logger.error(f"Error starting Airflow webserver: {e}")
            return False
            
    @staticmethod
    def start_api_server():
        """
        Start Airflow webserver (API server) - alias for start_webserver for backward compatibility
        
        Returns:
            bool: Success status
        """
        logger.info("'start_api_server' is deprecated, use 'start_webserver' instead")
        return AirflowUtils.start_webserver()
    
    @staticmethod
    def start_standalone():
        """
        Start Airflow in standalone mode (all components in one process)
        
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        # Check if Airflow is already running
        try:
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow standalone' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow standalone is already running")
                return True
        except subprocess.CalledProcessError:
            pass
        
        # Start Airflow in standalone mode
        try:
            logger.info("Starting Airflow in standalone mode...")
            subprocess.Popen([
                "/bin/bash", "-c", 
                f"source {AIRFLOW_VENV_DIR}/bin/activate && " +
                f"AIRFLOW_HOME={AIRFLOW_HOME} airflow standalone > {AIRFLOW_HOME}/logs/standalone.log 2>&1 &"
            ], env=env, shell=True)
            
            # Wait for Airflow to start
            time.sleep(5)
            
            # Check if Airflow is running
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow standalone' | grep -v grep"
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.info("Airflow standalone started successfully")
                return True
            else:
                logger.error("Failed to start Airflow standalone")
                return False
        except Exception as e:
            logger.error(f"Error starting Airflow standalone: {e}")
            return False
    
    @staticmethod
    def stop_airflow():
        """
        Stop all Airflow services
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Stopping Airflow services...")
            
            # Kill all Airflow processes
            subprocess.run([
                "/bin/bash", "-c", 
                "pkill -f 'airflow webserver' || true; " +
                "pkill -f 'airflow scheduler' || true; " +
                "pkill -f 'airflow standalone' || true"
            ], check=True)
            
            logger.info("Airflow services stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stopping Airflow services: {e}")
            return False
    
    @staticmethod
    def show_logs():
        """
        Show Airflow logs
        
        Returns:
            str: Log content
        """
        try:
            log_files = []
            log_dir = os.path.join(AIRFLOW_HOME, "logs")
            
            if os.path.exists(log_dir):
                for root, _, files in os.walk(log_dir):
                    for file in files:
                        if file.endswith('.log'):
                            log_files.append(os.path.join(root, file))
            
            if not log_files:
                return "No log files found"
            
            # Get the latest log file
            latest_log = max(log_files, key=os.path.getmtime)
            
            # Get the last 100 lines of the log file
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"tail -n 100 {latest_log}"
            ], capture_output=True, text=True, check=True)
            
            return result.stdout
        except Exception as e:
            logger.error(f"Error showing Airflow logs: {e}")
            return f"Error: {str(e)}"
    
    @staticmethod
    def check_status():
        """
        Check Airflow services status
        
        Returns:
            dict: Status of Airflow services
        """
        status = {
            'api_server': False,
            'scheduler': False,
            'standalone': False
        }
        
        try:
            # Check API server
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow webserver' | grep -v grep"
            ], capture_output=True, text=True)
            status['api_server'] = bool(result.stdout)
            
            # Check scheduler
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow scheduler' | grep -v grep"
            ], capture_output=True, text=True)
            status['scheduler'] = bool(result.stdout)
            
            # Check standalone
            result = subprocess.run([
                "/bin/bash", "-c", 
                f"ps aux | grep 'airflow standalone' | grep -v grep"
            ], capture_output=True, text=True)
            status['standalone'] = bool(result.stdout)
            
            return status
        except Exception as e:
            logger.error(f"Error checking Airflow status: {e}")
            return status
    
    @staticmethod
    def trigger_dag(dag_id="numerai_crypto_pipeline_v3", conf=None):
        """
        Trigger DAG execution
        
        Args:
            dag_id: DAG ID to trigger
            conf: DAG configuration
            
        Returns:
            bool: Success status
        """
        if not AirflowUtils.verify_airflow_installation():
            return False
            
        env = AirflowUtils.setup_environment()
        
        # Prepare command
        command = f"source {AIRFLOW_VENV_DIR}/bin/activate && AIRFLOW_HOME={AIRFLOW_HOME} "
        
        if conf:
            # Convert conf to JSON string
            conf_json = json.dumps(conf)
            command += f"airflow dags trigger -c '{conf_json}' {dag_id}"
        else:
            command += f"airflow dags trigger {dag_id}"
        
        try:
            logger.info(f"Triggering DAG: {dag_id}")
            subprocess.run([
                "/bin/bash", "-c", command
            ], env=env, check=True)
            
            logger.info(f"DAG {dag_id} triggered successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to trigger DAG {dag_id}: {e}")
            return False

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description='Airflow Utilities for Numerai Crypto Pipeline')
    parser.add_argument('--action', type=str, required=True, 
                      choices=['init', 'create-user', 'start-scheduler', 'start-api-server', 'start-webserver', 
                               'start-standalone', 'stop', 'status', 'logs', 'trigger-dag'],
                      help='Action to perform')
    parser.add_argument('--dag-id', type=str, default="numerai_crypto_pipeline_v3",
                      help='DAG ID to trigger')
    parser.add_argument('--username', type=str, default="admin",
                      help='Username for create-user action')
    parser.add_argument('--password', type=str, default=None,
                      help='Password for create-user action (if None, will be automatically generated)')
    parser.add_argument('--email', type=str, default="admin@example.com",
                      help='Email for create-user action')
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.action == 'init':
        result = AirflowUtils.initialize_airflow()
        return 0 if result else 1
        
    elif args.action == 'create-user':
        result = AirflowUtils.create_airflow_user(
            username=args.username, password=args.password, email=args.email)
        return 0 if result else 1
        
    elif args.action == 'start-scheduler':
        result = AirflowUtils.start_scheduler()
        return 0 if result else 1
        
    elif args.action == 'start-api-server':
        result = AirflowUtils.start_api_server()
        return 0 if result else 1
        
    elif args.action == 'start-webserver':
        result = AirflowUtils.start_webserver()
        return 0 if result else 1
        
    elif args.action == 'start-standalone':
        result = AirflowUtils.start_standalone()
        return 0 if result else 1
        
    elif args.action == 'stop':
        result = AirflowUtils.stop_airflow()
        return 0 if result else 1
        
    elif args.action == 'status':
        status = AirflowUtils.check_status()
        print(json.dumps(status, indent=2))
        return 0
        
    elif args.action == 'logs':
        logs = AirflowUtils.show_logs()
        print(logs)
        return 0
        
    elif args.action == 'trigger-dag':
        result = AirflowUtils.trigger_dag(dag_id=args.dag_id)
        return 0 if result else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())