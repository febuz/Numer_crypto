#!/usr/bin/env python3
"""
Workaround for data download and processing permission issues
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_date_directories():
    """Create date-based directories in user's home directory to avoid permission issues"""
    today = datetime.now().strftime("%Y%m%d")
    
    # Get directories from environment variables
    base_dir = os.environ.get("BASE_DIR", "/media/knight2/EDB/numer_crypto_temp")
    data_dir = os.environ.get("DATA_DIR", os.path.join(base_dir, "data"))
    numerai_dir = os.environ.get("NUMERAI_DATA_DIR", os.path.join(data_dir, "numerai"))
    yiedl_dir = os.environ.get("YIEDL_DATA_DIR", os.path.join(data_dir, "yiedl"))
    
    # User home directory based directories
    home_dir = os.path.expanduser("~")
    temp_dir = os.path.join(home_dir, ".numer_crypto_temp")
    temp_numerai_dir = os.path.join(temp_dir, "numerai", today)
    temp_yiedl_dir = os.path.join(temp_dir, "yiedl", today)
    
    # Create temp directories
    logger.info(f"Creating temporary directories in {temp_dir}")
    os.makedirs(temp_numerai_dir, exist_ok=True)
    os.makedirs(temp_yiedl_dir, exist_ok=True)
    
    # Create symbolic links from the original data directories to the temp directories
    logger.info(f"Setting up symbolic links from original data directories to temp directories")
    target_numerai_dir = os.path.join(numerai_dir, today)
    target_yiedl_dir = os.path.join(yiedl_dir, today)
    
    # Set environment variables to point to the temp directories
    os.environ["NUMERAI_DATA_DIR"] = temp_numerai_dir
    os.environ["YIEDL_DATA_DIR"] = temp_yiedl_dir
    
    logger.info(f"Set NUMERAI_DATA_DIR={temp_numerai_dir}")
    logger.info(f"Set YIEDL_DATA_DIR={temp_yiedl_dir}")
    
    # Write out the paths to a file so the shell script can source them
    env_file = os.path.join(temp_dir, "env_vars.sh")
    with open(env_file, "w") as f:
        f.write(f"export NUMERAI_DATA_DIR=\"{temp_numerai_dir}\"\n")
        f.write(f"export YIEDL_DATA_DIR=\"{temp_yiedl_dir}\"\n")
        f.write(f"export TEMP_NUMERAI_DIR=\"{temp_numerai_dir}\"\n")
        f.write(f"export TEMP_YIEDL_DIR=\"{temp_yiedl_dir}\"\n")
        f.write(f"export TARGET_NUMERAI_DIR=\"{target_numerai_dir}\"\n")
        f.write(f"export TARGET_YIEDL_DIR=\"{target_yiedl_dir}\"\n")
    
    logger.info(f"Wrote environment variables to {env_file}")
    logger.info(f"Use 'source {env_file}' to set these variables in your shell")
    
    return {
        "temp_dir": temp_dir,
        "temp_numerai_dir": temp_numerai_dir,
        "temp_yiedl_dir": temp_yiedl_dir,
        "target_numerai_dir": target_numerai_dir,
        "target_yiedl_dir": target_yiedl_dir,
        "env_file": env_file
    }

def copy_temp_to_target():
    """Copy files from temporary directories to target directories"""
    # Get paths from environment variables
    temp_numerai_dir = os.environ.get("TEMP_NUMERAI_DIR")
    temp_yiedl_dir = os.environ.get("TEMP_YIEDL_DIR")
    target_numerai_dir = os.environ.get("TARGET_NUMERAI_DIR")
    target_yiedl_dir = os.environ.get("TARGET_YIEDL_DIR")
    
    if not temp_numerai_dir or not temp_yiedl_dir or not target_numerai_dir or not target_yiedl_dir:
        logger.error("Missing environment variables. Run create_date_directories first.")
        return False
    
    # Create target directories if they don't exist
    os.makedirs(target_numerai_dir, exist_ok=True)
    os.makedirs(target_yiedl_dir, exist_ok=True)
    
    # Copy files from temp to target
    logger.info(f"Copying files from {temp_numerai_dir} to {target_numerai_dir}")
    for file in os.listdir(temp_numerai_dir):
        src = os.path.join(temp_numerai_dir, file)
        dst = os.path.join(target_numerai_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")
    
    logger.info(f"Copying files from {temp_yiedl_dir} to {target_yiedl_dir}")
    for file in os.listdir(temp_yiedl_dir):
        src = os.path.join(temp_yiedl_dir, file)
        dst = os.path.join(target_yiedl_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workaround for data directory permission issues")
    parser.add_argument("--create-dirs", action="store_true", help="Create date directories in user's home")
    parser.add_argument("--copy-files", action="store_true", help="Copy files from temp to target directories")
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_date_directories()
    elif args.copy_files:
        copy_temp_to_target()
    else:
        parser.print_help()