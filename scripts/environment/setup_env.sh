#!/bin/bash
# Consolidated script to set up both Python and Java environments for Numerai Crypto
# This script should be sourced: source setup_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}= Setting up Numerai Crypto environment =${NC}"
echo -e "${BLUE}=====================================${NC}"

# Define environment directories
ENV_DIR="/media/knight2/EDB/numer_crypto_temp/environment"
PYTHON_VERSION="3.12"  # Specify Python version for compatibility with all libraries
JAVA_HOME="/media/knight2/EDB/numer_crypto_temp/java17"
USE_AIRFLOW=${USE_AIRFLOW:-false}
AIRFLOW_VENV_DIR="${HOME}/airflow_env/venv"

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p "$ENV_DIR"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/log"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/md"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/raw"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/processed"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/models"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/submission"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/prediction"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/gravitator"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ================================
# STEP 1: PYTHON ENVIRONMENT SETUP
# ================================
echo -e "\n${BLUE}=== STEP 1: SETTING UP PYTHON ENVIRONMENT ===${NC}"

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    return 1
fi

# Check for Python
if ! command_exists python$PYTHON_VERSION && ! command_exists python3; then
    echo -e "${YELLOW}Python $PYTHON_VERSION or python3 not found.${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "sudo apt update && sudo apt install -y python3 python3-venv python3-dev python3-pyarrow python3-pandas"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "brew install python pyarrow"
    fi
    return 1
fi

# Check and install graphviz (required for Airflow DAG visualization)
if ! command_exists dot; then
    echo -e "${YELLOW}Graphviz not found (required for Airflow DAG visualization).${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}Installing graphviz...${NC}"
        echo -e "${YELLOW}You may need to run:${NC}"
        echo -e "sudo apt update && sudo apt install -y graphviz"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "${YELLOW}Installing graphviz...${NC}"
        echo -e "${YELLOW}You may need to run:${NC}"
        echo -e "brew install graphviz"
    fi
else
    echo -e "${GREEN}Graphviz is already installed${NC}"
fi

# Use available Python version
if command_exists python$PYTHON_VERSION; then
    PYTHON_CMD="python$PYTHON_VERSION"
else
    PYTHON_CMD="python3"
    echo -e "${YELLOW}Using python3 instead of python$PYTHON_VERSION${NC}"
fi

# Check if virtual environment already exists
if [ ! -d "$ENV_DIR/bin" ]; then
    echo -e "${GREEN}Creating Python virtual environment in $ENV_DIR...${NC}"
    $PYTHON_CMD -m venv "$ENV_DIR"
else
    echo -e "${GREEN}Using existing Python virtual environment in $ENV_DIR${NC}"
fi

# Activate virtual environment
source "$ENV_DIR/bin/activate"

# Check if activation worked
if [ "$VIRTUAL_ENV" != "$ENV_DIR" ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    return 1
fi

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install requirements
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_FILE="$REPO_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${GREEN}Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
    
    # CRITICAL: Ensure parquet support is properly installed
    echo -e "${GREEN}Ensuring parquet support is available...${NC}"
    pip install "pyarrow>=13.0.0" "fastparquet>=2023.10.1" "pandas>=2.0.0" --upgrade
    
    # Install Polars and its dependencies for high-performance data processing
    echo -e "${GREEN}Installing Polars and dependencies for feature generation...${NC}"
    pip install "polars[all]>=1.29.0" "dask>=2025.5.0" "toolz>=0.12.0" "scikit-learn>=1.0.2" "joblib>=1.2.0" --upgrade
else
    echo -e "${RED}Requirements file not found at $REQUIREMENTS_FILE${NC}"
    return 1
fi

# ============================
# STEP 2: JAVA ENVIRONMENT SETUP
# ============================
echo -e "\n${BLUE}=== STEP 2: SETTING UP JAVA ENVIRONMENT ===${NC}"

# Check if Java is installed
if command_exists java; then
    echo -e "${GREEN}Java is already installed:${NC}"
    java -version
    
    # Check Java version
    JAVA_VER=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    echo -e "${BLUE}Detected Java version: $JAVA_VER${NC}"
    
    if [ "$JAVA_VER" -ge 17 ]; then
        echo -e "${GREEN}Java 17 or higher is already installed.${NC}"
        return 0
    else
        echo -e "${YELLOW}Java version $JAVA_VER detected, but version 17+ is required for H2O.${NC}"
    fi
fi

# Define Java directories
JAVA_VERSION="17.0.9"
JAVA_BUILD="17.0.9+9"
JAVA_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-$JAVA_VERSION%2B9/OpenJDK17U-jdk_x64_linux_hotspot_$JAVA_BUILD.tar.gz"
JAVA_ARCHIVE="$JAVA_HOME/openjdk.tar.gz"

# Create Java directory
mkdir -p "$JAVA_HOME"

# Check if Java archive exists
if [ -f "$JAVA_ARCHIVE" ]; then
    echo -e "${GREEN}Using existing Java archive.${NC}"
else
    echo -e "${BLUE}Downloading Java 17...${NC}"
    curl -L "$JAVA_URL" -o "$JAVA_ARCHIVE"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download Java 17.${NC}"
        return 1
    fi
fi

# Extract Java archive
if [ -d "$JAVA_HOME/jdk-$JAVA_BUILD" ]; then
    echo -e "${GREEN}Java 17 already extracted.${NC}"
else
    echo -e "${BLUE}Extracting Java 17...${NC}"
    tar -xzf "$JAVA_ARCHIVE" -C "$JAVA_HOME"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract Java 17.${NC}"
        return 1
    fi
fi

# Set JAVA_HOME and PATH
export JAVA_HOME="$JAVA_HOME/jdk-$JAVA_BUILD"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify Java installation
echo -e "${BLUE}Verifying Java installation...${NC}"
java -version

if [ $? -ne 0 ]; then
    echo -e "${RED}Java installation verification failed.${NC}"
    return 1
fi

echo -e "${GREEN}Java 17 setup complete.${NC}"

# =============================
# STEP 3: AIRFLOW ENVIRONMENT (if needed)
# =============================
if [ "$USE_AIRFLOW" = true ]; then
    echo -e "\n${BLUE}=== STEP 3: SETTING UP AIRFLOW ENVIRONMENT ===${NC}"

    # Initialize Airflow using the same Python environment
    echo -e "${GREEN}Initializing Airflow in the same environment...${NC}"
    
    # Since we're using the same environment, just install Airflow dependencies
    echo -e "${BLUE}Installing Airflow dependencies...${NC}"
    
    # First check if we already have Airflow installed
    if python -c "import airflow" &> /dev/null; then
        echo -e "${GREEN}Airflow is already installed${NC}"
    else
        # Install Airflow with constraints
        AIRFLOW_VERSION=2.8.1
        PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
        
        echo -e "${BLUE}Installing Apache Airflow $AIRFLOW_VERSION (this may take a while)...${NC}"
        pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install Airflow${NC}"
            return 1
        fi
    fi
    
    # Install Slack provider for Airflow
    echo -e "${BLUE}Installing Slack provider for Airflow...${NC}"
    pip install apache-airflow-providers-slack
    
    # Set up Airflow directories
    echo -e "${BLUE}Setting up Airflow directories...${NC}"
    export AIRFLOW_HOME="/media/knight2/EDB/numer_crypto_temp/airflow"
    mkdir -p "$AIRFLOW_HOME/dags"
    mkdir -p "$AIRFLOW_HOME/logs"
    mkdir -p "$AIRFLOW_HOME/plugins"
    mkdir -p "$AIRFLOW_HOME/config"
    
    # Copy DAG files if they exist
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    AIRFLOW_DAGS_DIR="$REPO_DIR/airflow_dags"
    
    if [ -d "$AIRFLOW_DAGS_DIR" ]; then
        echo -e "${BLUE}Copying DAG files to Airflow dags directory...${NC}"
        for dag_file in "$AIRFLOW_DAGS_DIR"/*.py; do
            if [ -f "$dag_file" ]; then
                dag_name=$(basename "$dag_file")
                cp "$dag_file" "$AIRFLOW_HOME/dags/$dag_name"
                echo -e "${GREEN}Copied DAG file: $dag_name${NC}"
            fi
        done
    fi
    
    # Initialize Airflow database if needed
    if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
        echo -e "${BLUE}Initializing Airflow database...${NC}"
        airflow db init
        
        # Admin user is created automatically by Airflow 3.0
        echo -e "${GREEN}Airflow setup complete!${NC}"
        echo -e "${GREEN}Admin user is created automatically by Airflow 3.0${NC}"
        echo -e "${GREEN}The password is stored in $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated${NC}"
    else
        echo -e "${GREEN}Airflow is already initialized${NC}"
    fi
    else
        echo -e "${YELLOW}No Airflow environment found at $AIRFLOW_VENV_DIR.${NC}"
        echo -e "${YELLOW}Airflow will be initialized when you run the pipeline with --airflow-init${NC}"
        
        # Create temporary Python script to ensure parquet support in default Python
        TEMP_SCRIPT=$(mktemp)
        cat > "$TEMP_SCRIPT" << EOF
import os
import site
import sys

try:
    import pyarrow
    print("PyArrow already installed:", pyarrow.__version__)
except ImportError:
    print("PyArrow not found, will be installed")

try:
    import fastparquet
    print("Fastparquet already installed:", fastparquet.__version__)
except ImportError:
    print("Fastparquet not found, will be installed")

site_packages = site.getsitepackages()[0]
print("Site packages directory:", site_packages)
EOF

        echo -e "${YELLOW}Checking system Python for parquet support...${NC}"
        python3 "$TEMP_SCRIPT"
        
        echo -e "${YELLOW}Installing parquet support in system Python...${NC}"
        pip3 install "pyarrow>=13.0.0" "fastparquet>=2023.10.1" "pandas>=2.0.0"
        
        rm "$TEMP_SCRIPT"
    fi
fi

# Add environment activation to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}= Environment setup complete =${NC}"
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}Python version:${NC}"
python --version
echo -e "${GREEN}Pip version:${NC}"
pip --version
echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"
echo -e "${GREEN}Java version:${NC}"
java -version
echo -e "${GREEN}JAVA_HOME: $JAVA_HOME${NC}"

# Remind user to use source
echo -e "${YELLOW}Remember to use 'source' when running this script:${NC}"
echo -e "${YELLOW}source scripts/environment/setup_env.sh${NC}"