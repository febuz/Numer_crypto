#!/bin/bash
# Script to set up the Python environment for Numerai Crypto
# This script should be sourced: source setup_python_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define environment directory
ENV_DIR="/media/knight2/EDB/numer_crypto_temp/environment"
PYTHON_VERSION="3.12"  # Specify Python version for compatibility with all libraries

# Create directories if they don't exist
mkdir -p "$ENV_DIR"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/log"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/raw"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/processed"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/models"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/submission"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/prediction"

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    return 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
if ! command_exists python$PYTHON_VERSION && ! command_exists python3; then
    echo -e "${YELLOW}Python $PYTHON_VERSION or python3 not found.${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "sudo apt update && sudo apt install -y python3 python3-venv python3-dev"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "brew install python"
    fi
    return 1
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
pip install --upgrade pip

# Install requirements
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_FILE="$REPO_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${GREEN}Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
else
    echo -e "${RED}Requirements file not found at $REQUIREMENTS_FILE${NC}"
    return 1
fi

# Add environment activation to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo -e "${GREEN}Python environment setup complete.${NC}"
echo -e "${GREEN}Python version:${NC}"
python --version
echo -e "${GREEN}Pip version:${NC}"
pip --version
echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"

# Remind user to use source
echo -e "${YELLOW}Remember to use 'source' when running this script:${NC}"
echo -e "${YELLOW}source scripts/environment/setup_python_env.sh${NC}"