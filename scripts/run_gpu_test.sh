#!/bin/bash
# Script to run the GPU integration test

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Preparing to run full GPU integration test${NC}"

# Create reports directory if it doesn't exist
mkdir -p ../reports

# Choose Java 11 for H2O Sparkling Water compatibility
echo -e "${YELLOW}Setting up Java 11 for H2O compatibility${NC}"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java version
JAVA_VERSION=$(java -version 2>&1 | head -1)
echo -e "${GREEN}Using Java: $JAVA_VERSION${NC}"

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU may not be available.${NC}"
    echo "Continuing anyway, but tests may fail..."
else
    echo -e "${GREEN}GPU information:${NC}"
    nvidia-smi
fi

# Get Python environment
PYTHON_CMD=$(which python3)
echo -e "${GREEN}Using Python: $PYTHON_CMD${NC}"
$PYTHON_CMD --version

# Make sure script is executable
chmod +x test_gpu_integration.py

# Run the GPU integration test
echo -e "\n${YELLOW}Running GPU integration test...${NC}"
echo -e "${YELLOW}This may take several minutes to complete${NC}"
$PYTHON_CMD test_gpu_integration.py

# Check if test was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}GPU integration test completed successfully!${NC}"
    echo -e "${GREEN}Check the reports directory for detailed results.${NC}"
else
    echo -e "\n${RED}GPU integration test failed.${NC}"
    exit 1
fi