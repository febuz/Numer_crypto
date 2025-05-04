#!/bin/bash
# Script to set up a GPU testing environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gpu_test_env"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up new environment for GPU testing${NC}"

# Check and set JAVA_HOME to Java 11
echo -e "${YELLOW}Setting up Java 11 for this environment${NC}"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java version
JAVA_VERSION=$(java -version 2>&1 | head -1)
echo -e "${GREEN}Current Java version: $JAVA_VERSION${NC}"

# Remove existing environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}Removing existing environment: $ENV_PATH${NC}"
    rm -rf "$ENV_PATH"
fi

# Create new virtual environment
echo -e "${YELLOW}Creating new Python virtual environment at: $ENV_PATH${NC}"
python3 -m venv "$ENV_PATH"

# Activate the environment
echo -e "${YELLOW}Activating virtual environment${NC}"
source "$ENV_PATH/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip${NC}"
pip install --upgrade pip

# Install required packages
echo -e "${YELLOW}Installing required packages${NC}"
pip install numpy pandas scikit-learn matplotlib

# Install ML libraries with GPU support
echo -e "${YELLOW}Installing machine learning libraries with GPU support${NC}"
pip install xgboost lightgbm h2o

# Create an activation script
echo -e "${YELLOW}Creating activation script with Java 11 environment setup${NC}"
cat > "$ENV_PATH/bin/activate_gpu_test" << EOF
#!/bin/bash
source "$ENV_PATH/bin/activate"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=\$JAVA_HOME/bin:\$PATH
echo "Activated GPU testing environment with Java 11"
java -version
EOF

chmod +x "$ENV_PATH/bin/activate_gpu_test"

echo -e "${GREEN}Environment setup completed!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "${YELLOW}source $ENV_PATH/bin/activate_gpu_test${NC}"
echo -e "${GREEN}To run the GPU test, run:${NC}"
echo -e "${YELLOW}cd $SCRIPT_DIR/scripts && python test_gpu_integration.py${NC}"

# Deactivate the environment
deactivate