#!/bin/bash
# Configuration script for RAPIDS, Spark, and H2O on Ubuntu 24.04
# This script sets up an environment with GPU support for data science

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gpu_rapids_env"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

echo -e "${YELLOW}========================================================${NC}"
echo -e "${YELLOW}   Setting up GPU Environment for Ubuntu 24.04          ${NC}"
echo -e "${YELLOW}   RAPIDS + Spark + H2O Sparkling Water                 ${NC}"
echo -e "${YELLOW}========================================================${NC}"

# Check system compatibility
echo -e "\n${YELLOW}Checking system compatibility...${NC}"

# Verify Ubuntu 24.04
if ! grep -q "Ubuntu 24.04" /etc/os-release; then
    echo -e "${RED}This script is designed for Ubuntu 24.04. Current OS:${NC}"
    cat /etc/os-release
    echo -e "${YELLOW}Continuing anyway, but this might cause issues.${NC}"
fi

# Check for NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA driver not found. Please install NVIDIA drivers first.${NC}"
    echo "Visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html"
    exit 1
fi

# Display detected GPUs
echo -e "\n${YELLOW}Detected GPUs:${NC}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

# Check for CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}CUDA toolkit not found in PATH. Will install CUDA dependencies in conda environment.${NC}"
fi

# Check for installed Java versions
echo -e "\n${YELLOW}Checking for Java installations...${NC}"
JAVA11_PATH="/usr/lib/jvm/java-11-openjdk-amd64"
JAVA17_PATH="/usr/lib/jvm/java-17-openjdk-amd64"

if [ -d "$JAVA11_PATH" ]; then
    echo -e "${GREEN}Found Java 11 at $JAVA11_PATH${NC}"
    JAVA_AVAILABLE=1
else
    echo -e "${YELLOW}Java 11 not found at $JAVA11_PATH${NC}"
fi

if [ -d "$JAVA17_PATH" ]; then
    echo -e "${GREEN}Found Java 17 at $JAVA17_PATH${NC}"
    JAVA_AVAILABLE=1
else
    echo -e "${YELLOW}Java 17 not found at $JAVA17_PATH${NC}"
fi

if [ -z "$JAVA_AVAILABLE" ]; then
    echo -e "${YELLOW}Installing OpenJDK 11 and 17...${NC}"
    sudo apt update
    sudo apt install -y openjdk-11-jdk openjdk-17-jdk
fi

# Set up Java 11 as default for H2O compatibility
export JAVA_HOME=$JAVA11_PATH
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java version
if command -v java &> /dev/null; then
    echo -e "${GREEN}Default Java:${NC}"
    java -version
else
    echo -e "${RED}Java not found in PATH after setup. Please check your Java installation.${NC}"
    exit 1
fi

# Check if conda is installed
echo -e "\n${YELLOW}Checking for conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
    
    # Download and install Miniconda
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    
    wget $MINICONDA_URL -O $MINICONDA_INSTALLER
    bash $MINICONDA_INSTALLER -b -p $HOME/miniconda
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # Initialize conda for bash
    conda init bash
    
    echo -e "${GREEN}Miniconda installed. Please restart your shell or run 'source ~/.bashrc' before continuing.${NC}"
    exit 0
fi

# Remove existing environment if it exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "\n${YELLOW}Removing existing conda environment: $ENV_NAME${NC}"
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment with Python 3.10 (compatible with latest RAPIDS)
echo -e "\n${YELLOW}Creating new conda environment with Python 3.10...${NC}"
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
echo -e "\n${YELLOW}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install RAPIDS using conda
echo -e "\n${YELLOW}Installing RAPIDS (cuDF, cuML, cuGraph)...${NC}"
conda install -c rapidsai -c conda-forge -c nvidia \
    rapids=24.4 python=3.10 cudatoolkit=12.2 -y

# Install PySpark with dependencies
echo -e "\n${YELLOW}Installing PySpark and dependencies...${NC}"
conda install -c conda-forge \
    pyspark=3.5 \
    findspark \
    py4j \
    openjdk=11 \
    -y

# Install H2O and dependencies
echo -e "\n${YELLOW}Installing H2O and H2O Sparkling Water...${NC}"
pip install h2o pysparkling

# Install other ML libraries with GPU support
echo -e "\n${YELLOW}Installing ML libraries with GPU support...${NC}"
conda install -c conda-forge \
    xgboost=3.0.0 \
    lightgbm \
    scikit-learn \
    pandas \
    matplotlib \
    jupyter \
    ipykernel \
    -y

# Install development tools
echo -e "\n${YELLOW}Installing development tools...${NC}"
conda install -c conda-forge \
    pytest \
    pytest-cov \
    black \
    flake8 \
    -y

# Create activation script with proper environment setup
echo -e "\n${YELLOW}Creating activation script...${NC}"
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
mkdir -p "$(dirname "$ACTIVATE_SCRIPT")"

cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Environment variables for GPU + Rapids + Spark + H2O

# Set Java 11 for H2O compatibility
export JAVA_HOME=$JAVA11_PATH
export PATH=\$JAVA_HOME/bin:\$PATH

# CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Spark environment variables
export SPARK_HOME=\$CONDA_PREFIX/lib/python3.10/site-packages/pyspark
export PYSPARK_PYTHON=\$CONDA_PREFIX/bin/python
export PYSPARK_DRIVER_PYTHON=\$CONDA_PREFIX/bin/python
export PYTHONPATH=\$SPARK_HOME/python:\$SPARK_HOME/python/lib/py4j-*-src.zip:\$PYTHONPATH

# Print environment info
echo "GPU + Rapids + Spark + H2O environment activated"
echo "Java version:"
java -version
echo ""
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create script for using Java 17 with proper module options
echo -e "\n${YELLOW}Creating Java 17 setup script...${NC}"
JAVA17_SCRIPT="$SCRIPT_DIR/setup_java17_env.sh"

cat > "$JAVA17_SCRIPT" << EOF
#!/bin/bash
# Script to set up Java 17 with proper module options for H2O

# Set Java 17 as default
export JAVA_HOME=$JAVA17_PATH
export PATH=\$JAVA_HOME/bin:\$PATH

# Add Java 17 module options for H2O Sparkling Water
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED \\
--add-opens=java.base/java.net=ALL-UNNAMED \\
--add-opens=java.base/sun.net=ALL-UNNAMED"

echo "Java 17 environment configured:"
java -version
echo ""
EOF

chmod +x "$JAVA17_SCRIPT"

# Create a script for switching back to Java 11
echo -e "\n${YELLOW}Creating Java 11 setup script...${NC}"
JAVA11_SCRIPT="$SCRIPT_DIR/setup_java11_env.sh"

cat > "$JAVA11_SCRIPT" << EOF
#!/bin/bash
# Script to set up Java 11 for H2O

# Set Java 11 as default
export JAVA_HOME=$JAVA11_PATH
export PATH=\$JAVA_HOME/bin:\$PATH

# Clear any Java options
unset _JAVA_OPTIONS

echo "Java 11 environment configured:"
java -version
echo ""
EOF

chmod +x "$JAVA11_SCRIPT"

# Create a multi-GPU test script
echo -e "\n${YELLOW}Creating multi-GPU test wrapper script...${NC}"
MULTI_GPU_SCRIPT="$SCRIPT_DIR/run_multi_gpu_test.sh"

cat > "$MULTI_GPU_SCRIPT" << EOF
#!/bin/bash
# Script to run the multi-GPU test

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Set up Java 11 by default
source "$JAVA11_SCRIPT"

# Create reports directory
mkdir -p "$SCRIPT_DIR/reports"

# Run the test
python "$SCRIPT_DIR/tests/performance/test_multi_gpu_h2o.py" "\$@"

# Print completion message
if [ \$? -eq 0 ]; then
    echo -e "${GREEN}Multi-GPU test completed successfully!${NC}"
    echo -e "${GREEN}Check the reports directory for detailed results.${NC}"
else
    echo -e "${RED}Multi-GPU test failed.${NC}"
    exit 1
fi
EOF

chmod +x "$MULTI_GPU_SCRIPT"

# Create a Java comparison test script
echo -e "\n${YELLOW}Creating Java comparison test wrapper script...${NC}"
JAVA_COMP_SCRIPT="$SCRIPT_DIR/run_java_comparison_test.sh"

cat > "$JAVA_COMP_SCRIPT" << EOF
#!/bin/bash
# Script to run the Java comparison test

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Create reports directory
mkdir -p "$SCRIPT_DIR/reports"

# Run the test
python "$SCRIPT_DIR/tests/performance/test_java_gpu_comparison.py" "\$@"

# Print completion message
if [ \$? -eq 0 ]; then
    echo -e "${GREEN}Java comparison test completed successfully!${NC}"
    echo -e "${GREEN}Check the reports directory for detailed results.${NC}"
else
    echo -e "${RED}Java comparison test failed.${NC}"
    exit 1
fi
EOF

chmod +x "$JAVA_COMP_SCRIPT"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"

# Check RAPIDS
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')"
python -c "import cuml; print(f'cuML version: {cuml.__version__}')"

# Check XGBoost with GPU
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}'); print(f'CUDA enabled: {xgb.build_info().get(\"USE_CUDA\", \"no\")}')"

# Check PySpark
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"

# Check H2O
python -c "import h2o; print(f'H2O version: {h2o.__version__}')"

# Installation complete
echo -e "\n${GREEN}========================================================${NC}"
echo -e "${GREEN}   GPU Environment Setup Complete                        ${NC}"
echo -e "${GREEN}========================================================${NC}"
echo -e "\nTo activate the environment, run:"
echo -e "   ${YELLOW}conda activate $ENV_NAME${NC}"
echo -e "\nAvailable test scripts:"
echo -e "   ${YELLOW}$MULTI_GPU_SCRIPT${NC} - Run the multi-GPU test"
echo -e "   ${YELLOW}$JAVA_COMP_SCRIPT${NC} - Run the Java comparison test"
echo -e "\nTo switch Java versions:"
echo -e "   ${YELLOW}source $JAVA11_SCRIPT${NC} - Use Java 11"
echo -e "   ${YELLOW}source $JAVA17_SCRIPT${NC} - Use Java 17"