#!/bin/bash
# This script sets up the environment and runs the multi-GPU tests

# Set color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if EDB drive is mounted
if [ ! -d "/media/knight2/EDB" ]; then
    echo -e "${RED}Error: EDB drive not mounted at /media/knight2/EDB${NC}"
    echo -e "Please ensure the drive is properly mounted and try again."
    exit 1
fi

# Make sure the repository exists on the EDB drive
if [ ! -d "/media/knight2/EDB/repos/Numer_crypto" ]; then
    echo -e "${RED}Error: Repository not found at /media/knight2/EDB/repos/Numer_crypto${NC}"
    echo -e "Please ensure the repository is properly set up on the EDB drive."
    exit 1
fi

# Create the symbolic link if it doesn't exist
if [ ! -L "/home/knight2/repos/Numer_crypto_EDB" ]; then
    echo -e "${YELLOW}Creating symbolic link to EDB repository...${NC}"
    ln -sf /media/knight2/EDB/repos/Numer_crypto /home/knight2/repos/Numer_crypto_EDB
    echo -e "${GREEN}Created symbolic link /home/knight2/repos/Numer_crypto_EDB${NC}"
fi

# Create reports directory if it doesn't exist
if [ ! -d "/media/knight2/EDB/repos/Numer_crypto/reports" ]; then
    echo -e "${YELLOW}Creating reports directory...${NC}"
    mkdir -p /media/knight2/EDB/repos/Numer_crypto/reports
    echo -e "${GREEN}Created reports directory${NC}"
fi

# Create models directory if it doesn't exist
if [ ! -d "/media/knight2/EDB/repos/Numer_crypto/models/pytorch" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p /media/knight2/EDB/repos/Numer_crypto/models/pytorch
    echo -e "${GREEN}Created models directory${NC}"
fi

# Check for Python virtual environment for PyTorch
if [ ! -d "/media/knight2/EDB/repos/Numer_crypto/pytorch_env" ]; then
    echo -e "${YELLOW}Setting up PyTorch environment...${NC}"
    cd /media/knight2/EDB/repos/Numer_crypto
    python3 -m venv pytorch_env
    source pytorch_env/bin/activate
    pip install torch torchvision torchaudio matplotlib pandas numpy scikit-learn
    deactivate
    echo -e "${GREEN}PyTorch environment set up${NC}"
fi

# Check which test to run
echo -e "${YELLOW}Which multi-GPU test would you like to run?${NC}"
echo "1) H2O Sparkling Water (XGBoost)"
echo "2) PyTorch (Neural Network)"
echo "3) Both tests sequentially"
read -p "Enter your choice (1-3): " choice

if [ "$choice" = "1" ] || [ "$choice" = "3" ]; then
    echo -e "\n${YELLOW}Running H2O Sparkling Water multi-GPU test...${NC}"
    cd /media/knight2/EDB/repos/Numer_crypto
    ./scripts/setup/setup_h2o_sparkling_java17.sh
    source h2o_sparkling_java17_env/bin/activate.sparkling
    python tests/performance/test_multi_gpu_h2o.py --output-dir ./reports
    deactivate
    echo -e "${GREEN}H2O Sparkling Water test completed${NC}"
fi

if [ "$choice" = "2" ] || [ "$choice" = "3" ]; then
    echo -e "\n${YELLOW}Running PyTorch multi-GPU test...${NC}"
    cd /media/knight2/EDB/repos/Numer_crypto
    source pytorch_env/bin/activate
    python multi_gpu_pytorch.py --output-dir ./reports
    deactivate
    echo -e "${GREEN}PyTorch test completed${NC}"
fi

echo -e "\n${GREEN}All tests completed. Results are available in /media/knight2/EDB/repos/Numer_crypto/reports/${NC}"