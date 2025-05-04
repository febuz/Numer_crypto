#!/bin/bash
# Script to run all GPU tests sequentially
# This script activates the GPU test environment, runs the peak utilization test,
# and then the comprehensive integration test

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="$PROJECT_ROOT/gpu_test_env"
ACTIVATION_SCRIPT="$ENV_PATH/bin/activate_gpu_test"
REPORTS_DIR="$PROJECT_ROOT/reports"

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# Header
echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}         RUNNING ALL GPU UTILIZATION TESTS              ${NC}"
echo -e "${YELLOW}=========================================================${NC}"
echo "Started at: $(date)"
echo ""

# Check if the environment exists
if [ ! -f "$ACTIVATION_SCRIPT" ]; then
    echo -e "${RED}GPU test environment not found.${NC}"
    echo -e "${YELLOW}Creating test environment...${NC}"
    cd "$PROJECT_ROOT"
    bash setup_gpu_test_env.sh
fi

# Activate the environment
echo -e "${YELLOW}Activating GPU test environment...${NC}"
source "$ACTIVATION_SCRIPT"

# Run the peak GPU utilization test
echo -e "\n${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}         RUNNING PEAK GPU UTILIZATION TEST              ${NC}"
echo -e "${YELLOW}=========================================================${NC}"
cd "$SCRIPT_DIR"
python test_peak_gpu.py

# Check test result
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Peak GPU utilization test completed successfully!${NC}"
else
    echo -e "\n${RED}Peak GPU utilization test failed.${NC}"
    exit 1
fi

# Run the comprehensive GPU integration test
echo -e "\n${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}         RUNNING COMPREHENSIVE GPU INTEGRATION TEST     ${NC}"
echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Note: This test may take several minutes to complete.${NC}"
cd "$SCRIPT_DIR"
python test_gpu_integration.py

# Check test result
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Comprehensive GPU integration test completed successfully!${NC}"
else
    echo -e "\n${RED}Comprehensive GPU integration test failed.${NC}"
    # Continue even if this test fails
fi

# Final summary
echo -e "\n${YELLOW}=========================================================${NC}"
echo -e "${GREEN}All GPU tests completed!${NC}"
echo -e "${YELLOW}=========================================================${NC}"
echo "Finished at: $(date)"
echo -e "${GREEN}Test reports are available in: $REPORTS_DIR${NC}"

# Deactivate environment
deactivate