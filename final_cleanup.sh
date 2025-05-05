#!/bin/bash
# Final cleanup script to remove all redundant files from main directory

set -e  # Exit on error

# Text colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}  Final Cleanup of Redundant Main Directory Files  ${NC}"
echo -e "${YELLOW}==============================================${NC}"

# List of setup files to remove (they should already exist in scripts/setup)
SETUP_FILES=(
    "setup_env.sh"
    "setup_env_venv.sh"
    "setup_gpu_test_env.sh"
    "setup_h2o_sparkling_java17.sh"
    "setup_test_env_java11.sh"
    "setup_test_env_java17.sh"
    "setup_ubuntu_24.04_gpu.sh"
    "setup_venv.sh"
)

# List of test files to remove (they should already exist in tests)
TEST_FILES=(
    "test_h2o_sparkling_java11.py"
    "test_h2o_sparkling_minimal.py"
    "test_java11_h2o_spark.py"
    "test_java17_xgboost_gpu.py"
)

# First, verify that files exist in their target locations
echo -e "${YELLOW}Verifying files exist in their target locations...${NC}"
all_verified=true

for file in "${SETUP_FILES[@]}"; do
    target="scripts/setup/$file"
    if [ ! -f "$target" ]; then
        echo -e "${RED}Target file $target not found. Cannot safely remove $file${NC}"
        all_verified=false
    else
        echo -e "${GREEN}Verified: $target exists${NC}"
    fi
done

for file in "${TEST_FILES[@]}"; do
    if [[ "$file" == *"minimal"* ]]; then
        target="tests/functional/$file"
    else
        target="tests/performance/$file"
    fi
    
    if [ ! -f "$target" ] && [ -f "$file" ]; then
        echo -e "${RED}Target file $target not found. Cannot safely remove $file${NC}"
        all_verified=false
    elif [ -f "$target" ]; then
        echo -e "${GREEN}Verified: $target exists${NC}"
    fi
done

if [ "$all_verified" = false ]; then
    echo -e "\n${RED}Some files could not be verified in their target locations.${NC}"
    echo "Please fix these issues before continuing."
    exit 1
fi

# Remove redundant setup files from main directory
echo -e "\n${YELLOW}Removing redundant setup files from main directory...${NC}"
for file in "${SETUP_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing $file"
        git rm "$file" || echo -e "${RED}Failed to remove $file${NC}"
    fi
done

# Remove redundant test files from main directory
echo -e "\n${YELLOW}Removing redundant test files from main directory...${NC}"
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing $file"
        git rm "$file" || echo -e "${RED}Failed to remove $file${NC}"
    fi
done

# Final message
echo -e "\n${GREEN}Cleanup complete!${NC}"
echo "Check the changes with 'git status' before committing."