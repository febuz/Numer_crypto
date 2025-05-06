#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Switch to main branch
echo -e "${BLUE}===== Switching to main branch =====${NC}"
git checkout main
git pull origin main

# Ensure necessary directories exist
echo -e "${YELLOW}Creating necessary directory structure...${NC}"
mkdir -p scripts/setup
mkdir -p tests/functional
mkdir -p tests/performance
mkdir -p docs/setup
echo -e "${GREEN}Directory structure created${NC}"

# Move setup scripts to scripts/setup
echo -e "${YELLOW}Moving setup scripts to scripts/setup...${NC}"
git mv setup_env.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_env.sh${NC}"
git mv setup_env_venv.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_env_venv.sh${NC}"
git mv setup_env_and_run_multiGPU.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_env_and_run_multiGPU.sh${NC}"
git mv setup_gpu_test_env.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_gpu_test_env.sh${NC}"
git mv setup_h2o_sparkling_java17.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_h2o_sparkling_java17.sh${NC}"
git mv setup_test_env_java11.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_test_env_java11.sh${NC}"
git mv setup_test_env_java17.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_test_env_java17.sh${NC}"
git mv setup_ubuntu_24.04_gpu.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_ubuntu_24.04_gpu.sh${NC}"
git mv setup_venv.sh scripts/setup/ 2>/dev/null || echo -e "${RED}Could not move setup_venv.sh${NC}"
echo -e "${GREEN}Setup scripts moved${NC}"

# Move run scripts to scripts
echo -e "${YELLOW}Moving run scripts to scripts...${NC}"
git mv run_gpu_tests.sh scripts/ 2>/dev/null || echo -e "${RED}Could not move run_gpu_tests.sh${NC}"
git mv cleanup_tests.sh scripts/ 2>/dev/null || echo -e "${RED}Could not move cleanup_tests.sh${NC}"
echo -e "${GREEN}Run scripts moved${NC}"

# Move test scripts to tests directory
echo -e "${YELLOW}Moving test scripts to tests directory...${NC}"
git mv test_h2o_sparkling_java11.py tests/performance/ 2>/dev/null || echo -e "${RED}Could not move test_h2o_sparkling_java11.py${NC}"
git mv test_h2o_sparkling_minimal.py tests/functional/ 2>/dev/null || echo -e "${RED}Could not move test_h2o_sparkling_minimal.py${NC}"
git mv test_java11_h2o_spark.py tests/performance/ 2>/dev/null || echo -e "${RED}Could not move test_java11_h2o_spark.py${NC}"
git mv test_java17_xgboost_gpu.py tests/performance/ 2>/dev/null || echo -e "${RED}Could not move test_java17_xgboost_gpu.py${NC}"
echo -e "${GREEN}Test scripts moved${NC}"

# Update README.md to reflect new locations
echo -e "${YELLOW}Updating README.md...${NC}"
if [ -f "README.md" ]; then
    sed -i 's|./setup_|./scripts/setup/setup_|g' README.md
    sed -i 's|./run_|./scripts/run_|g' README.md
    sed -i 's|./test_|./tests/test_|g' README.md
    git add README.md
    echo -e "${GREEN}README.md updated${NC}"
fi

# Create a commit with the changes
echo -e "${YELLOW}Creating commit...${NC}"
git commit -m "Move scripts and tests to appropriate directories

- Moved all setup scripts to scripts/setup directory
- Moved run scripts to scripts directory
- Moved test scripts to tests directory
- Updated README.md to reflect new file locations

This commit improves the repository organization by ensuring that scripts
are in their appropriate directories rather than in the main directory.

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push changes to main
echo -e "${YELLOW}Pushing changes to main...${NC}"
git push origin main

# Switch back to feature branch
echo -e "${BLUE}===== Switching back to feature branch =====${NC}"
git checkout feature/crypto-analysis

echo -e "${GREEN}===== Cleanup complete! =====${NC}"