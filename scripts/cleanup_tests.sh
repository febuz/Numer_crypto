#!/bin/bash
# This script cleans up unnecessary test files

# Set color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# List of test files to keep
# These are the most important tests that demonstrate multi-GPU functionality
TESTS_TO_KEEP=(
    "test_multi_gpu_h2o.py"
    "test_peak_gpu.py"
    "test_gpu_utilization.py"
    "test_lightgbm_gpu.py"
    "test_h2o_sparkling.py"
    "test_h2o_sparkling_java17.py"
)

# Check if we're in the right directory
if [ ! -d "tests/performance" ]; then
    echo -e "${RED}Error: Must be run from the repository root!${NC}"
    exit 1
fi

# Create backup directory
BACKUP_DIR="tests/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${YELLOW}Created backup directory: $BACKUP_DIR${NC}"

# Move tests that aren't in the keep list to backup
echo -e "${YELLOW}Moving unnecessary test files to backup...${NC}"
for file in tests/performance/*.py; do
    if [[ "$file" == *"__init__.py" ]]; then
        continue  # Keep all __init__.py files
    fi
    
    filename=$(basename "$file")
    keep=false
    
    for keep_file in "${TESTS_TO_KEEP[@]}"; do
        if [[ "$filename" == "$keep_file" ]]; then
            keep=true
            break
        fi
    done
    
    if [ "$keep" = false ]; then
        echo "  Backing up: $file"
        cp "$file" "$BACKUP_DIR/"
        if [ -f "$file" ]; then
            # Create stub file
            echo "# This file has been moved to backup. See $BACKUP_DIR/$filename" > "$file"
        fi
    else
        echo "  Keeping: $file"
    fi
done

# Also backup unnecessary functional tests
echo -e "\n${YELLOW}Moving unnecessary functional tests to backup...${NC}"
for file in tests/functional/*.py; do
    if [[ "$file" == *"__init__.py" ]]; then
        continue  # Keep all __init__.py files
    fi
    
    # Only keep basic hardware and minimal tests
    if [[ "$file" != *"test_hardware.py" && "$file" != *"test_minimal.py" ]]; then
        filename=$(basename "$file")
        echo "  Backing up: $file"
        cp "$file" "$BACKUP_DIR/"
        if [ -f "$file" ]; then
            # Create stub file
            echo "# This file has been moved to backup. See $BACKUP_DIR/$filename" > "$file"
        fi
    else
        echo "  Keeping: $file"
    fi
done

echo -e "\n${GREEN}Cleanup complete! All files have been backed up to: $BACKUP_DIR${NC}"
echo -e "You can restore any file if needed."