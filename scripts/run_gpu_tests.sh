#!/bin/bash
# Main script to run GPU performance tests

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
TEST_TYPE="multi-gpu"  # Default test type
JAVA_VERSION=11        # Default Java version
ROWS=100000            # Default dataset rows
COLS=20                # Default dataset columns
GPU_ID=0               # Default GPU ID

function print_usage {
    echo "Usage: $0 [OPTIONS]"
    echo "Run GPU performance tests."
    echo ""
    echo "Options:"
    echo "  --test TYPE       Test type to run: peak, multi-gpu, java-comparison, all (default: multi-gpu)"
    echo "  --java VERSION    Java version to use: 11 or 17 (default: 11)"
    echo "  --rows COUNT      Number of rows in the dataset (default: 100000)"
    echo "  --cols COUNT      Number of features in the dataset (default: 20)"
    echo "  --gpu-id ID       GPU ID to use for single-GPU tests (default: 0)"
    echo "  --help            Display this help message and exit"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_TYPE="$2"
            shift 2
            ;;
        --java)
            JAVA_VERSION="$2"
            shift 2
            ;;
        --rows)
            ROWS="$2"
            shift 2
            ;;
        --cols)
            COLS="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate test type
if [[ "$TEST_TYPE" != "peak" && "$TEST_TYPE" != "multi-gpu" && "$TEST_TYPE" != "java-comparison" && "$TEST_TYPE" != "all" ]]; then
    echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
    print_usage
    exit 1
fi

# Validate Java version
if [[ "$JAVA_VERSION" != "11" && "$JAVA_VERSION" != "17" ]]; then
    echo -e "${RED}Invalid Java version: $JAVA_VERSION${NC}"
    print_usage
    exit 1
fi

# Ensure GPU test environment exists
if [ ! -d "$SCRIPT_DIR/gpu_test_env" ]; then
    echo -e "${YELLOW}GPU test environment not found. Setting up...${NC}"
    bash "$SCRIPT_DIR/setup_gpu_test_env.sh"
fi

# Set up Java environment
if [ "$JAVA_VERSION" -eq 11 ]; then
    echo -e "${YELLOW}Setting up Java 11 environment...${NC}"
    source "$SCRIPT_DIR/scripts/setup_java11_env.sh"
else
    echo -e "${YELLOW}Setting up Java 17 environment...${NC}"
    source "$SCRIPT_DIR/scripts/setup_java17_env.sh"
fi

# Activate the GPU test environment
echo -e "${YELLOW}Activating GPU test environment...${NC}"
source "$SCRIPT_DIR/gpu_test_env/bin/activate"

# Create reports directory
mkdir -p "$SCRIPT_DIR/reports"

# Run the tests
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Running ${TEST_TYPE} GPU test(s) with Java ${JAVA_VERSION}...${NC}"
python scripts/run_all_gpu_tests.py --test "$TEST_TYPE" --java-version "$JAVA_VERSION" --rows "$ROWS" --cols "$COLS" --gpu-id "$GPU_ID"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Tests completed successfully!${NC}"
    echo -e "${GREEN}Check the reports directory for detailed results.${NC}"
else
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
fi

# Deactivate environment
deactivate