#!/bin/bash
# Test runner for Numerai Crypto GPU tests
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directory for test results
OUTPUT_DIR="test_results"
mkdir -p $OUTPUT_DIR

# Print test header
echo -e "${YELLOW}===== Numerai Crypto GPU Tests =====${NC}"
echo "Starting tests at $(date)"
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local test_cmd=$2
    local output_file="$OUTPUT_DIR/${test_name}_result.log"
    
    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    echo "Command: $test_cmd"
    echo "Logging to: $output_file"
    
    # Run the test and capture output
    if $test_cmd > $output_file 2>&1; then
        echo -e "${GREEN}✅ Test passed: ${test_name}${NC}"
        return 0
    else
        echo -e "${RED}❌ Test failed: ${test_name}${NC}"
        echo "Last 10 lines of output:"
        tail -n 10 $output_file
        return 1
    fi
}

# Test 1: GPU Math Accelerator with small dataset
echo -e "${YELLOW}Test 1: GPU Math Accelerator (small dataset)${NC}"
run_test "gpu_accelerator_small" "python test_gpu_accelerator.py --rows 10000 --cols 100 --interactions 20"

# Test 2: Large dataset handling
echo -e "${YELLOW}Test 2: Large Dataset Handling${NC}"
run_test "large_dataset_handling" "python test_large_dataset_handling.py"

# Test 3: Crypto Ensemble Pipeline (minimal)
echo -e "${YELLOW}Test 3: Crypto Ensemble Pipeline (minimal)${NC}"
run_test "crypto_ensemble_minimal" "python test_crypto_ensemble_pipeline.py --skip-h2o"

# Test 4: Large Batch Processing (subset only)
echo -e "${YELLOW}Test 4: Large Batch Processing${NC}"
run_test "large_batch_processing" "python test_large_batch_processing.py --rows 1000000 --cols 1000 --subset-rows 100000 --subset-cols 200 --skip-generation"

# Print summary
echo ""
echo -e "${YELLOW}===== Test Summary =====${NC}"
passed=$(grep -c "✅ Test passed" <<< "$(cat $OUTPUT_DIR/*_result.log)")
failed=$(grep -c "❌ Test failed" <<< "$(cat $OUTPUT_DIR/*_result.log)")
echo "Tests passed: $passed"
echo "Tests failed: $failed"
echo "Completed at $(date)"