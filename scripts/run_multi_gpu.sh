#!/bin/bash
# Script to run the multi-GPU feature generation with optimal configuration
# 
# This script leverages all available GPUs for feature generation, distributing
# the workload efficiently across devices for maximum performance.
#
# It handles:
# - Environment activation
# - GPU process cleanup and memory clearing
# - Proper environment variable setting for CUDA
# - Execution of the multi-GPU feature generation script

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="/media/knight2/EDB/numer_crypto_temp/environment"

# Default settings
MAX_FEATURES=10000
USE_MULTI_GPU=true
USE_ALL_GPUS=true
FORCE_GPU=true
SKIP_CLEANUP=false
TEST_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-features)
      MAX_FEATURES="$2"
      shift 2
      ;;
    --single-gpu)
      USE_MULTI_GPU=false
      shift
      ;;
    --skip-cleanup)
      SKIP_CLEANUP=true
      shift
      ;;
    --test-mode)
      TEST_MODE=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--max-features COUNT] [--single-gpu] [--skip-cleanup] [--test-mode]"
      exit 1
      ;;
  esac
done

# Setup environment
echo -e "${BLUE}=== Multi-GPU Feature Generation ===${NC}"
echo -e "${BLUE}Max Features: ${MAX_FEATURES}${NC}"
echo -e "${BLUE}Use Multiple GPUs: ${USE_MULTI_GPU}${NC}"
echo -e "${BLUE}Skip Cleanup: ${SKIP_CLEANUP}${NC}"
echo -e "${BLUE}Test Mode: ${TEST_MODE}${NC}"

# Check if environment exists
if [ ! -f "${ENV_DIR}/bin/activate" ]; then
    echo -e "${RED}Python environment not found at ${ENV_DIR}${NC}"
    echo -e "${YELLOW}You may need to set up the environment first${NC}"
    exit 1
fi

# Activate environment
echo -e "${BLUE}Activating Python environment...${NC}"
source "${ENV_DIR}/bin/activate"

# Check if PyTorch with CUDA is available
echo -e "${BLUE}Checking PyTorch CUDA support...${NC}"
if ! python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"; then
    echo -e "${RED}Error importing PyTorch${NC}"
    deactivate
    exit 1
fi

# Check CUDA device availability
DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$DEVICE_COUNT" -le 0 ]; then
    echo -e "${RED}No CUDA devices detected${NC}"
    deactivate
    exit 1
fi

echo -e "${GREEN}Detected ${DEVICE_COUNT} CUDA devices${NC}"

# Function to kill GPU processes
function kill_gpu_processes() {
    echo -e "${BLUE}Killing all processes using GPU resources...${NC}"
    
    # Try using nvidia-smi to find processes
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$PIDS" ]; then
        echo -e "${BLUE}Found processes using GPU resources: ${PIDS}${NC}"
        
        # Kill each process
        for PID in $PIDS; do
            echo -e "${BLUE}Killing process with PID ${PID}${NC}"
            kill -9 "$PID" 2>/dev/null
        done
    else
        echo -e "${BLUE}No processes using GPU resources found${NC}"
    fi
}

# Function to clean GPU memory
function clean_gpu_memory() {
    echo -e "${BLUE}Cleaning GPU memory...${NC}"
    
    python -c "
import torch
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f'Found {device_count} CUDA devices')
    
    # Empty cache for each device
    for i in range(device_count):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(i)
        print(f'GPU {i} memory after cleanup: {memory_allocated / (1024**3):.2f} GB allocated')
"
}

# Kill GPU processes and clean memory if not skipped
if [ "$SKIP_CLEANUP" != "true" ]; then
    kill_gpu_processes
    clean_gpu_memory
else
    echo -e "${YELLOW}Skipping process cleanup and memory cleaning${NC}"
fi

# Remind user to check GPU monitoring (optional)
echo -e "${YELLOW}TIP: Open another terminal and run 'nvtop' or 'nvidia-smi -l 5' to monitor GPU usage during the run${NC}"

# Continue automatically without user intervention

# Run the multi-GPU feature generation
echo -e "${BLUE}Running multi-GPU feature generation...${NC}"

if [ "$USE_MULTI_GPU" = true ]; then
    # Run with multiple GPUs
    echo -e "${BLUE}Enabling multi-GPU mode with all ${DEVICE_COUNT} GPUs${NC}"
    
    # Need to set CUDA_VISIBLE_DEVICES explicitly for all GPUs
    CUDA_DEVICES=$(seq -s, 0 $((DEVICE_COUNT-1)))
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo -e "${BLUE}Set CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}${NC}"
    
    # Auto-detect the largest training file for better feature generation
    echo -e "${BLUE}Auto-detecting largest training dataset...${NC}"
    
    # Check for available training files in order of preference
    TRAIN_FILE=""
    BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
    
    # First check for the largest merged training files (7GB+)
    if [ -f "${BASE_DIR}/data/processed/train_merged_r1014_$(date +%Y%m%d).parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train_merged_r1014_$(date +%Y%m%d).parquet"
    elif [ -f "${BASE_DIR}/data/processed/train_merged_r1012_$(date +%Y%m%d).parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train_merged_r1012_$(date +%Y%m%d).parquet"
    else
        # Find the most recent large training file
        TRAIN_FILE=$(find "${BASE_DIR}/data/processed" -name "train_merged_*.parquet" -type f -exec ls -lt {} + | head -1 | awk '{print $NF}')
    fi
    
    # Fallback to train subdirectory
    if [ -z "$TRAIN_FILE" ] && [ -f "${BASE_DIR}/data/processed/train/train_data.parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train/train_data.parquet"
    fi
    
    # Final fallback to the smaller crypto_train.parquet
    if [ -z "$TRAIN_FILE" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/crypto_train.parquet"
    fi
    
    if [ -f "$TRAIN_FILE" ]; then
        FILE_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
        echo -e "${GREEN}Using training file: $TRAIN_FILE (${FILE_SIZE})${NC}"
        
        # Execute the multi-GPU feature generation script with all GPUs
        python "${SCRIPT_DIR}/run_multi_gpu_features.py" \
            --input-file "$TRAIN_FILE" \
            --max-features $MAX_FEATURES \
            --gpus "$CUDA_DEVICES"
    else
        echo -e "${RED}No training file found! Please run data download first.${NC}"
        exit 1
    fi
else
    # Run with single GPU
    echo -e "${BLUE}Using single best GPU mode${NC}"
    
    # Auto-detect the largest training file for single GPU mode too
    echo -e "${BLUE}Auto-detecting largest training dataset...${NC}"
    
    # Check for available training files in order of preference
    TRAIN_FILE=""
    BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
    
    # First check for the largest merged training files (7GB+)
    if [ -f "${BASE_DIR}/data/processed/train_merged_r1014_$(date +%Y%m%d).parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train_merged_r1014_$(date +%Y%m%d).parquet"
    elif [ -f "${BASE_DIR}/data/processed/train_merged_r1012_$(date +%Y%m%d).parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train_merged_r1012_$(date +%Y%m%d).parquet"
    else
        # Find the most recent large training file
        TRAIN_FILE=$(find "${BASE_DIR}/data/processed" -name "train_merged_*.parquet" -type f -exec ls -lt {} + | head -1 | awk '{print $NF}')
    fi
    
    # Fallback to train subdirectory
    if [ -z "$TRAIN_FILE" ] && [ -f "${BASE_DIR}/data/processed/train/train_data.parquet" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/train/train_data.parquet"
    fi
    
    # Final fallback to the smaller crypto_train.parquet
    if [ -z "$TRAIN_FILE" ]; then
        TRAIN_FILE="${BASE_DIR}/data/processed/crypto_train.parquet"
    fi
    
    if [ -f "$TRAIN_FILE" ]; then
        FILE_SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
        echo -e "${GREEN}Using training file: $TRAIN_FILE (${FILE_SIZE})${NC}"
        
        python "${SCRIPT_DIR}/run_multi_gpu_features.py" \
            --input-file "$TRAIN_FILE" \
            --max-features $MAX_FEATURES \
            --gpus "0"
    else
        echo -e "${RED}No training file found! Please run data download first.${NC}"
        exit 1
    fi
fi

# Check the exit code
EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✅ Feature generation completed successfully!${NC}"
else
    echo -e "${RED}❌ Feature generation failed with exit code ${EXIT_CODE}${NC}"
fi

# Deactivate environment
deactivate

# Return the exit code
exit ${EXIT_CODE}