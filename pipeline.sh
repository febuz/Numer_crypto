#!/bin/bash
# Numerai Crypto Complete Pipeline - Optimized for 23-minute execution
# A high-performance pipeline that uses Azure Synapse LightGBM exclusively for fast training
# and efficient model creation to complete within 23 minutes
# Run with: bash pipeline.sh (never use python pipeline.sh)

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import bash utilities
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"
source "$SCRIPT_DIR/scripts/bash_utils/help.sh"

# Check for and activate virtual environment if it exists
if [ -d "/media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_venv" ] && [ -f "/media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_venv/bin/activate" ]; then
    log_info "Activating Python virtual environment..."
    source /media/knight2/EDB/numer_crypto_temp/venv/numer_crypto_venv/bin/activate
    log_info "Using Python: $(which python)"
    log_info "Python version: $(python --version)"
fi

# Default configuration
BASE_DIR="/media/knight2/EDB/numer_crypto_temp"
TOURNAMENT="crypto"
MAX_FEATURES=25000
USE_GPU=true
FORCE_REPROCESS=false
FORCE_RETRAIN=false
MAX_ITERATIONS=1        # Reduced iterations for faster completion
FEATURES_PER_ITERATION=7500 # Reduced features for faster processing
SKIP_YIEDL=false        # By default, we INCLUDE Yiedl data
INCLUDE_HISTORICAL=true # By default, we INCLUDE historical data
GPU_MEMORY_LIMIT=45     # GPU memory limit in GB (for 2x 24GB GPUs with NVLink)
USE_AZURE_SYNAPSE=true  # ALWAYS use Azure Synapse LightGBM
SKIP_DOWNLOAD=false     # By default, we download data
USE_RANDOM_FEATURES=true # Use random baseline features for comparison
MONITOR_MEMORY=true     # Monitor memory usage and clean up when needed

# Enable Azure Synapse LightGBM for all training
export USE_AZURE_SYNAPSE_LIGHTGBM=1
export LIGHTGBM_SYNAPSE_MODE=1
export LIGHTGBM_USE_SYNAPSE=1

# Clear memory caches before starting
if command -v sync &> /dev/null; then
    log_info "Clearing filesystem cache..."
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null || true
fi

# Create standard directory structure
create_standard_directories "$BASE_DIR"

# Monitor memory usage
MONITOR_MEMORY=true
if [ "$MONITOR_MEMORY" = true ]; then
    log_info "Starting memory monitoring in background..."
    memory_monitor() {
        while true; do
            MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
            MEM_USED=$(free -m | awk '/^Mem:/{print $3}')
            MEM_USED_PCT=$((MEM_USED * 100 / MEM_TOTAL))
            
            # Log memory usage and trigger cleanup if necessary
            if [ $MEM_USED_PCT -gt 85 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') [WARNING] HIGH MEMORY USAGE: $MEM_USED_PCT% ($MEM_USED MB / $MEM_TOTAL MB)"
                sync
                echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
            elif [ $MEM_USED_PCT -gt 70 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Memory usage: $MEM_USED_PCT% ($MEM_USED MB / $MEM_TOTAL MB)"
            fi
            
            sleep 60  # Check every minute
        done
    }
    
    # Start monitor in background
    memory_monitor &
    MEMORY_MONITOR_PID=$!
    
    # Make sure to kill the monitor when the script exits
    trap "kill $MEMORY_MONITOR_PID 2>/dev/null || true" EXIT
fi

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-gpu)
                USE_GPU=false
                shift
                ;;
            --force-reprocess)
                FORCE_REPROCESS=true
                shift
                ;;
            --force-retrain)
                FORCE_RETRAIN=true
                shift
                ;;
            --max-iterations)
                MAX_ITERATIONS="$2"
                shift 2
                ;;
            --features-per-iteration)
                FEATURES_PER_ITERATION="$2"
                shift 2
                ;;
            --gpu-memory)
                GPU_MEMORY_LIMIT="$2"
                shift 2
                ;;
            --base-dir)
                BASE_DIR="$2"
                shift 2
                ;;
            --no-azure-synapse)
                USE_AZURE_SYNAPSE=false
                shift
                ;;
            --standard-lightgbm)
                USE_AZURE_SYNAPSE=false
                shift
                ;;
            --skip-yiedl)
                SKIP_YIEDL=true
                shift
                ;;
            --numerai-only)
                SKIP_YIEDL=true
                shift
                ;;
            --skip-historical)
                INCLUDE_HISTORICAL=false
                shift
                ;;
            --latest-only)
                INCLUDE_HISTORICAL=false
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --fix-permissions)
                fix_permissions "$BASE_DIR" "true"
                exit 0
                ;;
            --fix-permissions-aggressive)
                log_info "Using aggressive permission fixing (may require sudo)..."
                fix_permissions "$BASE_DIR" "true"
                exit 0
                ;;
            --no-random-features)
                USE_RANDOM_FEATURES=false
                shift
                ;;
            --no-memory-monitor)
                MONITOR_MEMORY=false
                shift
                ;;
            --help)
                echo "Usage: ./pipeline.sh [options]"
                echo ""
                echo "By default, this pipeline downloads BOTH Numerai and Yiedl data INCLUDING historical data."
                echo ""
                echo "Options:"
                echo "  --no-gpu                Disable GPU usage"
                echo "  --force-reprocess       Force data reprocessing"
                echo "  --force-retrain         Force model retraining"
                echo "  --max-iterations N      Set maximum feature evolution iterations (default: 1)"
                echo "  --features-per-iteration N  Set features per iteration (default: 7500)"
                echo "  --gpu-memory N          Set GPU memory limit in GB (default: 45)"
                echo "  --base-dir DIR          Set base directory for outputs"
                echo "  --no-azure-synapse      Use standard LightGBM instead of Azure Synapse variant"
                echo "  --standard-lightgbm     Same as --no-azure-synapse"
                echo "  --skip-yiedl            Skip Yiedl data download (not recommended)"
                echo "  --numerai-only          Same as --skip-yiedl"
                echo "  --skip-historical       Skip downloading historical data (not recommended)"
                echo "  --latest-only           Same as --skip-historical"
                echo "  --skip-download         Skip data download entirely"
                echo "  --no-random-features    Disable random baseline features for comparison"
                echo "  --no-memory-monitor     Disable memory monitoring and automatic cleanup"
                echo "  --fix-permissions       Fix permissions for all data directories and exit"
                echo "  --help                  Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Parse command line arguments
parse_args "$@"

# Build command for run_pipeline.sh
PIPELINE_CMD=("$SCRIPT_DIR/scripts/pipeline/run_pipeline.sh")
PIPELINE_CMD+=("--base-dir" "$BASE_DIR")

# Add conditional flags
if [ "$SKIP_DOWNLOAD" = true ]; then
    PIPELINE_CMD+=("--skip-download")
fi

if [ "$SKIP_YIEDL" = true ]; then
    PIPELINE_CMD+=("--skip-yiedl")
fi

if [ "$FORCE_REPROCESS" = true ]; then
    PIPELINE_CMD+=("--force-reprocess")
fi

if [ "$FORCE_RETRAIN" = true ]; then
    PIPELINE_CMD+=("--force-retrain")
fi

if [ "$USE_GPU" = false ]; then
    PIPELINE_CMD+=("--no-gpu")
fi

PIPELINE_CMD+=("--max-iterations" "$MAX_ITERATIONS")
PIPELINE_CMD+=("--features-per-iteration" "$FEATURES_PER_ITERATION")
PIPELINE_CMD+=("--gpu-memory" "$GPU_MEMORY_LIMIT")

if [ "$USE_AZURE_SYNAPSE" = false ]; then
    PIPELINE_CMD+=("--no-azure-synapse")
fi

if [ "$INCLUDE_HISTORICAL" = false ]; then
    PIPELINE_CMD+=("--skip-historical")
fi

# Print the command for debugging
log_info "Running pipeline command: ${PIPELINE_CMD[*]}"

# Run the pipeline using the modular pipeline script
"${PIPELINE_CMD[@]}"

exit 0