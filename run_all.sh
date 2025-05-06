#!/bin/bash
# Master runner script for Numerai Crypto models

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Set up paths
RUNNERS_DIR="${SCRIPT_DIR}/scripts/runners"
MODELS_DIR="${SCRIPT_DIR}/scripts/models"
LOGS_DIR="/media/knight2/EDB/cryptos/logs"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --no-gpu)
            NO_GPU="--no-gpu"
            shift
            ;;
        --gpu)
            GPU="--gpu"
            shift
            ;;
        --test)
            TEST="--test"
            shift
            ;;
        --help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL    Model to run (quick, advanced, ensemble, h2o, yiedl)"
            echo "  --gpu            Enable GPU acceleration"
            echo "  --no-gpu         Disable GPU acceleration"
            echo "  --test           Run in test mode"
            echo "  --help           Show this help message"
            echo ""
            echo "Available models:"
            echo "  quick            Quick model for rapid testing"
            echo "  advanced         Advanced model with full feature set"
            echo "  ensemble         Ensemble of multiple models"
            echo "  h2o              H2O AutoML model"
            echo "  h2o-simple       Simple H2O model"
            echo "  yiedl            Yiedl submission model"
            echo "  yiedl-quick      Quick Yiedl model"
            echo "  all              Run all models sequentially"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default model if not specified
if [ -z "$MODEL" ]; then
    MODEL="quick"
fi

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run a shell script
run_shell_script() {
    script_name="$1"
    shift
    log_file="${LOGS_DIR}/${script_name%.sh}_${TIMESTAMP}.log"
    
    echo "Running ${script_name} (logging to ${log_file})"
    "${RUNNERS_DIR}/${script_name}" "$@" 2>&1 | tee "${log_file}"
    return ${PIPESTATUS[0]}
}

# Function to run a Python script
run_python_script() {
    script_name="$1"
    shift
    log_file="${LOGS_DIR}/${script_name%.py}_${TIMESTAMP}.log"
    
    echo "Running ${script_name} (logging to ${log_file})"
    python "${MODELS_DIR}/${script_name}" "$@" 2>&1 | tee "${log_file}"
    return ${PIPESTATUS[0]}
}

# Run the selected model
case "${MODEL}" in
    "quick")
        run_python_script "run_quick_model.py" ${GPU} ${TEST} "$@"
        exit $?
        ;;
    "advanced")
        run_python_script "run_advanced_model.py" ${GPU} ${TEST} "$@"
        exit $?
        ;;
    "ensemble")
        run_shell_script "run_crypto_ensemble_v2.sh" ${GPU/--gpu/} ${TEST/--test/--test-first} "$@"
        exit $?
        ;;
    "h2o")
        run_shell_script "run_h2o_submission.sh" ${GPU/--gpu/} "$@"
        exit $?
        ;;
    "h2o-simple")
        run_shell_script "run_h2o_simple.sh" ${GPU/--gpu/} "$@"
        exit $?
        ;;
    "yiedl")
        run_shell_script "run_yiedl_submission.sh" ${GPU/--gpu/} "$@"
        exit $?
        ;;
    "yiedl-quick")
        run_shell_script "run_yiedl_quick.sh" ${GPU/--gpu/} "$@"
        exit $?
        ;;
    "all")
        # Run all models sequentially (only in production mode)
        if [ -n "$TEST" ]; then
            echo "Cannot run all models in test mode. Please run each model individually with --test."
            exit 1
        fi
        
        echo "Running all models sequentially..."
        
        # Python models
        run_python_script "run_quick_model.py" ${GPU} "$@"
        run_python_script "run_advanced_model.py" ${GPU} "$@"
        
        # Shell script models
        run_shell_script "run_crypto_ensemble_v2.sh" ${GPU/--gpu/} "$@"
        run_shell_script "run_h2o_submission.sh" ${GPU/--gpu/} "$@"
        run_shell_script "run_h2o_simple.sh" ${GPU/--gpu/} "$@"
        run_shell_script "run_yiedl_submission.sh" ${GPU/--gpu/} "$@"
        run_shell_script "run_yiedl_quick.sh" ${GPU/--gpu/} "$@"
        
        echo "All models completed. See logs in ${LOGS_DIR}"
        exit 0
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac