#!/bin/bash
# Main pipeline execution script for Numerai Crypto
# This script runs the full pipeline from environment setup to submission and supports Airflow orchestration

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source pipeline configuration
if [ -f "$SCRIPT_DIR/config/pipeline_config.py" ]; then
    echo -e "${BLUE}Loading pipeline configuration...${NC}"
    # Export key configuration values as environment variables
    export $(python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from config.pipeline_config import *
print(f'DATA_DIR={DATA_DIR}')
print(f'LOG_DIR={LOG_DIR}')
print(f'MAX_FEATURES={MAX_FEATURES}')
print(f'FEATURE_ENGINE={FEATURE_ENGINE}')
print(f'USE_GPU={USE_GPU}')
print(f'USE_AIRFLOW={USE_AIRFLOW}')
" | grep -v '^$')
fi

# Airflow configuration
AIRFLOW_HOME="/media/knight2/EDB/numer_crypto_temp/airflow"
AIRFLOW_VENV_DIR="/media/knight2/EDB/numer_crypto_temp/environment"
USE_AIRFLOW=${USE_AIRFLOW:-true}  # Default to true

# Default time limit for H2O models: 4 hours (in seconds)
H2O_TIME_LIMIT=14400
START_TIME=$(date +%s)

# REMOVED: Progress tracking functions have been removed as we've switched to Airflow for monitoring

# Process command-line arguments
SKIP_ENV_SETUP=false
SKIP_DATA_DOWNLOAD=false
SKIP_DATA_PROCESSING=false
SKIP_FEATURE_GENERATION=false
SKIP_MODEL_TRAINING=false
SKIP_HISTORICAL=false
MONITOR_MODE=false
BACKGROUND_MODE=false
USE_H2O=true
USE_GRAVITATOR=true
GRAVITATOR_SUBMIT=true
GRAVITATOR_ENSEMBLE_METHOD="mean_rank"
GRAVITATOR_SELECTION_METHOD="combined_rank"
GRAVITATOR_TOP_N=50
GRAVITATOR_MIN_IC=0.01
GRAVITATOR_MIN_SHARPE=0.5
GRAVITATOR_NO_NEUTRALIZE=false
PIT_DATE=""
FEATURE_ENGINE="polars" # Default feature engine
MAX_FEATURES=10000 # Default max features

# Environment setup options
USE_UV=true  # Default to using uv (faster package management)
USE_POETRY=false # Default to not using Poetry
USE_BOOTSTRAP=true # Default to using bootstrap mode when available

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --skip-env-setup           Skip environment setup"
    echo "  --skip-download            Skip data download step"
    echo "  --skip-processing          Skip data processing step"
    echo "  --skip-features            Skip feature generation step"
    echo "  --skip-training            Skip model training step"
    echo "  --skip-historical          Skip downloading historical data (download only latest data)"
    echo "  --time-limit SECONDS       Set time limit for H2O models in seconds (default: 14400 = 4 hours)"
    echo "  --use-h2o                  Include H2O Sparkling Water models in training"
    echo "  --monitor-only             Only monitor an existing pipeline (no execution)"
    echo "  --background               Run pipeline in background and show monitoring UI"
    echo "  --use-gravitator           Use the Data Gravitator for signal processing and ensemble creation"
    echo "  --submit-to-numerai        Submit gravitator results to Numerai Crypto tournament"
    echo "  --gravitator-ensemble      Ensemble method for gravitator (default: mean_rank)"
    echo "  --gravitator-selection     Selection method for gravitator (default: combined_rank)"
    echo "  --gravitator-top-n         Number of top signals to select (default: 50)"
    echo "  --gravitator-min-ic        Minimum IC threshold for signals (default: 0.01)"
    echo "  --gravitator-min-sharpe    Minimum Sharpe ratio for signals (default: 0.5)"
    echo "  --gravitator-no-neutralize Disable signal neutralization"
    echo "  --pit YYYYMMDD             Use point-in-time data from specific date (format: YYYYMMDD)"
    echo "  --feature-engine ENGINE    Feature generation engine: pandas, polars, pyspark (default: pandas)"
    echo "  --max-features NUMBER      Maximum number of features to generate (default: 10000)"
    echo "  --no-airflow               Disable Airflow orchestration (use sequential execution)"
    echo "  --help                     Display this help message"
    echo ""
    echo "Airflow 3.0.1 Options:"
    echo "  --airflow                  Use Airflow for pipeline orchestration (default)"
    echo "  --no-airflow               Disable Airflow orchestration"
    echo "  --airflow-init             Initialize Airflow environment"
    echo "  --airflow-standalone       Run Airflow in standalone mode (all components in one process)"
    echo "  --airflow-webserver        Start the Airflow webserver (Airflow 3.0+)"
    echo "  --airflow-api-server       Start the Airflow API server (legacy compatibility)"
    echo "  --airflow-scheduler        Start the Airflow scheduler"
    echo "  --airflow-stop             Stop all Airflow services"
    echo "  --airflow-status           Check Airflow services status"
    echo "  --airflow-create-user      Create default admin user for Airflow"
    echo "  --airflow-logs             Show Airflow logs"
    echo ""
    echo "Environment Performance Options:"
    echo "  --use-uv                  Use uv package manager for faster installations"
    echo "  --use-poetry              Use Poetry for dependency management"
    echo "  --no-bootstrap            Disable bootstrap mode for environment setup"
}

# Airflow functions using our bash script
airflow_init() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" init
    return $?
}

airflow_standalone() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" standalone
    return $?
}

airflow_api_server() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" api-server
    return $?
}

airflow_webserver() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" webserver
    return $?
}

airflow_scheduler() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" scheduler
    return $?
}

airflow_stop() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" stop
    return $?
}

airflow_status() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" status
    return $?
}

airflow_create_user() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" create-user
    return $?
}

airflow_setup_jwt() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" setup-jwt
    return $?
}

airflow_logs() {
    "$SCRIPT_DIR/scripts/airflow_ops.sh" logs
    return $?
}

# Run pipeline with Airflow orchestration
run_airflow_pipeline() {
    local config_json="{"
    
    # Add configuration based on command-line arguments
    if [ "$SKIP_DATA_DOWNLOAD" = true ]; then
        config_json="${config_json}\"skip_download\": true, "
    else
        config_json="${config_json}\"skip_download\": false, "
    fi
    
    if [ "$SKIP_DATA_PROCESSING" = true ]; then
        config_json="${config_json}\"skip_processing\": true, "
    else
        config_json="${config_json}\"skip_processing\": false, "
    fi
    
    if [ "$SKIP_FEATURE_GENERATION" = true ]; then
        config_json="${config_json}\"skip_features\": true, "
    else
        config_json="${config_json}\"skip_features\": false, "
    fi
    
    if [ "$SKIP_MODEL_TRAINING" = true ]; then
        config_json="${config_json}\"skip_training\": true, "
    else
        config_json="${config_json}\"skip_training\": false, "
    fi
    
    if [ "$SKIP_HISTORICAL" = true ]; then
        config_json="${config_json}\"skip_historical\": true, "
    else
        config_json="${config_json}\"skip_historical\": false, "
    fi
    
    # Add feature engine and model configuration
    config_json="${config_json}\"feature_engine\": \"${FEATURE_ENGINE}\", "
    config_json="${config_json}\"max_features\": ${MAX_FEATURES}, "
    config_json="${config_json}\"use_h2o\": ${USE_H2O}, "
    config_json="${config_json}\"use_gravitator\": ${USE_GRAVITATOR}, "
    config_json="${config_json}\"h2o_time_limit\": ${H2O_TIME_LIMIT}"
    
    # Close the JSON object
    config_json="${config_json}}"
    
    echo -e "${BLUE}Airflow configuration: ${config_json}${NC}"
    
    # Check if Airflow is properly installed
    check_airflow_installed() {
        if [ -d "$AIRFLOW_VENV_DIR" ]; then
            source "$AIRFLOW_VENV_DIR/bin/activate"
            local version=$(airflow version 2>/dev/null)
            deactivate
            
            if [[ "$version" == "3.0.1" ]]; then
                echo -e "${GREEN}Airflow 3.0.1 is installed${NC}"
                return 0
            else
                echo -e "${YELLOW}Airflow 3.0.1 is not properly installed (found: $version)${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Airflow environment directory not found: $AIRFLOW_VENV_DIR${NC}"
            return 1
        fi
    }
    
    # Check if Airflow database is initialized
    check_airflow_db_initialized() {
        if [ -f "$AIRFLOW_HOME/airflow.db" ]; then
            echo -e "${GREEN}Airflow database exists${NC}"
            return 0
        else
            echo -e "${YELLOW}Airflow database not initialized${NC}"
            return 1
        fi
    }
    
    # Check if Airflow is running
    check_airflow_running() {
        if pgrep -f "airflow standalone" > /dev/null || pgrep -f "airflow webserver" > /dev/null; then
            echo -e "${GREEN}Airflow is already running${NC}"
            return 0
        else
            echo -e "${YELLOW}Airflow is not running${NC}"
            return 1
        fi
    }
    
    # Make sure Airflow is installed
    if ! check_airflow_installed; then
        echo -e "${BLUE}Installing Airflow 3.0.1...${NC}"
        
        # Check if we're skipping environment setup
        if [ "$SKIP_ENV_SETUP" = true ]; then
            echo -e "${YELLOW}Environment setup is skipped (--skip-env-setup specified)${NC}"
            echo -e "${BLUE}Creating required directories...${NC}"
            
            # Just create necessary directories without running setup_env.sh
            mkdir -p "$AIRFLOW_HOME"
            mkdir -p "$AIRFLOW_HOME/dags"
            mkdir -p "$AIRFLOW_HOME/logs"
            mkdir -p "$AIRFLOW_HOME/plugins"
            return 0
        else
            # Source the environment setup script with USE_AIRFLOW=true
            USE_AIRFLOW=true source "$SCRIPT_DIR/scripts/environment/setup_env.sh"
            
            if ! check_airflow_installed; then
                echo -e "${RED}Failed to install Airflow 3.0.1${NC}"
                return 1
            fi
        fi
    fi
    
    # Make sure database is initialized
    if ! check_airflow_db_initialized; then
        echo -e "${BLUE}Initializing Airflow database...${NC}"
        source "$AIRFLOW_VENV_DIR/bin/activate"
        export AIRFLOW_HOME="$AIRFLOW_HOME"
        
        # Initialize the database
        airflow db migrate
        
        # Admin user is created automatically by Airflow 3.0
        # Do not set a hardcoded password for security reasons
        echo -e "${BLUE}Admin user is automatically created by Airflow 3.0${NC}"
        echo -e "${BLUE}Password is stored in $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated${NC}"
        
        deactivate
        
        if ! check_airflow_db_initialized; then
            echo -e "${RED}Failed to initialize Airflow database${NC}"
            return 1
        fi
    fi
    
    # Start Airflow if not running
    if ! check_airflow_running; then
        echo -e "${BLUE}Starting Airflow in standalone mode...${NC}"
        source "$AIRFLOW_VENV_DIR/bin/activate"
        export AIRFLOW_HOME="$AIRFLOW_HOME"
        
        # Ensure DAG directory exists and has our DAG
        mkdir -p "$AIRFLOW_HOME/dags"
        if [ ! -f "$AIRFLOW_HOME/dags/numerai_crypto_pipeline_v3.py" ] && [ -f "$SCRIPT_DIR/airflow_dags/numerai_crypto_pipeline_v3.py" ]; then
            echo -e "${BLUE}Copying DAG file to Airflow dags directory...${NC}"
            cp -f "$SCRIPT_DIR/airflow_dags/numerai_crypto_pipeline_v3.py" "$AIRFLOW_HOME/dags/"
        fi
        
        # Start Airflow standalone in the background
        nohup airflow standalone > "$AIRFLOW_HOME/logs/standalone.log" 2>&1 &
        
        # Wait a bit for Airflow to start
        echo -e "${BLUE}Waiting for Airflow to start...${NC}"
        sleep 15
        
        # Check if Airflow is running
        if check_airflow_running; then
            echo -e "${GREEN}Airflow started successfully${NC}"
            echo -e "${BLUE}Web UI available at http://localhost:8989${NC}"
        else
            echo -e "${RED}Failed to start Airflow${NC}"
            echo -e "${YELLOW}Check logs in $AIRFLOW_HOME/logs/standalone.log${NC}"
            return 1
        fi
        
        deactivate
    fi
    
    # Trigger the DAG
    echo -e "${BLUE}Triggering the Numerai Crypto pipeline DAG...${NC}"
    source "$AIRFLOW_VENV_DIR/bin/activate"
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Check if the DAG exists
    if ! airflow dags list | grep -q "numerai_crypto_pipeline_v3"; then
        echo -e "${YELLOW}DAG numerai_crypto_pipeline_v3 not found. Checking for DAG file...${NC}"
        
        if [ -f "$SCRIPT_DIR/airflow_dags/numerai_crypto_pipeline_v3.py" ]; then
            echo -e "${BLUE}Copying DAG file to Airflow dags directory...${NC}"
            cp -f "$SCRIPT_DIR/airflow_dags/numerai_crypto_pipeline_v3.py" "$AIRFLOW_HOME/dags/"
            
            # Wait for DAG to be detected
            echo -e "${BLUE}Waiting for DAG to be detected...${NC}"
            sleep 5
        else
            echo -e "${RED}DAG file not found in $SCRIPT_DIR/airflow_dags/${NC}"
            deactivate
            return 1
        fi
    fi
    
    # Trigger the DAG with configuration
    airflow dags trigger -c "$config_json" numerai_crypto_pipeline_v3
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to trigger DAG${NC}"
        deactivate
        return 1
    fi
    
    echo -e "${GREEN}Pipeline triggered successfully${NC}"
    echo -e "${BLUE}You can monitor the progress at http://localhost:8989/dags/numerai_crypto_pipeline_v3/grid${NC}"
    
    deactivate
    return 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-env-setup)
            SKIP_ENV_SETUP=true
            shift
            ;;
        --skip-download)
            SKIP_DATA_DOWNLOAD=true
            shift
            ;;
        --skip-processing)
            SKIP_DATA_PROCESSING=true
            shift
            ;;
        --skip-features)
            SKIP_FEATURE_GENERATION=true
            shift
            ;;
        --skip-training)
            SKIP_MODEL_TRAINING=true
            shift
            ;;
        --skip-historical)
            SKIP_HISTORICAL=true
            shift
            ;;
#        --numerai-only option removed
        --time-limit)
            H2O_TIME_LIMIT="$2"
            shift 2
            ;;
        --use-h2o)
            USE_H2O=true
            shift
            ;;
        --monitor-only)
            MONITOR_MODE=true
            shift
            ;;
        --background)
            BACKGROUND_MODE=true
            shift
            ;;
        --use-gravitator)
            USE_GRAVITATOR=true
            shift
            ;;
        --submit-to-numerai)
            GRAVITATOR_SUBMIT=true
            shift
            ;;
        --gravitator-ensemble)
            GRAVITATOR_ENSEMBLE_METHOD="$2"
            shift 2
            ;;
        --gravitator-selection)
            GRAVITATOR_SELECTION_METHOD="$2"
            shift 2
            ;;
        --gravitator-top-n)
            GRAVITATOR_TOP_N="$2"
            shift 2
            ;;
        --gravitator-min-ic)
            GRAVITATOR_MIN_IC="$2"
            shift 2
            ;;
        --gravitator-min-sharpe)
            GRAVITATOR_MIN_SHARPE="$2"
            shift 2
            ;;
        --gravitator-no-neutralize)
            GRAVITATOR_NO_NEUTRALIZE=true
            shift
            ;;
        --pit)
            PIT_DATE="$2"
            # Validate date format (YYYYMMDD, 8 digits)
            if [[ ! $PIT_DATE =~ ^[0-9]{8}$ ]]; then
                echo -e "${RED}Invalid point-in-time date format: $PIT_DATE${NC}"
                echo -e "${RED}Expected format: YYYYMMDD (e.g., 20250513)${NC}"
                exit 1
            fi
            shift 2
            ;;
        --feature-engine)
            FEATURE_ENGINE="$2"
            # Validate feature engine
            if [[ ! "$FEATURE_ENGINE" =~ ^(pandas|polars|pyspark)$ ]]; then
                echo -e "${RED}Invalid feature engine: $FEATURE_ENGINE${NC}"
                echo -e "${RED}Supported engines: pandas, polars, pyspark${NC}"
                exit 1
            fi
            shift 2
            ;;
        --max-features)
            MAX_FEATURES="$2"
            # Validate max features (must be a positive integer)
            if [[ ! "$MAX_FEATURES" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Invalid max features: $MAX_FEATURES${NC}"
                echo -e "${RED}Must be a positive integer${NC}"
                exit 1
            fi
            shift 2
            ;;
        --no-airflow)
            USE_AIRFLOW=false
            shift
            ;;
        --airflow-init)
            # Use our airflow_init function
            airflow_init
            exit $?
            ;;
        --airflow-standalone)
            airflow_standalone
            exit $?
            ;;
        --airflow-api-server)
            airflow_api_server
            exit $?
            ;;
        --airflow-webserver)
            airflow_webserver
            exit $?
            ;;
        --airflow-scheduler)
            airflow_scheduler
            exit $?
            ;;
        --airflow-stop)
            airflow_stop
            exit $?
            ;;
        --airflow-status)
            airflow_status
            exit $?
            ;;
        --airflow-create-user)
            airflow_create_user
            exit $?
            ;;
        --airflow-setup-jwt)
            airflow_setup_jwt
            exit $?
            ;;
        --airflow-logs)
            airflow_logs
            exit $?
            ;;
        --use-uv)
            USE_UV=true
            shift
            ;;
        --no-uv)
            USE_UV=false
            shift
            ;;
        --use-poetry)
            USE_POETRY=true
            shift
            ;;
        --no-bootstrap)
            USE_BOOTSTRAP=false
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Function to check remaining time for H2O models
check_h2o_time_limit() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local remaining=$((H2O_TIME_LIMIT - elapsed))
    
    echo -e "${BLUE}H2O Time elapsed: $elapsed seconds, Remaining: $remaining seconds${NC}"
    
    if [ $remaining -lt 600 ]; then
        echo -e "${RED}Warning: Less than 10 minutes remaining for H2O training!${NC}"
        return 1
    fi
    return 0
}

# Function to check symbol alignment
check_symbol_alignment() {
    echo -e "${BLUE}Checking symbol alignment between training and live data...${NC}"
    
    python3 -c "
import os
import sys
sys.path.append('$SCRIPT_DIR')
from utils.data.symbol_manager import SymbolManager

try:
    sm = SymbolManager()
    train_symbols = sm.get_training_symbols()
    live_symbols = sm.get_live_symbols()
    valid_symbols = sm.get_valid_symbols_for_features(min_history_days=20)
    
    print(f'Training symbols: {len(train_symbols[\"all\"])}')
    print(f'Live symbols: {len(live_symbols)}')
    print(f'Valid symbols for features: {len(valid_symbols)}')
    
    # Save mapping
    mapping_file = '/media/knight2/EDB/numer_crypto_temp/data/features/symbol_alignment.json'
    sm.save_symbol_mapping(mapping_file)
    print(f'Symbol mapping saved to: {mapping_file}')
    
    # Check if we have sufficient overlap
    if len(valid_symbols) < 100:
        print(f'WARNING: Insufficient valid symbols ({len(valid_symbols)})')
        sys.exit(1)
    
except Exception as e:
    print(f'Symbol alignment check failed: {e}')
    sys.exit(1)
"
    
    return $?
}

# Function to log the start of a step
log_step_start() {
    echo -e "\n${BLUE}====================================${NC}"
    echo -e "${BLUE}= STEP: $1${NC}"
    echo -e "${BLUE}====================================${NC}"
}

# Check and collect wheels if needed
check_wheels() {
    local SETUP_DIR="/media/knight2/EDB/numer_crypto_temp/setup"
    local WHEELS_DIR="${SETUP_DIR}/wheels"
    local COLLECT_SCRIPT="${WHEELS_DIR}/collect_wheels_with_updates.sh"
    local LATEST_LINK="${WHEELS_DIR}/latest"
    local REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
    local REQ_HASH=$(md5sum "${REQUIREMENTS_FILE}" | cut -d " " -f 1)
    local HASH_FILE="${WHEELS_DIR}/requirements_hash.md5"
    
    # Check if wheel collection script exists
    if [ ! -f "${COLLECT_SCRIPT}" ]; then
        echo -e "${YELLOW}Wheel collection script not found. Creating it...${NC}"
        
        # Ensure directory exists
        mkdir -p "${WHEELS_DIR}"
        
        # Copy the script from the repository
        cp "$SCRIPT_DIR/scripts/environment/collect_wheels_with_updates.sh" "${COLLECT_SCRIPT}" 2>/dev/null || \
        cat > "${COLLECT_SCRIPT}" << 'EOL'
#!/bin/bash
# Placeholder for collect_wheels_with_updates.sh
# This script will be replaced with the full version
echo "Please run the full wheel collection script"
exit 1
EOL
        
        chmod +x "${COLLECT_SCRIPT}"
    fi
    
    # Check if wheels are current
    local NEEDS_UPDATE=false
    
    # If no hash file or latest symlink, need to update
    if [ ! -f "${HASH_FILE}" ] || [ ! -L "${LATEST_LINK}" ]; then
        NEEDS_UPDATE=true
    else
        # Check if requirements file has changed
        local STORED_HASH=$(cat "${HASH_FILE}")
        if [ "${STORED_HASH}" != "${REQ_HASH}" ]; then
            echo -e "${YELLOW}Requirements file has changed since last wheel collection${NC}"
            NEEDS_UPDATE=true
        fi
        
        # Check if wheels directory exists and has wheels
        if [ ! -d "$(readlink "${LATEST_LINK}")" ] || [ -z "$(ls -A "$(readlink "${LATEST_LINK}")" 2>/dev/null)" ]; then
            echo -e "${YELLOW}Wheel directory is empty or not found${NC}"
            NEEDS_UPDATE=true
        fi
    fi
    
    # If update needed, run wheel collection
    if [ "${NEEDS_UPDATE}" = true ]; then
        echo -e "${BLUE}Collecting wheels and updating requirements...${NC}"
        bash "${COLLECT_SCRIPT}"
        
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Wheel collection failed, will continue with regular installation${NC}"
        else
            echo -e "${GREEN}Wheels collected successfully${NC}"
            # Reload requirements since they may have been updated
            echo -e "${BLUE}Updated requirements file. Continuing with setup...${NC}"
        fi
    else
        echo -e "${GREEN}Wheels are up-to-date${NC}"
    fi
}

# Check and cache system dependencies if needed
check_system_deps() {
    local SETUP_DIR="/media/knight2/EDB/numer_crypto_temp/setup"
    local SYSTEM_DEPS_DIR="${SETUP_DIR}/system_deps"
    local CACHE_SCRIPT="${SYSTEM_DEPS_DIR}/cache_system_deps.sh"
    local GRAPHVIZ_DIR="${SYSTEM_DEPS_DIR}/graphviz"
    local LAST_CACHE_FILE="${SYSTEM_DEPS_DIR}/last_cache_date.txt"
    local CURRENT_DATE=$(date +"%Y%m%d")
    
    # Check if cache script exists
    if [ ! -f "${CACHE_SCRIPT}" ]; then
        echo -e "${YELLOW}System dependencies cache script not found. Creating it...${NC}"
        
        # Ensure directory exists
        mkdir -p "${SYSTEM_DEPS_DIR}"
        
        # Copy the script from the repository
        cp "$SCRIPT_DIR/scripts/environment/cache_system_deps.sh" "${CACHE_SCRIPT}" 2>/dev/null || \
        cp /media/knight2/EDB/numer_crypto_temp/setup/system_deps/cache_system_deps.sh "${CACHE_SCRIPT}" 2>/dev/null || \
        cat > "${CACHE_SCRIPT}" << 'EOL'
#!/bin/bash
# Placeholder for cache_system_deps.sh
# This script will be replaced with the full version
echo "Please run the full system dependencies cache script"
exit 1
EOL
        
        chmod +x "${CACHE_SCRIPT}"
    fi
    
    # Check if system dependencies need refreshing
    local NEEDS_UPDATE=false
    
    # If no cache date file or graphviz directory, need to update
    if [ ! -f "${LAST_CACHE_FILE}" ] || [ ! -d "${GRAPHVIZ_DIR}" ]; then
        NEEDS_UPDATE=true
    else
        # Check if cache is recent (within last 30 days)
        local LAST_CACHE_DATE=$(cat "${LAST_CACHE_FILE}")
        local DAYS_DIFF=$(( ($(date -d "${CURRENT_DATE}" +%s) - $(date -d "${LAST_CACHE_DATE}" +%s)) / 86400 ))
        
        if [ $DAYS_DIFF -gt 30 ]; then
            echo -e "${YELLOW}System dependencies cache is more than 30 days old${NC}"
            NEEDS_UPDATE=true
        fi
    fi
    
    # If update needed, run system dependencies cache
    if [ "${NEEDS_UPDATE}" = true ]; then
        echo -e "${BLUE}Caching system dependencies...${NC}"
        bash "${CACHE_SCRIPT}"
        
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}System dependencies caching failed, will continue with regular installation${NC}"
        else
            echo -e "${GREEN}System dependencies cached successfully${NC}"
            echo "${CURRENT_DATE}" > "${LAST_CACHE_FILE}"
        fi
    else
        echo -e "${GREEN}System dependencies cache is up-to-date${NC}"
    fi
}

# Step 1: Set up environment
setup_environment() {
    log_step_start "Setting up environment"
    
    if [ "$SKIP_ENV_SETUP" = true ]; then
        echo -e "${YELLOW}Skipping environment setup (--skip-env-setup specified)${NC}"
        return 0
    fi
    
    # Update progress
    
    # Create directories
    mkdir -p /media/knight2/EDB/numer_crypto_temp/log
    mkdir -p /media/knight2/EDB/numer_crypto_temp/data/raw
    mkdir -p /media/knight2/EDB/numer_crypto_temp/data/processed
    mkdir -p /media/knight2/EDB/numer_crypto_temp/models
    mkdir -p /media/knight2/EDB/numer_crypto_temp/prediction
    mkdir -p /media/knight2/EDB/numer_crypto_temp/submission
    mkdir -p /media/knight2/EDB/numer_crypto_temp/setup/conflicts
    
    # Update progress
    
    # Check and cache system dependencies if needed
    echo -e "${BLUE}Checking system dependencies cache...${NC}"
    check_system_deps
    
    # Check and collect wheels if needed
    echo -e "${BLUE}Checking wheel collection status...${NC}"
    check_wheels
    
    # Source consolidated environment setup (this will affect the current shell)
    echo -e "${GREEN}Setting up the complete environment (Python, Java, and parquet support)...${NC}"
    export USE_AIRFLOW=$USE_AIRFLOW  # Pass the Airflow flag to the script
    export NON_INTERACTIVE=1  # Ensure non-interactive mode
    export USE_UV=true       # Enable uv for faster package installation
    
    # First check if we should use bootstrap mode (for faster startup)
    if [ "$USE_BOOTSTRAP" = true ] && [ -f "/media/knight2/EDB/numer_crypto_temp/environment/flags/core_deps_installed" ]; then
        echo -e "${GREEN}Using bootstrap mode for faster startup...${NC}"
        export BOOTSTRAP_MODE=true
        source "$SCRIPT_DIR/scripts/environment/setup_env.sh"
        
        # Load full environment if needed
        load_full_environment
    else
        # If bootstrap disabled or first run, use normal mode
        echo -e "${GREEN}Using full setup mode...${NC}"
        export BOOTSTRAP_MODE=false
        source "$SCRIPT_DIR/scripts/environment/setup_env.sh"
    fi
    
    # Check if environment setup was successful
    if [ $? -ne 0 ]; then
        echo -e "${RED}Environment setup failed${NC}"
        return 1
    fi
    
    # Update progress
    
    # Update progress
    
    echo -e "${GREEN}Environment setup complete${NC}"
    
    # REMOVED: Progress display
    
    return 0
}

# Step 2: Download data
download_data() {
    log_step_start "Downloading data"
    
    # Skip download if specified
    if [ "$SKIP_DATA_DOWNLOAD" = true ]; then
        echo -e "${YELLOW}Skipping data download (--skip-download specified)${NC}"
        return 0
    fi
    
    if [ "$BASELINE_ONLY" = true ]; then
        echo -e "${YELLOW}Skipping data download (--baseline-only specified)${NC}"
        return 0
    fi
    
    # Create data directory
    mkdir -p /media/knight2/EDB/numer_crypto_temp/data/raw
    
    # Check for 15:00 - force download of new data if after 15:00
    current_hour=$(date +%H)
    current_minute=$(date +%M)
    
    # Convert to 24-hour format number for comparison
    current_time=$((10#$current_hour * 100 + 10#$current_minute))
    
    if [ $current_time -ge 1500 ]; then
        echo -e "${BLUE}Current time is after 15:00 - checking for latest Numerai data release${NC}"
        # Force download of latest data
        FORCE_DOWNLOAD=true
        
        # Get the current date
        TODAY=$(date +%Y%m%d)
        
        # Check if we already have today's data
        NUMERAI_DIR="/media/knight2/EDB/numer_crypto_temp/data/numerai"
        TODAY_DIR="${NUMERAI_DIR}/${TODAY}"
        
        if [ -d "$TODAY_DIR" ]; then
            echo -e "${GREEN}Found today's data directory (${TODAY_DIR})${NC}"
            
            # Check the download time
            DATA_DOWNLOAD_TIME=$(stat -c %y "$TODAY_DIR" | cut -d' ' -f2 | cut -d':' -f1-2)
            DATA_DOWNLOAD_HOUR=$(echo $DATA_DOWNLOAD_TIME | cut -d':' -f1)
            DATA_DOWNLOAD_MINUTE=$(echo $DATA_DOWNLOAD_TIME | cut -d':' -f2)
            DATA_DOWNLOAD_TIME_NUM=$((10#$DATA_DOWNLOAD_HOUR * 100 + 10#$DATA_DOWNLOAD_MINUTE))
            
            if [ $DATA_DOWNLOAD_TIME_NUM -ge 1500 ]; then
                echo -e "${GREEN}Today's data was already downloaded after 15:00 (${DATA_DOWNLOAD_TIME})${NC}"
                FORCE_DOWNLOAD=false
            else
                echo -e "${YELLOW}Today's data was downloaded before 15:00 (${DATA_DOWNLOAD_TIME}) - will download latest release${NC}"
                
                # Delete the current day directory to ensure we download fresh
                echo -e "${BLUE}Removing existing data directory to ensure fresh download${NC}"
                rm -rf "$TODAY_DIR"
            fi
        else
            echo -e "${YELLOW}No data directory for today found - creating a clean download${NC}"
            mkdir -p "$NUMERAI_DIR"
        fi
    else
        FORCE_DOWNLOAD=false
    fi
    
    # Point-in-time data download
    if [ -n "$PIT_DATE" ]; then
        echo -e "${BLUE}Using point-in-time data from $PIT_DATE${NC}"
        
        DOWNLOAD_CMD="python3 $SCRIPT_DIR/scripts/download_data.py --pit $PIT_DATE"
        if [ "$SKIP_HISTORICAL" = true ]; then
            DOWNLOAD_CMD="$DOWNLOAD_CMD --skip-historical"
        fi
        
        eval $DOWNLOAD_CMD
        if [ $? -ne 0 ]; then
            echo -e "${RED}Data download failed for point-in-time $PIT_DATE${NC}"
            return 1
        fi
        
        echo -e "${GREEN}Data download complete for point-in-time $PIT_DATE${NC}"
        return 0
    fi
    
    # Skip historical data
    if [ "$SKIP_HISTORICAL" = true ] && [ "$FORCE_DOWNLOAD" = false ]; then
        echo -e "${BLUE}Downloading latest data only (skipping historical)${NC}"
        
        # Use our Python utility for downloading data
        python3 "$SCRIPT_DIR/scripts/pipeline_utils.py" --action download-data --skip-historical
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Data download failed with --skip-historical option${NC}"
            return 1
        fi
        
        echo -e "${GREEN}Data download complete (skipped historical data)${NC}"
        return 0
    fi
    
    # Force download or full data download
    if [ "$FORCE_DOWNLOAD" = true ]; then
        echo -e "${BLUE}Forcing download of latest Numerai data release (after 15:00)${NC}"
        
        # Use our Python utility for downloading data with force flag
        python3 "$SCRIPT_DIR/scripts/download_data.py" --force
        
        # Verify that we got the latest data
        TODAY=$(date +%Y%m%d)
        NUMERAI_DIR="/media/knight2/EDB/numer_crypto_temp/data/numerai"
        if [ -d "${NUMERAI_DIR}/${TODAY}" ]; then
            echo -e "${GREEN}Successfully downloaded today's data (${TODAY})${NC}"
            
            # List the downloaded files for verification
            echo -e "${BLUE}Files in ${NUMERAI_DIR}/${TODAY}:${NC}"
            ls -la "${NUMERAI_DIR}/${TODAY}"
        else
            echo -e "${YELLOW}Warning: Expected data directory for today (${TODAY}) was not created${NC}"
        fi
    else
        echo -e "${BLUE}Downloading Numerai and Yiedl data...${NC}"
        
        # Use our Python utility for downloading data
        python3 "$SCRIPT_DIR/scripts/download_data.py"
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data download failed${NC}"
        return 1
    fi
    
    # Ensure we have the latest live universe file
    echo -e "${BLUE}Checking for latest live universe file...${NC}"
    LATEST_LIVE_UNIVERSE=$(ls -t /media/knight2/EDB/numer_crypto_temp/data/raw/*/live_universe_r*.parquet 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LIVE_UNIVERSE" ]; then
        echo -e "${GREEN}Found latest live universe: $LATEST_LIVE_UNIVERSE${NC}"
        
        # Create a standard location for the live universe file
        cp "$LATEST_LIVE_UNIVERSE" /media/knight2/EDB/numer_crypto_temp/data/raw/numerai_live.parquet
        
        # Count symbols in the live universe
        UNIVERSE_SYMBOLS=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$LATEST_LIVE_UNIVERSE')
print(len(df['symbol'].dropna().unique()))
        ")
        
        echo -e "${GREEN}Live universe contains $UNIVERSE_SYMBOLS symbols${NC}"
    else
        echo -e "${YELLOW}Warning: No live universe file found after download${NC}"
    fi
    
    # Verify data files
    echo -e "${BLUE}Verifying downloaded data files...${NC}"
    
    python3 "$SCRIPT_DIR/scripts/pipeline_utils.py" --action verify-data > /tmp/data_verification.json
    
    # Display a summary of the verification
    echo -e "${BLUE}Data verification summary:${NC}"
    cat /tmp/data_verification.json | grep -E '"exists"|"size_human"|"size_warning"' | sed 's/",$/"/g' | sed 's/^[ \t]*//'
    
    echo -e "${GREEN}Data download complete${NC}"
    return 0
}

# Step 3: Process data
process_data() {
    log_step_start "Processing data"
    
    if [ "$SKIP_DATA_PROCESSING" = true ]; then
        echo -e "${YELLOW}Skipping data processing (--skip-processing specified)${NC}"
        return 0
    fi
    
    # Update progress
    
    # Check if data was already processed by the retriever
    if [ -f /media/knight2/EDB/numer_crypto_temp/data/merged/merged_train.parquet ]; then
        echo -e "${GREEN}Data already processed by NumeraiDataRetriever${NC}"
        return 0
    fi
    
    # Update progress
    
    # Check if placeholder files already exist
    if [ -f "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_train.parquet" ] && \
       [ -f "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_test.parquet" ] && \
       [ -f "/media/knight2/EDB/numer_crypto_temp/data/processed/crypto_live.parquet" ]; then
        echo -e "${GREEN}Using existing processed files in /media/knight2/EDB/numer_crypto_temp/data/processed/${NC}"
        return 0
    fi
    
    # Update progress
    echo -e "${BLUE}Starting full data processing...${NC}"
    
    # Run the process_data.py script
    python3 "$SCRIPT_DIR/scripts/process_data.py" ${SKIP_HISTORICAL:+--skip-historical}
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data processing failed${NC}"
        handle_pipeline_error "data_processing" "Failed to process data"
        return 1
    fi
    
    echo -e "${GREEN}Data processing complete${NC}"
    return 0
}

# Step 4: Generate features
generate_features() {
    log_step_start "Generating features"
    
    if [ "$SKIP_FEATURE_GENERATION" = true ]; then
        echo -e "${YELLOW}Skipping feature generation (--skip-features specified)${NC}"
        return 0
    fi
    
    # Check memory before feature generation
    echo -e "${BLUE}Checking memory availability...${NC}"
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from utils.memory_utils import check_memory_available
if not check_memory_available():
    print('⚠️  Low memory detected, running garbage collection...')
    sys.exit(1)
"
    
    # Update progress
    
    # Run feature generation based on selected engine
    case "$FEATURE_ENGINE" in
        pandas)
            echo -e "${BLUE}Running feature generation with pandas engine...${NC}"
            
            # Run feature generation script with pandas
            python3 "$SCRIPT_DIR/scripts/generate_features.py" --timeseries --max-features $MAX_FEATURES
            ;;
            
        polars)
            echo -e "${BLUE}Running feature generation with polars engine...${NC}"
            
            # Always try aligned generation first for polars to ensure live symbol coverage
            echo -e "${BLUE}Using aligned symbol generation for proper live coverage...${NC}"
            python3 "$SCRIPT_DIR/scripts/generate_aligned_features.py" --max-features $MAX_FEATURES
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Aligned feature generation completed${NC}"
                
                # Save alignment information for gravitator to use
                echo -e "${BLUE}Saving symbol alignment information...${NC}"
                ALIGNMENT_FILE="/media/knight2/EDB/numer_crypto_temp/data/features/symbol_alignment.json"
                if [ -f "$ALIGNMENT_FILE" ]; then
                    echo -e "${GREEN}Symbol alignment saved for gravitator${NC}"
                fi
            else
                # Fallback to standard polars generator
                echo -e "${YELLOW}Aligned generation failed, using standard polars...${NC}"
                python3 "$SCRIPT_DIR/features/polars_generator.py" --max-features $MAX_FEATURES
            fi
            ;;
            
        pyspark)
            echo -e "${BLUE}Running feature generation with pyspark engine...${NC}"
            
            # Run feature generation with pyspark
            FEATURE_START_TIME=$(date +%s)
            python3 "$SCRIPT_DIR/features/pyspark_generator.py" --max-features $MAX_FEATURES
            FEATURE_END_TIME=$(date +%s)
            FEATURE_DURATION=$((FEATURE_END_TIME - FEATURE_START_TIME))
            
            echo -e "${GREEN}PySpark feature generation completed in $FEATURE_DURATION seconds${NC}"
            ;;
    esac
    
    # Check if feature generation succeeded
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}$FEATURE_ENGINE feature generation failed, trying basic features...${NC}"
        
        # Try basic feature generation
        python3 "$SCRIPT_DIR/scripts/generate_features.py" --max-features $MAX_FEATURES
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Feature generation failed${NC}"
            return 1
        fi
        
        echo -e "${GREEN}Basic feature generation complete${NC}"
    else
        echo -e "${GREEN}Feature generation with $FEATURE_ENGINE complete${NC}"
    fi
    
    return 0
}

# Step 5: Train models
train_models() {
    log_step_start "Training models"
    
    if [ "$SKIP_MODEL_TRAINING" = true ]; then
        echo -e "${YELLOW}Skipping model training (--skip-training specified)${NC}"
        return 0
    fi
    
    # Update progress
    
    # Check if we already ran the integrated pipeline, which includes model training
    if ls /media/knight2/EDB/numer_crypto_temp/md/pipeline_report_*.md 1> /dev/null 2>&1; then
        echo -e "${GREEN}Models already trained by integrated pipeline${NC}"
        return 0
    fi
    
    # Update progress
    
    # Set up H2O environment if needed
    if [ "$USE_H2O" = true ]; then
        echo -e "${BLUE}Setting up H2O Sparkling Water environment...${NC}"
        
        # Check H2O time limit for the H2O models
        check_h2o_time_limit
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Not enough time remaining for H2O models, skipping H2O setup${NC}"
            USE_H2O=false
        fi
    fi
    
    # Run model training with GPU support
    echo -e "${BLUE}Training models using LightGBM and XGBoost...${NC}"
    
    # Build command with appropriate flags
    TRAIN_CMD="python3 $SCRIPT_DIR/scripts/train_models.py --use-gpu --parallel --multi-train"
    
    # Add H2O flag if needed
    if [ "$USE_H2O" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --include-h2o --h2o-time-limit $H2O_TIME_LIMIT"
    fi
    
    # Execute command with all models training in parallel
    $TRAIN_CMD --model-type all
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Standard model training failed, trying basic model training...${NC}"
        
        # Try training without GPU and parallel
        TRAIN_CMD="python3 $SCRIPT_DIR/scripts/train_models.py"
        
        # Add H2O flag if needed
        if [ "$USE_H2O" = true ]; then
            TRAIN_CMD="$TRAIN_CMD --include-h2o --h2o-time-limit $H2O_TIME_LIMIT"
        fi
        
        # Execute command
        $TRAIN_CMD
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}All model training methods failed${NC}"
            return 1
        fi
        
        echo -e "${GREEN}Basic model training complete${NC}"
    else
        echo -e "${GREEN}Model training complete${NC}"
    fi
    
    return 0
}

# Step 6: Generate predictions
generate_predictions() {
    log_step_start "Generating predictions"
    
    # Update progress
    
    # First run standard prediction generation
    echo -e "${BLUE}Generating primary predictions from trained models...${NC}"
    
    python3 "$SCRIPT_DIR/scripts/generate_predictions.py"
    
    # Store the result of the primary generation
    PRIMARY_RESULT=$?
    
    if [ $PRIMARY_RESULT -ne 0 ]; then
        echo -e "${YELLOW}Primary prediction generation failed, continuing with alternatives...${NC}"
    else
        echo -e "${GREEN}Primary prediction generation complete${NC}"
    fi
    
    # Now ensure we have multiple prediction files for the Data Gravitator
    echo -e "${BLUE}Generating additional prediction models for Data Gravitator...${NC}"
    
    # Call the script to generate multiple model outputs
    python3 "$SCRIPT_DIR/scripts/models/generate_model_outputs.py" --min-files 3
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Failed to generate additional model outputs${NC}"
        
        # If both primary and additional generation failed, we have a problem
        if [ $PRIMARY_RESULT -ne 0 ]; then
            echo -e "${RED}All prediction methods failed${NC}"
            return 1
        fi
    fi
    
    # Count prediction files from today
    TODAY=$(date +%Y%m%d)
    PREDICTION_COUNT=$(ls -1 /media/knight2/EDB/numer_crypto_temp/prediction/*${TODAY}*.csv 2>/dev/null | wc -l)
    
    echo -e "${GREEN}Generated ${PREDICTION_COUNT} prediction file(s) for Data Gravitator${NC}"
    
    return 0
}

# Step 7: Run Data Gravitator
run_gravitator() {
    log_step_start "Running Data Gravitator"
    
    if [ "$USE_GRAVITATOR" != "true" ]; then
        echo -e "${YELLOW}Skipping Data Gravitator (--use-gravitator not specified)${NC}"
        return 0
    fi
    
    # Update progress
    
    # Prepare Gravitator arguments
    GRAVITATOR_ARGS=""
    
    # Add all gravitator parameters
    GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-ensemble-method $GRAVITATOR_ENSEMBLE_METHOD"
    GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-selection-method $GRAVITATOR_SELECTION_METHOD"
    GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-top-n $GRAVITATOR_TOP_N"
    GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-min-ic $GRAVITATOR_MIN_IC"
    GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-min-sharpe $GRAVITATOR_MIN_SHARPE"
    
    # Add neutralization flag if needed
    if [ "$GRAVITATOR_NO_NEUTRALIZE" = "true" ]; then
        GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-no-neutralize"
    fi
    
    # Add submission flag if needed
    if [ "$GRAVITATOR_SUBMIT" = "true" ]; then
        GRAVITATOR_ARGS="$GRAVITATOR_ARGS --gravitator-submit"
    fi
    
    # Check for latest live universe file to ensure proper symbol coverage
    echo -e "${BLUE}Checking for latest live universe data...${NC}"
    LATEST_LIVE_UNIVERSE=$(ls -t /media/knight2/EDB/numer_crypto_temp/data/raw/*/live_universe_r*.parquet 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LIVE_UNIVERSE" ]; then
        echo -e "${GREEN}Found latest live universe: $LATEST_LIVE_UNIVERSE${NC}"
        
        # Count symbols in the universe
        UNIVERSE_SYMBOLS=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$LATEST_LIVE_UNIVERSE')
print(len(df['symbol'].dropna().unique()))
        ")
        
        echo -e "${BLUE}Live universe contains $UNIVERSE_SYMBOLS symbols${NC}"
        
        # Copy to ensure gravitator can find it
        cp "$LATEST_LIVE_UNIVERSE" /media/knight2/EDB/numer_crypto_temp/data/raw/numerai_live.parquet
    else
        echo -e "${YELLOW}Warning: No live universe file found - gravitator may not include all required symbols${NC}"
    fi
    
    # Check for symbol alignment file if we used aligned feature generation
    if [ -f "/media/knight2/EDB/numer_crypto_temp/data/features/symbol_alignment.json" ]; then
        echo -e "${GREEN}Using symbol alignment information from feature generation${NC}"
        GRAVITATOR_ARGS="$GRAVITATOR_ARGS --symbol-alignment /media/knight2/EDB/numer_crypto_temp/data/features/symbol_alignment.json"
    fi
    
    # Run the Gravitator
    echo -e "${BLUE}Running Data Gravitator for signal processing...${NC}"
    
    # Use today-only filter by default (only including predictions from today)
    TODAY_ARGS="--today-only"
    
    # Check if today's date filter should be applied
    USE_ALL_PREDICTIONS=false
    
    # If there are no predictions from today, use all available predictions
    TODAY_COUNT=$(ls -1 /media/knight2/EDB/numer_crypto_temp/prediction/*$(date +%Y%m%d)*.csv 2>/dev/null | wc -l)
    if [ "$TODAY_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}No predictions from today found. Using all available predictions.${NC}"
        TODAY_ARGS="--all-predictions"
        USE_ALL_PREDICTIONS=true
    else
        echo -e "${GREEN}Found $TODAY_COUNT prediction files from today.${NC}"
    fi
    
    python3 "$SCRIPT_DIR/scripts/gravitator_integration.py" \
        --base-dir /media/knight2/EDB/numer_crypto_temp \
        --tournament crypto \
        $GRAVITATOR_ARGS \
        $TODAY_ARGS
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Data Gravitator processing failed${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Data Gravitator processing complete${NC}"
    
    # Verify submission has all required symbols
    LATEST_SUBMISSION=$(ls -t /media/knight2/EDB/numer_crypto_temp/submission/gravitator_submission_*.csv 2>/dev/null | head -1)
    
    if [ -n "$LATEST_SUBMISSION" ]; then
        echo -e "${BLUE}Verifying gravitator submission: $LATEST_SUBMISSION${NC}"
        
        # Count symbols in submission
        SUBMISSION_SYMBOLS=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LATEST_SUBMISSION')
# Numerai crypto requires 'symbol' column
if 'symbol' in df.columns:
    print(len(df['symbol'].dropna().unique()))
elif 'id' in df.columns:
    print(f'ERROR: Found \"id\" column instead of required \"symbol\" column')
    print(len(df['id'].dropna().unique()))
else:
    print('0')  # No valid ID column found
        ")
        
        echo -e "${BLUE}Submission contains $SUBMISSION_SYMBOLS symbols${NC}"
        
        if [ "$SUBMISSION_SYMBOLS" -lt "100" ]; then
            echo -e "${YELLOW}Warning: Submission has fewer than 100 symbols (found $SUBMISSION_SYMBOLS)${NC}"
        fi
        
        # Check if it matches the universe count
        if [ -n "$UNIVERSE_SYMBOLS" ] && [ "$SUBMISSION_SYMBOLS" -ne "$UNIVERSE_SYMBOLS" ]; then
            echo -e "${YELLOW}Warning: Symbol count mismatch - universe has $UNIVERSE_SYMBOLS, submission has $SUBMISSION_SYMBOLS${NC}"
        fi
    fi
    
    return 0
}

# Step 8: Create submission
create_submission() {
    log_step_start "Creating submission"
    
    # Update progress
    
    # Check if gravitator output exists (use it if available)
    if [ "$USE_GRAVITATOR" = "true" ]; then
        GRAVITATOR_FILES=$(ls /media/knight2/EDB/numer_crypto_temp/submission/gravitator_submission_*.csv 2>/dev/null | wc -l)
        
        if [ "$GRAVITATOR_FILES" -gt 0 ]; then
            echo -e "${GREEN}Using Data Gravitator submission${NC}"
            
            # Find the most recent gravitator submission
            LATEST_SUBMISSION=$(ls -t /media/knight2/EDB/numer_crypto_temp/submission/gravitator_submission_*.csv | head -1)
            echo -e "${GREEN}Latest submission: $LATEST_SUBMISSION${NC}"
            
            # Verify the submission has all required symbols
            echo -e "${BLUE}Verifying gravitator submission has all required symbols...${NC}"
            
            # Check symbol count
            SUBMISSION_SYMBOL_COUNT=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LATEST_SUBMISSION')
# Numerai crypto requires 'symbol' column
if 'symbol' in df.columns:
    print(len(df['symbol'].dropna().unique()))
elif 'id' in df.columns:
    # This is incorrect - we need to warn the user
    print(f'ERROR: Found \"id\" column instead of required \"symbol\" column')
    print(len(df['id'].dropna().unique()))
else:
    print('0')  # No valid ID column found
            ")
            
            echo -e "${BLUE}Submission contains $SUBMISSION_SYMBOL_COUNT symbols${NC}"
            
            # Check if we have the latest live universe to compare
            LATEST_LIVE_UNIVERSE=$(ls -t /media/knight2/EDB/numer_crypto_temp/data/raw/*/live_universe_r*.parquet 2>/dev/null | head -1)
            
            if [ -n "$LATEST_LIVE_UNIVERSE" ]; then
                UNIVERSE_SYMBOL_COUNT=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$LATEST_LIVE_UNIVERSE')
print(len(df['symbol'].dropna().unique()))
                ")
                
                echo -e "${BLUE}Live universe contains $UNIVERSE_SYMBOL_COUNT symbols${NC}"
                
                if [ "$SUBMISSION_SYMBOL_COUNT" -eq "$UNIVERSE_SYMBOL_COUNT" ]; then
                    echo -e "${GREEN}✓ Submission contains all $UNIVERSE_SYMBOL_COUNT live universe symbols${NC}"
                elif [ "$SUBMISSION_SYMBOL_COUNT" -ge "100" ]; then
                    echo -e "${YELLOW}Submission has $SUBMISSION_SYMBOL_COUNT symbols (universe has $UNIVERSE_SYMBOL_COUNT) - meets minimum requirement${NC}"
                else
                    echo -e "${RED}WARNING: Submission only has $SUBMISSION_SYMBOL_COUNT symbols - may be invalid${NC}"
                fi
            fi
            
            return 0
        fi
    fi
    
    # If no gravitator output or gravitator not used, create standard submission
    echo -e "${BLUE}Creating standard submission...${NC}"
    
    # Ensure we have the latest live universe for submission
    LATEST_LIVE_UNIVERSE=$(ls -t /media/knight2/EDB/numer_crypto_temp/data/raw/*/live_universe_r*.parquet 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LIVE_UNIVERSE" ]; then
        echo -e "${GREEN}Using latest live universe for submission: $LATEST_LIVE_UNIVERSE${NC}"
        # Copy to standard location for submission scripts
        cp "$LATEST_LIVE_UNIVERSE" /media/knight2/EDB/numer_crypto_temp/data/raw/numerai_live.parquet
    fi
    
    # Create submission ensuring all symbols are included
    echo -e "${BLUE}Creating submission with all required symbols...${NC}"
    
    # Use the improved submission creation that handles all symbols
    python3 -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
submission_dir = Path('/media/knight2/EDB/numer_crypto_temp/submission')
submission_dir.mkdir(exist_ok=True)

# Load latest predictions if available
latest_pred = None
pred_files = list(Path('/media/knight2/EDB/numer_crypto_temp/prediction').glob('predictions_*.parquet'))
if pred_files:
    latest_pred = pd.read_parquet(max(pred_files, key=lambda f: f.stat().st_mtime))
    print(f'Loaded predictions with {len(latest_pred)} rows')

# Load latest live universe
live_universe_file = Path('/media/knight2/EDB/numer_crypto_temp/data/raw/numerai_live.parquet')
if not live_universe_file.exists():
    print('ERROR: No live universe file found')
    exit(1)

live_df = pd.read_parquet(live_universe_file)
required_symbols = set(live_df['symbol'].dropna().unique())
print(f'Live universe contains {len(required_symbols)} symbols')

# Create submission data
submission_data = []

if latest_pred is not None and 'symbol' in latest_pred.columns:
    # Use predictions where available
    pred_symbols = set(latest_pred['symbol'].unique())
    
    for symbol in required_symbols:
        if symbol in pred_symbols:
            pred_row = latest_pred[latest_pred['symbol'] == symbol].iloc[0]
            prediction = pred_row.get('prediction', 0.5)
        else:
            prediction = 0.5  # Neutral prediction for missing symbols
        
        submission_data.append({
            'symbol': symbol,
            'prediction': prediction
        })
else:
    # No predictions available - create neutral submission
    print('No predictions found - creating neutral submission')
    for symbol in required_symbols:
        submission_data.append({
            'symbol': symbol,
            'prediction': 0.5
        })

# Create submission DataFrame
submission_df = pd.DataFrame(submission_data)

# Check for duplicate symbols
if submission_df['symbol'].duplicated().any():
    dupes = submission_df['symbol'].duplicated().sum()
    print(f'WARNING: Found {dupes} duplicate symbols in submission - removing duplicates')
    submission_df = submission_df.drop_duplicates(subset=['symbol'], keep='first')

# Check for and fix NaN values
if submission_df['prediction'].isna().any():
    nans = submission_df['prediction'].isna().sum()
    print(f'WARNING: Found {nans} NaN prediction values - replacing with neutral 0.5')
    submission_df['prediction'] = submission_df['prediction'].fillna(0.5)

if submission_df['symbol'].isna().any():
    nans = submission_df['symbol'].isna().sum()
    print(f'WARNING: Found {nans} NaN symbol values - removing these rows')
    submission_df = submission_df.dropna(subset=['symbol'])

# Ensure predictions are in valid range
submission_df['prediction'] = submission_df['prediction'].clip(0.001, 0.999)

# Save submission
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
output_file = submission_dir / f'comprehensive_submission_{timestamp}.csv'
submission_df.to_csv(output_file, index=False)

print(f'Created submission with {len(submission_df)} symbols')
print(f'Saved to: {output_file}')

# Fix any submission format issues
try:
    # Import our submission format fixing function
    from utils.pipeline.gravitator_to_signals import fix_submission_format
    
    # Apply format fixes
    fixed_output_file = fix_submission_format(
        file_path=output_file,
        output_path=output_file,  # Overwrite with fixed version
        tournament='crypto'
    )
    
    print(f'Submission format validated and fixed: {fixed_output_file}')
except Exception as e:
    print(f'Note: Could not run final format validation - {str(e)}')

# Verify
unique_symbols = submission_df['symbol'].nunique()
print(f'Submission contains {unique_symbols} unique symbols')
"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Comprehensive submission creation failed${NC}"
        return 1
    fi
    
    # Find the newly created submission
    LATEST_SUBMISSION=$(ls -t /media/knight2/EDB/numer_crypto_temp/submission/comprehensive_submission_*.csv 2>/dev/null | head -1)
    
    if [ -n "$LATEST_SUBMISSION" ]; then
        echo -e "${GREEN}Created comprehensive submission: $LATEST_SUBMISSION${NC}"
        
        # Validate the submission
        SUBMISSION_SYMBOLS=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LATEST_SUBMISSION')
print(len(df['symbol'].dropna().unique()))
        ")
        
        echo -e "${GREEN}Submission contains $SUBMISSION_SYMBOLS symbols${NC}"
    fi
    
    
    return 0
}

# Run full pipeline
run_pipeline() {
    # REMOVED: Progress tracking initialization
    
    # Set up environment
    setup_environment || return 1
    
    # If using Airflow, trigger DAG
    if [ "$USE_AIRFLOW" = true ]; then
        log_step_start "Running with Airflow 3.0.1"
        
        # Run the Airflow pipeline directly
        run_airflow_pipeline
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Airflow pipeline execution failed${NC}"
            return 1
        fi
        
        echo -e "${GREEN}Airflow pipeline initiated successfully${NC}"
        echo -e "${BLUE}You can monitor progress at http://localhost:8989/dags/numerai_crypto_pipeline_v3/grid${NC}"
        
        return 0
    fi
    
    # Regular sequential pipeline
    download_data || return 1
    process_data || return 1
    
    # Check symbol alignment before feature generation
    if [ "$FEATURE_ENGINE" = "polars" ]; then
        check_symbol_alignment
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Symbol alignment check failed, but continuing...${NC}"
        fi
    fi
    
    generate_features || return 1
    train_models || return 1
    generate_predictions || return 1
    
    # Run Data Gravitator if enabled
    if [ "$USE_GRAVITATOR" = true ]; then
        run_gravitator || return 1
    fi
    
    # Create submission
    create_submission || return 1
    
    # Show final progress
    
    return 0
}

# Error handling function
handle_pipeline_error() {
    local stage="$1"
    local error_msg="$2"
    
    echo -e "${RED}Error in stage: $stage${NC}"
    echo -e "${RED}Error message: $error_msg${NC}"
    
    # Run error handler
    python3 "$SCRIPT_DIR/scripts/pipeline_error_handler.py" \
        --stage "$stage" \
        --error "$error_msg"
    
    # Create error report
    python3 "$SCRIPT_DIR/scripts/pipeline_error_handler.py" --create-report
    
    # Show recovery options
    echo -e "${YELLOW}Recovery options have been saved. Check error_report.md${NC}"
    
    exit 1
}

# Main execution
if [ "$MONITOR_MODE" = true ]; then
    # Monitoring mode redirects to Airflow UI
    log_step_start "Monitoring Pipeline"
    
    echo -e "${BLUE}Pipeline monitoring has been moved to Airflow${NC}"
    echo -e "${BLUE}Please use Airflow UI at http://localhost:8989/dags/numerai_crypto_pipeline_v3/grid${NC}"
    echo -e "${BLUE}You can also run with --airflow-status to check Airflow services status${NC}"
    
    # Check if Airflow is running
    if pgrep -f "airflow standalone" > /dev/null || pgrep -f "airflow webserver" > /dev/null; then
        echo -e "${GREEN}Airflow is currently running${NC}"
    else
        echo -e "${YELLOW}Airflow is not running. Use --airflow-standalone to start it.${NC}"
    fi
    
    exit 0
fi

# Check prerequisites before running
echo -e "${BLUE}Checking prerequisites...${NC}"
PREREQS=$(python3 "$SCRIPT_DIR/scripts/pipeline_error_handler.py" --check-prerequisites)

if [ $? -ne 0 ]; then
    echo -e "${RED}Prerequisites check failed${NC}"
    echo "$PREREQS"
    handle_pipeline_error "prerequisites" "Prerequisites not met"
fi

# Run the pipeline with error handling
run_pipeline

# Check result
if [ $? -ne 0 ]; then
    echo -e "${RED}Pipeline execution failed${NC}"
    handle_pipeline_error "pipeline" "Pipeline execution failed"
fi

echo -e "${GREEN}Pipeline execution completed successfully${NC}"
exit 0
