#!/bin/bash
# Environment setup utilities for Numerai Crypto Pipeline

# Import utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
source "$SCRIPT_DIR/scripts/bash_utils/logging.sh"
source "$SCRIPT_DIR/scripts/bash_utils/directory.sh"

# Check H2O Sparkling Water availability
check_sparkling_water() {
    log_info "Checking H2O Sparkling Water availability..."
    
    # Check if pysparkling is available
    python3 -c "
import sys
try:
    import pysparkling
    from pysparkling import H2OContext
    from pyspark.sql import SparkSession
    print('SPARKLING_WATER_AVAILABLE=true')
    sys.exit(0)
except ImportError as e:
    print('SPARKLING_WATER_AVAILABLE=false')
    print(f'Missing dependencies: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_success "H2O Sparkling Water is available"
        return 0
    else
        log_warning "H2O Sparkling Water is not available - H2O models will be skipped"
        return 1
    fi
}

# Environment setup
setup_environment() {
    local SKIP_ENV_SETUP=$1
    
    if [ "$SKIP_ENV_SETUP" = true ]; then
        log_info "Skipping environment setup"
        
        # Even if skipping env setup, ensure matplotlib is installed
        log_info "Checking if matplotlib is installed..."
        python3 -c "import matplotlib" 2>/dev/null
        if [ $? -ne 0 ]; then
            log_info "Installing matplotlib..."
            pip install matplotlib
        else
            log_info "Matplotlib is already installed"
        fi
        
        return 0
    fi

    log_info "Setting up Python and Java environment using setup_env.sh..."
    
    # Use the centralized setup script
    if [ -f "$SCRIPT_DIR/scripts/environment/setup_env.sh" ]; then
        # Source the setup script to inherit the environment
        source "$SCRIPT_DIR/scripts/environment/setup_env.sh"
        
        if [ $? -eq 0 ]; then
            log_success "Environment setup completed successfully"
            
            # Check GPU availability
            python3 "$SCRIPT_DIR/scripts/python_utils/gpu_utils.py" --get-info 2>/dev/null || log_warning "GPU check failed - continuing anyway"
        else
            log_error "Environment setup failed"
            return 1
        fi
    else
        log_error "setup_env.sh not found at $SCRIPT_DIR/scripts/environment/setup_env.sh"
        return 1
    fi
}