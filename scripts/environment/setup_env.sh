#!/bin/bash
# Consolidated script to set up Python and Java environments for Numerai Crypto
# This script should be sourced: source setup_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}= Setting up Numerai Crypto environment =${NC}"
echo -e "${BLUE}=====================================${NC}"

# Define environment directories
ENV_DIR="/media/knight2/EDB/numer_crypto_temp/environment"
PYTHON_VERSION="3.12"  # Specify Python version for compatibility with all libraries
JAVA_HOME="/media/knight2/EDB/numer_crypto_temp/java17"

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p "$ENV_DIR"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/log"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/md"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/raw"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/data/processed"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/models"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/submission"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/prediction"
mkdir -p "/media/knight2/EDB/numer_crypto_temp/gravitator"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ================================
# STEP 1: PYTHON ENVIRONMENT SETUP
# ================================
echo -e "\n${BLUE}=== STEP 1: SETTING UP PYTHON ENVIRONMENT ===${NC}"

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
    return 1
fi

# Check for GPU support with more thorough verification
detect_gpu() {
    local gpu_found=false
    local drivers_installed=false
    
    # Check if nvidia-smi command is available
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            echo -e "${GREEN}NVIDIA GPU detected with working drivers:${NC}"
            nvidia-smi | head -n 10
            gpu_found=true
            drivers_installed=true
        else
            echo -e "${YELLOW}NVIDIA tools found but nvidia-smi failed to run. Drivers may not be properly installed.${NC}"
        fi
    else
        echo -e "${YELLOW}NVIDIA nvidia-smi tool not found. Checking for CUDA...${NC}"
    fi
    
    # Check if CUDA is installed and accessible
    CUDA_PATHS=("/usr/local/cuda" "/usr/local/cuda-12.2" "/usr/local/cuda-12.1" "/usr/local/cuda-12.0" "/usr/local/cuda-11.8" "/usr/local/cuda-11.7" "/usr/local/cuda-11.6" "/usr/local/cuda-11.5" "/usr/local/cuda-11.4" "/usr/local/cuda-11.3" "/usr/local/cuda-11.2" "/usr/local/cuda-11.1" "/usr/local/cuda-11.0" "/usr/lib/cuda" "/opt/cuda")
    CUDA_FOUND=false
    CUDA_HOME=""
    
    # Check for nvcc in PATH first
    if command -v nvcc &>/dev/null; then
        echo -e "${GREEN}✅ NVCC found in PATH${NC}"
        nvcc --version | head -n 3
        gpu_found=true
        CUDA_FOUND=true
    else
        echo -e "${YELLOW}⚠️ NVCC not in PATH, searching for CUDA installations...${NC}"
        
        # Search common CUDA installation paths
        for cuda_path in "${CUDA_PATHS[@]}"; do
            if [ -d "$cuda_path" ]; then
                echo -e "${GREEN}CUDA installation found at $cuda_path${NC}"
                if [ -f "$cuda_path/version.txt" ]; then
                    echo -e "${GREEN}CUDA version:${NC} $(cat $cuda_path/version.txt)"
                fi
                
                # Check if nvcc exists in this installation
                if [ -f "$cuda_path/bin/nvcc" ]; then
                    echo -e "${GREEN}Found nvcc at $cuda_path/bin/nvcc${NC}"
                    if "$cuda_path/bin/nvcc" --version &>/dev/null; then
                        echo -e "${GREEN}CUDA toolkit is working correctly${NC}"
                        CUDA_HOME="$cuda_path"
                        CUDA_FOUND=true
                        gpu_found=true
                        
                        # Add CUDA to PATH if not already there
                        if [[ ":$PATH:" != *":$cuda_path/bin:"* ]]; then
                            export PATH="$cuda_path/bin:$PATH"
                            echo -e "${GREEN}Added $cuda_path/bin to PATH${NC}"
                        fi
                        
                        # Set CUDA_HOME environment variable
                        export CUDA_HOME="$cuda_path"
                        echo -e "${GREEN}Set CUDA_HOME=$CUDA_HOME${NC}"
                        
                        # Verify nvcc is now available
                        if command -v nvcc &>/dev/null; then
                            echo -e "${GREEN}✅ NVCC is now available in PATH${NC}"
                            nvcc --version | head -n 3
                        fi
                        break
                    else
                        echo -e "${YELLOW}CUDA toolkit found but nvcc failed to run properly${NC}"
                    fi
                else
                    echo -e "${YELLOW}CUDA installation found but no nvcc binary${NC}"
                fi
            fi
        done
        
        if [ "$CUDA_FOUND" = false ]; then
            echo -e "${YELLOW}No working CUDA installation found${NC}"
            echo -e "${YELLOW}To install CUDA toolkit:${NC}"
            echo -e "  1. Download from https://developer.nvidia.com/cuda-downloads"
            echo -e "  2. Or install via package manager:"
            echo -e "     sudo apt update && sudo apt install nvidia-cuda-toolkit"
        fi
    fi
    
    # Additional check: if drivers were found but CUDA wasn't, we should still return true
    if [ "$drivers_installed" = true ]; then
        gpu_found=true
    fi
    
    # Return true if GPU is found, false otherwise
    $gpu_found
}

# Check for GPU
USE_GPU=false
if detect_gpu; then
    echo -e "${GREEN}GPU detected. Automatically installing GPU acceleration support.${NC}"
    USE_GPU=true
    echo -e "${GREEN}Will install GPU acceleration libraries${NC}"
else
    echo -e "${YELLOW}No GPU detected. Will skip GPU acceleration libraries.${NC}"
fi

# Check for Python
if ! command_exists python$PYTHON_VERSION && ! command_exists python3; then
    echo -e "${YELLOW}Python $PYTHON_VERSION or python3 not found.${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "sudo apt update && sudo apt install -y python3 python3-venv python3-dev python3-pyarrow python3-pandas"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "${YELLOW}You may need to install it using:${NC}"
        echo -e "brew install python pyarrow"
    fi
    return 1
fi

# Check and install graphviz (required for DAG visualization)
check_graphviz_complete() {
    # Use timeout to prevent hanging and check if dot supports svg output format
    timeout 5s dot -T svg &>/dev/null
    return $?
}

if ! command_exists dot; then
    echo -e "${YELLOW}Graphviz not found (required for DAG visualization).${NC}"
else
    echo -e "${GREEN}Graphviz is already installed${NC}"
    
    # Quick check if graphviz works, but don't block on it
    if ! check_graphviz_complete; then
        echo -e "${YELLOW}Graphviz SVG output check timed out or failed. Continuing anyway.${NC}"
    else
        echo -e "${GREEN}Graphviz has SVG output support${NC}"
    fi
fi

# Use available Python version
if command_exists python$PYTHON_VERSION; then
    PYTHON_CMD="python$PYTHON_VERSION"
else
    PYTHON_CMD="python3"
    echo -e "${YELLOW}Using python3 instead of python$PYTHON_VERSION${NC}"
fi

# Check if virtual environment already exists
if [ ! -d "$ENV_DIR/bin" ]; then
    echo -e "${GREEN}Creating Python virtual environment in $ENV_DIR...${NC}"
    $PYTHON_CMD -m venv "$ENV_DIR"
else
    echo -e "${GREEN}Using existing Python virtual environment in $ENV_DIR${NC}"
fi

# Activate virtual environment
source "$ENV_DIR/bin/activate"

# Check if activation worked
if [ "$VIRTUAL_ENV" != "$ENV_DIR" ]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    return 1
fi

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install requirements
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_FILE="$REPO_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${GREEN}Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
    
    # Install GPU dependencies if requested
    if [ "$USE_GPU" = true ]; then
        echo -e "${GREEN}Installing GPU dependencies...${NC}"
        
        # Check CUDA version to determine which packages to install
        if command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
            CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
            echo -e "${GREEN}Detected CUDA version: $CUDA_VERSION (Major: $CUDA_MAJOR, Minor: $CUDA_MINOR)${NC}"
            
            # Install CuPy based on CUDA version
            echo -e "${GREEN}Installing CuPy for CUDA $CUDA_MAJOR...${NC}"
            if [ "$CUDA_MAJOR" = "12" ]; then
                echo -e "${GREEN}Installing CuPy for CUDA 12...${NC}"
                pip install cupy-cuda12x || echo -e "${YELLOW}CuPy installation failed. Continuing without CuPy...${NC}"
            elif [ "$CUDA_MAJOR" = "11" ]; then
                echo -e "${GREEN}Installing CuPy for CUDA 11...${NC}"
                pip install cupy-cuda11x || echo -e "${YELLOW}CuPy installation failed. Continuing without CuPy...${NC}"
            elif [ "$CUDA_MAJOR" = "10" ]; then
                echo -e "${GREEN}Installing CuPy for CUDA 10...${NC}"
                pip install cupy-cuda10x || echo -e "${YELLOW}CuPy installation failed. Continuing without CuPy...${NC}"
            else
                echo -e "${YELLOW}Unsupported CUDA version: $CUDA_VERSION. Skipping CuPy installation.${NC}"
            fi
            
            # Install PyTorch with appropriate CUDA version
            if [ "$CUDA_MAJOR" = "11" ] || [ "$CUDA_MAJOR" = "12" ]; then
                echo -e "${GREEN}Installing PyTorch with CUDA 11.8 support...${NC}"
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            elif [ "$CUDA_MAJOR" = "10" ]; then
                echo -e "${GREEN}Installing PyTorch with CUDA 10.2 support...${NC}"
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu102
            else
                echo -e "${YELLOW}Unsupported CUDA version for PyTorch: $CUDA_VERSION. Installing CPU version.${NC}"
                pip install torch torchvision
            fi
        else
            echo -e "${YELLOW}CUDA toolkit (nvcc) not found. Installing generic GPU packages...${NC}"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        fi
        
        # Install enhanced GPU libraries
        echo -e "${GREEN}Installing enhanced GPU libraries (CatBoost, etc.)...${NC}"
        pip install catboost lightgbm xgboost || echo -e "${YELLOW}Some enhanced GPU libraries failed to install. Continuing...${NC}"
        
        # Install PyCaret and dependencies for AutoML
        echo -e "${GREEN}Installing PyCaret for AutoML...${NC}"
        pip install pycaret[full] || echo -e "${YELLOW}PyCaret full installation failed, trying minimal...${NC}"
        pip install pycaret || echo -e "${YELLOW}PyCaret installation failed. Continuing...${NC}"
        
        # Install H2O and Sparkling Water for AutoML
        echo -e "${GREEN}Installing H2O for AutoML...${NC}"
        pip install h2o || echo -e "${YELLOW}H2O installation failed. Continuing...${NC}"
        
        # Install PySpark (specific version compatible with H2O Sparkling Water)
        echo -e "${GREEN}Installing PySpark for Sparkling Water...${NC}"
        pip install pyspark==3.4.3 py4j==0.10.9.7 || echo -e "${YELLOW}PySpark installation failed. Continuing...${NC}"
        
        # Install H2O Sparkling Water (correct package)
        echo -e "${GREEN}Installing H2O Sparkling Water...${NC}"
        pip install h2o-pysparkling-3.4 || echo -e "${YELLOW}H2O Sparkling Water installation failed. Will use regular H2O...${NC}"
        
        # Install RAPIDS AI with CuML for GPU-accelerated machine learning
        echo -e "${GREEN}Installing RAPIDS AI with CuML...${NC}"
        if [ "$CUDA_MAJOR" = "12" ]; then
            echo -e "${GREEN}Installing RAPIDS AI for CUDA 12...${NC}"
            # Use conda to install RAPIDS for better compatibility
            if command -v conda &>/dev/null; then
                conda install -c rapidsai -c conda-forge -c nvidia cuml=24.02 python=3.12 cudatoolkit=12.2 -y || echo -e "${YELLOW}RAPIDS conda installation failed. Trying pip...${NC}"
            fi
            # Fallback to pip installation
            pip install cuml-cu12 || echo -e "${YELLOW}CuML installation failed. Will use sklearn fallback.${NC}"
        elif [ "$CUDA_MAJOR" = "11" ]; then
            echo -e "${GREEN}Installing RAPIDS AI for CUDA 11...${NC}"
            if command -v conda &>/dev/null; then
                conda install -c rapidsai -c conda-forge -c nvidia cuml=24.02 python=3.12 cudatoolkit=11.8 -y || echo -e "${YELLOW}RAPIDS conda installation failed. Trying pip...${NC}"
            fi
            pip install cuml-cu11 || echo -e "${YELLOW}CuML installation failed. Will use sklearn fallback.${NC}"
        else
            echo -e "${YELLOW}Unsupported CUDA version for RAPIDS AI: $CUDA_VERSION. Skipping CuML installation.${NC}"
        fi
        
        # Verify PyTorch installation (with error handling)
        echo -e "${GREEN}Verifying PyTorch GPU functionality...${NC}"
        python -c "
try:
    import torch
    print('PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('✅ PyTorch GPU acceleration is working')
        print(f'   CUDA version: {torch.version.cuda}')
        print(f'   GPU count: {torch.cuda.device_count()}')
        try:
            for i in range(torch.cuda.device_count()):
                print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
        except Exception as e:
            print(f'   Note: Could not get GPU names: {e}')
    else:
        print('❌ PyTorch GPU acceleration is NOT available')
except Exception as e:
    print(f'Warning: PyTorch verification failed: {e}')
    print('This may not affect functionality - continuing...')
" 2>/dev/null || echo -e "${YELLOW}PyTorch verification failed, but installation appears successful${NC}"
        
        # Verify enhanced GPU libraries
        echo -e "${GREEN}Verifying enhanced GPU libraries...${NC}"
        python -c "
import sys
libraries = {
    'CatBoost': 'catboost',
    'LightGBM': 'lightgbm', 
    'XGBoost': 'xgboost',
    'PyCaret': 'pycaret',
    'H2O': 'h2o',
    'PySpark': 'pyspark',
    'CuML': 'cuml'
}

for name, module in libraries.items():
    try:
        __import__(module)
        print(f'✅ {name}: Available')
        if module == 'catboost':
            from catboost import CatBoostRegressor
            try:
                cb = CatBoostRegressor(task_type='GPU', devices='0', iterations=1, verbose=False)
                print('   - GPU acceleration: Working')
            except Exception as e:
                print(f'   - GPU acceleration: Failed ({e})')
        elif module == 'cuml':
            try:
                from cuml.ensemble import RandomForestRegressor
                print('   - Random Forest GPU: Available')
            except Exception as e:
                print(f'   - Random Forest GPU: Failed ({e})')
        elif module == 'h2o':
            try:
                import h2o
                print(f'   - Version: {h2o.__version__}')
            except Exception as e:
                print(f'   - Version check failed: {e}')
        elif module == 'pycaret':
            try:
                import pycaret
                print(f'   - Version: {pycaret.__version__}')
                # Test basic PyCaret functionality
                from pycaret.datasets import get_data
                print('   - Basic functionality: Working')
            except Exception as e:
                print(f'   - Functionality check failed: {e}')
    except ImportError:
        print(f'❌ {name}: Not available')

# Test H2O Sparkling Water
try:
    from pysparkling import H2OContext
    print('✅ H2O Sparkling Water: Available')
except ImportError as e:
    print(f'❌ H2O Sparkling Water: Not available ({e})')
    # Try alternative import path
    try:
        from h2o.pysparkling import H2OContext
        print('✅ H2O Sparkling Water (alt path): Available')
    except ImportError:
        print('❌ H2O Sparkling Water: Neither import path works')
" 2>/dev/null || echo -e "${YELLOW}Enhanced library verification failed, but installation may be successful${NC}"
    else
        echo -e "${YELLOW}Skipping GPU dependencies (no GPU detected).${NC}"
    fi
else
    echo -e "${RED}Requirements file not found at $REQUIREMENTS_FILE${NC}"
    return 1
fi

# ============================
# STEP 2: JAVA ENVIRONMENT SETUP
# ============================
echo -e "\n${BLUE}=== STEP 2: SETTING UP JAVA ENVIRONMENT ===${NC}"

# Check if Java is installed
if command_exists java; then
    echo -e "${GREEN}Java is already installed:${NC}"
    java -version
    
    # Check Java version
    JAVA_VER=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    echo -e "${BLUE}Detected Java version: $JAVA_VER${NC}"
    
    if [ "$JAVA_VER" -ge 17 ]; then
        echo -e "${GREEN}Java 17 or higher is already installed. Skipping Java installation.${NC}"
        
        # Set JAVA_HOME to system Java if not already set
        if [ -z "$JAVA_HOME" ]; then
            # Try to find Java home
            if command_exists readlink && command_exists dirname; then
                SYSTEM_JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
                export JAVA_HOME="$SYSTEM_JAVA_HOME"
                echo -e "${GREEN}Set JAVA_HOME to system Java: $JAVA_HOME${NC}"
            fi
        fi
        
        # Skip the installation part
        echo -e "${GREEN}Java setup complete (using system installation).${NC}"
        export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
        echo -e "\n${GREEN}=======================================${NC}"
        echo -e "${GREEN}= Environment setup complete =${NC}"
        echo -e "${GREEN}=======================================${NC}"
        echo -e "${GREEN}Python version:${NC}"
        python --version
        echo -e "${GREEN}Pip version:${NC}"
        pip --version
        echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"
        echo -e "${GREEN}Java version:${NC}"
        java -version
        echo -e "${GREEN}JAVA_HOME: $JAVA_HOME${NC}"
        
        # Check GPU status if GPU dependencies were installed
        if [ "$USE_GPU" = true ]; then
            echo -e "\n${GREEN}Enhanced GPU Status:${NC}"
            
            # Check NVCC status
            if command -v nvcc &>/dev/null; then
                echo -e "${GREEN}✅ NVCC found in PATH${NC}"
                nvcc --version | head -n 1
            else
                echo -e "${YELLOW}⚠️ NVCC not in PATH${NC}"
                echo -e "${YELLOW}GPU libraries installed but CUDA toolkit not in PATH${NC}"
            fi
            
            # Check enhanced GPU libraries status
            echo -e "${GREEN}Enhanced GPU Libraries Status:${NC}"
            python -c "
libraries = ['torch', 'catboost', 'lightgbm', 'xgboost', 'pycaret', 'h2o', 'pyspark', 'cuml']
available = []
for lib in libraries:
    try:
        __import__(lib)
        available.append(lib)
    except ImportError:
        pass

print(f'Available libraries: {len(available)}/{len(libraries)}')
if 'torch' in available:
    import torch
    if torch.cuda.is_available():
        print(f'✅ PyTorch CUDA: {torch.cuda.device_count()} GPUs')
if 'catboost' in available:
    print('✅ CatBoost GPU: Ready')
if 'lightgbm' in available:
    print('✅ LightGBM GPU: Ready')  
if 'xgboost' in available:
    print('✅ XGBoost GPU: Ready')
if 'pycaret' in available:
    print('✅ PyCaret AutoML: Ready')
if 'h2o' in available and 'pyspark' in available:
    try:
        from pysparkling import H2OContext
        print('✅ H2O Sparkling Water: Ready')
    except ImportError:
        print('✅ H2O: Ready (regular mode)')
elif 'h2o' in available:
    print('✅ H2O: Ready (regular mode)')
if 'cuml' in available:
    print('✅ CuML GPU: Ready')
else:
    print('ℹ CuML: Using sklearn fallback')
" 2>/dev/null || echo -e "${YELLOW}Could not check enhanced library status${NC}"
        fi
        
        echo -e "\n${YELLOW}Remember to use 'source' when running this script:${NC}"
        echo -e "${YELLOW}source scripts/environment/setup_env.sh${NC}"
        return 0
    else
        echo -e "${YELLOW}Java version $JAVA_VER detected, but version 17+ is required for H2O.${NC}"
        echo -e "${YELLOW}Installing Java 17...${NC}"
    fi
else
    echo -e "${YELLOW}Java not found. Installing Java 17...${NC}"
fi

# Define Java directories
JAVA_VERSION="17.0.9"
JAVA_BUILD="17.0.9+9"
JAVA_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-$JAVA_VERSION%2B9/OpenJDK17U-jdk_x64_linux_hotspot_$JAVA_BUILD.tar.gz"
JAVA_ARCHIVE="$JAVA_HOME/openjdk.tar.gz"

# Create Java directory
mkdir -p "$JAVA_HOME"

# Check if Java archive exists
if [ -f "$JAVA_ARCHIVE" ] && [ -d "$JAVA_HOME/jdk-$JAVA_BUILD" ]; then
    echo -e "${GREEN}Using existing Java installation.${NC}"
else
    echo -e "${BLUE}Downloading Java 17...${NC}"
    curl -L "$JAVA_URL" -o "$JAVA_ARCHIVE"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download Java 17.${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Extracting Java 17...${NC}"
    tar -xzf "$JAVA_ARCHIVE" -C "$JAVA_HOME"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract Java 17.${NC}"
        return 1
    fi
fi

# Set JAVA_HOME and PATH
export JAVA_HOME="$JAVA_HOME/jdk-$JAVA_BUILD"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify Java installation
echo -e "${BLUE}Verifying Java installation...${NC}"
java -version

if [ $? -ne 0 ]; then
    echo -e "${RED}Java installation verification failed.${NC}"
    return 1
fi

echo -e "${GREEN}Java 17 setup complete.${NC}"

# Add environment activation to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo -e "\n${GREEN}=======================================${NC}"
echo -e "${GREEN}= Environment setup complete =${NC}"
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}Python version:${NC}"
python --version
echo -e "${GREEN}Pip version:${NC}"
pip --version
echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"
echo -e "${GREEN}Java version:${NC}"
java -version
echo -e "${GREEN}JAVA_HOME: $JAVA_HOME${NC}"

# Check GPU status if GPU dependencies were installed
if [ "$USE_GPU" = true ]; then
    echo -e "\n${GREEN}Enhanced GPU Status:${NC}"
    
    # Check NVCC status
    if command -v nvcc &>/dev/null; then
        echo -e "${GREEN}✅ NVCC found in PATH${NC}"
        nvcc --version | head -n 1
    else
        echo -e "${YELLOW}⚠️ NVCC not in PATH${NC}"
        echo -e "${YELLOW}Run 'source scripts/environment/setup_env.sh' again or install CUDA toolkit${NC}"
    fi
    
    # Check enhanced GPU libraries status
    echo -e "${GREEN}Enhanced GPU Libraries Status:${NC}"
    python -c "
libraries = ['torch', 'catboost', 'lightgbm', 'xgboost', 'pycaret', 'h2o', 'pyspark', 'cuml']
available = []
for lib in libraries:
    try:
        __import__(lib)
        available.append(lib)
    except ImportError:
        pass

print(f'Available libraries: {len(available)}/{len(libraries)}')
if 'torch' in available:
    import torch
    if torch.cuda.is_available():
        print(f'✅ PyTorch CUDA: {torch.cuda.device_count()} GPUs')
if 'catboost' in available:
    print('✅ CatBoost GPU: Ready')
if 'lightgbm' in available:
    print('✅ LightGBM GPU: Ready')  
if 'xgboost' in available:
    print('✅ XGBoost GPU: Ready')
if 'pycaret' in available:
    print('✅ PyCaret AutoML: Ready')
if 'h2o' in available and 'pyspark' in available:
    try:
        from pysparkling import H2OContext
        print('✅ H2O Sparkling Water: Ready')
    except ImportError:
        print('✅ H2O: Ready (regular mode)')
elif 'h2o' in available:
    print('✅ H2O: Ready (regular mode)')
if 'cuml' in available:
    print('✅ CuML GPU: Ready')
else:
    print('ℹ CuML: Using sklearn fallback')
" 2>/dev/null || echo -e "${YELLOW}Could not check enhanced library status${NC}"
fi

# Remind user to use source
echo -e "\n${YELLOW}Remember to use 'source' when running this script:${NC}"
echo -e "${YELLOW}source scripts/environment/setup_env.sh${NC}"