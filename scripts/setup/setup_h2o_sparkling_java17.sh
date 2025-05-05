#!/bin/bash
# Setup script for H2O Sparkling Water with Java 17
# This script creates a dedicated environment with:
# - Java 17 with proper module permissions for H2O
# - PySpark 3.5.0
# - H2O 3.46.0.6
# - pysparkling 
# - XGBoost 3.0.0 with GPU support
# - Required dependencies

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="h2o_sparkling_java17_env"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

echo -e "${YELLOW}========================================================${NC}"
echo -e "${YELLOW}   Setting up H2O Sparkling Water with Java 17          ${NC}"
echo -e "${YELLOW}========================================================${NC}"

# Check for Java 17
JAVA17_PATH="/usr/lib/jvm/java-17-openjdk-amd64"
if [ ! -d "$JAVA17_PATH" ]; then
    echo -e "${YELLOW}Java 17 not found. Installing...${NC}"
    sudo apt update
    sudo apt install -y openjdk-17-jdk
fi

# Verify Java 17 installation
if [ -d "$JAVA17_PATH" ]; then
    echo -e "${GREEN}Java 17 found at $JAVA17_PATH${NC}"
else
    echo -e "${RED}Failed to install Java 17. Please install manually.${NC}"
    exit 1
fi

# Set Java 17 environment
export JAVA_HOME=$JAVA17_PATH
export PATH=$JAVA_HOME/bin:$PATH

# Set Java 17 module options for H2O Sparkling
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED \
--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED \
--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED \
--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED \
--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED \
--add-opens=java.base/java.net=ALL-UNNAMED \
--add-opens=java.base/sun.net=ALL-UNNAMED"

# Verify Java version
java -version

# Remove existing environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}Removing existing environment: $ENV_PATH${NC}"
    rm -rf "$ENV_PATH"
fi

# Create new Python virtual environment
echo -e "${YELLOW}Creating new Python virtual environment at: $ENV_PATH${NC}"
python3 -m venv "$ENV_PATH"

# Activate the environment
echo -e "${YELLOW}Activating virtual environment${NC}"
source "$ENV_PATH/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies${NC}"
pip install wheel setuptools

# Install specific H2O version (3.46.0.6)
echo -e "${YELLOW}Installing H2O 3.46.0.6 (required for Sparkling Water compatibility)${NC}"
pip install h2o==3.46.0.6

# Install PySpark and dependencies
echo -e "${YELLOW}Installing PySpark 3.5.0 and dependencies${NC}"
pip install pyspark==3.5.0 py4j==0.10.9.7

# Install additional required packages
echo -e "${YELLOW}Installing additional packages${NC}"
pip install numpy pandas scikit-learn matplotlib tabulate requests scipy

# Install GPU libraries
echo -e "${YELLOW}Installing GPU libraries (XGBoost, LightGBM)${NC}"
pip install xgboost==3.0.0 lightgbm==4.6.0 nvidia-ml-py py3nvml GPUtil

# Download Sparkling Water and set it up (since proper pysparkling might not be available via pip)
H2O_SPARKLING_VERSION="3.46.0.6-1-3.5"
SPARK_VERSION="3.5"
SCALA_VERSION="2.12"

TEMP_DIR="$SCRIPT_DIR/temp_download"
mkdir -p "$TEMP_DIR"

echo -e "${YELLOW}Downloading and setting up H2O Sparkling Water ${H2O_SPARKLING_VERSION}${NC}"
wget -P "$TEMP_DIR" "https://h2oai.jfrog.io/artifactory/h2o-release/ai/h2o/sparkling-water/sparkling-water_${SCALA_VERSION}/${H2O_SPARKLING_VERSION}/sparkling-water_${SCALA_VERSION}-${H2O_SPARKLING_VERSION}-dist.zip"

# Unzip Sparkling Water
echo -e "${YELLOW}Extracting Sparkling Water${NC}"
mkdir -p "$SCRIPT_DIR/sparkling-water"
unzip -q "$TEMP_DIR/sparkling-water_${SCALA_VERSION}-${H2O_SPARKLING_VERSION}-dist.zip" -d "$SCRIPT_DIR/sparkling-water"
SPARKLING_WATER_DIR="$SCRIPT_DIR/sparkling-water/sparkling-water_${SCALA_VERSION}-${H2O_SPARKLING_VERSION}-dist"

# Install PySparkling
echo -e "${YELLOW}Installing PySparkling${NC}"
pip install "$SPARKLING_WATER_DIR/py/dist/pysparkling-${H2O_SPARKLING_VERSION/%-1-3.5/}.zip"

# Clean up
rm -rf "$TEMP_DIR"

# Create activation script
echo -e "${YELLOW}Creating activation script${NC}"
cat > "$ENV_PATH/bin/activate.sparkling" << EOF
#!/bin/bash
# Activate environment with Java 17 for H2O Sparkling Water

# Activate Python environment
source "$ENV_PATH/bin/activate"

# Set Java 17 environment
export JAVA_HOME=$JAVA17_PATH
export PATH=\$JAVA_HOME/bin:\$PATH

# Set Java 17 module options required for H2O Sparkling
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED \\
--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED \\
--add-opens=java.base/java.net=ALL-UNNAMED \\
--add-opens=java.base/sun.net=ALL-UNNAMED"

# Set Spark environment variables
export SPARK_HOME=\$VIRTUAL_ENV/lib/python3.*/site-packages/pyspark
export PYTHONPATH=\$SPARK_HOME/python:\$SPARK_HOME/python/lib/py4j-*.zip:\$PYTHONPATH
export SPARKLING_WATER_HOME="$SPARKLING_WATER_DIR"
export PYSPARKLING_HOME="\$VIRTUAL_ENV/lib/python3.*/site-packages/pysparkling"

echo -e "${GREEN}H2O Sparkling Water environment with Java 17 activated${NC}"
echo -e "${GREEN}Java version:${NC}"
java -version
echo ""
echo -e "${GREEN}Verifying PySparkling:${NC}"
python -c "import pysparkling; print(f'PySparkling is available, version: {getattr(pysparkling, \"__version__\", \"Unknown\")}')"
EOF

chmod +x "$ENV_PATH/bin/activate.sparkling"

# Create a test script
echo -e "${YELLOW}Creating test script${NC}"
cat > "$SCRIPT_DIR/test_h2o_sparkling_java17_GPU.py" << EOF
#!/usr/bin/env python3
"""
H2O Sparkling Water with Java 17 and GPU Test
This script tests H2O Sparkling Water on all available GPUs using Java 17.
"""

import os
import sys
import time
import json
import threading
import subprocess
import numpy as np
import pandas as pd
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Set environment variable to ensure Java 17 module options are applied
os.environ["_JAVA_OPTIONS"] = (
    "--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED "
    "--add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED "
    "--add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED "
    "--add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED "
    "--add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/sun.net=ALL-UNNAMED"
)

# GPU monitoring utilities
def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in output.strip().split('\\n'):
            values = line.split(', ')
            if len(values) == 4:
                gpu_info.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'utilization_pct': float(values[2]),
                    'memory_used_mb': float(values[3])
                })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def monitor_gpus(stop_event, results_dict, interval=0.1):
    """Monitor all GPUs in a separate thread"""
    while not stop_event.is_set():
        gpu_info = get_gpu_info()
        timestamp = time.time()
        
        for gpu in gpu_info:
            gpu_idx = gpu['index']
            # Initialize list for this GPU if it doesn't exist
            if gpu_idx not in results_dict:
                results_dict[gpu_idx] = []
                
            results_dict[gpu_idx].append({
                'timestamp': timestamp,
                'gpu_index': gpu_idx,
                'name': gpu['name'],
                'utilization': gpu['utilization_pct'],
                'memory_used': gpu['memory_used_mb']
            })
        time.sleep(interval)

def create_synthetic_data(n_samples=10000, n_features=10):
    """Create synthetic dataset for testing"""
    print(f"Creating synthetic dataset with {n_samples} samples and {n_features} features")
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (np.random.randn(n_samples) > 0).astype(int)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df

def main():
    """Main function"""
    print("=" * 80)
    print("H2O SPARKLING WATER WITH JAVA 17 AND MULTI-GPU TEST")
    print("=" * 80)
    
    # Print Java version to confirm Java 17
    try:
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
        print(f"Java version:\\n{java_version}")
        
        if "17" not in java_version:
            print("Warning: Not using Java 17. Please run with Java 17.")
    except Exception as e:
        print(f"Error checking Java version: {e}")
    
    # Get available GPUs
    gpus = get_gpu_info()
    if not gpus:
        print("No GPUs detected. Exiting.")
        return 1
    
    print(f"Detected {len(gpus)} GPUs:")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Current utilization: {gpu['utilization_pct']}%")
        print(f"    Current memory usage: {gpu['memory_used_mb']} MB")
    
    # Create synthetic dataset
    df = create_synthetic_data(50000, 20)
    
    # Start GPU monitoring
    gpu_metrics = {}
    monitor_stop_event = Event()
    monitor_thread = threading.Thread(
        target=monitor_gpus,
        args=(monitor_stop_event, gpu_metrics)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Import and initialize Spark and H2O
    print("\\nInitializing Spark and H2O with Java 17...")
    try:
        from pyspark.sql import SparkSession
        import pysparkling
        from pyspark.sql.types import DoubleType, StructType, StructField
        from pyspark.ml.feature import VectorAssembler
        
        # Create Spark session with Java 17 config
        spark = SparkSession.builder \\
            .appName("H2OSparkling_Java17_GPUTest") \\
            .config("spark.executor.memory", "4g") \\
            .config("spark.driver.memory", "4g") \\
            .getOrCreate()
        
        # Initialize H2O
        from pysparkling import H2OContext
        h2o_context = H2OContext.getOrCreate()
        
        # Import H2O XGBoost
        from pysparkling.ml import H2OXGBoostEstimator
        import h2o
        print(f"H2O version: {h2o.__version__}")
        
        # Convert to Spark DataFrame
        print("Converting data to Spark DataFrame...")
        spark_df = spark.createDataFrame(df)
        
        # Prepare features vector
        feature_cols = [col for col in df.columns if col != 'target']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        spark_df = assembler.transform(spark_df)
        
        # Convert to H2O Frame
        print("Converting to H2O Frame...")
        h2o_frame = h2o_context.asH2OFrame(spark_df)
        h2o_frame['target'] = h2o_frame['target'].asfactor()
        
        # Train H2O XGBoost on each GPU in sequence to maximize utilization
        results = []
        
        for gpu in gpus:
            gpu_id = gpu['index']
            print(f"\\nTraining H2O XGBoost model on GPU {gpu_id}...")
            
            # Specify this GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Train model with GPU
            start_time = time.time()
            estimator = H2OXGBoostEstimator(
                featuresCols=["features"],
                labelCol="target",
                tree_method="gpu_hist",  # Use GPU
                gpu_id=0,  # Always use 0 since we're setting CUDA_VISIBLE_DEVICES
                ntrees=100,
                max_depth=10,
                learn_rate=0.1
            )
            
            try:
                model = estimator.fit(spark_df)
                end_time = time.time()
                
                results.append({
                    'gpu_id': gpu_id,
                    'success': True,
                    'training_time': end_time - start_time
                })
                
                print(f"Training on GPU {gpu_id} completed in {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error training on GPU {gpu_id}: {e}")
                results.append({
                    'gpu_id': gpu_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Stop monitoring
        monitor_stop_event.set()
        monitor_thread.join()
        
        # Print GPU utilization summary
        print("\\n" + "="*80)
        print("GPU UTILIZATION SUMMARY")
        print("="*80)
        
        for gpu_id, metrics in gpu_metrics.items():
            if not metrics:
                continue
                
            df_metrics = pd.DataFrame(metrics)
            peak_util = df_metrics['utilization'].max()
            avg_util = df_metrics['utilization'].mean()
            peak_memory = df_metrics['memory_used'].max()
            
            print(f"GPU {gpu_id}:")
            print(f"  Peak utilization: {peak_util:.2f}%")
            print(f"  Average utilization: {avg_util:.2f}%")
            print(f"  Peak memory usage: {peak_memory:.2f} MB")
        
        # Clean up
        h2o_context.stop()
        spark.stop()
        
        print("\\nTest completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Stop monitoring
        monitor_stop_event.set()
        if monitor_thread.is_alive():
            monitor_thread.join()
            
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x "$SCRIPT_DIR/test_h2o_sparkling_java17_GPU.py"

# Create a run script
echo -e "${YELLOW}Creating run script${NC}"
cat > "$SCRIPT_DIR/run_h2o_sparkling_java17_GPU.sh" << EOF
#!/bin/bash
# Script to run H2O Sparkling Water with Java 17 and GPU test

# Set up environment
source "$ENV_PATH/bin/activate.sparkling"

# Run test
python "$SCRIPT_DIR/test_h2o_sparkling_java17_GPU.py"

# Print completion message
if [ \$? -eq 0 ]; then
    echo -e "${GREEN}Test completed successfully!${NC}"
else
    echo -e "${RED}Test failed.${NC}"
    exit 1
fi
EOF

chmod +x "$SCRIPT_DIR/run_h2o_sparkling_java17_GPU.sh"

# Completion message
echo -e "\n${GREEN}H2O Sparkling Water with Java 17 environment setup completed!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "${YELLOW}source $ENV_PATH/bin/activate.sparkling${NC}"
echo -e "\n${GREEN}To run the multi-GPU test, run:${NC}"
echo -e "${YELLOW}$SCRIPT_DIR/run_h2o_sparkling_java17_GPU.sh${NC}"