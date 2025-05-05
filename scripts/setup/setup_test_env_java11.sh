#!/bin/bash
# Script to set up a new test environment with Java 11 for H2O Sparkling Water

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="test_env_java11"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up new environment with Java 11 for H2O Sparkling Water${NC}"

# Check and set JAVA_HOME to Java 11
echo -e "${YELLOW}Setting up Java 11 for this environment${NC}"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java version
if ! java -version 2>&1 | grep -q "openjdk version \"11"; then
    echo -e "${RED}Failed to use Java 11. Current version:${NC}"
    java -version
    echo -e "${YELLOW}Will export JAVA_HOME in the activation script${NC}"
fi

# Verify Java version
JAVA_VERSION=$(java -version 2>&1 | head -1)
echo -e "${GREEN}Current Java version: $JAVA_VERSION${NC}"

# Remove existing environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo -e "${YELLOW}Removing existing environment: $ENV_PATH${NC}"
    rm -rf "$ENV_PATH"
fi

# Create new virtual environment
echo -e "${YELLOW}Creating new Python virtual environment at: $ENV_PATH${NC}"
python3 -m venv "$ENV_PATH"

# Activate the environment
echo -e "${YELLOW}Activating virtual environment${NC}"
source "$ENV_PATH/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip${NC}"
pip install --upgrade pip

# Install setuptools
echo -e "${YELLOW}Installing setuptools${NC}"
pip install setuptools

# Install required packages
echo -e "${YELLOW}Installing required packages${NC}"
pip install numpy pandas scikit-learn==1.3.2 h2o==3.46.0.7 pyspark==3.5.0

# Install H2O Sparkling Water
echo -e "${YELLOW}Installing H2O Sparkling Water${NC}"
pip install h2o-pysparkling-3.5==3.46.0.6.post1

# Create a test file to verify the installation
echo -e "${YELLOW}Creating test script${NC}"
cat > "$SCRIPT_DIR/test_java11_h2o_spark.py" << EOF
#!/usr/bin/env python3
"""
Test script to verify Java 11 with H2O Sparkling Water
"""
import sys
import os
import subprocess

# Print Java version
java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode()
print(f"Java version:")
print(java_version)

# Print Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Import required packages
try:
    import h2o
    print(f"H2O version: {h2o.__version__}")
except ImportError:
    print("H2O not available")

try:
    import pyspark
    print(f"PySpark version: {pyspark.__version__}")
except ImportError:
    print("PySpark not available")

try:
    from pysparkling import H2OContext
    import pysparkling
    print(f"PySparkling imported successfully")
    print(f"PySparkling path: {pysparkling.__file__}")
except ImportError as e:
    print(f"PySparkling import error: {e}")

# Create a minimal Spark session and H2O context
try:
    from pyspark.sql import SparkSession
    
    # Create a simple Spark session
    spark = SparkSession.builder \
        .appName("H2OSparklingTest") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    print("Spark session created successfully")
    
    # Create a simple DataFrame
    data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    print("Sample DataFrame:")
    df.show()
    
    # Initialize H2O Sparkling
    try:
        hc = H2OContext.getOrCreate()
        print("H2O Context created successfully!")
        
        # Convert Spark DataFrame to H2O Frame
        h2o_df = hc.asH2OFrame(df)
        print("Converted to H2O Frame:")
        print(h2o_df)
        
        # Shutdown
        h2o.cluster().shutdown()
        spark.stop()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error with H2O Context: {e}")
        spark.stop()
except Exception as e:
    print(f"Error with Spark: {e}")
EOF

chmod +x "$SCRIPT_DIR/test_java11_h2o_spark.py"

echo -e "${GREEN}Environment setup completed!${NC}"
echo -e "${GREEN}To activate the environment, run:${NC}"
echo -e "${YELLOW}source $ENV_PATH/bin/activate${NC}"
echo -e "${GREEN}To test H2O Sparkling with Java 11, run:${NC}"
echo -e "${YELLOW}python $SCRIPT_DIR/test_java11_h2o_spark.py${NC}"

# Create an activate_java11 script to properly set JAVA_HOME
echo -e "${YELLOW}Creating activation script with Java 11 environment setup${NC}"
cat > "$ENV_PATH/bin/activate_java11" << EOF
#!/bin/bash
source "$ENV_PATH/bin/activate"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=\$JAVA_HOME/bin:\$PATH
echo "Activated environment with Java 11"
java -version
EOF

chmod +x "$ENV_PATH/bin/activate_java11"

echo -e "${GREEN}Environment setup completed!${NC}"
echo -e "${GREEN}To activate the environment with Java 11, run:${NC}"
echo -e "${YELLOW}source $ENV_PATH/bin/activate_java11${NC}"
echo -e "${GREEN}To test H2O Sparkling with Java 11, run:${NC}"
echo -e "${YELLOW}python $SCRIPT_DIR/test_java11_h2o_spark.py${NC}"

# Deactivate the environment
deactivate