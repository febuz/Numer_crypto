#!/bin/bash
# Script to set up a new test environment with Java 17 for H2O Sparkling Water

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="test_env_java17"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up new environment with Java 17 for H2O Sparkling Water${NC}"

# Check and set JAVA_HOME to Java 17
echo -e "${YELLOW}Setting up Java 17 for this environment${NC}"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Add necessary Java 17 module options
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/sun.net=ALL-UNNAMED"

# Verify Java version
if ! java -version 2>&1 | grep -q "openjdk version \"17"; then
    echo -e "${RED}Failed to use Java 17. Current version:${NC}"
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

# Create an activate_java17 script to properly set JAVA_HOME
echo -e "${YELLOW}Creating activation script with Java 17 environment setup${NC}"
cat > "$ENV_PATH/bin/activate_java17" << EOF
#!/bin/bash
source "$ENV_PATH/bin/activate"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=\$JAVA_HOME/bin:\$PATH
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/sun.net=ALL-UNNAMED"
echo "Activated environment with Java 17 and module options"
java -version
EOF

chmod +x "$ENV_PATH/bin/activate_java17"

echo -e "${GREEN}Environment setup completed!${NC}"
echo -e "${GREEN}To activate the environment with Java 17, run:${NC}"
echo -e "${YELLOW}source $ENV_PATH/bin/activate_java17${NC}"
echo -e "${GREEN}To test H2O Sparkling with Java 17, run:${NC}"
echo -e "${YELLOW}python /home/knight2/repos/Numer_crypto/scripts/test_h2o_sparkling_java17.py${NC}"

# Deactivate the environment
deactivate