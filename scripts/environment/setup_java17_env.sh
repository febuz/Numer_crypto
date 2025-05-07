#!/bin/bash
# Script to set up the Java 17 environment for H2O Sparkling Water
# This script should be sourced: source setup_java17_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set Java 17 as the Java to use
if [ -d "/usr/lib/jvm/java-17-openjdk-amd64" ]; then
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
    export PATH=$JAVA_HOME/bin:$PATH
    echo -e "${GREEN}Java 17 environment set up:${NC}"
    java -version
else
    echo -e "${RED}Java 17 installation not found at /usr/lib/jvm/java-17-openjdk-amd64${NC}"
    echo -e "${YELLOW}Available Java installations:${NC}"
    update-alternatives --list java 2>/dev/null || echo "No Java installations found via update-alternatives"
    echo -e "${YELLOW}Looking for Java installations in /usr/lib/jvm:${NC}"
    ls -la /usr/lib/jvm/ 2>/dev/null || echo "No Java installations found in /usr/lib/jvm"
    echo -e "${RED}Please install Java 17 or update this script with the correct Java 17 path${NC}"
    return 1
fi

# Verify Java version
if ! java -version 2>&1 | grep -q "openjdk version \"17"; then
    echo -e "${RED}Failed to set Java 17. Current version:${NC}"
    java -version
    return 1
fi

# Add Java 17 module options for H2O Sparkling Water
export _JAVA_OPTIONS="--add-opens=java.base/sun.net.www.protocol.http=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.https=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.file=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.ftp=ALL-UNNAMED --add-opens=java.base/sun.net.www.protocol.jar=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/sun.net=ALL-UNNAMED"
echo -e "${GREEN}Added Java 17 module options for H2O Sparkling Water${NC}"

echo -e "${GREEN}H2O Sparkling Water environment with Java 17 is ready to use.${NC}"
echo -e "${GREEN}You can run: python test_h2o_sparkling_java17.py${NC}"