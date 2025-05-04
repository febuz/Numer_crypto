#!/bin/bash
# Script to set up the Java 11 environment for H2O Sparkling Water
# This script should be sourced: source setup_java11_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set Java 11 as the Java to use
if [ -d "/usr/lib/jvm/java-11-openjdk-amd64" ]; then
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    export PATH=$JAVA_HOME/bin:$PATH
    echo -e "${GREEN}Java 11 environment set up:${NC}"
    java -version
else
    echo -e "${RED}Java 11 installation not found at /usr/lib/jvm/java-11-openjdk-amd64${NC}"
    echo -e "${YELLOW}Available Java installations:${NC}"
    update-alternatives --list java 2>/dev/null || echo "No Java installations found via update-alternatives"
    echo -e "${YELLOW}Looking for Java installations in /usr/lib/jvm:${NC}"
    ls -la /usr/lib/jvm/ 2>/dev/null || echo "No Java installations found in /usr/lib/jvm"
    echo -e "${RED}Please install Java 11 or update this script with the correct Java 11 path${NC}"
    return 1
fi

# Verify Java version
if ! java -version 2>&1 | grep -q "openjdk version \"11"; then
    echo -e "${RED}Failed to set Java 11. Current version:${NC}"
    java -version
    return 1
fi

echo -e "${GREEN}H2O Sparkling Water environment with Java 11 is ready to use.${NC}"
echo -e "${GREEN}You can run: python test_h2o_sparkling_minimal.py${NC}"