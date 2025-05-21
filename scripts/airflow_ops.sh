#!/bin/bash
# Helper script for Airflow operations for Numerai Crypto pipeline
# Supports modern Airflow 2.8+ commands and services

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Airflow configuration
AIRFLOW_HOME="/media/knight2/EDB/numer_crypto_temp/airflow"
AIRFLOW_VENV_DIR="/media/knight2/EDB/numer_crypto_temp/airflow_env"  # Use dedicated Airflow environment
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRFLOW_DAGS_DIR="$REPO_DIR/../airflow_dags"

# Function to show usage information
print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  init              Initialize Airflow environment"
    echo "  standalone        Run Airflow in standalone mode (all components in one process)"
    echo "  api-server        Start the Airflow webserver (API server)"
    echo "  scheduler         Start the Airflow scheduler"
    echo "  stop              Stop all Airflow services"
    echo "  status            Check Airflow service status"
    echo "  create-user       Create default admin user for Airflow"
    echo "  trigger [CONFIG]  Trigger the Numerai Crypto pipeline DAG with optional config"
    echo "  setup-jwt        Generate and configure JWT secret for Airflow webserver"
    echo "  logs              Show Airflow logs"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 standalone"
    echo "  $0 api-server"
    echo "  $0 trigger '{\"numerai_only\": true, \"skip_historical\": true}'"
}

# Function to initialize Airflow environment
init_airflow() {
    echo -e "${BLUE}Initializing Airflow environment at $AIRFLOW_HOME${NC}"
    
    # Create Airflow directories if they don't exist
    mkdir -p "$AIRFLOW_HOME/dags"
    mkdir -p "$AIRFLOW_HOME/logs"
    mkdir -p "$AIRFLOW_HOME/plugins"
    mkdir -p "$AIRFLOW_HOME/config"
    
    # Copy custom Airflow configuration if it exists
    CUSTOM_AIRFLOW_CFG="$REPO_DIR/../config/airflow/airflow.cfg"
    if [ -f "$CUSTOM_AIRFLOW_CFG" ]; then
        echo -e "${BLUE}Copying custom Airflow configuration...${NC}"
        cp "$CUSTOM_AIRFLOW_CFG" "$AIRFLOW_HOME/airflow.cfg"
        echo -e "${GREEN}Custom Airflow configuration applied${NC}"
    fi
    
    # Check if Python virtual environment for Airflow exists
    if [ ! -d "$AIRFLOW_VENV_DIR" ]; then
        echo -e "${YELLOW}Creating Python virtual environment for Airflow at $AIRFLOW_VENV_DIR${NC}"
        python3 -m venv "$AIRFLOW_VENV_DIR"
    fi
    
    # Activate the virtual environment
    source "$AIRFLOW_VENV_DIR/bin/activate"
    
    # Install required system dependencies first
    echo -e "${BLUE}Installing required dependencies...${NC}"
    pip install --upgrade pip wheel
    
    # Install Airflow with constraints 
    echo -e "${BLUE}Installing Airflow (this may take a while)...${NC}"
    AIRFLOW_VERSION=3.0.1
    PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
    
    # Install Airflow with constraints
    pip install "apache-airflow==${AIRFLOW_VERSION}"
    
    # Install the Slack provider for Airflow
    echo -e "${BLUE}Installing Slack provider for Airflow...${NC}"
    pip install apache-airflow-providers-slack
    
    # Export Airflow home
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Run Airflow database initialization
    echo -e "${BLUE}Initializing Airflow database...${NC}"
    airflow db migrate
    
    # Set up JWT secret for API server
    setup_jwt_secret
    
    # Create the admin user
    echo -e "${YELLOW}Creating default admin user...${NC}"
    # For Airflow 3.x, we need to create a user differently
    # First check if the admin user already exists
    if airflow connections get admin &> /dev/null; then
        echo -e "${YELLOW}Admin user already exists${NC}"
    else
        echo -e "${YELLOW}To create admin user, use the Airflow UI after starting${NC}"
    fi
    
    # Link DAG files from repository to Airflow dags directory
    echo -e "${BLUE}Linking DAG files to Airflow dags directory...${NC}"
    
    # Check if we have DAG files in the repo
    if [ -d "$AIRFLOW_DAGS_DIR" ]; then
        for dag_file in "$AIRFLOW_DAGS_DIR"/*.py; do
            if [ -f "$dag_file" ]; then
                # Get the basename of the file
                dag_name=$(basename "$dag_file")
                
                # Copy the file to the Airflow dags directory
                cp "$dag_file" "$AIRFLOW_HOME/dags/$dag_name"
                echo -e "${GREEN}Copied DAG file: $dag_name${NC}"
            fi
        done
    else
        echo -e "${YELLOW}No DAG directory found at $AIRFLOW_DAGS_DIR${NC}"
        echo -e "${YELLOW}Create $AIRFLOW_DAGS_DIR and place your DAG files there${NC}"
    fi
    
    echo -e "${GREEN}Airflow environment initialized successfully${NC}"
    echo -e "${BLUE}To start Airflow services, run one of:${NC}"
    echo -e "${BLUE}  $0 standalone${NC}"
    echo -e "${BLUE}  $0 api-server${NC}"
    echo -e "${BLUE}  $0 scheduler${NC}"
    
    # Deactivate virtual environment
    deactivate
}

# Function to set up JWT secret for Airflow
setup_jwt_secret() {
    echo -e "${BLUE}Setting up JWT secret for Airflow API server...${NC}"
    
    # Create a new JWT secret if it doesn't exist
    JWT_SECRET=$(openssl rand -hex 32)
    
    # Create or update api_auth section in airflow.cfg
    if [ -f "$AIRFLOW_HOME/airflow.cfg" ]; then
        # Check if api_auth section exists
        if grep -q "\[api_auth\]" "$AIRFLOW_HOME/airflow.cfg"; then
            # Update jwt_secret in existing section
            sed -i "/\[api_auth\]/,/\[.*\]/ s/jwt_secret = .*/jwt_secret = $JWT_SECRET/" "$AIRFLOW_HOME/airflow.cfg"
        else
            # Add api_auth section with jwt_secret at the end of the file
            echo -e "\n[api_auth]\njwt_secret = $JWT_SECRET" >> "$AIRFLOW_HOME/airflow.cfg"
        fi
    fi
    
    # Create or update airflow_local_settings.py
    mkdir -p "$AIRFLOW_HOME/config"
    echo "api_auth = {\"jwt_secret\": \"$JWT_SECRET\"}" > "$AIRFLOW_HOME/config/airflow_local_settings.py"
    
    echo -e "${GREEN}JWT secret set up successfully${NC}"
}

# Function to run Airflow in standalone mode
run_standalone() {
    echo -e "${BLUE}Starting Airflow in standalone mode${NC}"
    
    # Activate the virtual environment
    source "$AIRFLOW_VENV_DIR/bin/activate"
    
    # Export Airflow home
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Run Airflow standalone
    # Note: Unfortunately standalone mode doesn't accept a port parameter
    # but will use the port from airflow.cfg
    nohup airflow standalone > "$AIRFLOW_HOME/logs/standalone.log" 2>&1 &
    STANDALONE_PID=$!
    
    echo -e "${GREEN}Airflow standalone started with PID: $STANDALONE_PID${NC}"
    echo -e "${BLUE}All Airflow components are running in a single process${NC}"
    echo -e "${BLUE}Airflow UI is available at: http://localhost:8989${NC}"
    echo -e "${BLUE}Log file: $AIRFLOW_HOME/logs/standalone.log${NC}"
    
    # Deactivate virtual environment
    deactivate
}

# Function to start the Airflow webserver
run_webserver() {
    echo -e "${BLUE}Starting Airflow API server${NC}"
    
    # Activate the virtual environment
    source "$AIRFLOW_VENV_DIR/bin/activate"
    
    # Export Airflow home
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Start the API server in background (replaces webserver in Airflow 3.x)
    nohup airflow api-server -p 8989 > "$AIRFLOW_HOME/logs/api-server.log" 2>&1 &
    WEBSERVER_PID=$!
    
    echo -e "${GREEN}Airflow API server started with PID: $WEBSERVER_PID${NC}"
    echo -e "${BLUE}Airflow UI is available at: http://localhost:8989${NC}"
    echo -e "${BLUE}Log file: $AIRFLOW_HOME/logs/api-server.log${NC}"
    
    # Deactivate virtual environment
    deactivate
}

# Direct function for api-server (the preferred command in Airflow 3.x)
run_api_server() {
    run_webserver
}

# Function to start the Airflow scheduler
run_scheduler() {
    echo -e "${BLUE}Starting Airflow scheduler${NC}"
    
    # Activate the virtual environment
    source "$AIRFLOW_VENV_DIR/bin/activate"
    
    # Export Airflow home
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Start the scheduler in background
    nohup airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
    SCHEDULER_PID=$!
    
    echo -e "${GREEN}Airflow scheduler started with PID: $SCHEDULER_PID${NC}"
    echo -e "${BLUE}Log file: $AIRFLOW_HOME/logs/scheduler.log${NC}"
    
    # Deactivate virtual environment
    deactivate
}

# Function to trigger a DAG run
trigger_dag() {
    local dag_id="numerai_crypto_pipeline_v3"
    local conf="$1"
    
    # Check if numerai_crypto_pipeline_v3 exists, otherwise use numerai_crypto_pipeline_v2
    if [ ! -f "$AIRFLOW_HOME/dags/numerai_crypto_pipeline_v3.py" ]; then
        dag_id="numerai_crypto_pipeline"
    fi
    
    echo -e "${BLUE}Triggering DAG: $dag_id${NC}"
    
    # Activate the virtual environment
    source "$AIRFLOW_VENV_DIR/bin/activate"
    
    # Export Airflow home
    export AIRFLOW_HOME="$AIRFLOW_HOME"
    
    # Trigger the DAG with or without configuration
    if [ -n "$conf" ]; then
        airflow dags trigger -c "$conf" "$dag_id"
    else
        airflow dags trigger "$dag_id"
    fi
    
    echo -e "${GREEN}DAG triggered: $dag_id${NC}"
    echo -e "${BLUE}You can monitor the DAG run at: http://localhost:8989/dags/$dag_id/grid${NC}"
    
    # Deactivate virtual environment
    deactivate
}

# Function to stop all Airflow services
stop_airflow() {
    echo -e "${BLUE}Stopping all Airflow services${NC}"
    
    # Find and kill all Airflow processes
    pkill -f "airflow webserver" || true
    pkill -f "airflow scheduler" || true
    pkill -f "airflow triggerer" || true
    pkill -f "airflow standalone" || true
    pkill -f "airflow dag-processor" || true
    
    echo -e "${GREEN}All Airflow services stopped${NC}"
}

# Function to show Airflow service status
show_status() {
    echo -e "${BLUE}Checking Airflow service status${NC}"
    
    # Check processes
    echo -e "${BLUE}Airflow processes:${NC}"
    ps -ef | grep -E "airflow (webserver|scheduler|standalone|triggerer|dag-processor)" | grep -v grep
    
    # Count processes
    PROCESSES=$(ps -ef | grep -E "airflow (webserver|scheduler|standalone|triggerer|dag-processor)" | grep -v grep | wc -l)
    
    if [ "$PROCESSES" -eq 0 ]; then
        echo -e "${YELLOW}No Airflow processes found${NC}"
    else
        echo -e "${GREEN}$PROCESSES Airflow processes running${NC}"
        
        # Show UI URL
        echo -e "${BLUE}Airflow UI should be available at: http://localhost:8989${NC}"
    fi
}

# Function to show Airflow logs
show_logs() {
    echo -e "${BLUE}Recent Airflow logs:${NC}"
    
    # Show recent logs from all log files
    for log_file in "$AIRFLOW_HOME/logs"/*.log; do
        if [ -f "$log_file" ]; then
            echo -e "${GREEN}=== ${log_file} (last 20 lines) ===${NC}"
            tail -n 20 "$log_file"
            echo
        fi
    done
}

# Main script logic
case "$1" in
    init)
        init_airflow
        ;;
    standalone)
        run_standalone
        ;;
    api-server)
        run_api_server
        ;;
    webserver)
        run_webserver
        ;;
    scheduler)
        run_scheduler
        ;;
    stop)
        stop_airflow
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    setup-jwt)
        echo -e "${YELLOW}JWT setup is handled automatically by Airflow 3.0.${NC}"
        echo -e "${YELLOW}No manual setup required.${NC}"
        ;;
    create-user)
        echo -e "${BLUE}Creating default admin user...${NC}"
        source "$AIRFLOW_VENV_DIR/bin/activate"
        export AIRFLOW_HOME="$AIRFLOW_HOME"
        
        # In Airflow 3.0, there's no direct CLI command to create users
        # Instead, we create a small Python script to create a user and execute it
        TEMP_SCRIPT=$(mktemp)
        
        cat > "$TEMP_SCRIPT" << 'EOF'
from airflow.auth.managers.base_auth_manager import SessionSubclass
from airflow.auth.managers.models.resource_details import UserDetails
from airflow.models.resource import ResourceManagedBy
from airflow.providers.auth_manager.standard.auth_manager.auth_manager import StandardAuthManager

auth_manager = StandardAuthManager()

# Create admin user
user_details = UserDetails(
    username="admin",
    password=None,  # Password will be automatically generated
    email="admin@example.com",
    first_name="Admin",
    last_name="User",
    display_name="Admin User",
    managed_by=ResourceManagedBy.AIRFLOW,
    session_type=SessionSubclass.AIRFLOW
)

# Check if user exists
try:
    existing_user = auth_manager.get_user_details_by_username("admin")
    print("User 'admin' already exists, updating password...")
    # Reset password - it will be automatically generated and stored in the JSON file
    import secrets
    import string
    secure_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
    auth_manager.update_user_password(existing_user.username, secure_password)
except Exception:
    # Create user if it doesn't exist
    print("Creating new admin user...")
    auth_manager.create_user_details(user_details)

print("Admin user setup complete. Username: admin")
print("Password is stored in the simple_auth_manager_passwords.json.generated file")
EOF

        # Run the script
        python "$TEMP_SCRIPT"
        rm "$TEMP_SCRIPT"
        
        echo -e "${GREEN}User created or updated successfully${NC}"
        echo -e "${GREEN}Username: admin${NC}"
        echo -e "${GREEN}Password is stored in $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated${NC}"
        
        deactivate
        ;;
    trigger)
        trigger_dag "$2"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac

exit 0