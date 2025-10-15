#!/bin/bash

# Airflow entrypoint script for USDCOP RL Trading System
# Ensures proper initialization and dependency verification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[AIRFLOW INIT] Starting Airflow initialization...${NC}"

# Function to wait for database
wait_for_db() {
    echo -e "${YELLOW}[AIRFLOW INIT] Waiting for database connection...${NC}"
    while ! python -c "
import psycopg2
import os
import time
from urllib.parse import urlparse
try:
    # Parse the connection URL properly
    url = os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN']
    # Remove the postgresql+psycopg2:// prefix for psycopg2
    if url.startswith('postgresql+psycopg2://'):
        url = url.replace('postgresql+psycopg2://', 'postgresql://')

    # Parse the URL
    parsed = urlparse(url)

    # Connect using individual parameters
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password
    )
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
"; do
        echo -e "${YELLOW}[AIRFLOW INIT] Database not ready, waiting 5 seconds...${NC}"
        sleep 5
    done
    echo -e "${GREEN}[AIRFLOW INIT] Database connection established${NC}"
}

# Function to verify Python dependencies
verify_dependencies() {
    echo -e "${YELLOW}[AIRFLOW INIT] Verifying Python dependencies...${NC}"
    
    # Critical modules that must be available for DAGs
    REQUIRED_MODULES=(
        "scipy"
        "gymnasium"
    )
    
    for module in "${REQUIRED_MODULES[@]}"; do
        if python -c "import ${module}" 2>/dev/null; then
            echo -e "${GREEN}✓ ${module} - OK${NC}"
        else
            echo -e "${RED}✗ ${module} - FAILED${NC}"
            echo -e "${RED}[ERROR] Missing required module: ${module}${NC}"
            echo -e "${YELLOW}[DEBUG] Attempting to install ${module}...${NC}"
            pip install ${module} 2>/dev/null || true
            if python -c "import ${module}" 2>/dev/null; then
                echo -e "${GREEN}✓ ${module} - OK after installation${NC}"
            else
                echo -e "${RED}✗ ${module} - Still failed, but continuing...${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}[AIRFLOW INIT] Dependencies verification completed${NC}"
}

# Function to initialize Airflow database
init_airflow_db() {
    echo -e "${YELLOW}[AIRFLOW INIT] Initializing Airflow database...${NC}"
    
    # Check if database is already initialized
    if airflow db check 2>/dev/null; then
        echo -e "${GREEN}[AIRFLOW INIT] Database already initialized${NC}"
    else
        echo -e "${YELLOW}[AIRFLOW INIT] Running database initialization...${NC}"
        airflow db init
        echo -e "${GREEN}[AIRFLOW INIT] Database initialization completed${NC}"
    fi
}

# Function to create admin user if it doesn't exist
create_admin_user() {
    echo -e "${YELLOW}[AIRFLOW INIT] Checking for admin user...${NC}"
    
    if airflow users list | grep -q "admin"; then
        echo -e "${GREEN}[AIRFLOW INIT] Admin user already exists${NC}"
    else
        echo -e "${YELLOW}[AIRFLOW INIT] Creating admin user...${NC}"
        airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@usdcop.local \
            --password admin
        echo -e "${GREEN}[AIRFLOW INIT] Admin user created successfully${NC}"
    fi
}

# Function to upgrade database schema
upgrade_db() {
    echo -e "${YELLOW}[AIRFLOW INIT] Upgrading database schema...${NC}"
    airflow db upgrade
    echo -e "${GREEN}[AIRFLOW INIT] Database schema upgraded${NC}"
}

# Main initialization function
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  USDCOP RL Trading System - Airflow   ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Verify dependencies first
    verify_dependencies
    
    # Wait for database to be ready
    wait_for_db
    
    # Initialize based on command
    case "${1}" in
        webserver)
            init_airflow_db
            upgrade_db
            create_admin_user
            echo -e "${GREEN}[AIRFLOW INIT] Starting Airflow webserver...${NC}"
            exec airflow webserver
            ;;
        scheduler)
            # Wait a bit for webserver to initialize DB
            sleep 10
            echo -e "${GREEN}[AIRFLOW INIT] Starting Airflow scheduler...${NC}"
            exec airflow scheduler
            ;;
        worker)
            echo -e "${GREEN}[AIRFLOW INIT] Starting Airflow worker...${NC}"
            exec airflow celery worker
            ;;
        init)
            init_airflow_db
            upgrade_db
            create_admin_user
            echo -e "${GREEN}[AIRFLOW INIT] Initialization completed${NC}"
            ;;
        *)
            echo -e "${GREEN}[AIRFLOW INIT] Running custom command: $@${NC}"
            exec "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"