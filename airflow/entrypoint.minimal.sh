#!/bin/bash

# Minimal Airflow entrypoint for fast startup
# Prioritizes getting Airflow UI running quickly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[AIRFLOW MINIMAL] Starting fast Airflow initialization...${NC}"

# Function to wait for database (simplified)
wait_for_db() {
    echo -e "${YELLOW}[AIRFLOW MINIMAL] Waiting for database connection...${NC}"
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if python -c "
import psycopg2
import os
from urllib.parse import urlparse
try:
    url = os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN']
    if url.startswith('postgresql+psycopg2://'):
        url = url.replace('postgresql+psycopg2://', 'postgresql://')
    parsed = urlparse(url)
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
    exit(1)
" 2>/dev/null; then
            echo -e "${GREEN}[AIRFLOW MINIMAL] Database connection established${NC}"
            return 0
        fi
        echo -e "${YELLOW}[AIRFLOW MINIMAL] Database not ready, attempt $attempt/$max_attempts...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}[AIRFLOW MINIMAL] Database connection failed after $max_attempts attempts${NC}"
    exit 1
}

# Simplified dependency verification
verify_basic_dependencies() {
    echo -e "${YELLOW}[AIRFLOW MINIMAL] Verifying basic dependencies...${NC}"
    
    # Check only critical modules for L0 pipeline
    if python -c "import pandas, numpy, psycopg2, requests" 2>/dev/null; then
        echo -e "${GREEN}✓ Basic dependencies - OK${NC}"
    else
        echo -e "${RED}✗ Basic dependencies - FAILED${NC}"
        return 1
    fi
}

# Fast database initialization
init_airflow_db() {
    echo -e "${YELLOW}[AIRFLOW MINIMAL] Initializing Airflow database...${NC}"
    
    if airflow db check 2>/dev/null; then
        echo -e "${GREEN}[AIRFLOW MINIMAL] Database already initialized${NC}"
    else
        echo -e "${YELLOW}[AIRFLOW MINIMAL] Running database initialization...${NC}"
        airflow db init
        echo -e "${GREEN}[AIRFLOW MINIMAL] Database initialization completed${NC}"
    fi
}

# Create admin user quickly
create_admin_user() {
    echo -e "${YELLOW}[AIRFLOW MINIMAL] Creating admin user...${NC}"
    
    # Always try to create user - ignore if exists
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@usdcop.local \
        --password admin123 2>/dev/null || echo -e "${GREEN}[AIRFLOW MINIMAL] Admin user exists or created${NC}"
}

# Main function
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  USDCOP - Fast Airflow Startup       ${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Verify basic dependencies
    verify_basic_dependencies
    
    # Wait for database
    wait_for_db
    
    case "${1}" in
        webserver)
            init_airflow_db
            create_admin_user
            echo -e "${GREEN}[AIRFLOW MINIMAL] Starting Airflow webserver...${NC}"
            exec airflow webserver
            ;;
        scheduler)
            # Minimal wait for webserver
            sleep 5
            echo -e "${GREEN}[AIRFLOW MINIMAL] Starting Airflow scheduler...${NC}"
            exec airflow scheduler
            ;;
        worker)
            echo -e "${GREEN}[AIRFLOW MINIMAL] Starting Airflow worker...${NC}"
            exec airflow celery worker
            ;;
        init)
            init_airflow_db
            create_admin_user
            echo -e "${GREEN}[AIRFLOW MINIMAL] Initialization completed${NC}"
            ;;
        *)
            echo -e "${GREEN}[AIRFLOW MINIMAL] Running custom command: $@${NC}"
            exec "$@"
            ;;
    esac
}

# Run main function
main "$@"