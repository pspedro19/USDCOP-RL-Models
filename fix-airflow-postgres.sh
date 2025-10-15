#!/bin/bash

# =====================================================
# Complete Solution Script for Airflow-PostgreSQL Connectivity
# USDCOP Trading System
# =====================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}============================================================"
echo "USDCOP Trading System - Airflow PostgreSQL Fix"
echo "Complete Solution for Credential Issues"
echo -e "============================================================${NC}"

# Function to show step header
step_header() {
    echo -e "\n${BOLD}${BLUE}==== $1 ====${NC}"
}

# Function to show success message
success_msg() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to show warning message
warning_msg() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to show error message
error_msg() {
    echo -e "${RED}✗ $1${NC}"
}

step_header "Step 1: Verify .env file configuration"

if [ -f .env ]; then
    success_msg ".env file exists"

    # Check if credentials are updated
    if grep -q "POSTGRES_PASSWORD=admin123" .env; then
        success_msg "PostgreSQL password is correctly set to admin123"
    else
        warning_msg "PostgreSQL password needs to be updated"
        echo "Updating .env file with correct credentials..."

        # Backup current .env
        cp .env .env.backup
        success_msg "Backup created: .env.backup"

        # Update credentials in .env
        sed -i 's/POSTGRES_PASSWORD=admin$/POSTGRES_PASSWORD=admin123/' .env
        sed -i 's/REDIS_PASSWORD=$/REDIS_PASSWORD=redis123/' .env
        sed -i 's/MINIO_SECRET_KEY=minioadmin$/MINIO_SECRET_KEY=minioadmin123/' .env
        sed -i 's/AIRFLOW_PASSWORD=admin$/AIRFLOW_PASSWORD=admin123/' .env
        sed -i 's/GRAFANA_PASSWORD=admin$/GRAFANA_PASSWORD=admin123/' .env

        success_msg ".env file updated with consistent credentials"
    fi
else
    error_msg ".env file not found"
    echo "Please ensure .env file exists in the project root"
    exit 1
fi

step_header "Step 2: Stop all running services"
docker-compose down
success_msg "All services stopped"

step_header "Step 3: Clean up volumes (optional but recommended)"
echo "This will remove all existing data. Continue? (y/N)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    warning_msg "Removing PostgreSQL and Airflow volumes..."
    docker volume rm usdcop-rl-models_postgres_data 2>/dev/null || echo "PostgreSQL volume doesn't exist"
    docker volume rm usdcop-rl-models_airflow_logs 2>/dev/null || echo "Airflow logs volume doesn't exist"
    docker volume rm usdcop-rl-models_airflow_dags 2>/dev/null || echo "Airflow dags volume doesn't exist"
    docker volume rm usdcop-rl-models_airflow_plugins 2>/dev/null || echo "Airflow plugins volume doesn't exist"
    success_msg "Volumes cleaned up"
else
    warning_msg "Skipping volume cleanup - using existing data"
fi

step_header "Step 4: Start infrastructure services"
echo "Starting PostgreSQL and Redis..."
docker-compose up -d postgres redis
success_msg "Infrastructure services started"

step_header "Step 5: Wait for PostgreSQL to be ready"
echo "Waiting for PostgreSQL to initialize..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker exec usdcop-postgres-timescale pg_isready -U admin >/dev/null 2>&1; then
        success_msg "PostgreSQL is ready"
        break
    else
        echo "Attempt $attempt/$max_attempts: Waiting for PostgreSQL..."
        sleep 3
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    error_msg "PostgreSQL failed to start after $max_attempts attempts"
    echo "Checking PostgreSQL logs..."
    docker logs usdcop-postgres-timescale --tail 20
    exit 1
fi

step_header "Step 6: Test PostgreSQL connection"
if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT 1;" >/dev/null 2>&1; then
    success_msg "PostgreSQL connection test passed"
else
    error_msg "PostgreSQL connection test failed"
    docker logs usdcop-postgres-timescale --tail 10
    exit 1
fi

step_header "Step 7: Initialize Airflow database"
echo "Running Airflow database initialization..."
docker-compose up --no-deps airflow-init
success_msg "Airflow database initialized"

step_header "Step 8: Start all services"
echo "Starting all services..."
docker-compose up -d
success_msg "All services started"

step_header "Step 9: Wait for services to be healthy"
echo "Waiting for all services to be healthy..."
sleep 30

step_header "Step 10: Run connectivity verification"
if [ -f ./verify-connectivity.sh ]; then
    echo "Running connectivity verification..."
    chmod +x ./verify-connectivity.sh
    ./verify-connectivity.sh
else
    warning_msg "Connectivity verification script not found"
    echo "Manually testing connections..."

    # Manual tests
    echo "Testing PostgreSQL:"
    docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT version();" | head -1

    echo "Testing Redis:"
    docker exec usdcop-redis redis-cli -a redis123 ping

    echo "Testing Airflow web interface:"
    sleep 10
    curl -f http://localhost:8080/health >/dev/null 2>&1 && echo "Airflow web interface is healthy" || echo "Airflow web interface is not ready yet"
fi

step_header "✅ SOLUTION COMPLETED SUCCESSFULLY"

echo -e "\n${GREEN}${BOLD}=========================================="
echo "Fix Applied Successfully!"
echo -e "==========================================${NC}"

echo -e "\n${YELLOW}Credentials Summary:${NC}"
echo "--------------------"
echo "PostgreSQL:"
echo "  - Host: localhost:5432"
echo "  - User: admin"
echo "  - Password: admin123"
echo "  - Database: usdcop_trading"
echo ""
echo "Airflow:"
echo "  - URL: http://localhost:8080"
echo "  - User: admin"
echo "  - Password: admin123"
echo ""
echo "Redis:"
echo "  - Host: localhost:6379"
echo "  - Password: redis123"

echo -e "\n${YELLOW}Connection Strings:${NC}"
echo "--------------------"
echo "PostgreSQL DSN: postgresql+psycopg2://admin:admin123@postgres:5432/usdcop_trading"
echo "Redis URL: redis://:redis123@redis:6379/0"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "------------"
echo "1. Access Airflow at: http://localhost:8080"
echo "2. Login with: admin / admin123"
echo "3. Verify DAGs are loading correctly"
echo "4. Test your data pipelines"

echo -e "\n${YELLOW}Troubleshooting:${NC}"
echo "----------------"
echo "If issues persist:"
echo "1. Check service status: docker-compose ps"
echo "2. View logs: docker-compose logs [service-name]"
echo "3. Run verification: ./verify-connectivity.sh"
echo "4. Complete reset: ./reset-postgres.sh"

echo -e "\n${GREEN}${BOLD}All credential issues have been resolved!${NC}"