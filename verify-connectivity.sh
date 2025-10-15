#!/bin/bash

# =====================================================
# USDCOP Trading System - Connectivity Verification Script
# =====================================================

set -e

echo "=========================================="
echo "USDCOP Trading System - Connectivity Test"
echo "=========================================="

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded from .env"
else
    echo "✗ .env file not found"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test database connectivity
test_postgres() {
    echo -e "\n${YELLOW}Testing PostgreSQL connectivity...${NC}"

    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."

    # Test with docker exec if container is running
    if docker ps | grep -q "usdcop-postgres"; then
        echo "PostgreSQL container is running"

        # Test connection with psql
        if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT version();" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"

            # Test schemas
            echo "Checking schemas..."
            docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast', 'pg_temp_1', 'pg_toast_temp_1') ORDER BY schema_name;"

            return 0
        else
            echo -e "${RED}✗ PostgreSQL connection failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ PostgreSQL container is not running${NC}"
        return 1
    fi
}

# Function to test Redis connectivity
test_redis() {
    echo -e "\n${YELLOW}Testing Redis connectivity...${NC}"

    if docker ps | grep -q "usdcop-redis"; then
        echo "Redis container is running"

        if docker exec usdcop-redis redis-cli -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Redis connection successful${NC}"
            return 0
        else
            echo -e "${RED}✗ Redis connection failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Redis container is not running${NC}"
        return 1
    fi
}

# Function to test Airflow database connection
test_airflow_db() {
    echo -e "\n${YELLOW}Testing Airflow database connectivity...${NC}"

    # Construct the connection string
    AIRFLOW_DB_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres/${POSTGRES_DB}"
    echo "Connection string (masked): postgresql+psycopg2://${POSTGRES_USER}:****@postgres/${POSTGRES_DB}"

    # Test if airflow-webserver container exists and can connect
    if docker ps | grep -q "usdcop-airflow"; then
        echo "Airflow containers are running"

        # Check if we can access airflow CLI
        if docker exec usdcop-airflow-webserver airflow db check > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Airflow database connection successful${NC}"
            return 0
        else
            echo -e "${RED}✗ Airflow database connection failed${NC}"
            echo "Trying to check database directly..."
            docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT 1;" || true
            return 1
        fi
    else
        echo -e "${YELLOW}! Airflow containers are not running${NC}"
        return 1
    fi
}

# Function to show service status
show_service_status() {
    echo -e "\n${YELLOW}Service Status:${NC}"
    echo "===================="

    services=("usdcop-postgres-timescale" "usdcop-redis" "usdcop-airflow-webserver" "usdcop-airflow-scheduler" "usdcop-airflow-worker")

    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$service"; then
            status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$service" | awk '{print $2" "$3" "$4}')
            echo -e "${GREEN}✓${NC} $service: $status"
        else
            echo -e "${RED}✗${NC} $service: Not running"
        fi
    done
}

# Function to show connection details
show_connection_details() {
    echo -e "\n${YELLOW}Connection Details:${NC}"
    echo "===================="
    echo "PostgreSQL Host: localhost:5432"
    echo "PostgreSQL User: $POSTGRES_USER"
    echo "PostgreSQL Database: $POSTGRES_DB"
    echo "Redis Host: localhost:6379"
    echo "Airflow Web: http://localhost:8080"
    echo "Airflow User: $AIRFLOW_USER"
    echo ""
    echo "DSN: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
}

# Main execution
main() {
    show_service_status

    success_count=0
    total_tests=3

    if test_postgres; then
        ((success_count++))
    fi

    if test_redis; then
        ((success_count++))
    fi

    if test_airflow_db; then
        ((success_count++))
    fi

    show_connection_details

    echo -e "\n${YELLOW}Summary:${NC}"
    echo "========"
    echo "Tests passed: $success_count/$total_tests"

    if [ $success_count -eq $total_tests ]; then
        echo -e "${GREEN}All connectivity tests passed! ✓${NC}"
        exit 0
    else
        echo -e "${RED}Some connectivity tests failed! ✗${NC}"
        echo -e "\n${YELLOW}Troubleshooting tips:${NC}"
        echo "1. Make sure all containers are running: docker-compose ps"
        echo "2. Check logs: docker-compose logs postgres"
        echo "3. Verify environment variables in .env file"
        echo "4. Try restarting services: docker-compose restart"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "postgres"|"pg")
        test_postgres
        ;;
    "redis")
        test_redis
        ;;
    "airflow")
        test_airflow_db
        ;;
    "status")
        show_service_status
        ;;
    "details")
        show_connection_details
        ;;
    *)
        main
        ;;
esac