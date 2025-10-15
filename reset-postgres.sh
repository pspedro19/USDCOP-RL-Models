#!/bin/bash

# =====================================================
# PostgreSQL Reset Script for USDCOP Trading System
# This script completely resets PostgreSQL with known credentials
# =====================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "PostgreSQL Reset Script"
echo "USDCOP Trading System"
echo -e "==========================================${NC}"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓ Environment variables loaded from .env${NC}"
else
    echo -e "${RED}✗ .env file not found${NC}"
    exit 1
fi

# Confirmation prompt
echo -e "\n${YELLOW}WARNING: This will completely reset PostgreSQL database and all data will be lost!${NC}"
echo -e "${YELLOW}Current configuration:${NC}"
echo "  - PostgreSQL User: $POSTGRES_USER"
echo "  - PostgreSQL Database: $POSTGRES_DB"
echo "  - PostgreSQL Password: [HIDDEN]"
echo ""
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

echo -e "\n${BLUE}Step 1: Stopping all services...${NC}"
docker-compose down

echo -e "\n${BLUE}Step 2: Removing PostgreSQL data volume...${NC}"
docker volume rm usdcop-rl-models_postgres_data 2>/dev/null || echo "Volume does not exist or already removed"

echo -e "\n${BLUE}Step 3: Removing Airflow data volumes...${NC}"
docker volume rm usdcop-rl-models_airflow_logs 2>/dev/null || echo "Airflow logs volume does not exist"
docker volume rm usdcop-rl-models_airflow_dags 2>/dev/null || echo "Airflow dags volume does not exist"
docker volume rm usdcop-rl-models_airflow_plugins 2>/dev/null || echo "Airflow plugins volume does not exist"

echo -e "\n${BLUE}Step 4: Starting PostgreSQL service only...${NC}"
docker-compose up -d postgres

echo -e "\n${BLUE}Step 5: Waiting for PostgreSQL to be ready...${NC}"
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker exec usdcop-postgres-timescale pg_isready -U admin > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
        break
    else
        echo "Attempt $attempt/$max_attempts: PostgreSQL not ready yet..."
        sleep 2
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}✗ PostgreSQL failed to start after $max_attempts attempts${NC}"
    exit 1
fi

# Test the connection
echo -e "\n${BLUE}Step 6: Testing PostgreSQL connection...${NC}"
if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT version();" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"
else
    echo -e "${RED}✗ PostgreSQL connection failed${NC}"
    echo "Checking container logs..."
    docker logs usdcop-postgres-timescale --tail 20
    exit 1
fi

echo -e "\n${BLUE}Step 7: Verifying database structure...${NC}"
echo "Available schemas:"
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast', 'pg_temp_1', 'pg_toast_temp_1') ORDER BY schema_name;"

echo -e "\n${BLUE}Step 8: Starting Redis service...${NC}"
docker-compose up -d redis

echo -e "\n${BLUE}Step 9: Waiting for Redis to be ready...${NC}"
max_attempts=15
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker exec usdcop-redis redis-cli -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Redis is ready${NC}"
        break
    else
        echo "Attempt $attempt/$max_attempts: Redis not ready yet..."
        sleep 2
        ((attempt++))
    fi
done

echo -e "\n${BLUE}Step 10: Initializing Airflow database...${NC}"
docker-compose up --no-deps airflow-init

echo -e "\n${GREEN}=========================================="
echo "PostgreSQL Reset Completed Successfully!"
echo -e "==========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Start all services: docker-compose up -d"
echo "2. Verify connectivity: ./verify-connectivity.sh"
echo "3. Access Airflow: http://localhost:8080"
echo "   - Username: $AIRFLOW_USER"
echo "   - Password: [Check .env file]"
echo ""
echo -e "${YELLOW}Connection details:${NC}"
echo "PostgreSQL DSN: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
echo "Redis URL: redis://:${REDIS_PASSWORD}@localhost:6379/0"