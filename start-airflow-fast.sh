#!/bin/bash

# Fast Airflow Startup Script for USDCOP Trading System
# This script gets Airflow UI running quickly for L0 pipeline execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  USDCOP Trading System - Fast Airflow Setup ${NC}"
echo -e "${GREEN}===============================================${NC}"
echo

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check if Airflow UI is already running
if check_port 8080; then
    echo -e "${YELLOW}WARNING: Port 8080 is already in use!${NC}"
    echo -e "${YELLOW}This might be Airflow already running or another service.${NC}"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo "  1. Stop existing service and continue"
    echo "  2. Check what's running on port 8080"
    echo "  3. Exit and investigate manually"
    echo
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Stopping existing Docker containers...${NC}"
            docker-compose down 2>/dev/null || true
            sleep 2
            ;;
        2)
            echo -e "${BLUE}Services running on port 8080:${NC}"
            lsof -i :8080 || echo "No detailed information available"
            exit 1
            ;;
        3)
            echo -e "${BLUE}Exiting. Please investigate port 8080 usage manually.${NC}"
            exit 1
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
fi

echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running or not accessible.${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if docker-compose or docker compose is available
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}ERROR: Neither docker-compose nor docker compose is available.${NC}"
    echo "Please install Docker Compose and try again."
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo -e "${GREEN}✓ Docker Compose is available ($COMPOSE_CMD)${NC}"

echo
echo -e "${BLUE}Step 2: Cleaning up any existing containers...${NC}"
$COMPOSE_CMD down 2>/dev/null || true

echo
echo -e "${BLUE}Step 3: Starting minimal Airflow services...${NC}"
echo -e "${YELLOW}This uses a fast configuration with minimal dependencies.${NC}"
echo

# Start services with fast configuration
echo -e "${YELLOW}Starting infrastructure services (PostgreSQL, Redis, MinIO)...${NC}"
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml up -d postgres redis minio

echo -e "${YELLOW}Waiting for infrastructure to be ready...${NC}"
sleep 10

echo -e "${YELLOW}Initializing MinIO buckets...${NC}"
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml up -d minio-init

echo -e "${YELLOW}Starting Airflow services...${NC}"
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml up -d airflow-init

# Wait for init to complete
echo -e "${YELLOW}Waiting for Airflow initialization...${NC}"
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml wait airflow-init

# Start scheduler and webserver
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml up -d airflow-scheduler airflow-webserver

echo
echo -e "${BLUE}Step 4: Waiting for Airflow UI to be ready...${NC}"

# Wait for Airflow UI to be accessible
max_attempts=60
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200\|302"; then
        echo -e "${GREEN}✓ Airflow UI is ready!${NC}"
        break
    fi
    
    if [ $((attempt % 10)) -eq 0 ]; then
        echo -e "${YELLOW}Still waiting... (attempt $attempt/$max_attempts)${NC}"
    fi
    
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo -e "${RED}ERROR: Airflow UI did not become ready within expected time.${NC}"
    echo -e "${YELLOW}Checking container status:${NC}"
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml ps
    echo
    echo -e "${YELLOW}To debug, check logs with:${NC}"
    echo "  $COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml logs airflow-webserver"
    echo "  $COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml logs airflow-scheduler"
    exit 1
fi

echo
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  ✓ AIRFLOW IS READY!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo
echo -e "${BLUE}Access Information:${NC}"
echo -e "  • Airflow UI: ${GREEN}http://localhost:8080${NC}"
echo -e "  • Username: ${GREEN}admin${NC}"
echo -e "  • Password: ${GREEN}admin123${NC}"
echo
echo -e "${BLUE}Available Services:${NC}"
echo -e "  • PostgreSQL: localhost:5432"
echo -e "  • Redis: localhost:6379"
echo -e "  • MinIO: http://localhost:9001 (admin/minioadmin123)"
echo
echo -e "${YELLOW}Note: This is a minimal setup for L0 pipeline execution.${NC}"
echo -e "${YELLOW}Heavy ML dependencies are not installed for faster startup.${NC}"
echo
echo -e "${BLUE}To install full ML dependencies later:${NC}"
echo "  1. docker-compose exec airflow-webserver pip install -r /requirements.txt"
echo "  2. Restart the containers"
echo
echo -e "${BLUE}To stop all services:${NC}"
echo "  $COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml down"
echo
echo -e "${GREEN}Ready to execute L0 pipeline! → http://localhost:8080${NC}"
