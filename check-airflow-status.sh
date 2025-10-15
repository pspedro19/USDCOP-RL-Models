#!/bin/bash

# Airflow Status Check Script for USDCOP Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  USDCOP Airflow Status Check                ${NC}"
echo -e "${GREEN}===============================================${NC}"
echo

# Check if any docker compose files exist
if [[ ! -f "docker-compose.yml" ]]; then
    echo -e "${RED}ERROR: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${BLUE}1. Checking Docker containers...${NC}"

# Check container status
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}ERROR: Neither docker-compose nor docker compose found${NC}"
    exit 1
fi

echo -e "${YELLOW}Using: $COMPOSE_CMD${NC}"

# Check if fast mode is running
if $COMPOSE_CMD -f docker-compose.yml -f docker-compose.fast.yml ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✓ Fast mode containers detected${NC}"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.fast.yml"
    MODE="fast"
elif $COMPOSE_CMD ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✓ Standard mode containers detected${NC}"
    COMPOSE_FILES=""
    MODE="standard"
else
    echo -e "${YELLOW}No running containers detected${NC}"
    MODE="none"
fi

if [[ "$MODE" != "none" ]]; then
    echo
    echo -e "${BLUE}Container Status (${MODE} mode):${NC}"
    $COMPOSE_CMD $COMPOSE_FILES ps
fi

echo
echo -e "${BLUE}2. Checking port usage...${NC}"

# Check critical ports
ports=("5432:PostgreSQL" "6379:Redis" "8080:Airflow" "9000:MinIO")

for port_info in "${ports[@]}"; do
    port=$(echo $port_info | cut -d: -f1)
    service=$(echo $port_info | cut -d: -f2)
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Port $port ($service) - In Use${NC}"
    else
        echo -e "  ${RED}✗ Port $port ($service) - Free${NC}"
    fi
done

echo
echo -e "${BLUE}3. Checking Airflow UI accessibility...${NC}"

# Check Airflow UI
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 2>/dev/null | grep -q "200\|302\|401"; then
    echo -e "  ${GREEN}✓ Airflow UI is accessible at http://localhost:8080${NC}"
    
    # Try to get health status
    health_status=$(curl -s http://localhost:8080/health 2>/dev/null || echo "unavailable")
    if [[ "$health_status" == *"healthy"* ]] || [[ "$health_status" == *"ok"* ]]; then
        echo -e "  ${GREEN}✓ Airflow health check: OK${NC}"
    else
        echo -e "  ${YELLOW}⚠ Airflow health check: $health_status${NC}"
    fi
else
    echo -e "  ${RED}✗ Airflow UI is not accessible${NC}"
fi

echo
echo -e "${BLUE}4. Checking recent logs...${NC}"

if [[ "$MODE" != "none" ]]; then
    echo -e "${YELLOW}Last 5 lines from key services:${NC}"
    
    services=("airflow-webserver" "airflow-scheduler" "postgres")
    
    for service in "${services[@]}"; do
        echo
        echo -e "${BLUE}--- $service ---${NC}"
        $COMPOSE_CMD $COMPOSE_FILES logs --tail=5 $service 2>/dev/null || echo "Service not found or not running"
    done
fi

echo
echo -e "${BLUE}5. Quick diagnosis...${NC}"

if [[ "$MODE" == "none" ]]; then
    echo -e "  ${YELLOW}⚠ No containers running${NC}"
    echo -e "  ${BLUE}Suggested action:${NC} Run ./start-airflow-fast.sh"
elif ! curl -s http://localhost:8080 >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠ Containers running but Airflow UI not accessible${NC}"
    echo -e "  ${BLUE}Suggested actions:${NC}"
    echo "    - Check webserver logs: $COMPOSE_CMD $COMPOSE_FILES logs airflow-webserver"
    echo "    - Restart webserver: $COMPOSE_CMD $COMPOSE_FILES restart airflow-webserver"
else
    echo -e "  ${GREEN}✓ System appears to be running normally${NC}"
    echo -e "  ${BLUE}Access:${NC} http://localhost:8080 (admin/admin123)"
fi

echo
echo -e "${BLUE}Useful commands:${NC}"
echo "  View logs: $COMPOSE_CMD $COMPOSE_FILES logs -f [service]"
echo "  Restart service: $COMPOSE_CMD $COMPOSE_FILES restart [service]"
echo "  Stop all: $COMPOSE_CMD $COMPOSE_FILES down"
echo "  Start fast: ./start-airflow-fast.sh"
echo
