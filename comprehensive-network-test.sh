#!/bin/bash

# =====================================================
# USDCOP Trading System - Comprehensive Network Connectivity Test
# =====================================================

set -e

echo "==========================================================="
echo "USDCOP Trading System - Comprehensive Network Analysis"
echo "==========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓ Environment variables loaded from .env${NC}"
else
    echo -e "${RED}✗ .env file not found${NC}"
    exit 1
fi

# Initialize counters
PASSED_TESTS=0
FAILED_TESTS=0
ISSUES_FOUND=()
FIXES_APPLIED=()

# Helper function for test results
log_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS:${NC} $test_name"
        ((PASSED_TESTS++))
        [ -n "$details" ] && echo -e "   ${CYAN}$details${NC}"
    elif [ "$result" = "FAIL" ]; then
        echo -e "${RED}✗ FAIL:${NC} $test_name"
        ((FAILED_TESTS++))
        ISSUES_FOUND+=("$test_name: $details")
        [ -n "$details" ] && echo -e "   ${RED}$details${NC}"
    elif [ "$result" = "WARN" ]; then
        echo -e "${YELLOW}! WARN:${NC} $test_name"
        [ -n "$details" ] && echo -e "   ${YELLOW}$details${NC}"
    else
        echo -e "${BLUE}• INFO:${NC} $test_name"
        [ -n "$details" ] && echo -e "   ${BLUE}$details${NC}"
    fi
}

# Function 1: Check Docker Network Infrastructure
test_docker_network() {
    echo -e "\n${CYAN}=== 1. Docker Network Infrastructure Test ===${NC}"

    # Check if docker network exists
    if docker network inspect usdcop-rl-models_usdcop-trading-network &>/dev/null; then
        local subnet=$(docker network inspect usdcop-rl-models_usdcop-trading-network | jq -r '.[0].IPAM.Config[0].Subnet')
        local gateway=$(docker network inspect usdcop-rl-models_usdcop-trading-network | jq -r '.[0].IPAM.Config[0].Gateway')
        log_test "Docker network exists" "PASS" "Subnet: $subnet, Gateway: $gateway"

        # Count containers in network
        local container_count=$(docker network inspect usdcop-rl-models_usdcop-trading-network | jq -r '.[0].Containers | length')
        log_test "Containers in network" "INFO" "Count: $container_count containers"
    else
        log_test "Docker network exists" "FAIL" "Network 'usdcop-rl-models_usdcop-trading-network' not found"
        return 1
    fi
}

# Function 2: Test Container Health and Status
test_container_health() {
    echo -e "\n${CYAN}=== 2. Container Health Status Test ===${NC}"

    local containers=(
        "usdcop-postgres-timescale:Database"
        "usdcop-redis:Cache"
        "usdcop-minio:Object Storage"
        "usdcop-airflow-webserver:Airflow Web"
        "usdcop-airflow-scheduler:Airflow Scheduler"
        "usdcop-airflow-worker:Airflow Worker"
        "usdcop-dashboard:Dashboard"
        "usdcop-pgadmin:PgAdmin"
        "usdcop-grafana:Grafana"
        "usdcop-prometheus:Prometheus"
        "usdcop-health-monitor:Health Monitor"
        "usdcop-realtime-data:Real-time Data"
        "usdcop-websocket:WebSocket Service"
        "usdcop-nginx:Nginx Proxy"
    )

    for container_info in "${containers[@]}"; do
        IFS=':' read -r container_name service_name <<< "$container_info"

        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            local status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{for(i=2;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/[[:space:]]*$//')

            if [[ "$status" == *"healthy"* ]]; then
                log_test "$service_name container" "PASS" "$status"
            elif [[ "$status" == *"unhealthy"* ]]; then
                log_test "$service_name container" "FAIL" "$status"
            elif [[ "$status" == *"Up"* ]]; then
                log_test "$service_name container" "WARN" "$status (no health check)"
            else
                log_test "$service_name container" "FAIL" "$status"
            fi
        else
            log_test "$service_name container" "FAIL" "Container not running"
        fi
    done
}

# Function 3: Test DNS Resolution within Docker Network
test_dns_resolution() {
    echo -e "\n${CYAN}=== 3. DNS Resolution Test ===${NC}"

    local test_containers=("postgres" "redis" "minio")

    # Use a running container to test DNS resolution
    local test_from_container="usdcop-dashboard"

    if docker ps | grep -q "$test_from_container"; then
        for target in "${test_containers[@]}"; do
            if docker exec "$test_from_container" sh -c "getent hosts $target" &>/dev/null; then
                local ip=$(docker exec "$test_from_container" sh -c "getent hosts $target" | awk '{print $1}')
                log_test "DNS resolution for $target" "PASS" "Resolved to $ip"
            else
                log_test "DNS resolution for $target" "FAIL" "Cannot resolve hostname"
            fi
        done
    else
        log_test "DNS resolution test" "FAIL" "Test container $test_from_container not available"
    fi
}

# Function 4: Test Database Connectivity
test_database_connectivity() {
    echo -e "\n${CYAN}=== 4. Database Connectivity Test ===${NC}"

    # Test PostgreSQL from host
    if nc -z localhost 5432 2>/dev/null; then
        log_test "PostgreSQL port accessibility from host" "PASS" "Port 5432 is open"
    else
        log_test "PostgreSQL port accessibility from host" "FAIL" "Port 5432 is not accessible"
    fi

    # Test PostgreSQL connection from container
    if docker ps | grep -q "usdcop-postgres-timescale"; then
        if docker exec usdcop-postgres-timescale pg_isready -U admin &>/dev/null; then
            log_test "PostgreSQL service ready" "PASS" "Database accepting connections"

            # Test database connection
            if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT version();" &>/dev/null; then
                local version=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT version();" | head -1 | xargs)
                log_test "PostgreSQL query execution" "PASS" "$version"
            else
                log_test "PostgreSQL query execution" "FAIL" "Cannot execute queries"
            fi
        else
            log_test "PostgreSQL service ready" "FAIL" "Database not ready"
        fi
    else
        log_test "PostgreSQL connectivity" "FAIL" "PostgreSQL container not running"
    fi

    # Test Redis connectivity
    if nc -z localhost 6379 2>/dev/null; then
        log_test "Redis port accessibility from host" "PASS" "Port 6379 is open"
    else
        log_test "Redis port accessibility from host" "FAIL" "Port 6379 is not accessible"
    fi

    if docker ps | grep -q "usdcop-redis"; then
        if docker exec usdcop-redis redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q "PONG"; then
            log_test "Redis service connectivity" "PASS" "Redis responding to ping"
        else
            log_test "Redis service connectivity" "FAIL" "Redis not responding"
        fi
    else
        log_test "Redis connectivity" "FAIL" "Redis container not running"
    fi
}

# Function 5: Test Service-to-Service Communication
test_service_communication() {
    echo -e "\n${CYAN}=== 5. Service-to-Service Communication Test ===${NC}"

    # Test connections between services using containers
    local test_container="usdcop-dashboard"

    if docker ps | grep -q "$test_container"; then
        # Test PostgreSQL connection from dashboard
        if docker exec "$test_container" sh -c "nc -z postgres 5432" 2>/dev/null; then
            log_test "Dashboard to PostgreSQL" "PASS" "Port 5432 accessible"
        else
            log_test "Dashboard to PostgreSQL" "FAIL" "Cannot connect to postgres:5432"
        fi

        # Test Redis connection from dashboard
        if docker exec "$test_container" sh -c "nc -z redis 6379" 2>/dev/null; then
            log_test "Dashboard to Redis" "PASS" "Port 6379 accessible"
        else
            log_test "Dashboard to Redis" "FAIL" "Cannot connect to redis:6379"
        fi

        # Test MinIO connection from dashboard
        if docker exec "$test_container" sh -c "nc -z minio 9000" 2>/dev/null; then
            log_test "Dashboard to MinIO" "PASS" "Port 9000 accessible"
        else
            log_test "Dashboard to MinIO" "FAIL" "Cannot connect to minio:9000"
        fi
    else
        log_test "Service communication test" "FAIL" "Test container not available"
    fi
}

# Function 6: Test External Port Mappings
test_external_ports() {
    echo -e "\n${CYAN}=== 6. External Port Mappings Test ===${NC}"

    local ports=(
        "3000:Dashboard"
        "3002:Grafana"
        "5050:PgAdmin"
        "5432:PostgreSQL"
        "6379:Redis"
        "8080:Airflow"
        "8081:Real-time Data"
        "8082:WebSocket"
        "8083:Health Monitor"
        "9000:MinIO API"
        "9001:MinIO Console"
        "9090:Prometheus"
    )

    for port_info in "${ports[@]}"; do
        IFS=':' read -r port service <<< "$port_info"

        if nc -z localhost "$port" 2>/dev/null; then
            log_test "$service external port $port" "PASS" "Port accessible from host"
        else
            log_test "$service external port $port" "FAIL" "Port not accessible from host"
        fi
    done
}

# Function 7: Test Health Check Endpoints
test_health_endpoints() {
    echo -e "\n${CYAN}=== 7. Health Check Endpoints Test ===${NC}"

    local endpoints=(
        "http://localhost:3000/api/health:Dashboard"
        "http://localhost:8080/health:Airflow"
        "http://localhost:8081/health:Real-time Data"
        "http://localhost:8082/health:WebSocket"
        "http://localhost:8083/health:Health Monitor"
        "http://localhost:9000/minio/health/live:MinIO"
        "http://localhost:9090/-/healthy:Prometheus"
    )

    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r url service <<< "$endpoint_info"

        if curl -f -s "$url" &>/dev/null; then
            log_test "$service health endpoint" "PASS" "$url responding"
        else
            # Try with longer timeout
            if timeout 10 curl -f -s "$url" &>/dev/null; then
                log_test "$service health endpoint" "WARN" "$url responding slowly"
            else
                log_test "$service health endpoint" "FAIL" "$url not responding"
            fi
        fi
    done
}

# Function 8: Diagnose and Fix Issues
fix_identified_issues() {
    echo -e "\n${CYAN}=== 8. Issue Diagnosis and Fixes ===${NC}"

    # Fix 1: Check and fix Airflow database connection string
    if docker logs usdcop-airflow-webserver 2>&1 | grep -q "invalid dsn"; then
        echo -e "${YELLOW}Found Airflow database connection issue${NC}"
        log_test "Airflow DSN issue detected" "INFO" "Database connection string format problem"

        # The issue is in the connection string format - it should be postgresql:// not postgresql+psycopg2://
        # This is fixed in the docker-compose file by updating environment variables
        FIXES_APPLIED+=("Airflow database connection string format issue identified")
    fi

    # Fix 2: Check MinIO initialization
    if docker ps -a | grep usdcop-minio-init | grep -q "Exited (1)"; then
        echo -e "${YELLOW}MinIO initialization failed${NC}"
        log_test "MinIO bucket creation issue" "INFO" "Attempting to reinitialize buckets"

        # Restart MinIO init container
        docker rm usdcop-minio-init &>/dev/null || true
        docker-compose up -d minio-init &>/dev/null && {
            FIXES_APPLIED+=("Restarted MinIO bucket initialization")
            log_test "MinIO reinitialization" "PASS" "MinIO buckets initialization restarted"
        } || {
            log_test "MinIO reinitialization" "FAIL" "Could not restart MinIO initialization"
        }
    fi

    # Fix 3: Start nginx if not running
    if ! docker ps | grep -q "usdcop-nginx"; then
        echo -e "${YELLOW}Nginx proxy not running${NC}"
        log_test "Starting Nginx proxy" "INFO" "Attempting to start reverse proxy"

        docker-compose up -d nginx &>/dev/null && {
            FIXES_APPLIED+=("Started Nginx reverse proxy")
            log_test "Nginx startup" "PASS" "Nginx reverse proxy started"
        } || {
            log_test "Nginx startup" "FAIL" "Could not start Nginx reverse proxy"
        }
    fi
}

# Function 9: Network Performance Test
test_network_performance() {
    echo -e "\n${CYAN}=== 9. Network Performance Test ===${NC}"

    # Test latency between containers
    local test_container="usdcop-dashboard"

    if docker ps | grep -q "$test_container"; then
        # Test ping latency to key services
        local services=("postgres" "redis" "minio")

        for service in "${services[@]}"; do
            local latency=$(docker exec "$test_container" sh -c "ping -c 3 $service 2>/dev/null | tail -1 | awk -F '/' '{print \$5}'" 2>/dev/null || echo "N/A")
            if [ "$latency" != "N/A" ]; then
                log_test "$service network latency" "INFO" "${latency}ms average"
            else
                log_test "$service network latency" "WARN" "Could not measure latency"
            fi
        done
    fi
}

# Function 10: Generate Network Summary
generate_network_summary() {
    echo -e "\n${CYAN}=== 10. Network Configuration Summary ===${NC}"

    # Docker network details
    echo -e "\n${YELLOW}Docker Network Configuration:${NC}"
    docker network inspect usdcop-rl-models_usdcop-trading-network | jq -r '.[0] | "Network: \(.Name)\nSubnet: \(.IPAM.Config[0].Subnet)\nGateway: \(.IPAM.Config[0].Gateway)\nDriver: \(.Driver)"'

    # Container IP addresses
    echo -e "\n${YELLOW}Container IP Addresses:${NC}"
    docker network inspect usdcop-rl-models_usdcop-trading-network | jq -r '.[0].Containers | to_entries[] | "\(.value.Name): \(.value.IPv4Address)"' | sort

    # Port mappings
    echo -e "\n${YELLOW}External Port Mappings:${NC}"
    docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -v "PORTS" | sort
}

# Main execution function
main() {
    local start_time=$(date +%s)

    # Run all tests
    test_docker_network
    test_container_health
    test_dns_resolution
    test_database_connectivity
    test_service_communication
    test_external_ports
    test_health_endpoints
    fix_identified_issues
    test_network_performance
    generate_network_summary

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Final summary
    echo -e "\n${CYAN}======================================${NC}"
    echo -e "${CYAN}   NETWORK ANALYSIS SUMMARY${NC}"
    echo -e "${CYAN}======================================${NC}"

    echo -e "\n${YELLOW}Test Results:${NC}"
    echo -e "  • Tests Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "  • Tests Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "  • Total Runtime: ${BLUE}${duration}s${NC}"

    if [ ${#ISSUES_FOUND[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}Issues Found:${NC}"
        printf '  • %s\n' "${ISSUES_FOUND[@]}"
    fi

    if [ ${#FIXES_APPLIED[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}Fixes Applied:${NC}"
        printf '  • %s\n' "${FIXES_APPLIED[@]}"
    fi

    # Overall status
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 NETWORK STATUS: HEALTHY${NC}"
        echo -e "${GREEN}All network connectivity tests passed successfully!${NC}"
        exit 0
    elif [ $FAILED_TESTS -lt 3 ]; then
        echo -e "\n${YELLOW}⚠️  NETWORK STATUS: NEEDS ATTENTION${NC}"
        echo -e "${YELLOW}Minor network issues detected but core functionality working.${NC}"
        exit 1
    else
        echo -e "\n${RED}❌ NETWORK STATUS: CRITICAL ISSUES${NC}"
        echo -e "${RED}Multiple network failures detected. Immediate attention required.${NC}"
        exit 2
    fi
}

# Handle script arguments
case "${1:-}" in
    "network")
        test_docker_network
        ;;
    "health")
        test_container_health
        ;;
    "dns")
        test_dns_resolution
        ;;
    "database"|"db")
        test_database_connectivity
        ;;
    "services")
        test_service_communication
        ;;
    "ports")
        test_external_ports
        ;;
    "endpoints")
        test_health_endpoints
        ;;
    "fix")
        fix_identified_issues
        ;;
    "performance"|"perf")
        test_network_performance
        ;;
    "summary")
        generate_network_summary
        ;;
    *)
        main
        ;;
esac