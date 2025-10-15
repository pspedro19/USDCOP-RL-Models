#!/bin/bash

# Enhanced WebSocket Service Deployment Script
# ============================================
# This script provides easy deployment options for the Enhanced WebSocket Service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration
SERVICE_NAME="enhanced-websocket-service"
SERVICE_VERSION="4.0.0"
COMPOSE_FILE="docker-compose.enhanced-websocket.yml"
ENV_FILE=".env.enhanced-websocket"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  Enhanced WebSocket Service Deployment v${SERVICE_VERSION}"
    echo "  L0 Integration & Real-time Market Data"
    echo "=================================================="
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file $ENV_FILE not found."
        print_status "Copying template environment file..."
        cp "$ENV_FILE" .env
        print_warning "Please edit .env file with your configuration before continuing."
        exit 1
    fi

    print_status "Prerequisites check passed ✓"
}

create_directories() {
    print_status "Creating necessary directories..."

    # Create data directories
    mkdir -p /data/ready-signals
    mkdir -p /home/GlobalForex/USDCOP-RL-Models/data/backups
    mkdir -p ./logs
    mkdir -p ./sql/init

    # Set proper permissions
    chmod 755 /data/ready-signals
    chmod 755 /home/GlobalForex/USDCOP-RL-Models/data/backups
    chmod 755 ./logs

    print_status "Directories created ✓"
}

create_database_init() {
    print_status "Creating database initialization scripts..."

    cat > ./sql/init/01-create-tables.sql << EOF
-- Enhanced WebSocket Service Database Schema
-- ==========================================

-- Market data table for OHLC data
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(12,6),
    high DECIMAL(12,6),
    low DECIMAL(12,6),
    close DECIMAL(12,6),
    volume INTEGER DEFAULT 0,
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, datetime, source)
);

-- Real-time market data table for tick data
CREATE TABLE IF NOT EXISTS realtime_market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    bid DECIMAL(12,6),
    ask DECIMAL(12,6),
    last DECIMAL(12,6),
    volume INTEGER DEFAULT 0,
    spread DECIMAL(12,6),
    session_date DATE,
    trading_session BOOLEAN DEFAULT true,
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System health monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100),
    status VARCHAR(50),
    details JSONB,
    response_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market sessions table
CREATE TABLE IF NOT EXISTS market_sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE UNIQUE NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_datetime ON market_data(symbol, datetime);
CREATE INDEX IF NOT EXISTS idx_market_data_datetime ON market_data(datetime);
CREATE INDEX IF NOT EXISTS idx_realtime_data_symbol_time ON realtime_market_data(symbol, time);
CREATE INDEX IF NOT EXISTS idx_realtime_data_time ON realtime_market_data(time);
CREATE INDEX IF NOT EXISTS idx_system_health_service_time ON system_health(service_name, created_at);

-- Create Colombian timezone function
CREATE OR REPLACE FUNCTION get_colombia_time()
RETURNS TIMESTAMP WITH TIME ZONE AS \$\$
BEGIN
    RETURN NOW() AT TIME ZONE 'America/Bogota';
END;
\$\$ LANGUAGE plpgsql;

-- Create market hours check function
CREATE OR REPLACE FUNCTION is_market_open()
RETURNS BOOLEAN AS \$\$
DECLARE
    current_time_cot TIME;
    current_day INTEGER;
BEGIN
    -- Get current time in Colombia timezone
    current_time_cot := (NOW() AT TIME ZONE 'America/Bogota')::TIME;
    current_day := EXTRACT(dow FROM (NOW() AT TIME ZONE 'America/Bogota'));

    -- Check if it's a weekday (1=Monday, 5=Friday)
    IF current_day < 1 OR current_day > 5 THEN
        RETURN FALSE;
    END IF;

    -- Check if time is within market hours (8:00 AM - 12:55 PM)
    IF current_time_cot >= '08:00:00' AND current_time_cot <= '12:55:00' THEN
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
\$\$ LANGUAGE plpgsql;

COMMENT ON TABLE market_data IS 'OHLC market data for USDCOP with 5-minute intervals';
COMMENT ON TABLE realtime_market_data IS 'Real-time tick data from WebSocket feeds';
COMMENT ON TABLE system_health IS 'Service health monitoring and status tracking';
COMMENT ON TABLE market_sessions IS 'Daily market session tracking';
EOF

    print_status "Database initialization scripts created ✓"
}

validate_environment() {
    print_status "Validating environment configuration..."

    # Source the environment file
    set -a
    source .env
    set +a

    # Check required variables
    required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "TWELVEDATA_API_KEY_G1_1"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ] || [ "${!var}" = "your_api_key_here" ]; then
            print_error "Required environment variable $var is not set or has default value"
            print_warning "Please edit .env file with proper values"
            exit 1
        fi
    done

    print_status "Environment validation passed ✓"
}

deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."

    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull

    # Build the service
    docker-compose -f "$COMPOSE_FILE" build

    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d

    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 10

    # Check service health
    check_service_health
}

deploy_standalone() {
    print_status "Deploying standalone Docker container..."

    # Build the image
    docker build -f Dockerfile.enhanced-websocket -t "$SERVICE_NAME:$SERVICE_VERSION" .

    # Stop existing container if running
    docker stop "$SERVICE_NAME" 2>/dev/null || true
    docker rm "$SERVICE_NAME" 2>/dev/null || true

    # Run the container
    docker run -d \
        --name "$SERVICE_NAME" \
        -p 8080:8080 \
        --env-file .env \
        -v /data/ready-signals:/data/ready-signals \
        -v "$(pwd)/data/backups:/data/backups" \
        -v "$(pwd)/logs:/app/logs" \
        --restart unless-stopped \
        "$SERVICE_NAME:$SERVICE_VERSION"

    # Wait for service to start
    print_status "Waiting for service to start..."
    sleep 10

    # Check service health
    check_service_health
}

check_service_health() {
    print_status "Checking service health..."

    # Wait up to 60 seconds for service to be healthy
    for i in {1..12}; do
        if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
            print_status "Service is healthy ✓"

            # Get service status
            echo -e "${BLUE}Service Status:${NC}"
            curl -s http://localhost:8080/health | python3 -m json.tool
            return 0
        fi

        print_status "Waiting for service to be ready... ($i/12)"
        sleep 5
    done

    print_error "Service health check failed"
    print_status "Checking logs..."
    if [ "$1" = "compose" ]; then
        docker-compose -f "$COMPOSE_FILE" logs enhanced-websocket
    else
        docker logs "$SERVICE_NAME"
    fi
    exit 1
}

show_status() {
    print_status "Service Status:"
    curl -s http://localhost:8080/status | python3 -m json.tool || print_error "Could not get service status"
}

show_logs() {
    if [ "$1" = "compose" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f enhanced-websocket
    else
        docker logs -f "$SERVICE_NAME"
    fi
}

stop_services() {
    print_status "Stopping services..."

    if [ "$1" = "compose" ]; then
        docker-compose -f "$COMPOSE_FILE" down
    else
        docker stop "$SERVICE_NAME" 2>/dev/null || true
        docker rm "$SERVICE_NAME" 2>/dev/null || true
    fi

    print_status "Services stopped ✓"
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy-compose    Deploy using Docker Compose (recommended)"
    echo "  deploy-standalone Deploy standalone Docker container"
    echo "  status           Show service status"
    echo "  logs             Show service logs"
    echo "  logs-compose     Show Docker Compose logs"
    echo "  stop             Stop standalone container"
    echo "  stop-compose     Stop Docker Compose services"
    echo "  health           Check service health"
    echo "  validate         Validate configuration"
    echo "  setup            Setup directories and database scripts"
    echo ""
    echo "Examples:"
    echo "  $0 deploy-compose    # Deploy with full stack"
    echo "  $0 status           # Check service status"
    echo "  $0 logs             # View service logs"
}

# Main script logic
case "${1:-}" in
    deploy-compose)
        print_header
        check_prerequisites
        create_directories
        create_database_init
        validate_environment
        deploy_docker_compose
        print_status "Deployment completed successfully!"
        print_status "Service available at: http://localhost:8080"
        print_status "Health check: http://localhost:8080/health"
        print_status "Status endpoint: http://localhost:8080/status"
        ;;

    deploy-standalone)
        print_header
        check_prerequisites
        create_directories
        validate_environment
        deploy_standalone
        print_status "Deployment completed successfully!"
        print_status "Service available at: http://localhost:8080"
        ;;

    status)
        show_status
        ;;

    logs)
        show_logs
        ;;

    logs-compose)
        show_logs compose
        ;;

    stop)
        stop_services
        ;;

    stop-compose)
        stop_services compose
        ;;

    health)
        curl -s http://localhost:8080/health | python3 -m json.tool
        ;;

    validate)
        validate_environment
        print_status "Configuration is valid ✓"
        ;;

    setup)
        print_header
        create_directories
        create_database_init
        print_status "Setup completed ✓"
        ;;

    *)
        show_usage
        exit 1
        ;;
esac