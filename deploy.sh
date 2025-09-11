#!/bin/bash

# USDCOP RL Trading System - Production Deployment Script
# This script automates the deployment process with safety checks and optimizations

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="usdcop-rl-trading"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found. Please copy from $ENV_FILE.example and configure it."
        exit 1
    fi
    
    log "Prerequisites check passed ✓"
}

# Validate environment variables
validate_env() {
    log "Validating environment variables..."
    
    # Load environment variables
    source "$ENV_FILE"
    
    # Check critical variables
    REQUIRED_VARS=(
        "POSTGRES_PASSWORD"
        "MINIO_ACCESS_KEY"
        "MINIO_SECRET_KEY"
        "JWT_SECRET"
        "AIRFLOW_FERNET_KEY"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set in $ENV_FILE"
            exit 1
        fi
    done
    
    # Validate JWT secret length
    if [[ ${#JWT_SECRET} -lt 32 ]]; then
        error "JWT_SECRET must be at least 32 characters long"
        exit 1
    fi
    
    log "Environment validation passed ✓"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    DIRS=(
        "data/postgres"
        "data/redis"
        "data/minio"
        "data/influxdb"
        "data/grafana"
        "data/prometheus"
        "logs/nginx"
        "logs/api"
        "logs/dashboard"
        "ssl"
    )
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$dir"
        info "Created directory: $dir"
    done
    
    log "Directories created ✓"
}

# Generate SSL certificates (self-signed for development)
generate_ssl() {
    log "Checking SSL certificates..."
    
    if [[ ! -f "nginx/ssl/usdcop.crt" ]] || [[ ! -f "nginx/ssl/usdcop.key" ]]; then
        warning "SSL certificates not found. Generating self-signed certificates..."
        
        mkdir -p nginx/ssl
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/usdcop.key \
            -out nginx/ssl/usdcop.crt \
            -subj "/C=CO/ST=Bogota/L=Bogota/O=USDCOP Trading/OU=IT Department/CN=usdcop.local"
        
        log "Self-signed SSL certificates generated ✓"
    else
        log "SSL certificates found ✓"
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build with BuildKit for better performance
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    docker-compose -f "$COMPOSE_FILE" build --parallel --no-cache
    
    log "Docker images built successfully ✓"
}

# Test DAG imports
test_dag_imports() {
    log "Testing DAG imports..."
    
    # Build Airflow image first if it doesn't exist
    if ! docker images | grep -q "usdcop-rl-trading_airflow-webserver"; then
        info "Building Airflow image..."
        docker-compose -f "$COMPOSE_FILE" build airflow-webserver
    fi
    
    # Test DAG imports in the container
    info "Running DAG import tests..."
    if docker-compose -f "$COMPOSE_FILE" run --rm airflow-webserver python /opt/airflow/test_dag_imports.py; then
        log "DAG import tests passed ✓"
    else
        error "DAG import tests failed. Please check the logs above and fix import errors."
        exit 1
    fi
}

# Initialize databases
init_databases() {
    log "Initializing databases..."
    
    # Start database services first
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis influxdb
    
    # Wait for PostgreSQL to be ready
    info "Waiting for PostgreSQL to be ready..."
    timeout 60 bash -c 'until docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U postgres; do sleep 2; done'
    
    # Wait for Redis to be ready
    info "Waiting for Redis to be ready..."
    timeout 60 bash -c 'until docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping; do sleep 2; done'
    
    # Initialize Airflow database
    info "Initializing Airflow database..."
    docker-compose -f "$COMPOSE_FILE" run --rm airflow-webserver init
    
    log "Databases initialized ✓"
}

# Deploy application
deploy() {
    log "Deploying USDCOP RL Trading System..."
    
    # Pull latest images for base services
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Start all services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log "Deployment completed successfully ✓"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    SERVICES=(
        "trading-dashboard:3000"
        "backend-api:8000"
        "postgres:5432"
        "redis:6379"
        "minio:9000"
        "influxdb:8086"
    )
    
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if docker-compose -f "$COMPOSE_FILE" ps "$name" | grep -q "Up"; then
            info "$name is running ✓"
        else
            warning "$name is not running properly"
        fi
    done
    
    log "Health check completed"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    
    # Show running containers
    info "Running containers:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    # Show service URLs
    info "Service URLs:"
    echo "📊 Trading Dashboard: https://dashboard.usdcop.local"
    echo "🔌 API Endpoint: https://api.usdcop.local"
    echo "🌬️ Airflow UI: https://airflow.usdcop.local"
    echo "📈 Grafana: https://monitoring.usdcop.local/grafana"
    echo "🔍 Prometheus: https://monitoring.usdcop.local/prometheus"
    echo "💾 MinIO Console: http://localhost:9001"
    echo ""
    
    # Show logs command
    info "To view logs:"
    echo "docker-compose -f $COMPOSE_FILE logs -f [service_name]"
    echo ""
    
    # Show stop command
    info "To stop all services:"
    echo "docker-compose -f $COMPOSE_FILE down"
    echo ""
}

# Backup function
backup() {
    log "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup databases
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U postgres usdcop_trading > "$BACKUP_DIR/postgres_backup.sql"
    
    # Backup volumes
    docker run --rm -v usdcop-rl-trading_minio-data:/data -v "$PWD/$BACKUP_DIR":/backup alpine tar czf /backup/minio_backup.tar.gz -C /data .
    
    log "Backup created in $BACKUP_DIR ✓"
}

# Update function
update() {
    log "Updating USDCOP RL Trading System..."
    
    # Create backup before update
    backup
    
    # Pull latest changes
    git pull origin main
    
    # Rebuild and redeploy
    build_images
    deploy
    
    log "Update completed ✓"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Stop all services
    docker-compose -f "$COMPOSE_FILE" down -v
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    log "Cleanup completed ✓"
}

# Main function
main() {
    echo ""
    echo "🚀 USDCOP RL Trading System Deployment Script"
    echo "=============================================="
    echo ""
    
    case "${1:-deploy}" in
        "deploy"|"")
            check_root
            check_prerequisites
            validate_env
            create_directories
            generate_ssl
            build_images
            test_dag_imports
            init_databases
            deploy
            show_status
            ;;
        "status")
            show_status
            ;;
        "update")
            check_root
            check_prerequisites
            validate_env
            update
            show_status
            ;;
        "backup")
            check_prerequisites
            backup
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the full system (default)"
            echo "  status   - Show deployment status"
            echo "  update   - Update and redeploy the system"
            echo "  backup   - Create a backup"
            echo "  cleanup  - Stop services and cleanup"
            echo "  help     - Show this help message"
            echo ""
            ;;
        *)
            error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"