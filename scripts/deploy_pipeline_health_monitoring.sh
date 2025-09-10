#!/bin/bash

# Pipeline Health Monitoring Deployment Script
# ============================================
# 
# Complete deployment script for the Pipeline Health Monitoring system
# Supports standalone deployment or integration with existing trading system

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE_HEALTH="${PROJECT_ROOT}/docker-compose-pipeline-health.yml"
COMPOSE_FILE_MAIN="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env"

# Default configuration
DEFAULT_MODE="standalone"
DEFAULT_PROFILE="minimal"
HEALTH_CHECK_TIMEOUT=60

# Usage function
usage() {
    cat << EOF
Pipeline Health Monitoring Deployment Script

Usage: $0 [OPTIONS]

Options:
    -m, --mode MODE           Deployment mode: standalone, integrated (default: standalone)
    -p, --profile PROFILE     Profile: minimal, with-dashboard, with-monitoring, full (default: minimal)
    -e, --env-file FILE       Environment file path (default: .env)
    --check-only             Only run health checks without deployment
    --stop                   Stop all services
    --restart                Restart all services
    --logs                   Show logs for all services
    --status                 Show status of all services
    -h, --help               Show this help message

Modes:
    standalone               Deploy health monitoring with its own infrastructure
    integrated               Deploy health monitoring connecting to existing trading system

Profiles:
    minimal                  API service only
    with-dashboard          Include dashboard
    with-monitoring         Include Prometheus
    full                    All services

Examples:
    $0                                      # Deploy minimal standalone
    $0 -m integrated -p with-dashboard      # Deploy with main system + dashboard
    $0 --status                            # Check service status
    $0 --stop                             # Stop all services

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Environment setup
setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating .env file from example..."
        if [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
            cp "${PROJECT_ROOT}/.env.example" "$ENV_FILE"
        else
            create_default_env_file
        fi
    fi
    
    # Source environment variables
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
        log_success "Environment loaded from $ENV_FILE"
    fi
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/config"
    
    log_success "Environment setup completed"
}

# Create default .env file
create_default_env_file() {
    cat > "$ENV_FILE" << 'EOF'
# Pipeline Health Monitoring Configuration
# ========================================

# Application Environment
APP_ENV=production

# Database Configuration
POSTGRES_PASSWORD=postgres123
DB_PASSWORD=trading123
DB_NAME=trading_db
DB_USER=trading_user

# Redis Configuration
REDIS_PASSWORD=redis123

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# Airflow Configuration
AIRFLOW_FERNET_KEY=your-fernet-key-here
AIRFLOW_SECRET_KEY=your-secret-key-here
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin123

# Pipeline Health API Configuration
PIPELINE_HEALTH_API_PORT=8002
NEXT_PUBLIC_PIPELINE_HEALTH_API=http://localhost:8002

# Monitoring Ports (for standalone mode)
POSTGRES_HEALTH_PORT=5433
REDIS_HEALTH_PORT=6380
MINIO_HEALTH_API_PORT=9002
MINIO_HEALTH_CONSOLE_PORT=9003
PIPELINE_DASHBOARD_PORT=3001
PROMETHEUS_HEALTH_PORT=9091

# Monitoring Configuration
HEALTH_CHECK_INTERVAL=30
METRICS_RETENTION_HOURS=168
LOG_LEVEL=INFO

EOF
}

# Dependency checks
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Network setup
setup_networks() {
    log_info "Setting up Docker networks..."
    
    # Create trading network if it doesn't exist (for integrated mode)
    if [[ "$MODE" == "integrated" ]]; then
        if ! docker network inspect trading-network &> /dev/null; then
            log_info "Creating trading network..."
            docker network create trading-network --driver bridge --subnet=172.28.0.0/16
            log_success "Trading network created"
        else
            log_info "Trading network already exists"
        fi
    fi
    
    log_success "Network setup completed"
}

# Service health checks
check_service_health() {
    local service_name=$1
    local max_attempts=12  # 60 seconds with 5-second intervals
    local attempt=0
    
    log_info "Checking health of $service_name..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE_HEALTH" ps "$service_name" | grep -q "healthy\|Up"; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting for $service_name to be healthy (attempt $attempt/$max_attempts)..."
        sleep 5
    done
    
    log_error "$service_name failed to become healthy"
    return 1
}

# Deploy services
deploy_services() {
    log_info "Deploying Pipeline Health Monitoring services..."
    
    # Build the compose command based on mode and profile
    local compose_cmd="docker-compose -f $COMPOSE_FILE_HEALTH"
    
    if [[ "$MODE" == "integrated" ]]; then
        compose_cmd="$compose_cmd -f $COMPOSE_FILE_MAIN"
    fi
    
    # Set profiles based on configuration
    local profiles=()
    case $PROFILE in
        "minimal")
            # No additional profiles needed for minimal
            ;;
        "with-dashboard")
            profiles+=("with-dashboard")
            ;;
        "with-monitoring")
            profiles+=("with-monitoring")
            ;;
        "full")
            profiles+=("with-dashboard" "with-monitoring")
            ;;
    esac
    
    if [[ "$MODE" == "standalone" ]]; then
        profiles+=("standalone")
    fi
    
    # Add profiles to command
    for profile in "${profiles[@]}"; do
        compose_cmd="$compose_cmd --profile $profile"
    done
    
    # Pull latest images
    log_info "Pulling latest images..."
    eval "$compose_cmd pull"
    
    # Build and start services
    log_info "Building and starting services..."
    eval "$compose_cmd up -d --build"
    
    # Wait for core services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check health of core service
    if ! check_service_health "pipeline-health-api"; then
        log_error "Core service failed to start properly"
        show_logs
        exit 1
    fi
    
    log_success "Services deployed successfully"
}

# Stop services
stop_services() {
    log_info "Stopping Pipeline Health Monitoring services..."
    
    local compose_cmd="docker-compose -f $COMPOSE_FILE_HEALTH"
    
    if [[ "$MODE" == "integrated" ]]; then
        compose_cmd="$compose_cmd -f $COMPOSE_FILE_MAIN"
    fi
    
    eval "$compose_cmd down"
    
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting Pipeline Health Monitoring services..."
    stop_services
    sleep 5
    deploy_services
}

# Show logs
show_logs() {
    log_info "Showing service logs..."
    
    local compose_cmd="docker-compose -f $COMPOSE_FILE_HEALTH"
    
    if [[ "$MODE" == "integrated" ]]; then
        compose_cmd="$compose_cmd -f $COMPOSE_FILE_MAIN"
    fi
    
    eval "$compose_cmd logs -f --tail=50"
}

# Show status
show_status() {
    log_info "Service Status:"
    echo "===================="
    
    local compose_cmd="docker-compose -f $COMPOSE_FILE_HEALTH"
    
    if [[ "$MODE" == "integrated" ]]; then
        compose_cmd="$compose_cmd -f $COMPOSE_FILE_MAIN"
    fi
    
    eval "$compose_cmd ps"
    
    # Show service endpoints
    echo
    log_info "Service Endpoints:"
    echo "===================="
    echo "Pipeline Health API: http://localhost:${PIPELINE_HEALTH_API_PORT:-8002}"
    
    if [[ "$PROFILE" == "with-dashboard" || "$PROFILE" == "full" ]]; then
        echo "Dashboard: http://localhost:${PIPELINE_DASHBOARD_PORT:-3001}"
    fi
    
    if [[ "$PROFILE" == "with-monitoring" || "$PROFILE" == "full" ]]; then
        echo "Prometheus: http://localhost:${PROMETHEUS_HEALTH_PORT:-9091}"
    fi
    
    if [[ "$MODE" == "standalone" ]]; then
        echo "PostgreSQL: localhost:${POSTGRES_HEALTH_PORT:-5433}"
        echo "Redis: localhost:${REDIS_HEALTH_PORT:-6380}"
        echo "MinIO Console: http://localhost:${MINIO_HEALTH_CONSOLE_PORT:-9003}"
    fi
}

# Health checks only
run_health_checks() {
    log_info "Running health checks..."
    
    # Check if Python dependencies are available
    log_info "Checking Python dependencies..."
    if command -v python3 &> /dev/null; then
        if python3 "${PROJECT_ROOT}/scripts/run_pipeline_health_service.py" --check-only; then
            log_success "Python environment is ready"
        else
            log_warning "Python environment has issues"
        fi
    else
        log_warning "Python3 not available for dependency checks"
    fi
    
    # Check if services are running
    if docker-compose -f "$COMPOSE_FILE_HEALTH" ps | grep -q "Up"; then
        log_success "Services are running"
        show_status
    else
        log_warning "Services are not running"
    fi
}

# Main deployment flow
main() {
    cd "$PROJECT_ROOT"
    
    log_info "Pipeline Health Monitoring Deployment"
    log_info "======================================"
    log_info "Mode: $MODE"
    log_info "Profile: $PROFILE"
    log_info "Environment: $ENV_FILE"
    echo
    
    # Setup
    setup_environment
    check_dependencies
    setup_networks
    
    # Deploy
    deploy_services
    
    # Final status
    echo
    log_success "Deployment completed successfully!"
    show_status
    
    echo
    log_info "To view logs: $0 --logs"
    log_info "To check status: $0 --status"
    log_info "To stop services: $0 --stop"
}

# Parse command line arguments
MODE="$DEFAULT_MODE"
PROFILE="$DEFAULT_PROFILE"
ACTION="deploy"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --check-only)
            ACTION="check"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$MODE" != "standalone" && "$MODE" != "integrated" ]]; then
    log_error "Invalid mode: $MODE"
    usage
    exit 1
fi

if [[ "$PROFILE" != "minimal" && "$PROFILE" != "with-dashboard" && "$PROFILE" != "with-monitoring" && "$PROFILE" != "full" ]]; then
    log_error "Invalid profile: $PROFILE"
    usage
    exit 1
fi

# Execute action
case $ACTION in
    "deploy")
        main
        ;;
    "check")
        setup_environment
        run_health_checks
        ;;
    "stop")
        setup_environment
        stop_services
        ;;
    "restart")
        setup_environment
        restart_services
        ;;
    "logs")
        setup_environment
        show_logs
        ;;
    "status")
        setup_environment
        show_status
        ;;
    *)
        log_error "Invalid action: $ACTION"
        exit 1
        ;;
esac