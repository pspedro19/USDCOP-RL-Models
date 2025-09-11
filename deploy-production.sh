#!/bin/bash

# =====================================================
# USDCOP Trading System - Production Deployment Script
# =====================================================
# Comprehensive deployment script for production environment
# Features:
# - Environment validation
# - Dependency checks
# - Service orchestration
# - Health monitoring
# - Rollback capabilities
# =====================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="usdcop-trading-production"
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"
TIMEOUT=300
HEALTH_CHECK_INTERVAL=10
MAX_RETRIES=30

# Logging
LOG_FILE="deployment_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_banner() {
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "                 USDCOP Trading System"
    echo "               Production Deployment v1.0"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

show_help() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  start      Start all services"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  status     Show service status"
    echo "  logs       Show service logs"
    echo "  backup     Create system backup"
    echo "  restore    Restore from backup"
    echo "  cleanup    Clean up unused resources"
    echo "  validate   Validate environment and configuration"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
    echo "  -f, --force    Force operation (skip confirmations)"
    echo "  --no-build     Skip building images"
    echo "  --dev          Use development configuration"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOYMENT_ENV    Deployment environment (production|staging|development)"
    echo "  BACKUP_DIR        Directory for backups (default: ./backups)"
    echo "  DATA_DIR          Directory for data persistence (default: ./data)"
    echo ""
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - JSON parsing will be limited"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install missing dependencies and try again"
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

check_docker_daemon() {
    log_info "Checking Docker daemon..."
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        log_error "Please start Docker and try again"
        exit 1
    fi
    
    log_success "Docker daemon is running"
}

validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file $ENV_FILE not found"
        log_error "Please create the environment file from .env.production.example"
        exit 1
    fi
    
    # Check if docker-compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file $COMPOSE_FILE not found"
        exit 1
    fi
    
    # Validate environment variables
    source "$ENV_FILE"
    
    local required_vars=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "MINIO_ROOT_PASSWORD"
        "AIRFLOW_FERNET_KEY"
        "AIRFLOW_SECRET_KEY"
        "GRAFANA_ADMIN_PASSWORD"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    log_success "Environment configuration validated"
}

create_directories() {
    log_info "Creating required directories..."
    
    # Load environment to get directory paths
    source "$ENV_FILE"
    
    local dirs=(
        "${DATA_DIR:-./data}/postgres"
        "${DATA_DIR:-./data}/redis"
        "${DATA_DIR:-./data}/minio"
        "${DATA_DIR:-./data}/prometheus"
        "${DATA_DIR:-./data}/grafana"
        "${DATA_DIR:-./data}/loki"
        "${DATA_DIR:-./data}/consul"
        "${DATA_DIR:-./data}/app"
        "${DATA_DIR:-./data}/models"
        "${LOG_DIR:-./logs}/app"
        "${LOG_DIR:-./logs}/nginx"
        "${BACKUP_DIR:-./backups}/postgres"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set appropriate permissions
    chmod 755 "${DATA_DIR:-./data}"
    chmod 755 "${LOG_DIR:-./logs}"
    chmod 755 "${BACKUP_DIR:-./backups}"
    
    log_success "Directories created and configured"
}

wait_for_service() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts="$3"
    
    log_info "Waiting for $service_name to be healthy..."
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep $HEALTH_CHECK_INTERVAL
        ((attempt++))
    done
    
    log_error "$service_name failed to become healthy after $max_attempts attempts"
    return 1
}

check_service_health() {
    log_info "Performing health checks..."
    
    local services=(
        "PostgreSQL:http://localhost:5432"
        "Redis:http://localhost:6379"
        "MinIO:http://localhost:9000/minio/health/live"
        "Trading App:http://localhost:8000/health"
        "Dashboard:http://localhost:3000"
        "Prometheus:http://localhost:9090/-/healthy"
        "Grafana:http://localhost:3001/api/health"
    )
    
    local failed_services=()
    
    for service_info in "${services[@]}"; do
        IFS=':' read -ra parts <<< "$service_info"
        local service_name="${parts[0]}"
        local health_url="${parts[1]}:${parts[2]}"
        
        if ! wait_for_service "$service_name" "$health_url" 5; then
            failed_services+=("$service_name")
        fi
    done
    
    if [ ${#failed_services[@]} -ne 0 ]; then
        log_warning "Some services failed health checks: ${failed_services[*]}"
        return 1
    fi
    
    log_success "All services passed health checks"
    return 0
}

start_services() {
    log_info "Starting USDCOP Trading System..."
    
    # Validate environment first
    validate_environment
    create_directories
    
    # Pull latest images if not skipping build
    if [[ "$SKIP_BUILD" != "true" ]]; then
        log_info "Pulling latest images..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    fi
    
    # Start infrastructure services first
    log_info "Starting infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        consul postgres redis minio
    
    # Wait for infrastructure to be ready
    sleep 30
    
    # Start initialization services
    log_info "Running initialization services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up \
        postgres-init minio-init
    
    # Start application services
    log_info "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        airflow-init
    
    # Wait for Airflow init to complete
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f airflow-init
    
    # Start remaining services
    log_info "Starting remaining services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    # Wait for services to be ready
    sleep 60
    
    # Perform health checks
    if check_service_health; then
        log_success "USDCOP Trading System started successfully!"
        show_service_urls
    else
        log_warning "System started but some services may not be fully ready"
        log_info "Check logs with: $0 logs"
    fi
}

stop_services() {
    log_info "Stopping USDCOP Trading System..."
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    
    log_success "USDCOP Trading System stopped"
}

restart_services() {
    log_info "Restarting USDCOP Trading System..."
    
    stop_services
    sleep 10
    start_services
}

show_status() {
    log_info "Service Status:"
    echo ""
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    
    echo ""
    log_info "Docker System Info:"
    docker system df
}

show_logs() {
    local service="$1"
    
    if [[ -n "$service" ]]; then
        log_info "Showing logs for service: $service"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "$service"
    else
        log_info "Showing logs for all services:"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f
    fi
}

show_service_urls() {
    source "$ENV_FILE"
    
    echo ""
    log_success "Service URLs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Trading Dashboard:    http://localhost:${DASHBOARD_PORT:-3000}"
    echo "📊 Grafana Monitoring:   http://localhost:${GRAFANA_PORT:-3001}"
    echo "🔄 Airflow:              http://localhost:${AIRFLOW_PORT:-8081}"
    echo "🗄️  PgAdmin:              http://localhost:${PGADMIN_PORT:-5050}"
    echo "📈 Prometheus:           http://localhost:${PROMETHEUS_PORT:-9090}"
    echo "💾 MinIO Console:        http://localhost:${MINIO_CONSOLE_PORT:-9001}"
    echo "🔍 Jaeger Tracing:       http://localhost:16686"
    echo "🏥 Service Discovery:    http://localhost:${CONSUL_HTTP_PORT:-8500}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Default Credentials:"
    echo "   Grafana:    admin / ${GRAFANA_ADMIN_PASSWORD}"
    echo "   Airflow:    ${AIRFLOW_ADMIN_USER} / ${AIRFLOW_ADMIN_PASSWORD}"
    echo "   PgAdmin:    ${PGADMIN_EMAIL} / ${PGADMIN_PASSWORD}"
    echo "   MinIO:      ${MINIO_ROOT_USER} / ${MINIO_ROOT_PASSWORD}"
}

create_backup() {
    log_info "Creating system backup..."
    
    source "$ENV_FILE"
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${BACKUP_DIR:-./backups}/backup_${backup_timestamp}"
    
    mkdir -p "$backup_path"
    
    # Backup PostgreSQL
    log_info "Backing up PostgreSQL databases..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres \
        pg_dumpall -U postgres > "$backup_path/postgres_backup.sql"
    
    # Backup configuration files
    log_info "Backing up configuration files..."
    cp "$ENV_FILE" "$backup_path/"
    cp "$COMPOSE_FILE" "$backup_path/"
    cp -r config/ "$backup_path/" 2>/dev/null || true
    
    # Create backup manifest
    cat > "$backup_path/manifest.json" << EOF
{
    "backup_timestamp": "$backup_timestamp",
    "system_version": "1.0.0",
    "services": [
        "postgres",
        "redis",
        "minio",
        "airflow",
        "trading-app",
        "trading-dashboard"
    ],
    "notes": "Production backup created by deployment script"
}
EOF
    
    # Compress backup
    tar -czf "${backup_path}.tar.gz" -C "${BACKUP_DIR:-./backups}" "backup_${backup_timestamp}"
    rm -rf "$backup_path"
    
    log_success "Backup created: ${backup_path}.tar.gz"
}

cleanup_system() {
    log_info "Cleaning up system resources..."
    
    # Clean up Docker resources
    docker system prune -f
    docker volume prune -f
    docker network prune -f
    
    # Clean up old logs
    find "${LOG_DIR:-./logs}" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clean up old backups (keep last 5)
    find "${BACKUP_DIR:-./backups}" -name "backup_*.tar.gz" -type f | \
        sort -r | tail -n +6 | xargs rm -f 2>/dev/null || true
    
    log_success "System cleanup completed"
}

# Parse command line arguments
COMMAND=""
VERBOSE=false
FORCE=false
SKIP_BUILD=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            ENV_FILE=".env.dev"
            COMPOSE_FILE="docker-compose.yml"
            shift
            ;;
        start|stop|restart|status|logs|backup|restore|cleanup|validate)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_banner
    
    # Check if running as root (not recommended for production)
    if [[ $EUID -eq 0 ]] && [[ "$FORCE" != "true" ]]; then
        log_warning "Running as root is not recommended for production"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check dependencies
    check_dependencies
    check_docker_daemon
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Execute command
    case "$COMMAND" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        backup)
            create_backup
            ;;
        cleanup)
            cleanup_system
            ;;
        validate)
            validate_environment
            log_success "Environment validation completed"
            ;;
        "")
            log_error "No command specified"
            show_help
            exit 1
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Execute main function
main "$@"