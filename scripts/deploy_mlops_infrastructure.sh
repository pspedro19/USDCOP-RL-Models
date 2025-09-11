#!/bin/bash
# ==================================================================
# MLOps Infrastructure Deployment Script
# ==================================================================
# Complete deployment script for USDCOP Trading System with
# automated bucket provisioning and infrastructure validation
# ==================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_ROOT}/config"
ENVIRONMENT="${ENVIRONMENT:-production}"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.mlops.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "         USDCOP Trading System MLOps Infrastructure"
    echo "              Automated Deployment Script"
    echo "=================================================================="
    echo -e "${NC}"
    echo "Environment: ${ENVIRONMENT}"
    echo "Project Root: ${PROJECT_ROOT}"
    echo "Compose File: ${COMPOSE_FILE}"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v pip >/dev/null 2>&1 || missing_tools+=("pip")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again"
        return 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    # Check configuration files
    local required_configs=(
        "${CONFIG_DIR}/minio-buckets.yaml"
        "${COMPOSE_FILE}"
    )
    
    for config in "${required_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            log_error "Required configuration file not found: $config"
            return 1
        fi
    done
    
    log_success "All prerequisites satisfied"
    return 0
}

# Setup environment
setup_environment() {
    log_step "Setting up environment..."
    
    # Create necessary directories
    local dirs=(
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/data"
        "${PROJECT_ROOT}/temp"
        "${PROJECT_ROOT}/reports"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    # Set default environment variables if not provided
    export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
    export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin123}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres123}"
    export AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-airflow123}"
    export MLFLOW_PASSWORD="${MLFLOW_PASSWORD:-mlflow123}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-grafana123}"
    export INFLUXDB_PASSWORD="${INFLUXDB_PASSWORD:-influxdb123}"
    
    # Generate Airflow secrets if not provided
    if [[ -z "${AIRFLOW_FERNET_KEY:-}" ]]; then
        export AIRFLOW_FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
        log_info "Generated Airflow Fernet key"
    fi
    
    if [[ -z "${AIRFLOW_SECRET_KEY:-}" ]]; then
        export AIRFLOW_SECRET_KEY=$(openssl rand -hex 32)
        log_info "Generated Airflow secret key"
    fi
    
    # Save environment configuration
    cat > "${PROJECT_ROOT}/.env" << EOF
# USDCOP Trading System Environment Configuration
# Generated on $(date)

ENVIRONMENT=${ENVIRONMENT}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
AIRFLOW_PASSWORD=${AIRFLOW_PASSWORD}
AIRFLOW_FERNET_KEY=${AIRFLOW_FERNET_KEY}
AIRFLOW_SECRET_KEY=${AIRFLOW_SECRET_KEY}
MLFLOW_PASSWORD=${MLFLOW_PASSWORD}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
INFLUXDB_PASSWORD=${INFLUXDB_PASSWORD}
LOG_LEVEL=INFO
EOF
    
    log_success "Environment setup completed"
}

# Install Python dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    # Install bucket provisioner dependencies
    pip install --upgrade pip
    pip install -r "${PROJECT_ROOT}/requirements.bucket-init.txt"
    
    log_success "Dependencies installed"
}

# Build Docker images
build_images() {
    log_step "Building Docker images..."
    
    # Build bucket init container
    log_info "Building bucket provisioner init container..."
    docker build \
        -f "${PROJECT_ROOT}/docker/Dockerfile.bucket-init" \
        -t usdcop/bucket-provisioner:latest \
        "${PROJECT_ROOT}"
    
    log_success "Docker images built successfully"
}

# Validate configuration
validate_configuration() {
    log_step "Validating bucket configuration..."
    
    # Validate YAML syntax
    python3 -c "
import yaml
import sys
try:
    with open('${CONFIG_DIR}/minio-buckets.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ YAML configuration is valid')
except Exception as e:
    print(f'✗ YAML validation failed: {e}')
    sys.exit(1)
"
    
    log_success "Configuration validation passed"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_step "Deploying MLOps infrastructure..."
    
    # Pull required images
    log_info "Pulling required Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Start infrastructure services
    log_info "Starting infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" up -d minio postgres redis
    
    # Wait for core services to be healthy
    log_info "Waiting for core services to be healthy..."
    wait_for_service "minio" "http://localhost:9000/minio/health/live" 60
    wait_for_service "postgres" "pg_isready -h localhost -p 5432 -U postgres" 60
    wait_for_service "redis" "redis-cli -h localhost -p 6379 ping" 60
    
    log_success "Core infrastructure services are running"
}

# Run bucket provisioning
provision_buckets() {
    log_step "Provisioning MinIO buckets..."
    
    # Run bucket provisioner init container
    log_info "Running bucket provisioner..."
    docker-compose -f "$COMPOSE_FILE" run --rm bucket-init
    
    # Validate bucket provisioning
    log_info "Validating bucket provisioning..."
    python3 "${PROJECT_ROOT}/scripts/mlops/bucket_provisioner.py" \
        --config "${CONFIG_DIR}/minio-buckets.yaml" \
        --environment "$ENVIRONMENT" \
        --action validate \
        --output "${PROJECT_ROOT}/reports/bucket_validation_$(date +%Y%m%d_%H%M%S).json" \
        --verbose
    
    log_success "Bucket provisioning completed"
}

# Deploy application services
deploy_application_services() {
    log_step "Deploying application services..."
    
    # Start MLflow
    log_info "Starting MLflow..."
    docker-compose -f "$COMPOSE_FILE" up -d mlflow
    wait_for_service "mlflow" "http://localhost:5000/health" 60
    
    # Start Airflow
    log_info "Starting Airflow services..."
    docker-compose -f "$COMPOSE_FILE" up -d airflow-webserver airflow-scheduler
    wait_for_service "airflow" "http://localhost:8080/health" 120
    
    # Start monitoring services
    log_info "Starting monitoring services..."
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana influxdb
    
    # Start application services
    log_info "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" up -d trading-dashboard nginx
    
    log_success "Application services deployed"
}

# Run comprehensive validation
run_validation() {
    log_step "Running comprehensive infrastructure validation..."
    
    # Run infrastructure validator
    docker-compose -f "$COMPOSE_FILE" run --rm infrastructure-validator
    
    # Run detailed bucket validation
    log_info "Running detailed bucket validation..."
    python3 "${PROJECT_ROOT}/scripts/mlops/bucket_validator.py" \
        --config "${CONFIG_DIR}/minio-buckets.yaml" \
        --environment "$ENVIRONMENT" \
        --action validate \
        --output "${PROJECT_ROOT}/reports/comprehensive_validation_$(date +%Y%m%d_%H%M%S).html" \
        --format html \
        --verbose
    
    log_success "Comprehensive validation completed"
}

# Wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local health_check="$2"
    local timeout="$3"
    local elapsed=0
    local interval=5
    
    log_info "Waiting for $service_name to be ready..."
    
    while [[ $elapsed -lt $timeout ]]; do
        if eval "$health_check" >/dev/null 2>&1; then
            log_success "$service_name is ready (${elapsed}s)"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_info "Waiting for $service_name... (${elapsed}s/${timeout}s)"
    done
    
    log_error "$service_name failed to become ready within ${timeout}s"
    return 1
}

# Display service URLs
display_service_urls() {
    log_step "Deployment completed successfully!"
    
    echo ""
    echo -e "${GREEN}Service URLs:${NC}"
    echo "  📊 Trading Dashboard:  http://localhost:3000"
    echo "  🔄 Airflow:           http://localhost:8080"
    echo "  📈 MLflow:            http://localhost:5000"
    echo "  📊 Grafana:           http://localhost:3100"
    echo "  📈 Prometheus:        http://localhost:9090"
    echo "  💾 MinIO Console:     http://localhost:9001"
    echo "  🌐 Nginx:             http://localhost"
    echo ""
    echo -e "${GREEN}Credentials:${NC}"
    echo "  MinIO:     Access Key: ${MINIO_ACCESS_KEY}, Secret Key: ${MINIO_SECRET_KEY}"
    echo "  Grafana:   admin / ${GRAFANA_PASSWORD}"
    echo "  Airflow:   admin / admin"
    echo ""
    echo -e "${GREEN}Validation Reports:${NC}"
    echo "  Reports saved in: ${PROJECT_ROOT}/reports/"
    echo ""
    echo -e "${BLUE}To monitor the system:${NC}"
    echo "  docker-compose -f docker-compose.mlops.yml logs -f"
    echo ""
    echo -e "${BLUE}To validate buckets:${NC}"
    echo "  python3 scripts/mlops/bucket_validator.py --config config/minio-buckets.yaml --environment $ENVIRONMENT --action validate"
    echo ""
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code: $exit_code"
        log_info "Cleaning up partial deployment..."
        
        # Stop services
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
        
        log_info "Cleanup completed"
    fi
    
    exit $exit_code
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy USDCOP Trading System MLOps Infrastructure"
    echo ""
    echo "OPTIONS:"
    echo "  -e, --environment ENVIRONMENT    Set deployment environment (default: production)"
    echo "  -c, --config-only                Only validate configuration, don't deploy"
    echo "  -b, --buckets-only               Only provision buckets, don't deploy services"
    echo "  -v, --validate-only              Only run validation, don't deploy"
    echo "  -s, --skip-build                 Skip Docker image building"
    echo "  -f, --force                      Force deployment even if services are running"
    echo "  -h, --help                       Show this help message"
    echo ""
    echo "ENVIRONMENT VARIABLES:"
    echo "  ENVIRONMENT                      Deployment environment (production, staging, development)"
    echo "  MINIO_ACCESS_KEY                 MinIO access key (default: minioadmin)"
    echo "  MINIO_SECRET_KEY                 MinIO secret key (default: minioadmin123)"
    echo "  POSTGRES_PASSWORD                PostgreSQL password"
    echo "  AIRFLOW_PASSWORD                 Airflow database password"
    echo "  GRAFANA_PASSWORD                 Grafana admin password"
    echo ""
    echo "EXAMPLES:"
    echo "  # Full deployment"
    echo "  $0"
    echo ""
    echo "  # Deploy to staging environment"
    echo "  $0 --environment staging"
    echo ""
    echo "  # Only provision buckets"
    echo "  $0 --buckets-only"
    echo ""
    echo "  # Validate configuration only"
    echo "  $0 --config-only"
    echo ""
}

# Parse command line arguments
CONFIG_ONLY=false
BUCKETS_ONLY=false
VALIDATE_ONLY=false
SKIP_BUILD=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--config-only)
            CONFIG_ONLY=true
            shift
            ;;
        -b|--buckets-only)
            BUCKETS_ONLY=true
            shift
            ;;
        -v|--validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    print_banner
    
    # Always check prerequisites and validate configuration
    check_prerequisites
    validate_configuration
    
    if [[ "$CONFIG_ONLY" == true ]]; then
        log_success "Configuration validation completed successfully"
        exit 0
    fi
    
    setup_environment
    install_dependencies
    
    if [[ "$SKIP_BUILD" != true ]]; then
        build_images
    fi
    
    if [[ "$VALIDATE_ONLY" == true ]]; then
        # Only run validation without deploying
        deploy_infrastructure
        provision_buckets
        run_validation
        exit 0
    fi
    
    # Check if services are already running
    if [[ "$FORCE" != true ]] && docker-compose -f "$COMPOSE_FILE" ps -q | grep -q .; then
        log_warn "Services are already running. Use --force to redeploy."
        log_info "Current running services:"
        docker-compose -f "$COMPOSE_FILE" ps
        exit 0
    fi
    
    # Full deployment
    deploy_infrastructure
    provision_buckets
    
    if [[ "$BUCKETS_ONLY" != true ]]; then
        deploy_application_services
        run_validation
        display_service_urls
    else
        log_success "Bucket provisioning completed successfully"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi