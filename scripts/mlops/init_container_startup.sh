#!/bin/bash
# ==================================================================
# MLOps Init Container Startup Script
# ==================================================================
# This script runs as an init container to provision MinIO buckets
# before the main application services start
# ==================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
BUCKET_CONFIG_FILE="${BUCKET_CONFIG_FILE:-minio-buckets.yaml}"
ENVIRONMENT="${ENVIRONMENT:-production}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to wait for MinIO to be ready
wait_for_minio() {
    local endpoint="${MINIO_ENDPOINT:-localhost:9000}"
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for MinIO to be ready at ${endpoint}..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "http://${endpoint}/minio/health/ready" > /dev/null 2>&1; then
            log_success "MinIO is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: MinIO not ready yet, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "MinIO failed to become ready after $max_attempts attempts"
    return 1
}

# Function to check required environment variables
check_environment() {
    local required_vars=(
        "MINIO_ENDPOINT"
        "MINIO_ACCESS_KEY"
        "MINIO_SECRET_KEY"
    )
    
    log_info "Checking required environment variables..."
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_success "All required environment variables are set"
    return 0
}

# Function to validate configuration file
validate_config() {
    local config_file="${CONFIG_DIR}/${BUCKET_CONFIG_FILE}"
    
    log_info "Validating configuration file: ${config_file}"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        return 1
    fi
    
    # Check if it's valid YAML
    if ! python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
        log_error "Invalid YAML configuration file: $config_file"
        return 1
    fi
    
    log_success "Configuration file is valid"
    return 0
}

# Function to install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create temporary requirements file if it doesn't exist
    local temp_requirements="/tmp/bucket_provisioner_requirements.txt"
    cat > "$temp_requirements" << EOF
boto3>=1.26.0
minio>=7.1.0
PyYAML>=6.0
requests>=2.28.0
EOF
    
    pip install --no-cache-dir -r "$temp_requirements"
    rm -f "$temp_requirements"
    
    log_success "Dependencies installed successfully"
}

# Function to run bucket provisioning
run_bucket_provisioning() {
    local config_file="${CONFIG_DIR}/${BUCKET_CONFIG_FILE}"
    local provisioner_script="${SCRIPT_DIR}/bucket_provisioner.py"
    local output_file="/tmp/bucket_provision_results.json"
    
    log_info "Starting bucket provisioning for environment: ${ENVIRONMENT}"
    
    # Check if provisioner script exists
    if [[ ! -f "$provisioner_script" ]]; then
        log_error "Bucket provisioner script not found: $provisioner_script"
        return 1
    fi
    
    # Run the provisioner
    if python3 "$provisioner_script" \
        --config "$config_file" \
        --environment "$ENVIRONMENT" \
        --action provision \
        --output "$output_file" \
        --verbose; then
        
        log_success "Bucket provisioning completed successfully"
        
        # Display results summary
        if [[ -f "$output_file" ]]; then
            log_info "Provisioning results:"
            python3 -c "
import json
with open('$output_file') as f:
    results = json.load(f)
summary = results.get('summary', {})
print(f\"  Created: {summary.get('created', 0)} buckets\")
print(f\"  Updated: {summary.get('updated', 0)} buckets\")
print(f\"  Skipped: {summary.get('skipped', 0)} buckets\")
print(f\"  Failed:  {summary.get('failed', 0)} buckets\")
"
        fi
        
        return 0
    else
        log_error "Bucket provisioning failed"
        
        # Display error details if available
        if [[ -f "$output_file" ]]; then
            log_error "Error details:"
            python3 -c "
import json
try:
    with open('$output_file') as f:
        results = json.load(f)
    errors = results.get('details', {}).get('errors', [])
    for error in errors:
        print(f\"  - {error}\")
except:
    pass
"
        fi
        
        return 1
    fi
}

# Function to validate bucket accessibility
validate_buckets() {
    local config_file="${CONFIG_DIR}/${BUCKET_CONFIG_FILE}"
    local provisioner_script="${SCRIPT_DIR}/bucket_provisioner.py"
    local output_file="/tmp/bucket_validation_results.json"
    
    log_info "Validating bucket accessibility..."
    
    if python3 "$provisioner_script" \
        --config "$config_file" \
        --environment "$ENVIRONMENT" \
        --action validate \
        --output "$output_file" \
        --verbose; then
        
        log_success "Bucket validation completed"
        
        # Display validation results
        if [[ -f "$output_file" ]]; then
            python3 -c "
import json
with open('$output_file') as f:
    results = json.load(f)
summary = results.get('summary', {})
print(f\"  Accessible: {summary.get('healthy', 0)} buckets\")
print(f\"  Issues:     {summary.get('unhealthy', 0)} buckets\")
print(f\"  Success Rate: {summary.get('success_rate', 0):.1f}%\")
"
        fi
        
        return 0
    else
        log_error "Bucket validation failed"
        return 1
    fi
}

# Function to create health check file
create_health_check() {
    local health_file="/tmp/bucket_init_complete"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$health_file" << EOF
{
    "status": "completed",
    "timestamp": "$timestamp",
    "environment": "$ENVIRONMENT",
    "init_container": "mlops-bucket-provisioner"
}
EOF
    
    log_info "Created health check file: $health_file"
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Init container failed with exit code: $exit_code"
    fi
    
    # Create failure marker if needed
    if [[ $exit_code -ne 0 ]]; then
        echo "{\"status\": \"failed\", \"exit_code\": $exit_code}" > /tmp/bucket_init_failed
    fi
    
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    log_info "=== MLOps Bucket Provisioner Init Container ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Config file: ${BUCKET_CONFIG_FILE}"
    log_info "MinIO endpoint: ${MINIO_ENDPOINT:-not-set}"
    
    # Step 1: Check environment variables
    if ! check_environment; then
        log_error "Environment check failed"
        exit 1
    fi
    
    # Step 2: Install Python dependencies
    if ! install_dependencies; then
        log_error "Failed to install dependencies"
        exit 1
    fi
    
    # Step 3: Validate configuration
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 1
    fi
    
    # Step 4: Wait for MinIO to be ready
    if ! wait_for_minio; then
        log_error "MinIO readiness check failed"
        exit 1
    fi
    
    # Step 5: Run bucket provisioning
    if ! run_bucket_provisioning; then
        log_error "Bucket provisioning failed"
        exit 1
    fi
    
    # Step 6: Validate bucket accessibility
    if ! validate_buckets; then
        log_warn "Bucket validation had issues (continuing anyway)"
    fi
    
    # Step 7: Create health check file
    create_health_check
    
    log_success "=== Init container completed successfully ==="
    exit 0
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi