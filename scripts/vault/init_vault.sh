#!/bin/bash
# =============================================================================
# USD/COP Trading System - Vault Initialization Script
# =============================================================================
#
# This script initializes HashiCorp Vault for the USDCOP Trading System:
# - Enables KV v2 secrets engine
# - Enables AppRole authentication
# - Creates policies for trading and airflow services
# - Creates AppRoles with appropriate policies
# - Optionally migrates secrets from .env file
#
# Prerequisites:
# - Vault server running and unsealed
# - VAULT_ADDR environment variable set
# - VAULT_TOKEN with admin privileges (or root token for initial setup)
#
# Usage:
#   ./scripts/vault/init_vault.sh                    # Initialize Vault
#   ./scripts/vault/init_vault.sh --migrate-env      # Initialize and migrate .env
#   ./scripts/vault/init_vault.sh --help             # Show help
#
# Author: Pedro @ Lean Tech Solutions
# Version: 1.0.0
# Date: 2026-01-17
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICIES_DIR="${PROJECT_ROOT}/config/vault/policies"

# Default values
VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"
MIGRATE_ENV=false
ENV_FILE="${PROJECT_ROOT}/.env"
SECRETS_MOUNT_PATH="secret"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

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

check_vault_connection() {
    log_info "Checking Vault connection at ${VAULT_ADDR}..."

    if ! vault status > /dev/null 2>&1; then
        log_error "Cannot connect to Vault at ${VAULT_ADDR}"
        log_error "Make sure Vault is running and VAULT_ADDR is set correctly"
        exit 1
    fi

    # Check if Vault is sealed
    if vault status 2>&1 | grep -q "Sealed.*true"; then
        log_error "Vault is sealed. Please unseal Vault first."
        exit 1
    fi

    log_success "Vault is running and unsealed"
}

check_vault_auth() {
    log_info "Checking Vault authentication..."

    if [ -z "${VAULT_TOKEN}" ]; then
        log_warning "VAULT_TOKEN not set. Trying to use cached token..."
    fi

    if ! vault token lookup > /dev/null 2>&1; then
        log_error "Not authenticated to Vault. Set VAULT_TOKEN environment variable."
        exit 1
    fi

    log_success "Vault authentication verified"
}

# -----------------------------------------------------------------------------
# Secrets Engine Setup
# -----------------------------------------------------------------------------

enable_kv_secrets_engine() {
    log_info "Enabling KV v2 secrets engine at '${SECRETS_MOUNT_PATH}'..."

    # Check if already enabled
    if vault secrets list | grep -q "^${SECRETS_MOUNT_PATH}/"; then
        log_warning "Secrets engine already enabled at '${SECRETS_MOUNT_PATH}'"

        # Check if it's KV v2
        if vault secrets list -detailed | grep "${SECRETS_MOUNT_PATH}/" | grep -q "version:2"; then
            log_success "KV v2 secrets engine already configured"
            return 0
        else
            log_warning "Existing secrets engine is not KV v2. Skipping to avoid data loss."
            log_warning "If you want to upgrade, manually disable and re-enable the secrets engine."
            return 0
        fi
    fi

    # Enable KV v2
    vault secrets enable -path="${SECRETS_MOUNT_PATH}" -version=2 kv

    log_success "KV v2 secrets engine enabled at '${SECRETS_MOUNT_PATH}'"
}

# -----------------------------------------------------------------------------
# Authentication Setup
# -----------------------------------------------------------------------------

enable_approle_auth() {
    log_info "Enabling AppRole authentication..."

    # Check if already enabled
    if vault auth list | grep -q "^approle/"; then
        log_warning "AppRole auth already enabled"
        return 0
    fi

    vault auth enable approle

    log_success "AppRole authentication enabled"
}

# -----------------------------------------------------------------------------
# Policy Management
# -----------------------------------------------------------------------------

create_policies() {
    log_info "Creating Vault policies..."

    # Trading service policy
    if [ -f "${POLICIES_DIR}/trading-policy.hcl" ]; then
        log_info "Creating 'trading-service' policy..."
        vault policy write trading-service "${POLICIES_DIR}/trading-policy.hcl"
        log_success "Policy 'trading-service' created"
    else
        log_error "Policy file not found: ${POLICIES_DIR}/trading-policy.hcl"
        exit 1
    fi

    # Airflow service policy
    if [ -f "${POLICIES_DIR}/airflow-policy.hcl" ]; then
        log_info "Creating 'airflow-service' policy..."
        vault policy write airflow-service "${POLICIES_DIR}/airflow-policy.hcl"
        log_success "Policy 'airflow-service' created"
    else
        log_error "Policy file not found: ${POLICIES_DIR}/airflow-policy.hcl"
        exit 1
    fi

    # Create a minimal policy for Airflow DAG runners (child tokens)
    log_info "Creating 'airflow-dag-runner' policy..."
    vault policy write airflow-dag-runner - <<EOF
# Minimal policy for Airflow DAG runner tokens
# Read-only access to trading secrets for individual DAG runs

path "secret/data/trading/twelvedata/*" {
  capabilities = ["read"]
}

path "secret/data/trading/database" {
  capabilities = ["read"]
}

path "secret/data/trading/redis" {
  capabilities = ["read"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
EOF
    log_success "Policy 'airflow-dag-runner' created"

    log_success "All policies created successfully"
}

# -----------------------------------------------------------------------------
# AppRole Creation
# -----------------------------------------------------------------------------

create_approles() {
    log_info "Creating AppRoles..."

    # Trading service AppRole
    log_info "Creating 'trading-service' AppRole..."
    vault write auth/approle/role/trading-service \
        token_policies="trading-service" \
        token_ttl=1h \
        token_max_ttl=4h \
        secret_id_ttl=24h \
        secret_id_num_uses=0

    # Get and display role-id (stored for services to use)
    TRADING_ROLE_ID=$(vault read -field=role_id auth/approle/role/trading-service/role-id)
    log_success "Trading service AppRole created"
    log_info "Trading Role ID: ${TRADING_ROLE_ID}"

    # Generate secret-id for trading service
    log_info "Generating secret-id for trading service..."
    TRADING_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/trading-service/secret-id)
    log_success "Trading Secret ID generated (save securely!)"

    # Airflow service AppRole
    log_info "Creating 'airflow-service' AppRole..."
    vault write auth/approle/role/airflow-service \
        token_policies="airflow-service" \
        token_ttl=1h \
        token_max_ttl=8h \
        secret_id_ttl=24h \
        secret_id_num_uses=0

    # Get and display role-id
    AIRFLOW_ROLE_ID=$(vault read -field=role_id auth/approle/role/airflow-service/role-id)
    log_success "Airflow service AppRole created"
    log_info "Airflow Role ID: ${AIRFLOW_ROLE_ID}"

    # Generate secret-id for airflow service
    log_info "Generating secret-id for airflow service..."
    AIRFLOW_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/airflow-service/secret-id)
    log_success "Airflow Secret ID generated (save securely!)"

    # Save credentials to a secure file
    CREDS_FILE="${PROJECT_ROOT}/secrets/vault_approle_credentials.txt"
    mkdir -p "$(dirname "${CREDS_FILE}")"

    cat > "${CREDS_FILE}" << EOF
# Vault AppRole Credentials - KEEP SECURE!
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# WARNING: These credentials provide access to sensitive secrets

# Trading Service
VAULT_ROLE_ID_TRADING=${TRADING_ROLE_ID}
VAULT_SECRET_ID_TRADING=${TRADING_SECRET_ID}

# Airflow Service
VAULT_ROLE_ID_AIRFLOW=${AIRFLOW_ROLE_ID}
VAULT_SECRET_ID_AIRFLOW=${AIRFLOW_SECRET_ID}

# Example usage in .env or docker-compose:
# VAULT_ROLE_ID=\${VAULT_ROLE_ID_TRADING}
# VAULT_SECRET_ID=\${VAULT_SECRET_ID_TRADING}
EOF

    chmod 600 "${CREDS_FILE}"
    log_success "AppRole credentials saved to: ${CREDS_FILE}"
    log_warning "Store these credentials securely and never commit to git!"
}

# -----------------------------------------------------------------------------
# Secret Migration from .env
# -----------------------------------------------------------------------------

migrate_env_secrets() {
    log_info "Migrating secrets from ${ENV_FILE}..."

    if [ ! -f "${ENV_FILE}" ]; then
        log_error "Environment file not found: ${ENV_FILE}"
        log_warning "Skipping migration. You'll need to add secrets manually."
        return 1
    fi

    # Source the env file safely
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}" 2>/dev/null || true
    set +a

    # Migrate database credentials
    log_info "Migrating database credentials..."
    if [ -n "${POSTGRES_PASSWORD:-}" ]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/database" \
            user="${POSTGRES_USER:-trading_user}" \
            password="${POSTGRES_PASSWORD}" \
            host="${POSTGRES_HOST:-localhost}" \
            port="${POSTGRES_PORT:-5432}" \
            database="${POSTGRES_DB:-usdcop_trading}"
        log_success "Database credentials stored"
    else
        log_warning "POSTGRES_PASSWORD not found in .env, skipping database secrets"
    fi

    # Migrate Redis credentials
    log_info "Migrating Redis credentials..."
    if [ -n "${REDIS_PASSWORD:-}" ]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/redis" \
            password="${REDIS_PASSWORD}" \
            host="${REDIS_HOST:-localhost}" \
            port="${REDIS_PORT:-6379}"
        log_success "Redis credentials stored"
    else
        log_warning "REDIS_PASSWORD not found in .env, skipping Redis secrets"
    fi

    # Migrate MinIO credentials
    log_info "Migrating MinIO credentials..."
    if [ -n "${MINIO_SECRET_KEY:-}" ]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/minio" \
            access_key="${MINIO_ACCESS_KEY:-minioadmin}" \
            secret_key="${MINIO_SECRET_KEY}"
        log_success "MinIO credentials stored"
    else
        log_warning "MINIO_SECRET_KEY not found in .env, skipping MinIO secrets"
    fi

    # Migrate JWT secret
    log_info "Migrating JWT secret..."
    if [ -n "${JWT_SECRET:-}" ]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/jwt" \
            secret="${JWT_SECRET}"
        log_success "JWT secret stored"
    else
        log_warning "JWT_SECRET not found in .env, skipping JWT secret"
    fi

    # Migrate Airflow credentials
    log_info "Migrating Airflow credentials..."
    if [ -n "${AIRFLOW_FERNET_KEY:-}" ] || [ -n "${AIRFLOW_SECRET_KEY:-}" ]; then
        vault kv put "${SECRETS_MOUNT_PATH}/airflow" \
            fernet_key="${AIRFLOW_FERNET_KEY:-}" \
            secret_key="${AIRFLOW_SECRET_KEY:-}" \
            user="${AIRFLOW_USER:-admin}" \
            password="${AIRFLOW_PASSWORD:-}"
        log_success "Airflow credentials stored"
    else
        log_warning "Airflow keys not found in .env, skipping Airflow secrets"
    fi

    # Migrate TwelveData API keys
    log_info "Migrating TwelveData API keys..."

    # Legacy keys (TWELVEDATA_API_KEY_1 through TWELVEDATA_API_KEY_8)
    LEGACY_KEYS=""
    for i in $(seq 1 8); do
        VAR_NAME="TWELVEDATA_API_KEY_${i}"
        VAL="${!VAR_NAME:-}"
        if [ -n "${VAL}" ] && [[ "${VAL}" != YOUR_* ]] && [[ "${VAL}" != CHANGE_ME* ]]; then
            LEGACY_KEYS="${LEGACY_KEYS} api_key_${i}=${VAL}"
        fi
    done

    if [ -n "${LEGACY_KEYS}" ]; then
        # shellcheck disable=SC2086
        vault kv put "${SECRETS_MOUNT_PATH}/trading/twelvedata" ${LEGACY_KEYS}
        log_success "Legacy TwelveData keys stored"
    else
        log_warning "No valid legacy TwelveData keys found"
    fi

    # Group 1 keys (API_KEY_G1_1 through API_KEY_G1_8)
    G1_KEYS=""
    for i in $(seq 1 8); do
        VAR_NAME="API_KEY_G1_${i}"
        VAL="${!VAR_NAME:-}"
        if [ -n "${VAL}" ] && [[ "${VAL}" != YOUR_* ]] && [[ "${VAL}" != CHANGE_ME* ]]; then
            G1_KEYS="${G1_KEYS} api_key_${i}=${VAL}"
        fi
    done

    if [ -n "${G1_KEYS}" ]; then
        # shellcheck disable=SC2086
        vault kv put "${SECRETS_MOUNT_PATH}/trading/twelvedata/g1" ${G1_KEYS}
        log_success "TwelveData Group 1 keys stored"
    else
        log_warning "No valid TwelveData Group 1 keys found"
    fi

    # Group 2 keys (API_KEY_G2_1 through API_KEY_G2_8)
    G2_KEYS=""
    for i in $(seq 1 8); do
        VAR_NAME="API_KEY_G2_${i}"
        VAL="${!VAR_NAME:-}"
        if [ -n "${VAL}" ] && [[ "${VAL}" != YOUR_* ]] && [[ "${VAL}" != CHANGE_ME* ]]; then
            G2_KEYS="${G2_KEYS} api_key_${i}=${VAL}"
        fi
    done

    if [ -n "${G2_KEYS}" ]; then
        # shellcheck disable=SC2086
        vault kv put "${SECRETS_MOUNT_PATH}/trading/twelvedata/g2" ${G2_KEYS}
        log_success "TwelveData Group 2 keys stored"
    else
        log_warning "No valid TwelveData Group 2 keys found"
    fi

    # Migrate LLM API keys
    log_info "Migrating LLM API keys..."
    if [ -n "${DEEPSEEK_API_KEY:-}" ] && [[ "${DEEPSEEK_API_KEY}" != YOUR_* ]]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/llm/deepseek" \
            api_key="${DEEPSEEK_API_KEY}"
        log_success "DeepSeek API key stored"
    fi

    if [ -n "${ANTHROPIC_API_KEY:-}" ] && [[ "${ANTHROPIC_API_KEY}" != YOUR_* ]]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/llm/anthropic" \
            api_key="${ANTHROPIC_API_KEY}"
        log_success "Anthropic API key stored"
    fi

    # Migrate notification credentials
    log_info "Migrating notification credentials..."
    if [ -n "${SLACK_WEBHOOK_URL:-}" ] && [[ "${SLACK_WEBHOOK_URL}" != *YOUR* ]]; then
        vault kv put "${SECRETS_MOUNT_PATH}/trading/notifications" \
            slack_webhook="${SLACK_WEBHOOK_URL}" \
            email_smtp_server="${EMAIL_SMTP_SERVER:-}" \
            email_smtp_user="${EMAIL_SMTP_USER:-}" \
            email_smtp_password="${EMAIL_SMTP_PASSWORD:-}"
        log_success "Notification credentials stored"
    else
        log_warning "No valid notification credentials found"
    fi

    log_success "Secret migration completed"
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

show_help() {
    cat << EOF
USD/COP Trading System - Vault Initialization Script

Usage: $(basename "$0") [OPTIONS]

Options:
    --migrate-env       Migrate secrets from .env file to Vault
    --env-file FILE     Path to .env file (default: PROJECT_ROOT/.env)
    --mount-path PATH   KV secrets mount path (default: secret)
    --help              Show this help message

Environment Variables:
    VAULT_ADDR          Vault server URL (default: http://localhost:8200)
    VAULT_TOKEN         Vault authentication token (required)

Examples:
    # Initialize Vault with default settings
    ./scripts/vault/init_vault.sh

    # Initialize and migrate secrets from .env
    ./scripts/vault/init_vault.sh --migrate-env

    # Use custom mount path
    ./scripts/vault/init_vault.sh --mount-path trading-secrets

EOF
}

main() {
    echo "=============================================================="
    echo " USD/COP Trading System - Vault Initialization"
    echo "=============================================================="
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --migrate-env)
                MIGRATE_ENV=true
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --mount-path)
                SECRETS_MOUNT_PATH="$2"
                shift 2
                ;;
            --help)
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

    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Vault address: ${VAULT_ADDR}"
    log_info "Secrets mount path: ${SECRETS_MOUNT_PATH}"
    echo ""

    # Pre-flight checks
    check_vault_connection
    check_vault_auth
    echo ""

    # Enable secrets engine
    enable_kv_secrets_engine
    echo ""

    # Enable authentication
    enable_approle_auth
    echo ""

    # Create policies
    create_policies
    echo ""

    # Create AppRoles
    create_approles
    echo ""

    # Optionally migrate secrets from .env
    if [ "${MIGRATE_ENV}" = true ]; then
        echo ""
        migrate_env_secrets
    fi

    echo ""
    echo "=============================================================="
    echo " Vault Initialization Complete!"
    echo "=============================================================="
    echo ""
    log_info "Next steps:"
    echo "  1. Store AppRole credentials securely (see secrets/vault_approle_credentials.txt)"
    echo "  2. Update docker-compose.yml or .env with VAULT_ROLE_ID and VAULT_SECRET_ID"
    echo "  3. If you didn't use --migrate-env, manually add secrets to Vault"
    echo "  4. Test with: vault kv get ${SECRETS_MOUNT_PATH}/trading/database"
    echo ""
    log_warning "Remember to:"
    echo "  - Revoke the root/admin token after setup"
    echo "  - Set up Vault auto-unseal in production"
    echo "  - Enable audit logging: vault audit enable file file_path=/var/log/vault/audit.log"
    echo ""
}

# Run main function
main "$@"
