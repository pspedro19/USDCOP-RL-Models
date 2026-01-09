#!/bin/bash
# =====================================================
# USDCOP Trading System - Database Initialization Script
# =====================================================
# File: docker/init-db.sh
# Purpose: Orchestrates database schema initialization on container startup
# Compatible with: timescale/timescaledb:latest-pg15, postgres:15
# =====================================================
# Usage:
#   - Mounted to /docker-entrypoint-initdb.d/ in PostgreSQL container
#   - Runs automatically on FIRST container startup (empty data volume)
#   - Idempotent: safe to run multiple times
# =====================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPTS_DIR="/docker-entrypoint-initdb.d"
LOG_PREFIX="[USDCOP-DB-INIT]"
MARKER_TABLE="audit.init_log"

# Colors for output (if terminal supports it)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}${LOG_PREFIX}${NC} [INFO] $1"
}

log_success() {
    echo -e "${GREEN}${LOG_PREFIX}${NC} [SUCCESS] $1"
}

log_warn() {
    echo -e "${YELLOW}${LOG_PREFIX}${NC} [WARN] $1"
}

log_error() {
    echo -e "${RED}${LOG_PREFIX}${NC} [ERROR] $1"
}

# Check if PostgreSQL is ready
wait_for_postgres() {
    local max_attempts=30
    local attempt=1

    log_info "Waiting for PostgreSQL to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if pg_isready -h localhost -U "${POSTGRES_USER:-postgres}" > /dev/null 2>&1; then
            log_success "PostgreSQL is ready!"
            return 0
        fi
        log_info "Attempt $attempt/$max_attempts - PostgreSQL not ready, waiting..."
        sleep 2
        ((attempt++))
    done

    log_error "PostgreSQL did not become ready in time"
    return 1
}

# Check if database is already initialized
is_initialized() {
    # Check if the marker table exists
    local result
    result=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc \
        "SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'audit' AND table_name = 'init_log'
        );" 2>/dev/null || echo "f")

    if [ "$result" = "t" ]; then
        return 0  # Already initialized
    else
        return 1  # Not initialized
    fi
}

# Check if a specific table exists
table_exists() {
    local schema=$1
    local table=$2
    local result
    result=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc \
        "SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = '$schema' AND table_name = '$table'
        );" 2>/dev/null || echo "f")

    [ "$result" = "t" ]
}

# Execute a SQL file with error handling
execute_sql_file() {
    local file=$1
    local filename=$(basename "$file")
    local start_time=$(date +%s%3N)

    log_info "Executing: $filename"

    # Execute the SQL file
    if psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -f "$file" 2>&1; then
        local end_time=$(date +%s%3N)
        local duration=$((end_time - start_time))
        log_success "$filename completed in ${duration}ms"

        # Log to audit table if it exists
        if table_exists "audit" "init_log"; then
            psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -c \
                "INSERT INTO audit.init_log (script_name, duration_ms, message)
                 VALUES ('$filename', $duration, 'Executed successfully');" 2>/dev/null || true
        fi
        return 0
    else
        local error_msg="Failed to execute $filename"
        log_error "$error_msg"

        # Log error to audit table if it exists
        if table_exists "audit" "init_log"; then
            psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -c \
                "INSERT INTO audit.init_log (script_name, success, message, error_detail)
                 VALUES ('$filename', FALSE, 'Execution failed', '$error_msg');" 2>/dev/null || true
        fi
        return 1
    fi
}

# Execute a Python file with error handling (for data seeding)
execute_python_file() {
    local file=$1
    local filename=$(basename "$file")

    log_info "Executing Python script: $filename"

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_warn "Python3 not available, skipping $filename"
        return 0
    fi

    # Execute the Python file
    if python3 "$file" 2>&1; then
        log_success "$filename completed"
        return 0
    else
        log_error "Failed to execute $filename"
        return 1
    fi
}

# =============================================================================
# MAIN INITIALIZATION LOGIC
# =============================================================================

main() {
    log_info "=============================================="
    log_info "USDCOP Trading System - Database Initialization"
    log_info "=============================================="
    log_info "Database: ${POSTGRES_DB:-postgres}"
    log_info "User: ${POSTGRES_USER:-postgres}"
    log_info "Scripts Directory: $SCRIPTS_DIR"
    log_info "=============================================="

    # Wait for PostgreSQL to be ready
    wait_for_postgres || exit 1

    # Check if already initialized
    if is_initialized; then
        log_info "Database already initialized (audit.init_log table exists)"
        log_info "Checking for new scripts to run..."

        # Even if initialized, we can run new scripts that haven't been executed
        # This makes the script truly idempotent
    else
        log_info "Fresh database detected, running full initialization..."
    fi

    # Find and sort SQL files
    local sql_files=()
    if [ -d "$SCRIPTS_DIR" ]; then
        while IFS= read -r -d '' file; do
            sql_files+=("$file")
        done < <(find "$SCRIPTS_DIR" -maxdepth 1 -name "*.sql" -type f -print0 | sort -z)
    fi

    if [ ${#sql_files[@]} -eq 0 ]; then
        log_warn "No SQL files found in $SCRIPTS_DIR"
    else
        log_info "Found ${#sql_files[@]} SQL file(s) to process"

        # Execute each SQL file in order
        local failed=0
        for sql_file in "${sql_files[@]}"; do
            local filename=$(basename "$sql_file")

            # Check if this script was already executed successfully
            if table_exists "audit" "init_log"; then
                local already_run
                already_run=$(psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -tAc \
                    "SELECT EXISTS (
                        SELECT 1 FROM audit.init_log
                        WHERE script_name = '$filename' AND success = TRUE
                    );" 2>/dev/null || echo "f")

                if [ "$already_run" = "t" ]; then
                    log_info "Skipping $filename (already executed successfully)"
                    continue
                fi
            fi

            # Execute the SQL file
            if ! execute_sql_file "$sql_file"; then
                ((failed++))
                # Continue with other files even if one fails
                log_warn "Continuing with remaining scripts..."
            fi
        done

        if [ $failed -gt 0 ]; then
            log_warn "$failed script(s) failed to execute"
        fi
    fi

    # Find and execute Python files (for data seeding)
    local py_files=()
    if [ -d "$SCRIPTS_DIR" ]; then
        while IFS= read -r -d '' file; do
            py_files+=("$file")
        done < <(find "$SCRIPTS_DIR" -maxdepth 1 -name "*.py" -type f -print0 | sort -z)
    fi

    if [ ${#py_files[@]} -gt 0 ]; then
        log_info "Found ${#py_files[@]} Python file(s) to process"
        for py_file in "${py_files[@]}"; do
            execute_python_file "$py_file" || true
        done
    fi

    # Final status report
    log_info "=============================================="
    log_info "Database Initialization Complete"
    log_info "=============================================="

    # Show table summary
    psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -c \
        "SELECT
            schemaname AS schema,
            COUNT(*) AS tables,
            pg_size_pretty(SUM(pg_total_relation_size(schemaname || '.' || tablename))) AS size
         FROM pg_tables
         WHERE schemaname IN ('public', 'dw', 'staging', 'audit')
         GROUP BY schemaname
         ORDER BY schemaname;" 2>/dev/null || true

    # Show hypertables if TimescaleDB is active
    psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" -c \
        "SELECT hypertable_name, num_chunks
         FROM timescaledb_information.hypertables
         ORDER BY hypertable_name;" 2>/dev/null || true

    log_success "USDCOP Database initialization completed!"
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Only run main if this script is being executed directly
# (not sourced by another script)
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
