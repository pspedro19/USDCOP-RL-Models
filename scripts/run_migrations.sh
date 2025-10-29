#!/bin/bash
# =====================================================
# Migration Execution Script
# Description: Runs database migrations in order with validation
# Author: Migration System
# Date: 2025-10-22
# =====================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MIGRATIONS_DIR="${PROJECT_ROOT}/postgres/migrations"
LOG_DIR="/tmp/usdcop-backups"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/migration_$(date +%Y%m%d_%H%M%S).log"

# Database connection parameters
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-usdcop_trading}"
DB_USER="${POSTGRES_USER:-admin}"
DB_PASSWORD="${POSTGRES_PASSWORD:-admin123}"

# Docker container name (if running in docker)
DOCKER_CONTAINER="${DOCKER_CONTAINER:-usdcop-postgres-timescale}"

# =====================================================
# HELPER FUNCTIONS
# =====================================================

print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =====================================================
# FUNCTIONS
# =====================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if migrations directory exists
    if [ ! -d "$MIGRATIONS_DIR" ]; then
        print_error "Migrations directory not found: $MIGRATIONS_DIR"
        exit 1
    fi
    print_success "Migrations directory found"

    # Check if log directory exists, create if not
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        print_info "Created log directory: $LOG_DIR"
    fi

    # Check if docker container is running
    if command -v docker &> /dev/null; then
        if docker ps | grep -q "$DOCKER_CONTAINER"; then
            print_success "Docker container '$DOCKER_CONTAINER' is running"
            USE_DOCKER=true
        else
            print_warning "Docker container '$DOCKER_CONTAINER' not found, will try direct connection"
            USE_DOCKER=false
        fi
    else
        print_warning "Docker not found, will try direct connection"
        USE_DOCKER=false
    fi

    echo ""
}

test_db_connection() {
    print_header "Testing Database Connection"

    if [ "$USE_DOCKER" = true ]; then
        if docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
            print_success "Database connection successful (Docker)"
            return 0
        else
            print_error "Cannot connect to database via Docker"
            return 1
        fi
    else
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
            print_success "Database connection successful (Direct)"
            return 0
        else
            print_error "Cannot connect to database directly"
            return 1
        fi
    fi

    echo ""
}

execute_sql_file() {
    local sql_file=$1
    local migration_name=$(basename "$sql_file" .sql)

    print_header "Executing Migration: $migration_name"
    print_info "File: $sql_file"

    # Create temporary log file for this migration
    local temp_log="${LOG_DIR}/${migration_name}_$(date +%Y%m%d_%H%M%S).log"

    # Execute the migration
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" < "$sql_file" 2>&1 | tee "$temp_log"
        local exit_code=${PIPESTATUS[0]}
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file" 2>&1 | tee "$temp_log"
        local exit_code=${PIPESTATUS[0]}
    fi

    # Append to main log
    cat "$temp_log" >> "$LOG_FILE"

    # Check exit code
    if [ $exit_code -eq 0 ]; then
        print_success "Migration $migration_name completed successfully"
        echo "Migration: $migration_name - SUCCESS - $(date)" >> "$LOG_FILE"
        return 0
    else
        print_error "Migration $migration_name failed"
        echo "Migration: $migration_name - FAILED - $(date)" >> "$LOG_FILE"
        return 1
    fi
}

backup_database() {
    print_header "Creating Database Backup"

    local backup_file="${LOG_DIR}/backup_pre_migration_$(date +%Y%m%d_%H%M%S).sql"

    if [ "$USE_DOCKER" = true ]; then
        docker exec "$DOCKER_CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" -t market_data > "$backup_file" 2>&1
        local exit_code=$?
    else
        PGPASSWORD="$DB_PASSWORD" pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t market_data > "$backup_file" 2>&1
        local exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        print_success "Backup created: $backup_file"
        local backup_size=$(du -h "$backup_file" | cut -f1)
        print_info "Backup size: $backup_size"
        echo ""
        return 0
    else
        print_warning "Backup failed, but continuing with migration"
        echo ""
        return 1
    fi
}

show_table_info() {
    print_header "Current Table Information"

    local query="
    SELECT
        COUNT(*) as total_rows,
        MIN(timestamp) as earliest_data,
        MAX(timestamp) as latest_data,
        pg_size_pretty(pg_total_relation_size('market_data')) as table_size
    FROM market_data;
    "

    if [ "$USE_DOCKER" = true ]; then
        docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "$query"
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$query"
    fi

    echo ""
}

estimate_migration_time() {
    print_header "Estimating Migration Time"

    local query="SELECT COUNT(*) FROM market_data;"

    if [ "$USE_DOCKER" = true ]; then
        local row_count=$(docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "$query" | tr -d ' ')
    else
        local row_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "$query" | tr -d ' ')
    fi

    # Rough estimate: 100,000 rows per second for column addition
    # Plus time for index creation
    local seconds_estimate=$((row_count / 100000 + 30))
    local minutes=$((seconds_estimate / 60))

    print_info "Total rows: $row_count"
    print_info "Estimated time: ~${minutes} minutes (${seconds_estimate} seconds)"
    echo ""
}

confirm_migration() {
    print_warning "This will modify the market_data table structure."
    print_warning "A backup will be created before proceeding."
    echo ""
    read -p "Continue with migration? (yes/no): " confirmation

    if [ "$confirmation" != "yes" ]; then
        print_error "Migration cancelled by user"
        exit 0
    fi
    echo ""
}

run_all_migrations() {
    print_header "Running All Migrations"

    local migrations=(
        "001_add_ohlcv_columns.sql"
        "002_add_optimized_indexes.sql"
        "003_add_constraints.sql"
    )

    local failed_migrations=0

    for migration in "${migrations[@]}"; do
        local migration_file="${MIGRATIONS_DIR}/${migration}"

        if [ ! -f "$migration_file" ]; then
            print_warning "Migration file not found: $migration_file (skipping)"
            continue
        fi

        if ! execute_sql_file "$migration_file"; then
            print_error "Migration failed: $migration"
            failed_migrations=$((failed_migrations + 1))

            # Ask if we should continue
            read -p "Continue with remaining migrations? (yes/no): " continue_choice
            if [ "$continue_choice" != "yes" ]; then
                print_error "Migration process stopped by user"
                return 1
            fi
        fi

        echo ""
    done

    if [ $failed_migrations -eq 0 ]; then
        print_success "All migrations completed successfully!"
        return 0
    else
        print_error "$failed_migrations migration(s) failed"
        return 1
    fi
}

show_summary() {
    print_header "Migration Summary"

    local query="
    SELECT
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_name = 'market_data'
    ORDER BY ordinal_position;
    "

    print_info "Current market_data schema:"

    if [ "$USE_DOCKER" = true ]; then
        docker exec "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "$query"
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$query"
    fi

    echo ""
    print_info "Log file: $LOG_FILE"
    print_info "All migration logs saved to: $LOG_DIR"
}

# =====================================================
# MAIN EXECUTION
# =====================================================

main() {
    print_header "USDCOP Trading System - Database Migration"
    echo "Starting migration process at $(date)"
    echo ""

    # Step 1: Prerequisites
    check_prerequisites

    # Step 2: Test connection
    if ! test_db_connection; then
        print_error "Cannot proceed without database connection"
        exit 1
    fi

    # Step 3: Show current state
    show_table_info
    estimate_migration_time

    # Step 4: Confirm
    confirm_migration

    # Step 5: Backup
    backup_database

    # Step 6: Run migrations
    if run_all_migrations; then
        print_success "Migration process completed successfully!"
    else
        print_error "Migration process completed with errors"
        print_warning "Check log file for details: $LOG_FILE"
        exit 1
    fi

    # Step 7: Show summary
    show_summary

    print_header "Migration Complete"
    echo "Finished at $(date)"
}

# =====================================================
# SCRIPT ENTRY POINT
# =====================================================

# Check if we're running the rollback
if [ "$1" == "--rollback" ]; then
    print_header "ROLLBACK MODE"
    print_warning "This will revert all OHLCV migrations!"

    if ! test_db_connection; then
        print_error "Cannot proceed without database connection"
        exit 1
    fi

    backup_database

    rollback_file="${MIGRATIONS_DIR}/rollback_ohlcv.sql"
    if [ -f "$rollback_file" ]; then
        execute_sql_file "$rollback_file"
    else
        print_error "Rollback file not found: $rollback_file"
        exit 1
    fi

    exit 0
fi

# Run main migration
main

exit 0
