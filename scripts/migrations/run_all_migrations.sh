#!/bin/bash
# =============================================================================
# USDCOP Trading System - Master Migration Runner
# =============================================================================
#
# This script applies all pending database migrations in order.
# It is IDEMPOTENT - safe to run multiple times.
#
# Usage:
#   ./run_all_migrations.sh                    # Run against Docker container
#   ./run_all_migrations.sh --local            # Run against local PostgreSQL
#   ./run_all_migrations.sh --dry-run          # Show what would be applied
#
# Author: Trading Team
# Version: 1.0.0
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Database connection (Docker default)
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${POSTGRES_USER:-admin}"
DB_PASSWORD="${POSTGRES_PASSWORD:-admin123}"
DB_NAME="${POSTGRES_DB:-usdcop_trading}"
CONTAINER_NAME="${CONTAINER_NAME:-usdcop-postgres-timescale}"

# Flags
DRY_RUN=false
USE_DOCKER=true
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_DOCKER=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local      Run against local PostgreSQL (not Docker)"
            echo "  --dry-run    Show what would be applied without executing"
            echo "  --verbose    Show detailed output"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

run_sql() {
    local sql="$1"
    if [ "$USE_DOCKER" = true ]; then
        docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "$sql" 2>/dev/null
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql" 2>/dev/null
    fi
}

run_sql_file() {
    local file="$1"
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" < "$file" 2>&1
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$file" 2>&1
    fi
}

is_migration_applied() {
    local migration_name="$1"
    local result=$(run_sql "SELECT 1 FROM _applied_migrations WHERE migration_name = '$migration_name';" 2>/dev/null | grep -c "1 row" || echo "0")
    [ "$result" != "0" ]
}

mark_migration_applied() {
    local migration_name="$1"
    local checksum="$2"
    run_sql "INSERT INTO _applied_migrations (migration_name, checksum) VALUES ('$migration_name', '$checksum') ON CONFLICT (migration_name) DO NOTHING;" > /dev/null 2>&1
}

get_file_checksum() {
    local file="$1"
    if command -v md5sum &> /dev/null; then
        md5sum "$file" | awk '{print $1}'
    elif command -v md5 &> /dev/null; then
        md5 -q "$file"
    else
        echo "no-checksum"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "============================================================"
echo "   USDCOP Trading System - Database Migration Runner"
echo "============================================================"
echo ""

# Check Docker container is running
if [ "$USE_DOCKER" = true ]; then
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Docker container '$CONTAINER_NAME' is not running!"
        log_info "Start it with: docker-compose up -d postgres-timescale"
        exit 1
    fi
    log_info "Using Docker container: $CONTAINER_NAME"
else
    log_info "Using local PostgreSQL: $DB_HOST:$DB_PORT"
fi

# Create migrations tracking table if not exists
log_info "Ensuring migrations tracking table exists..."
run_sql "
CREATE TABLE IF NOT EXISTS _applied_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    checksum VARCHAR(64),
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    applied_by VARCHAR(100) DEFAULT CURRENT_USER
);
CREATE INDEX IF NOT EXISTS idx_migrations_name ON _applied_migrations(migration_name);
COMMENT ON TABLE _applied_migrations IS 'Tracks applied database migrations for idempotent execution';
" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    log_error "Failed to create migrations tracking table"
    exit 1
fi

log_success "Migrations tracking table ready"
echo ""

# =============================================================================
# DEFINE MIGRATION ORDER
# =============================================================================
# Critical: These must be in dependency order!

declare -a MIGRATIONS=(
    # Phase 1: Core Extensions & Base Tables
    "init-scripts/00-init-extensions.sql"
    "init-scripts/01-essential-usdcop-init.sql"
    "init-scripts/02-macro-indicators-schema.sql"

    # Phase 2: Feature Store (CRITICAL for V7.1)
    "init-scripts/03-inference-features-views-v2.sql"

    # Phase 3: Model Registry & MLOps
    "init-scripts/05-model-registry.sql"
    "init-scripts/06-experiment-registry.sql"
    "init-scripts/10-multi-model-schema.sql"
    "init-scripts/11-paper-trading-tables.sql"
    "init-scripts/12-trades-metadata.sql"
    "init-scripts/15-forecasting-schema.sql"
    "init-scripts/20-signalbridge-schema.sql"

    # Phase 4: Incremental Migrations
    "database/migrations/020_feature_snapshot_improvements.sql"
    "database/migrations/021_drift_audit.sql"
    "database/migrations/022_experiment_registry.sql"
    "database/migrations/025_lineage_tables.sql"
    "database/migrations/026_v_macro_unified.sql"

    # Phase 5: V7.1 Event-Driven Architecture (CRITICAL)
    "database/migrations/033_event_triggers.sql"

    # Phase 6: Two-Vote Promotion System
    "database/migrations/034_promotion_proposals.sql"
    "database/migrations/035_approval_audit_log.sql"
    "database/migrations/036_model_registry_enhanced.sql"
    "database/migrations/037_experiment_contracts.sql"
)

# =============================================================================
# APPLY MIGRATIONS
# =============================================================================

APPLIED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

echo "============================================================"
echo "   Applying Migrations"
echo "============================================================"
echo ""

for migration in "${MIGRATIONS[@]}"; do
    migration_path="$PROJECT_ROOT/$migration"
    migration_name=$(basename "$migration")

    # Check if file exists
    if [ ! -f "$migration_path" ]; then
        log_warning "Migration file not found: $migration"
        continue
    fi

    # Check if already applied
    if is_migration_applied "$migration_name"; then
        if [ "$VERBOSE" = true ]; then
            echo -e "  ${YELLOW}SKIP${NC}  $migration_name (already applied)"
        fi
        ((SKIPPED_COUNT++))
        continue
    fi

    # Get checksum
    checksum=$(get_file_checksum "$migration_path")

    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${BLUE}WOULD APPLY${NC}  $migration_name"
        ((APPLIED_COUNT++))
        continue
    fi

    # Apply migration
    echo -ne "  Applying $migration_name... "

    output=$(run_sql_file "$migration_path" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        mark_migration_applied "$migration_name" "$checksum"
        echo -e "${GREEN}OK${NC}"
        ((APPLIED_COUNT++))
    else
        echo -e "${RED}FAILED${NC}"
        if [ "$VERBOSE" = true ]; then
            echo "$output" | head -20
        fi
        ((FAILED_COUNT++))
        # Continue with other migrations (don't fail completely)
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "   Migration Summary"
echo "============================================================"
echo ""
echo -e "  Applied:  ${GREEN}$APPLIED_COUNT${NC}"
echo -e "  Skipped:  ${YELLOW}$SKIPPED_COUNT${NC} (already applied)"
echo -e "  Failed:   ${RED}$FAILED_COUNT${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN - No changes were made"
fi

if [ $FAILED_COUNT -gt 0 ]; then
    log_error "Some migrations failed. Check output above."
    exit 1
fi

if [ $APPLIED_COUNT -gt 0 ]; then
    log_success "All pending migrations applied successfully!"
else
    log_info "Database is up to date - no new migrations to apply."
fi

echo ""
echo "============================================================"
echo "   Verification"
echo "============================================================"
echo ""

# Quick verification
log_info "Checking critical tables..."

tables_to_check=(
    "inference_features_5m"
    "event_dead_letter_queue"
    "circuit_breaker_state"
    "model_registry"
)

for table in "${tables_to_check[@]}"; do
    exists=$(run_sql "SELECT 1 FROM information_schema.tables WHERE table_name = '$table';" 2>/dev/null | grep -c "1 row" || echo "0")
    if [ "$exists" != "0" ]; then
        echo -e "  ${GREEN}✓${NC} $table"
    else
        echo -e "  ${RED}✗${NC} $table (missing)"
    fi
done

# Check triggers
log_info "Checking NOTIFY triggers..."
trigger_count=$(run_sql "SELECT COUNT(*) FROM pg_trigger WHERE tgname LIKE '%notify%';" 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "0")
echo -e "  Found ${BLUE}$trigger_count${NC} NOTIFY triggers"

echo ""
log_success "Migration runner completed!"
echo ""
