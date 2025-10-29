#!/bin/bash
# =====================================================
# Migration Status Checker
# Description: Check current migration status and readiness
# Author: Migration System
# Date: 2025-10-22
# =====================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-usdcop_trading}"
DB_USER="${POSTGRES_USER:-admin}"
DB_PASSWORD="${POSTGRES_PASSWORD:-admin123}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-usdcop-postgres-timescale}"

# Helper functions
print_header() {
    echo -e "${BLUE}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Check if using docker
if docker ps | grep -q "$DOCKER_CONTAINER" 2>/dev/null; then
    USE_DOCKER=true
    PSQL_CMD="docker exec $DOCKER_CONTAINER psql -U $DB_USER -d $DB_NAME"
else
    USE_DOCKER=false
    PSQL_CMD="PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
fi

print_header "USDCOP Database Migration Status Check"
echo ""

# Test connection
print_info "Testing database connection..."
if $PSQL_CMD -c "SELECT 1;" > /dev/null 2>&1; then
    print_success "Connected to database"
else
    print_error "Cannot connect to database"
    exit 1
fi
echo ""

# Check table structure
print_header "1. Table Structure"

query="
SELECT
    column_name,
    data_type,
    is_nullable,
    CASE
        WHEN column_default IS NULL THEN 'No default'
        ELSE column_default
    END as default_value
FROM information_schema.columns
WHERE table_name = 'market_data'
ORDER BY ordinal_position;
"

$PSQL_CMD -c "$query"
echo ""

# Check for OHLCV columns
print_header "2. OHLCV Column Status"

query="
SELECT
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'open') as has_open,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'high') as has_high,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'low') as has_low,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'close') as has_close,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'timeframe') as has_timeframe,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'spread') as has_spread,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'data_quality') as has_data_quality;
"

result=$($PSQL_CMD -t -c "$query")

if echo "$result" | grep -q "t | t | t | t | t | t | t"; then
    print_success "All OHLCV columns exist - Migration 001 completed"
elif echo "$result" | grep -q "f.*f.*f.*f"; then
    print_warning "OHLCV columns missing - Migration 001 NOT run"
else
    print_warning "Partial OHLCV columns exist - Migration may be incomplete"
fi
echo ""

# Check data completeness
print_header "3. Data Completeness"

query="
SELECT
    COUNT(*) as total_rows,
    COUNT(CASE WHEN open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL THEN 1 END) as complete_ohlc_rows,
    COUNT(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL THEN 1 END) as incomplete_rows,
    ROUND(
        100.0 * COUNT(CASE WHEN open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL THEN 1 END) / NULLIF(COUNT(*), 0),
        2
    ) as completeness_percent
FROM market_data;
"

$PSQL_CMD -c "$query" 2>/dev/null || print_warning "OHLCV columns do not exist yet"
echo ""

# Check indexes
print_header "4. Index Status"

query="
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexname::text)) as size
FROM pg_indexes
WHERE tablename = 'market_data'
ORDER BY indexname;
"

index_count=$($PSQL_CMD -t -c "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'market_data';")

print_info "Total indexes: $index_count"
$PSQL_CMD -c "$query"

if [ "$index_count" -gt 10 ]; then
    print_success "Optimized indexes present - Migration 002 likely completed"
else
    print_warning "Limited indexes - Migration 002 may not be run"
fi
echo ""

# Check constraints
print_header "5. Constraint Status"

query="
SELECT
    conname as constraint_name,
    contype as type,
    CASE contype
        WHEN 'c' THEN 'CHECK'
        WHEN 'f' THEN 'FOREIGN KEY'
        WHEN 'p' THEN 'PRIMARY KEY'
        WHEN 'u' THEN 'UNIQUE'
        WHEN 't' THEN 'TRIGGER'
        ELSE contype::text
    END as constraint_type
FROM pg_constraint
WHERE conrelid = 'market_data'::regclass
ORDER BY conname;
"

constraint_count=$($PSQL_CMD -t -c "SELECT COUNT(*) FROM pg_constraint WHERE conrelid = 'market_data'::regclass AND contype = 'c';")

print_info "Total CHECK constraints: $constraint_count"
$PSQL_CMD -c "$query"

if [ "$constraint_count" -gt 5 ]; then
    print_success "Validation constraints present - Migration 003 likely completed"
else
    print_warning "Limited constraints - Migration 003 may not be run"
fi
echo ""

# Check table size
print_header "6. Storage Statistics"

query="
SELECT
    pg_size_pretty(pg_total_relation_size('market_data')) as total_size,
    pg_size_pretty(pg_relation_size('market_data')) as table_size,
    pg_size_pretty(pg_total_relation_size('market_data') - pg_relation_size('market_data')) as index_size;
"

$PSQL_CMD -c "$query"
echo ""

# Check data quality
print_header "7. Data Quality Check"

query="
SELECT
    MIN(timestamp) as earliest_data,
    MAX(timestamp) as latest_data,
    COUNT(*) as total_rows,
    COUNT(DISTINCT symbol) as unique_symbols
FROM market_data;
"

$PSQL_CMD -c "$query"
echo ""

# Migration readiness assessment
print_header "8. Migration Readiness Assessment"

# Get column status
has_ohlcv=$($PSQL_CMD -t -c "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'market_data' AND column_name = 'open');" | xargs)

if [ "$has_ohlcv" = "t" ]; then
    print_success "OHLCV columns already exist"
    print_info "Status: Migrations already applied or partially applied"
    print_info "Action: Review data completeness or run rollback if needed"
else
    print_success "System ready for migration"
    print_info "Status: Original schema with price column"
    print_info "Action: Run ./scripts/run_migrations.sh to apply migrations"
fi
echo ""

# Summary
print_header "Summary"

row_count=$($PSQL_CMD -t -c "SELECT COUNT(*) FROM market_data;" | xargs)
table_size=$($PSQL_CMD -t -c "SELECT pg_size_pretty(pg_total_relation_size('market_data'));" | xargs)

echo "Total Rows: $row_count"
echo "Table Size: $table_size"
echo "Indexes: $index_count"
echo "Constraints: $constraint_count"
echo ""

if [ "$has_ohlcv" = "t" ]; then
    print_success "MIGRATIONS APPLIED"
else
    print_warning "MIGRATIONS PENDING"
fi

echo ""
print_info "To run migrations: ./scripts/run_migrations.sh"
print_info "To rollback: ./scripts/run_migrations.sh --rollback"
echo ""
