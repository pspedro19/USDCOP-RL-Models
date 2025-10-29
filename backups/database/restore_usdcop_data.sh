#!/bin/bash
# ================================================
# USD/COP OHLCV Data Restoration Script
# ================================================
# Purpose: Restore USD/COP market data from backup
# Usage: ./restore_usdcop_data.sh <backup_file.csv.gz>
# ================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="usdcop-postgres-timescale"
DB_NAME="usdcop_trading"
DB_USER="admin"
TABLE_NAME="usdcop_m5_ohlcv"

# Print with color
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if backup file provided
if [ -z "$1" ]; then
    print_error "No backup file specified!"
    echo ""
    echo "Usage: $0 <backup_file.csv.gz>"
    echo ""
    echo "Available backups:"
    ls -lh *.csv.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Validate backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    print_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

print_info "Starting restoration process..."
print_info "Backup file: $BACKUP_FILE"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    print_error "PostgreSQL container '$CONTAINER_NAME' is not running!"
    exit 1
fi

# Extract backup to temp file
TEMP_CSV="/tmp/usdcop_restore_$$.csv"
print_info "Extracting backup..."
gunzip -c "$BACKUP_FILE" > "$TEMP_CSV"

# Get record count
RECORD_COUNT=$(wc -l < "$TEMP_CSV")
RECORD_COUNT=$((RECORD_COUNT - 1))  # Subtract header
print_info "Records to restore: $RECORD_COUNT"

# Ask for confirmation
echo ""
print_warning "This will restore $RECORD_COUNT records to table '$TABLE_NAME'"
print_warning "Existing data with the same timestamps will be updated/skipped"
echo ""
read -p "Continue? (yes/no): " -r
echo ""
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    print_info "Restoration cancelled by user"
    rm "$TEMP_CSV"
    exit 0
fi

# Copy temp file into container
print_info "Copying data to container..."
docker cp "$TEMP_CSV" "$CONTAINER_NAME:/tmp/restore_data.csv"

# Restore data using COPY with ON CONFLICT handling
print_info "Restoring data to database..."
docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" << EOF
-- Create temporary table
CREATE TEMP TABLE temp_restore (
    time TIMESTAMPTZ,
    symbol TEXT,
    open NUMERIC(12,6),
    high NUMERIC(12,6),
    low NUMERIC(12,6),
    close NUMERIC(12,6),
    volume BIGINT,
    source TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

-- Load data into temp table
\COPY temp_restore FROM '/tmp/restore_data.csv' CSV HEADER

-- Insert with conflict resolution (update on conflict)
INSERT INTO $TABLE_NAME
SELECT * FROM temp_restore
ON CONFLICT (time, symbol)
DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    updated_at = NOW();

-- Drop temp table
DROP TABLE temp_restore;
EOF

# Cleanup
print_info "Cleaning up temporary files..."
rm "$TEMP_CSV"
docker exec "$CONTAINER_NAME" rm /tmp/restore_data.csv

# Verify restoration
print_info "Verifying restoration..."
FINAL_COUNT=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM $TABLE_NAME" | tr -d ' ')
DATE_RANGE=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT MIN(time) || ' to ' || MAX(time) FROM $TABLE_NAME")

echo ""
print_info "✅ Restoration completed successfully!"
echo ""
echo "  Total records in table: $FINAL_COUNT"
echo "  Date range: $DATE_RANGE"
echo ""
