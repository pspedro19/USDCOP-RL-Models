#!/bin/bash
##############################################################################
# Auto-restore OHLCV Data from Backup on Database Initialization
# Este script se ejecuta automáticamente cuando PostgreSQL inicia por primera vez
# Restaura datos a la tabla UNIFICADA: usdcop_m5_ohlcv
##############################################################################

set -e

echo "🔄 Checking if usdcop_m5_ohlcv needs to be restored..."

# Check if usdcop_m5_ohlcv table has data
RECORD_COUNT=$(psql -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP';" 2>/dev/null || echo "0")

if [ "$RECORD_COUNT" -gt "1000" ]; then
    echo "✅ usdcop_m5_ohlcv already has $RECORD_COUNT records. Skipping restore."
    exit 0
fi

echo "📦 usdcop_m5_ohlcv is empty. Restoring from backup..."

# Find the latest backup
BACKUP_DIR="/docker-entrypoint-initdb.d/data-backups"
LATEST_BACKUP=$(ls -t $BACKUP_DIR/*/usdcop_m5_ohlcv.csv.gz 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "⚠️  No usdcop_m5_ohlcv backup found. Database will remain empty."
    echo "ℹ️  To create a backup, run from host machine:"
    echo "ℹ️  sudo /home/azureuser/USDCOP-RL-Models/scripts/export_usdcop_ohlcv_backup.sh"
    exit 0
else
    echo "📁 Found OHLCV backup: $LATEST_BACKUP"
    echo "🔄 Decompressing and loading data..."

    # Decompress and import directly into PostgreSQL
    zcat "$LATEST_BACKUP" | psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
        COPY usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source, created_at)
        FROM STDIN
        WITH (FORMAT csv, HEADER true);
    "
fi

# Verify data was loaded
FINAL_COUNT=$(psql -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP';")
echo "✅ Successfully loaded $FINAL_COUNT records into usdcop_m5_ohlcv table"

# Update statistics
psql -U $POSTGRES_USER -d $POSTGRES_DB -c "ANALYZE usdcop_m5_ohlcv;"

# Show date range
psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    'Data range:' as info,
    MIN(time)::date as first_date,
    MAX(time)::date as last_date,
    COUNT(*) as total_records
FROM usdcop_m5_ohlcv
WHERE symbol = 'USD/COP';
"

echo "🎉 OHLCV data restore complete!"
