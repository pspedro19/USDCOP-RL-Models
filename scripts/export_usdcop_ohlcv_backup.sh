#!/bin/bash
###############################################################################
# Export usdcop_m5_ohlcv data to backup for auto-restore
# Este script exporta los datos actuales de usdcop_m5_ohlcv para backup
###############################################################################

set -e

echo "🔄 Exporting usdcop_m5_ohlcv data to backup..."

# Create backup directory with timestamp
BACKUP_DIR="/home/azureuser/USDCOP-RL-Models/init-scripts/data-backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Export data from PostgreSQL
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
COPY (
    SELECT
        time,
        symbol,
        open,
        high,
        low,
        close,
        volume,
        source,
        created_at
    FROM usdcop_m5_ohlcv
    WHERE symbol = 'USD/COP'
    ORDER BY time
) TO STDOUT WITH CSV HEADER
" | gzip > "$BACKUP_DIR/usdcop_m5_ohlcv.csv.gz"

# Get record count
RECORD_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP';")

echo "✅ Exported $RECORD_COUNT records to:"
echo "   $BACKUP_DIR/usdcop_m5_ohlcv.csv.gz"
echo ""
echo "📊 Backup info:"
ls -lh "$BACKUP_DIR/usdcop_m5_ohlcv.csv.gz"
echo ""
echo "🎉 Backup complete! This backup will be used for auto-restore on next database init."
