#!/bin/bash
# =============================================================================
# Auto-Initialization Script for USDCOP Trading System
# =============================================================================
# Este script se ejecuta automáticamente al inicializar los servicios
# y garantiza que la tabla market_data esté disponible con datos

set -e  # Exit on any error

echo "🚀 USDCOP TRADING SYSTEM - AUTO INITIALIZATION"
echo "============================================="

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec usdcop-postgres-timescale pg_isready -U admin -d usdcop_trading >/dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi

    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - waiting 5 seconds..."
    sleep 5
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ PostgreSQL not ready after $max_attempts attempts"
    exit 1
fi

# Wait a bit more for full initialization
sleep 10

# Run the backup/restore system auto-initialization
echo "🔄 Running backup/restore system initialization..."
python3 /home/GlobalForex/USDCOP-RL-Models/scripts/backup_restore_system.py auto-init

# Check final status
echo ""
echo "📊 Final system status:"
python3 /home/GlobalForex/USDCOP-RL-Models/scripts/backup_restore_system.py stats

echo ""
echo "✅ Auto-initialization completed!"
echo "============================================="