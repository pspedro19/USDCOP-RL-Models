#!/bin/bash
# =============================================================================
# Inference API Entrypoint
# =============================================================================
# Runs database migrations before starting the API server.
# This prevents schema drift issues in production.
# =============================================================================

set -e

echo "=============================================="
echo "USDCOP Inference API - Startup"
echo "=============================================="

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
MAX_RETRIES=30
RETRY_COUNT=0

until python -c "
import asyncio
import asyncpg
import os

async def check():
    try:
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            user=os.getenv('POSTGRES_USER', 'admin'),
            password=os.getenv('POSTGRES_PASSWORD', 'admin123'),
            database=os.getenv('POSTGRES_DB', 'usdcop_trading'),
        )
        await conn.close()
        return True
    except Exception as e:
        print(f'  Waiting... ({e})')
        return False

asyncio.run(check())
" 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: PostgreSQL not available after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "  PostgreSQL not ready, retrying in 2s... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "PostgreSQL is ready!"

# Run database migrations
echo ""
echo "Running database migrations..."
if [ -f /app/scripts/db_migrate.py ]; then
    python /app/scripts/db_migrate.py || {
        echo "WARNING: Migration script failed, continuing anyway..."
    }
else
    echo "Migration script not found, skipping..."
fi

# Validate schema
echo ""
echo "Validating database schema..."
if [ -f /app/scripts/db_migrate.py ]; then
    python /app/scripts/db_migrate.py --validate || {
        echo "WARNING: Schema validation failed, some features may not work"
    }
fi

# Start the API server
echo ""
echo "Starting Uvicorn server..."
echo "=============================================="
exec uvicorn services.inference_api.main:app \
    --host 0.0.0.0 \
    --port ${API_PORT:-8000} \
    --log-level ${LOG_LEVEL:-info}
