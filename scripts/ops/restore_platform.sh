#!/bin/bash
# =============================================================================
# USD/COP Trading Platform - Full Restoration Script
# =============================================================================
# This script restores the complete platform from a fresh deployment
# Run this after docker-compose up -d to initialize the database
#
# Usage: ./scripts/restore_platform.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 1. WAIT FOR SERVICES
# =============================================================================
log_info "Waiting for PostgreSQL to be ready..."
until docker exec usdcop-timescaledb pg_isready -U trading_user -d trading_db > /dev/null 2>&1; do
    sleep 2
    echo -n "."
done
echo ""
log_info "PostgreSQL is ready!"

# =============================================================================
# 2. RESTORE DATABASE BACKUP
# =============================================================================
BACKUP_FILE="$PROJECT_DIR/data/backups/usdcop_backup.sql.gz"

if [ -f "$BACKUP_FILE" ]; then
    log_info "Restoring database from backup: $BACKUP_FILE"

    # Decompress and restore
    gunzip -c "$BACKUP_FILE" | docker exec -i usdcop-timescaledb psql -U trading_user -d trading_db

    log_info "Database restored successfully!"
else
    log_warn "Backup file not found: $BACKUP_FILE"
    log_warn "Starting with empty database. Run init scripts instead."

    # Run init scripts if no backup
    for sql_file in "$PROJECT_DIR"/init-scripts/*.sql; do
        if [ -f "$sql_file" ]; then
            log_info "Executing: $(basename $sql_file)"
            docker exec -i usdcop-timescaledb psql -U trading_user -d trading_db < "$sql_file"
        fi
    done
fi

# =============================================================================
# 3. COPY MODEL FILES TO AIRFLOW CONTAINER
# =============================================================================
log_info "Copying model files to Airflow container..."

# Create models directory in container
docker exec usdcop-airflow mkdir -p /opt/airflow/ml_models/ppo_v20_production

# Copy V19 model
if [ -f "$PROJECT_DIR/models/ppo_v1_20251226_054154.zip" ]; then
    log_info "Copying V19 model..."
    cat "$PROJECT_DIR/models/ppo_v1_20251226_054154.zip" | \
        docker exec -i usdcop-airflow tee /opt/airflow/ml_models/ppo_v1_20251226_054154.zip > /dev/null
fi

# Copy V20 models
if [ -d "$PROJECT_DIR/models/ppo_v20_production" ]; then
    log_info "Copying V20 models..."
    for model_file in "$PROJECT_DIR/models/ppo_v20_production"/*; do
        if [ -f "$model_file" ]; then
            filename=$(basename "$model_file")
            cat "$model_file" | \
                docker exec -i usdcop-airflow tee "/opt/airflow/ml_models/ppo_v20_production/$filename" > /dev/null
        fi
    done
fi

log_info "Model files copied successfully!"

# =============================================================================
# 4. VERIFY MODEL REGISTRY
# =============================================================================
log_info "Verifying model registry in database..."

# Check if config.models table exists and has entries
MODEL_COUNT=$(docker exec usdcop-timescaledb psql -U trading_user -d trading_db -t -c \
    "SELECT COUNT(*) FROM config.models WHERE is_active = true" 2>/dev/null || echo "0")

if [ "$MODEL_COUNT" -lt "2" ]; then
    log_warn "Model registry incomplete. Creating entries..."

    docker exec usdcop-timescaledb psql -U trading_user -d trading_db << 'EOF'
-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS config;

-- Create models table if not exists
CREATE TABLE IF NOT EXISTS config.models (
    model_id VARCHAR(50) PRIMARY KEY,
    display_name VARCHAR(100),
    model_path VARCHAR(255),
    model_type VARCHAR(50),
    observation_dim INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert V19 model
INSERT INTO config.models (model_id, display_name, model_path, model_type, observation_dim, is_active)
VALUES ('ppo_v19_prod', 'PPO V19 (Original)', '/opt/airflow/ml_models/ppo_v1_20251226_054154.zip', 'PPO', 15, true)
ON CONFLICT (model_id) DO UPDATE SET
    model_path = EXCLUDED.model_path,
    is_active = EXCLUDED.is_active;

-- Insert V20 model
INSERT INTO config.models (model_id, display_name, model_path, model_type, observation_dim, is_active)
VALUES ('ppo_v20_prod', 'PPO V20 (Macro)', '/opt/airflow/ml_models/ppo_v20_production/final_model.zip', 'PPO', 15, true)
ON CONFLICT (model_id) DO UPDATE SET
    model_path = EXCLUDED.model_path,
    is_active = EXCLUDED.is_active;
EOF
    log_info "Model registry created!"
fi

# =============================================================================
# 5. ENABLE AIRFLOW DAGS
# =============================================================================
log_info "Enabling Airflow DAGs..."

docker exec usdcop-airflow airflow dags unpause v3.l0_ohlcv_realtime 2>/dev/null || true
docker exec usdcop-airflow airflow dags unpause v3.l0_macro_unified 2>/dev/null || true
docker exec usdcop-airflow airflow dags unpause v3.l5_multi_model_inference 2>/dev/null || true

log_info "DAGs enabled!"

# =============================================================================
# 6. VERIFY INSTALLATION
# =============================================================================
log_info "Running verification checks..."

# Check OHLCV data
OHLCV_COUNT=$(docker exec usdcop-timescaledb psql -U trading_user -d trading_db -t -c \
    "SELECT COUNT(*) FROM usdcop_m5_ohlcv" 2>/dev/null || echo "0")
log_info "OHLCV records: $OHLCV_COUNT"

# Check models
MODELS=$(docker exec usdcop-timescaledb psql -U trading_user -d trading_db -t -c \
    "SELECT model_id FROM config.models WHERE is_active = true" 2>/dev/null || echo "none")
log_info "Active models: $MODELS"

# Check inferences
INF_COUNT=$(docker exec usdcop-timescaledb psql -U trading_user -d trading_db -t -c \
    "SELECT COUNT(*) FROM trading.model_inferences" 2>/dev/null || echo "0")
log_info "Inference records: $INF_COUNT"

echo ""
log_info "============================================"
log_info "Platform restoration complete!"
log_info "============================================"
echo ""
log_info "Next steps:"
echo "  1. Access dashboard: http://localhost:3000"
echo "  2. Access Airflow: http://localhost:8080 (admin/admin)"
echo "  3. Access pgAdmin: http://localhost:5050"
echo ""
log_info "Market hours: Mon-Fri 8:00-12:55 COT (13:00-17:55 UTC)"
