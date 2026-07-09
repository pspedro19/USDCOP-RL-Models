#!/bin/bash
# =============================================================================
# Activate Vol-Targeting Paper Trading (Paso 1.4)
# =============================================================================
# Run this ONCE after Docker stack is up.
# Prerequisites: docker-compose up -d (PostgreSQL + Airflow running)
#
# What it does:
#   1. Runs migration 041 (creates tables + trigger + view)
#   2. Verifies tables exist
#   3. Triggers L5c manually for today's signal
#   4. Waits and verifies signal was persisted
#   5. Unpauses both DAGs for daily operation
#
# Usage:
#   bash scripts/activate_paper_trading.sh
# =============================================================================

set -e

POSTGRES_CONTAINER="usdcop-postgres-timescale"
AIRFLOW_CONTAINER="usdcop-airflow-webserver"

echo "============================================"
echo "  Vol-Targeting Paper Trading Activation"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Step 1: Run migration
echo ""
echo "[1/5] Running migration 041_forecast_vol_targeting.sql..."
docker exec -i ${POSTGRES_CONTAINER} psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-usdcop_trading}" \
    < database/migrations/041_forecast_vol_targeting.sql

echo "  Migration completed."

# Step 2: Verify tables
echo ""
echo "[2/5] Verifying tables..."
docker exec ${POSTGRES_CONTAINER} psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-usdcop_trading}" -c "
    SELECT table_name FROM information_schema.tables
    WHERE table_name IN ('forecast_vol_targeting_signals', 'forecast_paper_trading')
    ORDER BY table_name;
"

TABLES_OK=$(docker exec ${POSTGRES_CONTAINER} psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-usdcop_trading}" -t -c "
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_name IN ('forecast_vol_targeting_signals', 'forecast_paper_trading');
")

if [ "$(echo $TABLES_OK | tr -d ' ')" != "2" ]; then
    echo "  ERROR: Expected 2 tables, found ${TABLES_OK}. Aborting."
    exit 1
fi
echo "  Both tables verified."

# Step 3: Trigger L5c manually
echo ""
echo "[3/5] Triggering forecast_l5_02_vol_targeting (force_run=true)..."
docker exec ${AIRFLOW_CONTAINER} airflow dags trigger forecast_l5_02_vol_targeting \
    --conf '{"force_run": true}'

echo "  DAG triggered. Waiting 30s for execution..."
sleep 30

# Step 4: Verify signal
echo ""
echo "[4/5] Checking forecast_vol_targeting_signals..."
docker exec ${POSTGRES_CONTAINER} psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-usdcop_trading}" -c "
    SELECT signal_date, forecast_direction, clipped_leverage, position_size,
           realized_vol_21d, config_version
    FROM forecast_vol_targeting_signals
    ORDER BY signal_date DESC
    LIMIT 3;
"

SIGNAL_COUNT=$(docker exec ${POSTGRES_CONTAINER} psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-usdcop_trading}" -t -c "
    SELECT COUNT(*) FROM forecast_vol_targeting_signals;
")

if [ "$(echo $SIGNAL_COUNT | tr -d ' ')" = "0" ]; then
    echo "  WARNING: No signals found. The DAG may still be running."
    echo "  Check Airflow UI or run:"
    echo "    docker exec ${AIRFLOW_CONTAINER} airflow dags list-runs -d forecast_l5_02_vol_targeting"
else
    echo "  Signal(s) found: ${SIGNAL_COUNT}"
fi

# Step 5: Unpause both DAGs
echo ""
echo "[5/5] Unpausing DAGs for daily operation..."
docker exec ${AIRFLOW_CONTAINER} airflow dags unpause forecast_l5_02_vol_targeting
docker exec ${AIRFLOW_CONTAINER} airflow dags unpause forecast_l6_03_paper_trading

echo ""
echo "============================================"
echo "  Paper Trading ACTIVATED"
echo "============================================"
echo ""
echo "  L5c: forecast_l5_02_vol_targeting"
echo "    Schedule: 18:30 UTC (13:30 COT) Mon-Fri"
echo "    Generates daily vol-targeting signal"
echo ""
echo "  L6:  forecast_l6_03_paper_trading"
echo "    Schedule: 00:00 UTC+1 (19:00 COT) Mon-Fri"
echo "    Evaluates yesterday's signal vs actual return"
echo ""
echo "  Checkpoints:"
echo "    Day 20: First DA check (>50% = continue)"
echo "    Day 40: Pause gate (DA<48% or MaxDD>15%)"
echo "    Day 60: Final gate (DA>51%, binomial p<0.10)"
echo ""
echo "  Monitoring:"
echo "    SELECT * FROM v_paper_trading_performance;"
echo ""
echo "  Last 10 days:"
echo "    SELECT signal_date, signal_direction, signal_leverage,"
echo "           actual_return_1d, strategy_return, running_da_pct"
echo "    FROM forecast_paper_trading"
echo "    ORDER BY signal_date DESC LIMIT 10;"
echo "============================================"
