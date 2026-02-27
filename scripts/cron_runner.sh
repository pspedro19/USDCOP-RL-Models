#!/usr/bin/env bash
# ==============================================================================
# cron_runner.sh — Master cron orchestrator for USDCOP Trading System
# ==============================================================================
# Usage: ./scripts/cron_runner.sh <task>
#   Tasks: m5_fetch | macro_update | weekly_signal | monitor | week_end
#
# System timezone is America/Lima (UTC-5) = same as COT. Cron times are LOCAL (COT).
# Logs are written to logs/cron/<task>_<YYYYMMDD>.log
#
# Author: Pedro @ Lean Tech Solutions
# Version: 1.0.0
# Date: 2026-02-17
# ==============================================================================

set -euo pipefail

# ==============================================================================
# PATHS
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/cron"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

# Load .env if present (for POSTGRES_*, TWELVEDATA_*, FRED_* vars)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Override POSTGRES_HOST for local execution (Docker uses 'timescaledb')
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_DB="${POSTGRES_DB:-usdcop_trading}"
export POSTGRES_USER="${POSTGRES_USER:-admin}"
# POSTGRES_PASSWORD must be set in .env

# Python path: include project root and src/ for imports
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Use system Python (3.12)
PYTHON="${PYTHON:-python3}"

# ==============================================================================
# TASK ROUTING
# ==============================================================================

TASK="${1:-}"
DATE_STAMP="$(date +%Y%m%d)"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

if [ -z "${TASK}" ]; then
    echo "Usage: $0 <m5_fetch|macro_update|weekly_signal|monitor|week_end>"
    exit 1
fi

LOG_FILE="${LOG_DIR}/${TASK}_${DATE_STAMP}.log"

# Log header
{
    echo "================================================================"
    echo "  CRON TASK: ${TASK}"
    echo "  Started:   ${TIMESTAMP}"
    echo "  PID:       $$"
    echo "  Python:    $(${PYTHON} --version 2>&1)"
    echo "  Host:      ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    echo "================================================================"
} >> "${LOG_FILE}" 2>&1

# ==============================================================================
# EXECUTE
# ==============================================================================

run_task() {
    local script="${1}"
    local full_path="${PROJECT_ROOT}/scripts/${script}"

    if [ ! -f "${full_path}" ]; then
        echo "[ERROR] Script not found: ${full_path}" >> "${LOG_FILE}" 2>&1
        exit 1
    fi

    echo "[RUN] ${PYTHON} ${full_path}" >> "${LOG_FILE}" 2>&1
    ${PYTHON} "${full_path}" >> "${LOG_FILE}" 2>&1
    local rc=$?

    echo "[EXIT] Return code: ${rc}" >> "${LOG_FILE}" 2>&1
    echo "[END]  $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}" 2>&1
    echo "" >> "${LOG_FILE}" 2>&1
    return ${rc}
}

case "${TASK}" in
    m5_fetch)
        run_task "cron_m5_fetch.py"
        ;;
    macro_update)
        run_task "cron_macro_update.py"
        ;;
    weekly_signal)
        # Weekly signal is a heavyweight task — delegate to existing pipeline script
        # which handles model loading, prediction, confidence scoring, vol targeting
        run_task "run_pipeline_standalone.py"
        ;;
    monitor)
        run_task "cron_monitor.py"
        ;;
    week_end)
        run_task "cron_week_end.py"
        ;;
    *)
        echo "[ERROR] Unknown task: ${TASK}" >> "${LOG_FILE}" 2>&1
        echo "Valid tasks: m5_fetch, macro_update, weekly_signal, monitor, week_end"
        exit 1
        ;;
esac

exit $?
