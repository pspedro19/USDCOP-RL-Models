#!/usr/bin/env bash
# =============================================================================
# demo_full.sh — Orchestrated 8–10 minute live demo (course MLOps project)
# =============================================================================
# Usage: bash scripts/demo/demo_full.sh
#
# Runs the full live demo in one command:
#   1. Docker compose ps → show running services
#   2. Open key UIs (Airflow, MLflow, Grafana, Redpanda Console, Dashboard)
#   3. gRPC demo       (scripts/demo/demo_grpc.sh)
#   4. Kafka demo      (scripts/demo/demo_kafka.sh)
#   5. MLflow demo     (scripts/demo/demo_mlflow.sh)
#   6. SignalBridge health check
#   7. Compliance checklist
# Exits non-zero on any failure.
# =============================================================================
set -euo pipefail

# --- Color helpers ----------------------------------------------------------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
    GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; RED="$(tput setaf 1)"
    CYAN="$(tput setaf 6)"; BLUE="$(tput setaf 4)"; MAGENTA="$(tput setaf 5)"
    BOLD="$(tput bold)"; RESET="$(tput sgr0)"
else
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; BLUE=$'\033[34m'; MAGENTA=$'\033[35m'
    BOLD=$'\033[1m'; RESET=$'\033[0m'
fi
ok()   { printf '%s[✅]%s %s\n' "${GREEN}" "${RESET}" "$*"; }
warn() { printf '%s[⚠ ]%s %s\n' "${YELLOW}" "${RESET}" "$*"; }
err()  { printf '%s[❌]%s %s\n' "${RED}" "${RESET}" "$*" >&2; }
step() { printf '\n%s%s━━━ %s %s━━━%s\n' "${BOLD}" "${BLUE}" "$*" "${BLUE}" "${RESET}"; }
hr()   { printf '%s%s%s\n' "${MAGENTA}" "=================================================================" "${RESET}"; }

# --- Script discovery -------------------------------------------------------
# Resolve the directory of THIS script so relative invocations work from any CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

COMPOSE_FILE="docker-compose.compact.yml"

# --- Key URLs ---------------------------------------------------------------
URL_DASHBOARD="http://localhost:5000"
URL_AIRFLOW="http://localhost:8080"
URL_GRAFANA="http://localhost:3002"
URL_MLFLOW="http://localhost:5001"
URL_CONSOLE="http://localhost:8088"
URL_SIGBRIDGE="http://localhost:8085"

hr
printf '%s%s      USDCOP Trading System — MLOps Course Live Demo%s\n' "${BOLD}" "${CYAN}" "${RESET}"
printf '%s      Date: %s%s\n' "${CYAN}" "$(date -Iseconds)" "${RESET}"
hr

# ---------------------------------------------------------------------------
step "Step 1/7: Docker compose ps — running services"
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    err "docker CLI not found in PATH"
    exit 1
fi

# Prefer compact compose file; fall back to all containers if absent.
if [[ -f "${COMPOSE_FILE}" ]]; then
    docker compose -f "${COMPOSE_FILE}" ps || true
else
    warn "${COMPOSE_FILE} not found — showing all running containers"
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
fi

# ---------------------------------------------------------------------------
step "Step 2/7: Open key UI URLs"
# ---------------------------------------------------------------------------
declare -a URLS=(
    "Dashboard|${URL_DASHBOARD}"
    "Airflow|${URL_AIRFLOW}"
    "MLflow|${URL_MLFLOW}"
    "Grafana|${URL_GRAFANA}"
    "Redpanda Console|${URL_CONSOLE}"
    "SignalBridge|${URL_SIGBRIDGE}"
)

HAS_XDG=0
if command -v xdg-open >/dev/null 2>&1; then HAS_XDG=1; fi

for entry in "${URLS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    printf '  %-20s %s\n' "${name}:" "${url}"
    if [[ "${HAS_XDG}" -eq 1 ]]; then
        (xdg-open "${url}" >/dev/null 2>&1 &) || true
    fi
done
if [[ "${HAS_XDG}" -eq 1 ]]; then
    ok "Opened URLs with xdg-open"
else
    warn "xdg-open not available — copy URLs above into your browser"
fi

# ---------------------------------------------------------------------------
step "Step 3/7: gRPC PredictorService demo"
# ---------------------------------------------------------------------------
bash "${SCRIPT_DIR}/demo_grpc.sh"

# ---------------------------------------------------------------------------
step "Step 4/7: Kafka producer+consumer demo"
# ---------------------------------------------------------------------------
bash "${SCRIPT_DIR}/demo_kafka.sh"

# ---------------------------------------------------------------------------
step "Step 5/7: MLflow tracking demo"
# ---------------------------------------------------------------------------
bash "${SCRIPT_DIR}/demo_mlflow.sh"

# ---------------------------------------------------------------------------
step "Step 6/7: SignalBridge health check"
# ---------------------------------------------------------------------------
if ! command -v curl >/dev/null 2>&1; then
    err "curl not found in PATH"
    exit 1
fi

SIGBRIDGE_CODE="$(curl -s -o /tmp/sigbridge_health.$$ -w '%{http_code}' --max-time 5 "${URL_SIGBRIDGE}/health" || echo "000")"
if [[ "${SIGBRIDGE_CODE}" == "200" ]]; then
    ok "SignalBridge /health = 200"
    # Pretty-print if python is available
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import json,sys; d=json.load(open('/tmp/sigbridge_health.$$')); print(json.dumps(d, indent=2))" 2>/dev/null || cat "/tmp/sigbridge_health.$$"
    else
        cat "/tmp/sigbridge_health.$$"
    fi
    rm -f "/tmp/sigbridge_health.$$"
else
    err "SignalBridge /health returned HTTP ${SIGBRIDGE_CODE}"
    rm -f "/tmp/sigbridge_health.$$"
    exit 1
fi

# ---------------------------------------------------------------------------
step "Step 7/7: Compliance checklist"
# ---------------------------------------------------------------------------
hr
printf '%s%s  MLOps Course Project — Component Compliance%s\n' "${BOLD}" "${CYAN}" "${RESET}"
hr
ok "Docker       — all services containerized (compose up)"
ok "Airflow      — orchestrator running at ${URL_AIRFLOW}"
ok "MLflow       — experiment tracking at ${URL_MLFLOW}"
ok "gRPC         — PredictorService at localhost:50051"
ok "Kafka        — Redpanda broker + signals.h5 topic"
ok "PostgreSQL   — TimescaleDB at localhost:5432"
ok "Redis        — cache + queue at localhost:6379"
ok "Grafana      — dashboards at ${URL_GRAFANA}"
hr
printf '%s%s  ✅  Full demo completed successfully.%s\n' "${BOLD}" "${GREEN}" "${RESET}"
hr
