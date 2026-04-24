#!/usr/bin/env bash
# =============================================================================
# demo_mlflow.sh — Live demo of MLflow tracking UI + experiment summary
# =============================================================================
# Usage: bash scripts/demo/demo_mlflow.sh
#
# Checks MLflow is reachable at localhost:5001, backfills training runs if
# the tracking store is empty, then prints a summary of experiments + runs.
# Tracking URL: http://localhost:5001
# =============================================================================
set -euo pipefail

# --- Color helpers ----------------------------------------------------------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
    GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; RED="$(tput setaf 1)"
    CYAN="$(tput setaf 6)"; BOLD="$(tput bold)"; RESET="$(tput sgr0)"
else
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
fi
ok()   { printf '%s[✅]%s %s\n' "${GREEN}" "${RESET}" "$*"; }
warn() { printf '%s[⚠ ]%s %s\n' "${YELLOW}" "${RESET}" "$*"; }
err()  { printf '%s[❌]%s %s\n' "${RED}" "${RESET}" "$*" >&2; }
step() { printf '\n%s%s==> %s%s\n' "${BOLD}" "${CYAN}" "$*" "${RESET}"; }

MLFLOW_URL="http://localhost:5001"
LOG_SCRIPT="scripts/log_training_to_mlflow.py"

step "Step 1: Check MLflow is up (${MLFLOW_URL})"
if ! command -v curl >/dev/null 2>&1; then
    err "curl not found in PATH"
    exit 1
fi

HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${MLFLOW_URL}/" || echo "000")"
if [[ "${HTTP_CODE}" == "000" ]]; then
    err "MLflow not reachable at ${MLFLOW_URL}"
    echo "Start it with: docker compose -f docker-compose.compact.yml up -d mlflow"
    exit 1
fi
ok "MLflow HTTP ${HTTP_CODE} — tracking server is up"

step "Step 2: Count experiments + runs via REST API"
# MLflow 2.x REST: /api/2.0/mlflow/experiments/search
EXP_JSON="$(curl -s --max-time 5 -X POST "${MLFLOW_URL}/api/2.0/mlflow/experiments/search" \
    -H 'Content-Type: application/json' -d '{"max_results": 100}' || echo '{}')"

# Count experiments (jq if available, else fall back to python, else grep)
count_json_array() {
    local json="$1" key="$2"
    if command -v jq >/dev/null 2>&1; then
        echo "${json}" | jq -r ".${key} // [] | length"
    elif command -v python3 >/dev/null 2>&1; then
        echo "${json}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('${key}') or []))" 2>/dev/null || echo "0"
    else
        # Crude fallback: count occurrences of the key's element markers
        echo "${json}" | grep -o "\"${key}\"" | wc -l
    fi
}

EXP_COUNT="$(count_json_array "${EXP_JSON}" "experiments")"
EXP_COUNT="${EXP_COUNT:-0}"
echo "  Experiments found: ${BOLD}${EXP_COUNT}${RESET}"

# Total run count across all experiments
TOTAL_RUNS=0
if [[ "${EXP_COUNT}" != "0" ]] && command -v python3 >/dev/null 2>&1; then
    TOTAL_RUNS="$(python3 <<PY 2>/dev/null || echo 0
import json, urllib.request
try:
    req = urllib.request.Request(
        "${MLFLOW_URL}/api/2.0/mlflow/experiments/search",
        data=b'{"max_results": 100}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    exps = json.loads(urllib.request.urlopen(req, timeout=5).read()).get("experiments", []) or []
    total = 0
    for e in exps:
        body = json.dumps({"experiment_ids": [e["experiment_id"]], "max_results": 1000}).encode()
        r = urllib.request.Request(
            "${MLFLOW_URL}/api/2.0/mlflow/runs/search",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        runs = json.loads(urllib.request.urlopen(r, timeout=5).read()).get("runs", []) or []
        total += len(runs)
    print(total)
except Exception:
    print(0)
PY
)"
fi
echo "  Runs across all experiments: ${BOLD}${TOTAL_RUNS}${RESET}"

step "Step 3: Backfill runs if tracking store is empty"
if [[ "${TOTAL_RUNS:-0}" == "0" ]]; then
    warn "No runs found. Attempting to backfill from local training artifacts..."
    if [[ -f "${LOG_SCRIPT}" ]]; then
        if python "${LOG_SCRIPT}"; then
            ok "Backfill script completed"
        else
            warn "Backfill script exited non-zero (continuing)"
        fi
    else
        warn "${LOG_SCRIPT} does not exist — skipping backfill."
        echo "     Create it to auto-populate MLflow from existing model artifacts."
    fi
else
    ok "Tracking store already has ${TOTAL_RUNS} run(s) — no backfill needed"
fi

step "Step 4: Open the UI"
ok "MLflow UI : ${BOLD}${MLFLOW_URL}${RESET}"
echo "  Filter by experiment to see metrics, params, and artifacts."

step "Summary"
ok "MLflow demo OK"
echo "  - Tracking : ${MLFLOW_URL}"
echo "  - Experiments: ${EXP_COUNT}"
echo "  - Runs       : ${TOTAL_RUNS}"
