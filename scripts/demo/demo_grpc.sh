#!/usr/bin/env bash
# =============================================================================
# demo_grpc.sh — Live demo of the gRPC PredictorService
# =============================================================================
# Usage: bash scripts/demo/demo_grpc.sh
#
# Validates usdcop-grpc-predictor container health and runs the bundled
# client_example.py inside the container to show a real Predict() round-trip.
# Contract: services/grpc_predictor/proto/predictor.proto
# Address : localhost:50051 (host) / grpc-predictor:50051 (in-network)
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

CONTAINER="usdcop-grpc-predictor"
COMPOSE_FILE="docker-compose.compact.yml"

step "Step 1: Check ${CONTAINER} is running"
if ! command -v docker >/dev/null 2>&1; then
    err "docker CLI not found in PATH"
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    err "Container ${CONTAINER} is NOT running."
    cat <<EOF
${YELLOW}To start it:${RESET}
    docker compose -f ${COMPOSE_FILE} up -d grpc-predictor
    # or: make course-up
EOF
    exit 1
fi
ok "Container ${CONTAINER} is up"

step "Step 2: Health-check the container"
HEALTH="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${CONTAINER}" 2>/dev/null || echo "unknown")"
case "${HEALTH}" in
    healthy) ok "Health = healthy" ;;
    starting) warn "Health = starting (proceeding anyway)" ;;
    none) warn "No HEALTHCHECK defined (proceeding)" ;;
    *) warn "Health = ${HEALTH} (proceeding)" ;;
esac

step "Step 3: Run client_example.py inside the container"
echo "${CYAN}Contract:${RESET} PredictorService.Predict(features) + HealthCheck()"
echo "${CYAN}Address :${RESET} grpc-predictor:50051 (in-network)"
echo ""

if ! docker exec "${CONTAINER}" test -f /app/client_example.py 2>/dev/null \
   && ! docker exec "${CONTAINER}" test -f client_example.py 2>/dev/null; then
    err "client_example.py not found inside ${CONTAINER}"
    echo "Expected at /app/client_example.py or CWD of the container."
    exit 1
fi

if docker exec "${CONTAINER}" python client_example.py; then
    ok "gRPC round-trip completed"
else
    err "client_example.py failed"
    exit 1
fi

step "Summary"
ok "gRPC Predictor demo OK"
echo "  - Service : PredictorService (localhost:50051)"
echo "  - Methods : Predict(features), HealthCheck()"
echo "  - Proto   : services/grpc_predictor/proto/predictor.proto"
