#!/usr/bin/env bash
# =============================================================================
# demo_kafka.sh — Live demo of Kafka/Redpanda producer+consumer round-trip
# =============================================================================
# Usage: bash scripts/demo/demo_kafka.sh
#
# Validates Redpanda health, prints Console URL, runs the producer in --demo
# mode and consumes 3 messages from topic signals.h5 to show end-to-end flow.
# Broker  : localhost:19092 (external) / redpanda:9092 (in-network)
# Topic   : signals.h5 (JSON messages)
# Console : http://localhost:8088
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

REDPANDA="usdcop-redpanda"
PRODUCER="usdcop-kafka-producer"
CONSUMER="usdcop-kafka-consumer"
CONSOLE_URL="http://localhost:8088"
TOPIC="signals.h5"

step "Step 1: Check Redpanda broker"
if ! docker ps --format '{{.Names}}' | grep -q "^${REDPANDA}$"; then
    err "Container ${REDPANDA} is NOT running."
    echo "Start it with: docker compose -f docker-compose.compact.yml up -d redpanda"
    exit 1
fi

# Health via cluster info (rpk is shipped inside Redpanda image)
if docker exec "${REDPANDA}" rpk cluster info >/dev/null 2>&1; then
    ok "Redpanda cluster reachable"
else
    warn "rpk cluster info failed — proceeding, but broker may not be ready"
fi

step "Step 2: Redpanda Console"
ok "Open the Console UI in your browser to watch messages live:"
echo "     ${BOLD}${CONSOLE_URL}${RESET}"
echo "     Topic to watch: ${BOLD}${TOPIC}${RESET}"

step "Step 3: Run producer in --demo mode"
if ! docker ps --format '{{.Names}}' | grep -q "^${PRODUCER}$"; then
    err "Container ${PRODUCER} is NOT running."
    echo "Start it with: docker compose -f docker-compose.compact.yml up -d kafka-producer"
    exit 1
fi
if docker exec "${PRODUCER}" python producer.py --demo; then
    ok "Producer published demo messages to '${TOPIC}'"
else
    err "Producer failed"
    exit 1
fi

step "Step 4: Consume 3 messages (timeout 20s)"
if ! docker ps --format '{{.Names}}' | grep -q "^${CONSUMER}$"; then
    err "Container ${CONSUMER} is NOT running."
    echo "Start it with: docker compose -f docker-compose.compact.yml up -d kafka-consumer"
    exit 1
fi
if docker exec "${CONSUMER}" python consumer.py --count 3 --timeout 20; then
    ok "Consumer read messages from '${TOPIC}'"
else
    err "Consumer failed"
    exit 1
fi

step "Summary"
ok "Kafka round-trip OK"
echo "  - Broker   : localhost:19092 (external) / redpanda:9092 (in-network)"
echo "  - Topic    : ${TOPIC} (JSON)"
echo "  - Console  : ${CONSOLE_URL}"
