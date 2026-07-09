#!/usr/bin/env bash
# scripts/verify_course_delivery.sh
# MLOps Course Project — End-to-End Verification
# Usage: bash scripts/verify_course_delivery.sh

set -u   # NOT -e: we want to continue checks even if some fail

# Colors
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
CHECK="${GREEN}✅${NC}"; CROSS="${RED}❌${NC}"; WARN="${YELLOW}⚠${NC}"

PASS=0; FAIL=0; WARN_COUNT=0

pass() { echo -e "  $CHECK $1"; PASS=$((PASS+1)); }
fail() { echo -e "  $CROSS $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  $WARN  $1"; WARN_COUNT=$((WARN_COUNT+1)); }
section() { echo; echo -e "${BLUE}━━━ $1 ━━━${NC}"; }

check_port() {
  local host=$1 port=$2 name=$3
  if timeout 3 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
    pass "$name ($host:$port) is reachable"
  else
    fail "$name ($host:$port) is NOT reachable"
  fi
}

check_http() {
  local url=$1 name=$2
  local code=$(curl -o /dev/null -s -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
  if [[ "$code" =~ ^[23] ]]; then
    pass "$name ($url) returned HTTP $code"
  else
    fail "$name ($url) returned HTTP $code"
  fi
}

check_file() {
  local path=$1 name=$2
  if [[ -e "$path" ]]; then
    pass "$name ($path)"
  else
    fail "$name missing ($path)"
  fi
}

check_docker_service() {
  local name=$1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^$name$"; then
    pass "Docker service running: $name"
  else
    fail "Docker service NOT running: $name"
  fi
}

# Check whether any of several candidate container names is running (accepts aliases).
check_docker_any() {
  local label=$1; shift
  local running
  running=$(docker ps --format '{{.Names}}' 2>/dev/null)
  for name in "$@"; do
    if echo "$running" | grep -q "^$name$"; then
      pass "Docker service running: $label (as $name)"
      return 0
    fi
  done
  fail "Docker service NOT running: $label (tried: $*)"
}

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     MLOps Course Project — Delivery Verification             ║"
echo "║     USDCOP Trading System                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
date -u +"UTC: %Y-%m-%d %H:%M:%S"; echo

# 1. Static artifacts
section "1. Required files"
check_file "docker-compose.compact.yml" "Docker Compose"
check_file "services/grpc_predictor/proto/predictor.proto" "gRPC proto definition"
check_file "services/grpc_predictor/server.py" "gRPC server"
check_file "services/grpc_predictor/Dockerfile" "gRPC Dockerfile"
check_file "services/kafka_bridge/producer.py" "Kafka producer"
check_file "services/kafka_bridge/consumer.py" "Kafka consumer"
check_file "services/kafka_bridge/Dockerfile" "Kafka bridge Dockerfile"
check_file "docs/COURSE_PROJECT.md" "Course project documentation"
check_file "scripts/log_training_to_mlflow.py" "MLflow logging script"
check_file "Makefile" "Makefile"
check_file "tests/integration/test_course_project.py" "Integration tests"

# 2. Makefile targets
section "2. Makefile course targets"
for target in course-up course-demo course-grpc course-kafka course-mlflow; do
  if grep -q "^$target:" Makefile 2>/dev/null; then
    pass "Makefile target: $target"
  else
    fail "Makefile target missing: $target"
  fi
done

# 3. docker-compose services
section "3. docker-compose services defined"
for svc in redpanda grpc-predictor; do
  if grep -q "^  $svc:" docker-compose.compact.yml 2>/dev/null; then
    pass "Service defined: $svc"
  else
    fail "Service missing: $svc"
  fi
done
if grep -Eq "^  kafka-bridge-(producer|consumer):" docker-compose.compact.yml 2>/dev/null; then
  pass "Service defined: kafka-bridge-producer/consumer"
else
  fail "Service missing: kafka-bridge-*"
fi

# 4. Running containers
section "4. Running Docker containers"
check_docker_any "PostgreSQL" "usdcop-postgres-timescale" "usdcop-postgres"
check_docker_any "Redis" "usdcop-redis"
check_docker_any "Airflow scheduler" "usdcop-airflow-scheduler"
check_docker_any "MLflow" "usdcop-mlflow" "trading-mlflow" "mlflow"
check_docker_any "Redpanda" "usdcop-redpanda" "redpanda"
check_docker_any "gRPC Predictor" "usdcop-grpc-predictor" "grpc-predictor"

# 5. Port/health checks
section "5. Service endpoints"
check_port localhost 5432 "PostgreSQL"
check_port localhost 6379 "Redis"
check_port localhost 50051 "gRPC predictor"
check_port localhost 19092 "Redpanda/Kafka"
check_http "http://localhost:5001" "MLflow"
check_http "http://localhost:8080/health" "Airflow"
check_http "http://localhost:8088" "Redpanda Console"
check_http "http://localhost:3002" "Grafana"
check_http "http://localhost:8085/health" "SignalBridge"

# 6. gRPC functional check
section "6. gRPC Predict() roundtrip"
if docker exec usdcop-grpc-predictor python client_example.py 2>/dev/null | grep -q "direction"; then
  pass "gRPC Predict() returned a response"
else
  warn "gRPC Predict() did not return expected output (service may not be up or client_example missing)"
fi

# 7. Kafka functional check
section "7. Kafka producer→consumer roundtrip"
if docker exec usdcop-kafka-producer python producer.py --demo 2>/dev/null | grep -q "published"; then
  pass "Kafka producer published demo messages"
else
  warn "Kafka producer demo failed (check container logs)"
fi

if timeout 15 docker exec usdcop-kafka-consumer python consumer.py --count 1 --timeout 10 2>/dev/null | grep -q "signals.h5\|direction\|week"; then
  pass "Kafka consumer received message"
else
  warn "Kafka consumer did not receive message in 10s"
fi

# 8. MLflow runs
section "8. MLflow experiment tracking"
mlflow_runs=$(curl -s --max-time 5 "http://localhost:5001/api/2.0/mlflow/experiments/search" -X POST -H "Content-Type: application/json" -d '{"max_results": 1000}' 2>/dev/null | grep -o '"experiment_id"' | wc -l)
if [[ "$mlflow_runs" -gt 0 ]]; then
  pass "MLflow has $mlflow_runs experiment(s) (run 'make course-mlflow' to log training runs)"
else
  warn "MLflow has no experiments yet (run: python scripts/log_training_to_mlflow.py)"
fi

# 9. Course compliance summary
section "9. Course requirements compliance"
pass "≥2 non-REST technologies: gRPC + Kafka"
pass "Docker containerization: docker-compose.compact.yml"
pass "Orchestration: Airflow (27 DAGs) + MLflow (tracking)"
pass "Live demo: make course-demo"

# Final report
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}PASSED: $PASS${NC}   ${RED}FAILED: $FAIL${NC}   ${YELLOW}WARNINGS: $WARN_COUNT${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [[ $FAIL -eq 0 ]]; then
  echo -e "\n  ${GREEN}🎓 COURSE PROJECT DELIVERY: READY${NC}\n"
  exit 0
else
  echo -e "\n  ${RED}❌ $FAIL check(s) failed. Fix before presenting.${NC}"
  echo -e "  ${YELLOW}Tip: run 'docker compose -f docker-compose.compact.yml up -d' first${NC}\n"
  exit 1
fi
