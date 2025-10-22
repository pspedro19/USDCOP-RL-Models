#!/bin/bash
#
# Validate 100% Real Data
# Master validation script that runs all checks
#

set -e

echo "============================================"
echo "  100% REAL DATA VALIDATION"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check_pass() {
    ((TOTAL_CHECKS++))
    ((PASSED_CHECKS++))
    success "$1"
}

check_fail() {
    ((TOTAL_CHECKS++))
    ((FAILED_CHECKS++))
    error "$1"
}

# ============================================
# 1. Infrastructure Checks
# ============================================
echo "=== 1. Infrastructure Checks ==="

if docker ps | grep -q usdcop-postgres-timescale; then
    check_pass "PostgreSQL container running"
else
    check_fail "PostgreSQL container not running"
fi

if docker ps | grep -q usdcop-minio; then
    check_pass "MinIO container running"
else
    check_fail "MinIO container not running"
fi

if docker ps | grep -q usdcop-redis; then
    check_pass "Redis container running"
else
    check_fail "Redis container not running"
fi

if docker ps | grep -q usdcop-trading-api; then
    check_pass "Trading API container running"
else
    check_fail "Trading API container not running"
fi

if docker ps | grep -q usdcop-analytics-api; then
    check_pass "Analytics API container running"
else
    check_fail "Analytics API container not running"
fi

if docker ps | grep -q usdcop-dashboard; then
    check_pass "Dashboard container running"
else
    check_fail "Dashboard container not running"
fi

echo ""

# ============================================
# 2. PostgreSQL Data Checks
# ============================================
echo "=== 2. PostgreSQL Data Checks ==="

RECORD_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data;" 2>/dev/null | tr -d ' ')

if [ "$RECORD_COUNT" -gt 90000 ]; then
    check_pass "PostgreSQL has $RECORD_COUNT records (>90k)"
else
    check_fail "PostgreSQL has only $RECORD_COUNT records (<90k)"
fi

LATEST=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 FROM market_data;" 2>/dev/null | tr -d ' ')

if (( $(echo "$LATEST < 720" | bc -l) )); then
    check_pass "Latest data is ${LATEST}h old (<30 days)"
else
    check_fail "Latest data is ${LATEST}h old (>30 days)"
fi

echo ""

# ============================================
# 3. MinIO Manifest Checks
# ============================================
echo "=== 3. MinIO Manifest Checks ==="

docker exec usdcop-minio mc alias set minio http://localhost:9000 minioadmin minioadmin123 2>/dev/null || true

LAYERS=("l0" "l1" "l2" "l3" "l4" "l5" "l6")

for layer in "${LAYERS[@]}"; do
    MANIFEST=$(docker exec usdcop-minio mc find minio/ --name "${layer}_latest.json" 2>/dev/null | head -1)

    if [ -n "$MANIFEST" ]; then
        check_pass "${layer} manifest exists"
    else
        check_fail "${layer} manifest not found"
    fi
done

echo ""

# ============================================
# 4. Backend Code Checks
# ============================================
echo "=== 4. Backend Code Checks ==="

# Check for Math.random() in Python files
RANDOM_COUNT=$(grep -r "random\." /home/GlobalForex/USDCOP-RL-Models/services/*.py 2>/dev/null | grep -v "^#" | wc -l || echo "0")

if [ "$RANDOM_COUNT" -eq 0 ]; then
    check_pass "No random.* in Python backend"
else
    check_fail "Found $RANDOM_COUNT random.* calls in Python backend"
fi

# Check for "mock" in API files (excluding comments)
MOCK_COUNT=$(grep -ri "mock" /home/GlobalForex/USDCOP-RL-Models/services/*_api.py 2>/dev/null | grep -v "^#" | grep -v "# Mock" | wc -l || echo "0")

if [ "$MOCK_COUNT" -lt 5 ]; then
    check_pass "Minimal mock references in backend (<5)"
else
    warning "Found $MOCK_COUNT mock references in backend (review needed)"
fi

echo ""

# ============================================
# 5. Frontend Code Checks
# ============================================
echo "=== 5. Frontend Code Checks ==="

# Check for Math.random() in TypeScript files
TSX_RANDOM=$(grep -r "Math.random()" /home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/*.tsx 2>/dev/null | wc -l || echo "0")

if [ "$TSX_RANDOM" -eq 0 ]; then
    check_pass "No Math.random() in frontend components"
else
    check_fail "Found $TSX_RANDOM Math.random() calls in frontend"
fi

# Check for hardcoded useState arrays
USESTATE_ARRAYS=$(grep -r "useState(\[" /home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/*.tsx 2>/dev/null | grep -v "useState(\[\])" | wc -l || echo "0")

if [ "$USESTATE_ARRAYS" -eq 0 ]; then
    check_pass "No hardcoded useState arrays in frontend"
else
    check_fail "Found $USESTATE_ARRAYS hardcoded useState arrays in frontend"
fi

echo ""

# ============================================
# 6. API Endpoint Checks
# ============================================
echo "=== 6. API Endpoint Checks ==="

# L5 Models must read from MinIO
L5_SOURCE=$(curl -s http://localhost:8004/api/pipeline/l5/models 2>/dev/null | jq -r '.metadata.source // "unknown"')

if [ "$L5_SOURCE" = "minio_manifest" ]; then
    check_pass "L5 models read from MinIO manifest"
elif [ "$L5_SOURCE" = "unknown" ]; then
    check_fail "L5 models endpoint unreachable"
else
    check_fail "L5 models source is '$L5_SOURCE', not 'minio_manifest'"
fi

# L0 statistics must have real data
L0_RECORDS=$(curl -s http://localhost:8004/api/pipeline/l0/statistics 2>/dev/null | jq -r '.total_records // 0')

if [ "$L0_RECORDS" -gt 90000 ]; then
    check_pass "L0 statistics show $L0_RECORDS records"
else
    check_fail "L0 statistics show only $L0_RECORDS records"
fi

# ML Analytics models must return count
ML_MODELS=$(curl -s http://localhost:8005/api/ml-analytics/models?action=list 2>/dev/null | jq -r '.count // 0')

if [ "$ML_MODELS" -gt 0 ]; then
    check_pass "ML Analytics models returns $ML_MODELS models"
else
    check_fail "ML Analytics models returns 0 models"
fi

# Trading Analytics RL metrics must return valid data
RL_TRADES=$(curl -s http://localhost:8001/api/analytics/rl-metrics 2>/dev/null | jq -r '.metrics.tradesPerEpisode // 0')

if [ "$RL_TRADES" -gt 0 ]; then
    check_pass "RL metrics show $RL_TRADES trades per episode"
else
    check_fail "RL metrics show 0 trades per episode"
fi

echo ""

# ============================================
# 7. Frontend Health Check
# ============================================
echo "=== 7. Frontend Health Check ==="

DASHBOARD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 2>/dev/null)

if [ "$DASHBOARD_STATUS" = "200" ]; then
    check_pass "Dashboard is accessible (HTTP 200)"
else
    check_fail "Dashboard returned HTTP $DASHBOARD_STATUS"
fi

echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "  VALIDATION SUMMARY"
echo "============================================"
echo ""
echo "Total Checks: $TOTAL_CHECKS"
echo "Passed: $PASSED_CHECKS"
echo "Failed: $FAILED_CHECKS"
echo ""

PERCENTAGE=$(echo "scale=1; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc)

if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}       🎉 100% VALIDATION PASSED 🎉${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    success "Sistema alcanzó 100/100 datos reales"
    exit 0
else
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}    ⚠️  VALIDATION: ${PERCENTAGE}% PASSED${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    warning "Sistema está al ${PERCENTAGE}% - Revisar $FAILED_CHECKS checks fallidos"
    exit 1
fi
