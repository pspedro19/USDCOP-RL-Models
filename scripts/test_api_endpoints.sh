#!/bin/bash
#
# API Endpoints Test
# Validates all backend APIs return real data (no mock/hardcoded)
#

set -e

echo "============================================"
echo "  API Endpoints Test"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
info() { echo "ℹ️  $1"; }

# Configuration
API_BASE="http://localhost"
PIPELINE_API_PORT=8004
ML_ANALYTICS_PORT=8005
TRADING_ANALYTICS_PORT=8001
TRADING_API_PORT=8000

PASSED=0
FAILED=0

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local jq_check=$3
    local expected=$4

    info "Testing: $name"

    RESPONSE=$(curl -s "$url" 2>&1)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>&1)

    # Check HTTP status
    if [ "$HTTP_CODE" != "200" ]; then
        error "$name failed with HTTP $HTTP_CODE"
        ((FAILED++))
        return 1
    fi

    # Check JSON structure
    if [ -n "$jq_check" ]; then
        RESULT=$(echo "$RESPONSE" | jq -r "$jq_check" 2>/dev/null)

        if [ -z "$RESULT" ] || [ "$RESULT" = "null" ]; then
            error "$name returned null/empty for: $jq_check"
            ((FAILED++))
            return 1
        fi

        # Check expected value if provided
        if [ -n "$expected" ] && [ "$RESULT" != "$expected" ]; then
            warning "$name returned '$RESULT', expected '$expected'"
        fi
    fi

    success "$name passed"
    ((PASSED++))
    return 0
}

# ============================================
# Pipeline Data API Tests (port 8004)
# ============================================
echo "=== Pipeline Data API (port $PIPELINE_API_PORT) ==="

test_endpoint \
    "L0 Statistics" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l0/statistics" \
    ".total_records" \
    ""

test_endpoint \
    "L0 Raw Data" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l0/raw-data?limit=10" \
    ".data | length" \
    ""

test_endpoint \
    "L1 Episodes" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l1/episodes?limit=10" \
    ".episodes | length" \
    ""

test_endpoint \
    "L1 Quality Report" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l1/quality-report" \
    ".quality_score" \
    ""

test_endpoint \
    "L2 Prepared Data" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l2/prepared-data?limit=100" \
    ".pass" \
    ""

test_endpoint \
    "L3 Features" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l3/features?limit=100" \
    ".metadata.features_count" \
    ""

test_endpoint \
    "L3 Forward IC" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l3/forward-ic?limit=100" \
    ".summary.pass" \
    ""

test_endpoint \
    "L4 Contract" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l4/contract" \
    ".features_count" \
    "17"

test_endpoint \
    "L4 Quality Check" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l4/quality-check" \
    ".overall_pass" \
    ""

test_endpoint \
    "L5 Models (MUST read from MinIO)" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l5/models" \
    ".metadata.source" \
    "minio_manifest"

test_endpoint \
    "L6 Backtest Results (MUST read from MinIO)" \
    "${API_BASE}:${PIPELINE_API_PORT}/api/pipeline/l6/backtest-results" \
    ".kpis.top_bar.CAGR" \
    ""

echo ""

# ============================================
# ML Analytics API Tests (port 8005)
# ============================================
echo "=== ML Analytics API (port $ML_ANALYTICS_PORT) ==="

test_endpoint \
    "Models List" \
    "${API_BASE}:${ML_ANALYTICS_PORT}/api/ml-analytics/models?action=list" \
    ".count" \
    ""

test_endpoint \
    "Health Summary" \
    "${API_BASE}:${ML_ANALYTICS_PORT}/api/ml-analytics/health?action=summary" \
    ".data | length" \
    ""

test_endpoint \
    "Predictions Data" \
    "${API_BASE}:${ML_ANALYTICS_PORT}/api/ml-analytics/predictions?action=data&limit=10" \
    ".count" \
    ""

test_endpoint \
    "Predictions Metrics" \
    "${API_BASE}:${ML_ANALYTICS_PORT}/api/ml-analytics/predictions?action=metrics" \
    ".data.metrics.mse" \
    ""

echo ""

# ============================================
# Trading Analytics API Tests (port 8001)
# ============================================
echo "=== Trading Analytics API (port $TRADING_ANALYTICS_PORT) ==="

test_endpoint \
    "RL Metrics" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/rl-metrics?days=30" \
    ".metrics.tradesPerEpisode" \
    ""

test_endpoint \
    "Performance KPIs" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/performance-kpis?days=90" \
    ".kpis.sharpeRatio" \
    ""

test_endpoint \
    "Production Gates" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/production-gates?days=90" \
    ".gates | length" \
    "6"

test_endpoint \
    "Risk Metrics" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/risk-metrics?days=30" \
    ".risk_metrics.portfolioValue" \
    ""

test_endpoint \
    "Session P&L" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/session-pnl" \
    ".session_pnl" \
    ""

test_endpoint \
    "Market Conditions" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/market-conditions?days=30" \
    ".conditions | length" \
    "6"

test_endpoint \
    "Spread Proxy" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/spread-proxy?days=30" \
    ".statistics.mean_bps" \
    ""

test_endpoint \
    "Session Progress" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/session-progress" \
    ".status" \
    ""

test_endpoint \
    "Order Flow" \
    "${API_BASE}:${TRADING_ANALYTICS_PORT}/api/analytics/order-flow?window=60" \
    ".order_flow.imbalance" \
    ""

echo ""

# ============================================
# Summary
# ============================================
echo "=== Summary ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    success "All API endpoint tests PASSED"
    exit 0
else
    error "$FAILED API endpoint tests FAILED"
    exit 1
fi
