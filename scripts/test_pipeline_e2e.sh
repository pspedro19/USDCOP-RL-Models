#!/bin/bash
#
# End-to-End Pipeline Test
# Tests L0→L6 pipeline execution and validates manifests
#

set -e

echo "============================================"
echo "  E2E Pipeline Test - L0→L6"
echo "============================================"
echo ""

# Configuration
MAX_WAIT_MINUTES=30
CHECK_INTERVAL_SECONDS=60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo "ℹ️  $1"
}

# Step 1: Verify infrastructure
echo "=== Step 1: Verify Infrastructure ==="
info "Checking Docker containers..."

if docker ps | grep -q usdcop-postgres-timescale; then
    success "PostgreSQL running"
else
    error "PostgreSQL not running"
    exit 1
fi

if docker ps | grep -q usdcop-minio; then
    success "MinIO running"
else
    error "MinIO not running"
    exit 1
fi

if docker ps | grep -q airflow; then
    success "Airflow running"
else
    warning "Airflow not running - attempting to start..."
    docker-compose up -d airflow-webserver airflow-scheduler
    sleep 10
fi

echo ""

# Step 2: Check PostgreSQL data
echo "=== Step 2: Check PostgreSQL Data ==="
info "Querying market_data table..."

RECORD_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data;" 2>/dev/null | tr -d ' ')

if [ "$RECORD_COUNT" -gt 0 ]; then
    success "PostgreSQL has $RECORD_COUNT records"
else
    error "PostgreSQL market_data table is empty"
    exit 1
fi

echo ""

# Step 3: Trigger DAGs
echo "=== Step 3: Trigger Pipeline DAGs ==="

DAGS=(
    "usdcop_m5__01_l0_intelligent_acquire"
    "usdcop_m5__02_l1_standardize"
    "usdcop_m5__03_l2_prepare"
    "usdcop_m5__04_l3_feature"
    "usdcop_m5__05_l4_rlready"
    "usdcop_m5__06_l5_serving"
    "usdcop_m5__07_l6_backtest_referencia"
)

for dag in "${DAGS[@]}"; do
    info "Triggering $dag..."

    if docker exec airflow-webserver airflow dags trigger "$dag" 2>/dev/null; then
        success "$dag triggered"
    else
        warning "$dag trigger failed - may already be running"
    fi

    sleep 5
done

echo ""

# Step 4: Wait for DAGs to complete
echo "=== Step 4: Wait for DAG Completion ==="
info "Waiting up to $MAX_WAIT_MINUTES minutes for DAGs to complete..."

for i in $(seq 1 $MAX_WAIT_MINUTES); do
    echo -n "Check $i/$MAX_WAIT_MINUTES... "

    # Count failed DAGs
    FAILED_COUNT=$(docker exec airflow-webserver airflow dags list-runs 2>/dev/null | grep -c FAILED || true)

    # Count running DAGs
    RUNNING_COUNT=$(docker exec airflow-webserver airflow dags list-runs 2>/dev/null | grep -c RUNNING || true)

    if [ "$FAILED_COUNT" -gt 0 ]; then
        error "Found $FAILED_COUNT failed DAGs"
        docker exec airflow-webserver airflow dags list-runs | grep FAILED
        exit 1
    fi

    if [ "$RUNNING_COUNT" -eq 0 ]; then
        success "All DAGs completed"
        break
    fi

    echo "$RUNNING_COUNT DAGs still running"
    sleep $CHECK_INTERVAL_SECONDS
done

if [ "$RUNNING_COUNT" -gt 0 ]; then
    warning "Some DAGs still running after $MAX_WAIT_MINUTES minutes"
fi

echo ""

# Step 5: Verify MinIO manifests
echo "=== Step 5: Verify MinIO Manifests ==="
info "Checking for manifest files..."

# Setup MinIO alias
docker exec usdcop-minio mc alias set minio http://localhost:9000 minioadmin minioadmin123 2>/dev/null

LAYERS=("l0" "l1" "l2" "l3" "l4" "l5" "l6")
MANIFEST_COUNT=0

for layer in "${LAYERS[@]}"; do
    # Try to find manifest in any bucket matching the layer
    MANIFEST=$(docker exec usdcop-minio mc find minio/ --name "${layer}_latest.json" 2>/dev/null | head -1)

    if [ -n "$MANIFEST" ]; then
        success "$layer manifest found: $MANIFEST"

        # Verify manifest content
        VALIDATION_STATUS=$(docker exec usdcop-minio mc cat "$MANIFEST" 2>/dev/null | jq -r '.validation_status // "unknown"')

        if [ "$VALIDATION_STATUS" = "passed" ]; then
            success "$layer validation passed"
            ((MANIFEST_COUNT++))
        else
            warning "$layer validation status: $VALIDATION_STATUS"
        fi
    else
        error "$layer manifest not found"
    fi
done

echo ""

# Step 6: Summary
echo "=== Summary ==="
echo "PostgreSQL Records: $RECORD_COUNT"
echo "Manifests Found: $MANIFEST_COUNT/7"
echo ""

if [ "$MANIFEST_COUNT" -eq 7 ]; then
    success "Pipeline E2E test PASSED - All 7 manifests created successfully"
    exit 0
elif [ "$MANIFEST_COUNT" -gt 0 ]; then
    warning "Pipeline E2E test PARTIAL - Only $MANIFEST_COUNT/7 manifests created"
    exit 1
else
    error "Pipeline E2E test FAILED - No manifests created"
    exit 1
fi
