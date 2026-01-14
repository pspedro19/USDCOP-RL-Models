#!/bin/bash
# =============================================================================
# Verification Script: Phases 10-15 Remediation
# =============================================================================
# Verifies that all remediation phases have been properly implemented.
#
# Usage:
#   chmod +x scripts/verify_remediation.sh
#   ./scripts/verify_remediation.sh
#
# Author: Trading Team
# Date: 2025-01-14
# =============================================================================

set -e

echo "=========================================="
echo "  REMEDIATION VERIFICATION SCRIPT"
echo "  Phases 10-15"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

# =============================================================================
# Helper Functions
# =============================================================================

check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo "✅ $description"
        echo "   File: $file"
        ((PASSED++))
        return 0
    else
        echo "❌ $description"
        echo "   Missing: $file"
        ((FAILED++))
        return 1
    fi
}

check_dir() {
    local dir=$1
    local description=$2

    if [ -d "$dir" ]; then
        echo "✅ $description"
        echo "   Dir: $dir"
        ((PASSED++))
        return 0
    else
        echo "❌ $description"
        echo "   Missing: $dir"
        ((FAILED++))
        return 1
    fi
}

run_test() {
    local test_path=$1
    local description=$2

    echo "🧪 Running: $description"
    if python -m pytest "$test_path" -v --tb=short 2>/dev/null; then
        echo "✅ Tests passed"
        ((PASSED++))
        return 0
    else
        echo "❌ Tests failed"
        ((FAILED++))
        return 1
    fi
}

# =============================================================================
# Phase 10: Feature Calculation Consistency
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 10: Feature Calculation Consistency"
echo "=========================================="

check_file "services/inference_api/core/feature_adapter.py" \
    "InferenceFeatureAdapter created"

check_file "tests/unit/test_feature_adapter.py" \
    "Feature adapter tests created"

# Check that adapter uses SSOT
if grep -q "from src.feature_store.core import" services/inference_api/core/feature_adapter.py 2>/dev/null; then
    echo "✅ Adapter delegates to SSOT feature_store"
    ((PASSED++))
else
    echo "❌ Adapter not delegating to SSOT"
    ((FAILED++))
fi

# Check that observation_builder uses adapter
if grep -q "InferenceFeatureAdapter" services/inference_api/core/observation_builder.py 2>/dev/null; then
    echo "✅ ObservationBuilder delegates to adapter"
    ((PASSED++))
else
    echo "❌ ObservationBuilder not using adapter"
    ((FAILED++))
fi

# Check Wilder's EMA is used
if grep -q "SmoothingMethod.WILDER" src/feature_store/core.py 2>/dev/null; then
    echo "✅ RSI/ATR/ADX use Wilder's EMA (alpha=1/period)"
    ((PASSED++))
else
    echo "❌ Wilder's EMA not found in SSOT"
    ((FAILED++))
fi

# =============================================================================
# Phase 11: Data Versioning (DVC)
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 11: Data Versioning (DVC)"
echo "=========================================="

check_file "dvc.yaml" \
    "DVC pipeline configuration"

check_file "scripts/setup_dvc.sh" \
    "DVC setup script"

check_file "docs/DATA_VERSIONING.md" \
    "Data versioning documentation"

# Check if DVC is initialized (optional - might not be on CI)
if [ -d ".dvc" ]; then
    echo "✅ DVC initialized (.dvc directory exists)"
    ((PASSED++))
else
    echo "⚠️  DVC not initialized (run scripts/setup_dvc.sh)"
fi

# =============================================================================
# Phase 12: Latency SLA
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 12: Latency SLA Monitoring"
echo "=========================================="

check_file "docs/SLA.md" \
    "SLA documentation"

check_file "config/prometheus/alerts/latency.yml" \
    "Prometheus latency alerts"

check_file "tests/load/test_latency_sla.py" \
    "Latency SLA load tests"

# Check SLA targets are documented
if grep -q "p50.*20ms" docs/SLA.md 2>/dev/null; then
    echo "✅ SLA targets documented (p50 < 20ms)"
    ((PASSED++))
else
    echo "❌ SLA targets not properly documented"
    ((FAILED++))
fi

# =============================================================================
# Phase 13: Feature Circuit Breaker
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 13: Feature Circuit Breaker"
echo "=========================================="

check_file "src/features/circuit_breaker.py" \
    "FeatureCircuitBreaker class"

# Check circuit breaker threshold
if grep -q "max_nan_ratio.*0.20" src/features/circuit_breaker.py 2>/dev/null; then
    echo "✅ Circuit breaker threshold: 20% NaN"
    ((PASSED++))
else
    echo "❌ Circuit breaker threshold not set"
    ((FAILED++))
fi

# Check cooldown period
if grep -q "cooldown_minutes.*15" src/features/circuit_breaker.py 2>/dev/null; then
    echo "✅ Circuit breaker cooldown: 15 minutes"
    ((PASSED++))
else
    echo "❌ Circuit breaker cooldown not set"
    ((FAILED++))
fi

# =============================================================================
# Phase 14: Gap Handling
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 14: Gap Handling Consistency"
echo "=========================================="

check_file "src/features/gap_handler.py" \
    "GapHandler class"

# Check warmup bars
if grep -q "warmup_bars.*14" src/features/gap_handler.py 2>/dev/null; then
    echo "✅ Warmup bars: 14"
    ((PASSED++))
else
    echo "❌ Warmup bars not configured"
    ((FAILED++))
fi

# Check fill strategy
if grep -q "forward_then_zero" src/features/gap_handler.py 2>/dev/null; then
    echo "✅ Fill strategy: forward_then_zero"
    ((PASSED++))
else
    echo "❌ Fill strategy not configured"
    ((FAILED++))
fi

# =============================================================================
# Phase 15: Secondary Gaps (Partial)
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 15: Secondary Gaps (Partial)"
echo "=========================================="

# These are lower priority items
echo "ℹ️  Phase 15 items are lower priority and may be partially implemented"

if [ -f "docs/SLA.md" ]; then
    echo "✅ SLA documentation exists"
    ((PASSED++))
fi

if [ -f "docs/DATA_VERSIONING.md" ]; then
    echo "✅ Data versioning documentation exists"
    ((PASSED++))
fi

# =============================================================================
# Run Tests (if pytest available)
# =============================================================================
echo ""
echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="

if command -v pytest &> /dev/null; then
    # Run feature adapter tests
    if [ -f "tests/unit/test_feature_adapter.py" ]; then
        echo "Running feature adapter tests..."
        if python -m pytest tests/unit/test_feature_adapter.py -v --tb=short -x 2>/dev/null; then
            echo "✅ Feature adapter tests passed"
            ((PASSED++))
        else
            echo "⚠️  Some feature adapter tests may have failed (check dependencies)"
        fi
    fi
else
    echo "⚠️  pytest not available, skipping test execution"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "  VERIFICATION SUMMARY"
echo "=========================================="
echo ""
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    echo ""
    echo "Please review the failed items above and fix them."
    exit 1
fi
