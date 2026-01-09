#!/bin/bash
# USD/COP Trading System - Test Suite Runner
# ==========================================
#
# Quick test execution commands
# Author: Pedro @ Lean Tech Solutions
# Date: 2025-12-16

set -e

echo ""
echo "============================================================"
echo " USD/COP Trading System - Test Suite"
echo "============================================================"
echo ""

show_help() {
    echo "Usage: ./RUN_TESTS.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all          - Run all tests (unit + integration)"
    echo "  unit         - Run only unit tests (fast)"
    echo "  integration  - Run only integration tests"
    echo "  parity       - Run CRITICAL feature parity test"
    echo "  quick        - Run quick smoke tests"
    echo "  coverage     - Run all tests with coverage report"
    echo ""
    echo "Examples:"
    echo "  ./RUN_TESTS.sh unit"
    echo "  ./RUN_TESTS.sh parity"
    echo "  ./RUN_TESTS.sh coverage"
    echo ""
}

run_all() {
    echo "Running all tests..."
    pytest tests/ -v
}

run_unit() {
    echo "Running unit tests only..."
    pytest tests/unit/ -v
}

run_integration() {
    echo "Running integration tests..."
    pytest tests/integration/ -v
}

run_parity() {
    echo "Running CRITICAL feature parity test..."
    pytest tests/integration/test_feature_parity.py::TestFeatureParityLegacy::test_features_match_legacy -v
}

run_quick() {
    echo "Running quick smoke tests..."
    pytest tests/unit/test_config_loader.py tests/unit/test_feature_builder.py::TestObservationSpace -v
}

run_coverage() {
    echo "Running all tests with coverage..."
    pytest tests/ --cov=services --cov-report=html --cov-report=term
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
}

# Main script
case "$1" in
    all)
        run_all
        ;;
    unit)
        run_unit
        ;;
    integration)
        run_integration
        ;;
    parity)
        run_parity
        ;;
    quick)
        run_quick
        ;;
    coverage)
        run_coverage
        ;;
    *)
        show_help
        ;;
esac
