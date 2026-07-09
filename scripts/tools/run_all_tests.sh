#!/bin/bash
# ============================================================================
# Run All Tests for USDCOP Trading System
# ============================================================================
#
# Executes all test suites with coverage reporting
# Exit code: 0 if all tests pass, non-zero if any fail
#
# Usage:
#   ./scripts/run_all_tests.sh              # Run all tests
#   ./scripts/run_all_tests.sh unit         # Run only unit tests
#   ./scripts/run_all_tests.sh integration  # Run only integration tests
#   ./scripts/run_all_tests.sh load         # Run only load tests
#   ./scripts/run_all_tests.sh e2e          # Run only e2e tests
#   ./scripts/run_all_tests.sh --fast       # Skip slow tests
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="${PROJECT_ROOT}/tests"
REPORTS_DIR="${PROJECT_ROOT}/test-reports"
COVERAGE_DIR="${REPORTS_DIR}/coverage"
HTML_REPORT="${REPORTS_DIR}/test-report.html"
COVERAGE_THRESHOLD=80

# Create reports directory
mkdir -p "${REPORTS_DIR}"
mkdir -p "${COVERAGE_DIR}"

# Print header
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}USDCOP Trading System - Test Suite Execution${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install -r tests/requirements-test.txt"
    exit 1
fi

# Parse arguments
TEST_SUITE="all"
SKIP_SLOW=""
PARALLEL=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        unit|integration|load|e2e)
            TEST_SUITE="$1"
            shift
            ;;
        --fast)
            SKIP_SLOW="-m 'not slow'"
            shift
            ;;
        --parallel)
            PARALLEL="-n auto"
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [unit|integration|load|e2e] [--fast] [--parallel] [--verbose]"
            echo ""
            echo "Options:"
            echo "  unit          Run only unit tests"
            echo "  integration   Run only integration tests"
            echo "  load          Run only load tests"
            echo "  e2e           Run only end-to-end tests"
            echo "  --fast        Skip slow tests"
            echo "  --parallel    Run tests in parallel"
            echo "  --verbose     Verbose output"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set test path based on suite
case $TEST_SUITE in
    unit)
        TEST_PATH="${TESTS_DIR}/unit"
        TEST_MARKER="-m unit"
        echo -e "${YELLOW}Running Unit Tests...${NC}"
        ;;
    integration)
        TEST_PATH="${TESTS_DIR}/integration"
        TEST_MARKER="-m integration"
        echo -e "${YELLOW}Running Integration Tests...${NC}"
        ;;
    load)
        TEST_PATH="${TESTS_DIR}/load"
        TEST_MARKER="-m load"
        echo -e "${YELLOW}Running Load Tests...${NC}"
        ;;
    e2e)
        TEST_PATH="${TESTS_DIR}/e2e"
        TEST_MARKER="-m e2e"
        echo -e "${YELLOW}Running End-to-End Tests...${NC}"
        ;;
    all)
        TEST_PATH="${TESTS_DIR}"
        TEST_MARKER=""
        echo -e "${YELLOW}Running All Test Suites...${NC}"
        ;;
esac

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Test Suite:     ${TEST_SUITE}"
echo "  Test Path:      ${TEST_PATH}"
echo "  Skip Slow:      $([ -n "${SKIP_SLOW}" ] && echo 'Yes' || echo 'No')"
echo "  Parallel:       $([ -n "${PARALLEL}" ] && echo 'Yes' || echo 'No')"
echo "  Coverage Target: ${COVERAGE_THRESHOLD}%"
echo ""

# Set environment for testing
export TESTING=true
export LOG_LEVEL=WARNING
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Load test environment variables if .env.test exists
if [ -f "${PROJECT_ROOT}/.env.test" ]; then
    echo -e "${BLUE}Loading test environment from .env.test${NC}"
    export $(cat "${PROJECT_ROOT}/.env.test" | grep -v '^#' | xargs)
fi

# Run tests with coverage
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Executing Tests...${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

START_TIME=$(date +%s)

pytest "${TEST_PATH}" \
    ${TEST_MARKER} \
    ${SKIP_SLOW} \
    ${PARALLEL} \
    ${VERBOSE} \
    --cov="${PROJECT_ROOT}/services" \
    --cov="${PROJECT_ROOT}/app" \
    --cov-report=html:"${COVERAGE_DIR}" \
    --cov-report=term-missing \
    --cov-report=xml:"${COVERAGE_DIR}/coverage.xml" \
    --cov-fail-under=${COVERAGE_THRESHOLD} \
    --html="${HTML_REPORT}" \
    --self-contained-html \
    --tb=short \
    --strict-markers \
    --color=yes \
    || TEST_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Test Execution Complete${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Check test results
if [ "${TEST_EXIT_CODE:-0}" -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${GREEN}Duration: ${DURATION} seconds${NC}"
    echo ""
    echo -e "${BLUE}Reports generated:${NC}"
    echo "  HTML Report:     ${HTML_REPORT}"
    echo "  Coverage Report: ${COVERAGE_DIR}/index.html"
    echo "  Coverage XML:    ${COVERAGE_DIR}/coverage.xml"
    echo ""

    # Display coverage summary
    if [ -f "${COVERAGE_DIR}/index.html" ]; then
        echo -e "${BLUE}Coverage Summary:${NC}"
        echo "  Open: file://${COVERAGE_DIR}/index.html"
    fi

    exit 0
else
    echo -e "${RED}✗ Tests failed!${NC}"
    echo ""
    echo -e "${RED}Duration: ${DURATION} seconds${NC}"
    echo -e "${RED}Exit code: ${TEST_EXIT_CODE}${NC}"
    echo ""
    echo -e "${YELLOW}Check the reports for details:${NC}"
    echo "  HTML Report:     ${HTML_REPORT}"
    echo "  Coverage Report: ${COVERAGE_DIR}/index.html"
    echo ""

    exit ${TEST_EXIT_CODE}
fi
