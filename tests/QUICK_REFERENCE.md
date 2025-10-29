# USDCOP Test Suite - Quick Reference

## Installation

```bash
pip install -r tests/requirements-test.txt
cp .env.test.example .env.test  # Configure test environment
```

## Run Tests

### All Tests
```bash
./scripts/run_all_tests.sh
```

### By Suite
```bash
./scripts/run_all_tests.sh unit         # Unit tests only
./scripts/run_all_tests.sh integration  # Integration tests only
./scripts/run_all_tests.sh load         # Load tests only
./scripts/run_all_tests.sh e2e          # E2E tests only
```

### Options
```bash
./scripts/run_all_tests.sh --fast       # Skip slow tests
./scripts/run_all_tests.sh --parallel   # Parallel execution
./scripts/run_all_tests.sh --verbose    # Verbose output
```

### Direct pytest
```bash
pytest tests/                           # All tests
pytest tests/unit/                      # Unit tests only
pytest -m unit                          # By marker
pytest -m "not slow"                    # Skip slow tests
pytest --cov=services --cov-report=html # With coverage
pytest -v tests/unit/test_realtime_ingestion.py::TestTwelveDataFetching::test_fetch_latest_data_success
```

## Test Categories

| Category    | Location                      | Speed | Dependencies    |
|-------------|-------------------------------|-------|-----------------|
| Unit        | `tests/unit/`                 | Fast  | None (mocked)   |
| Integration | `tests/integration/`          | Med   | DB, Redis       |
| Load        | `tests/load/`                 | Slow  | WebSocket       |
| E2E         | `tests/e2e/`                  | Slow  | Full system     |

## Coverage Target

- **Minimum**: 80%
- **View report**: `open test-reports/coverage/index.html`

## Common Commands

```bash
# Quick smoke test (unit tests only)
pytest -m unit

# Full test with coverage
./scripts/run_all_tests.sh

# Integration tests (requires DB)
pytest -m integration

# Performance tests
pytest -m load

# Skip slow tests
pytest -m "not slow"

# Specific test file
pytest tests/unit/test_realtime_ingestion.py

# Verbose output
pytest -v tests/

# Show print statements
pytest -s tests/

# Stop on first failure
pytest -x tests/

# Parallel execution
pytest -n auto tests/
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.load` - Load tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow tests

## Reports Location

- Test Report: `test-reports/test-report.html`
- Coverage: `test-reports/coverage/index.html`
- Coverage XML: `test-reports/coverage/coverage.xml`

## Environment

Test environment is configured in `.env.test`:
- Test database: `TEST_DATABASE_URL`
- Test Redis: `TEST_REDIS_URL`
- Test WebSocket: `TEST_WEBSOCKET_URL`

## Troubleshooting

### Database not found
```bash
docker exec -it usdcop-postgres-timescale psql -U admin -c "CREATE DATABASE usdcop_trading_test;"
```

### Redis connection error
```bash
docker ps | grep redis  # Check if running
redis-cli ping          # Test connection
```

### Import errors
```bash
pip install -r tests/requirements-test.txt
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

## Performance Metrics

- **Latency p95**: < 100ms
- **Throughput**: > 100 msg/sec
- **Memory**: < 512 MB
- **Coverage**: > 80%

## CI/CD

For automated testing in CI/CD:
```bash
./scripts/run_all_tests.sh --fast --parallel
```

## More Info

See `tests/README.md` for detailed documentation.
