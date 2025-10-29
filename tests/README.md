# USDCOP Trading System - Test Suite

Comprehensive testing suite for the USDCOP real-time trading system.

## Overview

This test suite provides 4 types of tests with >80% code coverage target:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated component tests with mocks
2. **Integration Tests** (`tests/integration/`) - Database and Redis integration tests
3. **Load Tests** (`tests/load/`) - Performance and scalability tests
4. **E2E Tests** (`tests/e2e/`) - End-to-end pipeline validation

## Quick Start

### Installation

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Copy test environment configuration
cp .env.test.example .env.test
# Edit .env.test with your test database credentials
```

### Running Tests

```bash
# Run all tests
./scripts/run_all_tests.sh

# Run specific test suite
./scripts/run_all_tests.sh unit
./scripts/run_all_tests.sh integration
./scripts/run_all_tests.sh load
./scripts/run_all_tests.sh e2e

# Skip slow tests
./scripts/run_all_tests.sh --fast

# Run tests in parallel
./scripts/run_all_tests.sh --parallel

# Verbose output
./scripts/run_all_tests.sh --verbose
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_realtime_ingestion.py

# Run specific test
pytest tests/unit/test_realtime_ingestion.py::TestTwelveDataFetching::test_fetch_latest_data_success

# Run with coverage
pytest tests/ --cov=services --cov-report=html

# Run marked tests
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Skip slow tests
```

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared fixtures and configuration
├── requirements-test.txt          # Test dependencies
├── README.md                      # This file
│
├── unit/                          # Unit Tests (fast, mocked)
│   ├── __init__.py
│   └── test_realtime_ingestion.py # Real-time service unit tests
│       ├── TestTwelveDataFetching    # API fetch tests
│       ├── TestDatabaseInsertion     # DB insertion tests
│       ├── TestRedisPublishing       # Redis pub/sub tests
│       ├── TestCircuitBreaker        # Circuit breaker tests
│       ├── TestLeaderElection        # Leader election tests
│       └── TestMarketHours           # Market hours validation
│
├── integration/                   # Integration Tests (DB/Redis)
│   ├── __init__.py
│   └── test_rt_to_db.py          # Real-time to DB integration
│       ├── TestCompleteDataFlow      # API → DB → Redis flow
│       ├── TestOnConflictBehavior    # UPSERT behavior
│       ├── TestGapDetection          # Gap detection and backfill
│       ├── TestDataConsistency       # Data integrity tests
│       ├── TestRedisPubSub           # Redis pub/sub integration
│       └── TestPerformance           # Query performance tests
│
├── load/                          # Load Tests (performance)
│   ├── __init__.py
│   └── test_websocket_load.py    # WebSocket load tests
│       ├── TestConcurrentConnections # 100+ concurrent clients
│       ├── TestThroughputLatency     # Latency measurements
│       ├── TestMemoryLeaks           # Memory stability
│       └── TestErrorHandling         # Error handling under load
│
└── e2e/                           # End-to-End Tests (full pipeline)
    ├── __init__.py
    └── test_full_pipeline.py     # Complete pipeline tests
        ├── TestCompletePipeline      # L0 → RT → Dashboard
        ├── TestMarketHoursBehavior   # Market hours logic
        ├── TestSystemRecovery        # Crash recovery
        ├── TestDashboardAPIs         # API endpoint validation
        ├── TestMultiSymbolSupport    # Multi-symbol support
        └── TestUserJourney           # Complete user flows
```

## Test Coverage

### Unit Tests Coverage

- TwelveData API fetch with mock responses
- Database insertion with mock connections
- Redis publishing with mock client
- Circuit breaker state transitions (closed, open, half-open)
- Leader election (acquire, release, renewal)
- Market hours validation
- **Target: >80% code coverage**

### Integration Tests Coverage

- Complete API → DB → Redis → WebSocket flow
- ON CONFLICT UPSERT behavior
- Gap detection and backfill logic
- Real-time data insertion
- Redis pub/sub message flow
- Data consistency checks
- Query performance with indexes
- **Target: All critical paths tested**

### Load Tests Coverage

- 100+ concurrent WebSocket connections
- Message throughput (messages/second)
- Latency measurements (p50, p95, p99)
- Memory leak detection over time
- Connection stability under load
- Error handling under stress
- **Target: p95 latency < 100ms, throughput > 100 msg/sec**

### E2E Tests Coverage

- L0 pipeline execution and data availability
- Real-time service activation during market hours
- Complete data flow through all layers
- Dashboard API endpoints
- System recovery after crashes
- Multi-symbol support (if applicable)
- Complete user journeys
- **Target: All user-facing features tested**

## Fixtures and Utilities

### Shared Fixtures (conftest.py)

- `db_pool` - Async PostgreSQL connection pool
- `redis_client` - Redis client for testing
- `clean_db` - Clean database before/after tests
- `clean_redis` - Clean Redis keys before/after tests
- `sample_market_data` - Sample market data
- `sample_ohlcv_data` - Sample OHLCV bar data
- `mock_websocket_message` - Mock WebSocket message
- `market_hours_config` - Market hours configuration
- `circuit_breaker_states` - Circuit breaker state configs
- `leader_election_config` - Leader election config
- `performance_thresholds` - Performance testing thresholds
- `generate_market_ticks` - Generate test market ticks

## Test Markers

Tests are marked for easy filtering:

- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)
- `@pytest.mark.integration` - Integration tests (require DB/Redis)
- `@pytest.mark.load` - Load and performance tests (slow)
- `@pytest.mark.e2e` - End-to-end tests (slow)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.smoke` - Smoke tests for quick validation

## Performance Thresholds

Default thresholds (configurable in `.env.test`):

- **Latency**: p95 < 100ms
- **Throughput**: > 100 messages/second
- **Memory**: < 512 MB
- **CPU**: < 80%
- **Coverage**: > 80%

## Test Reports

After running tests, reports are generated in `test-reports/`:

- `test-report.html` - HTML test execution report
- `coverage/index.html` - Code coverage report
- `coverage/coverage.xml` - Coverage XML (for CI/CD)

Open reports:
```bash
# Test report
open test-reports/test-report.html

# Coverage report
open test-reports/coverage/index.html
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: timescale/timescaledb:latest-pg14
        env:
          POSTGRES_PASSWORD: test123
          POSTGRES_DB: usdcop_trading_test
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run tests
        run: ./scripts/run_all_tests.sh --fast
        env:
          DATABASE_URL: postgresql://postgres:test123@localhost:5432/usdcop_trading_test
          REDIS_URL: redis://localhost:6379/1

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./test-reports/coverage/coverage.xml
```

## Troubleshooting

### Common Issues

#### Database Connection Error

```
asyncpg.exceptions.PostgresConnectionError: Connection refused
```

**Solution**: Ensure PostgreSQL is running and TEST_DATABASE_URL is correct in `.env.test`

```bash
# Check PostgreSQL status
docker ps | grep postgres

# Create test database
docker exec -it usdcop-postgres-timescale psql -U admin -c "CREATE DATABASE usdcop_trading_test;"
```

#### Redis Connection Error

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution**: Ensure Redis is running

```bash
# Check Redis status
docker ps | grep redis

# Test Redis connection
redis-cli ping
```

#### Import Errors

```
ModuleNotFoundError: No module named 'pytest'
```

**Solution**: Install test dependencies

```bash
pip install -r tests/requirements-test.txt
```

#### Slow Tests Timeout

```
FAILED tests/load/test_websocket_load.py::test_memory_stability - Timeout
```

**Solution**: Increase timeout or skip slow tests

```bash
# Skip slow tests
./scripts/run_all_tests.sh --fast

# Or increase timeout in pytest.ini
timeout = 600  # 10 minutes
```

## Best Practices

1. **Write Independent Tests**: Tests should not depend on each other
2. **Use Fixtures**: Leverage fixtures for setup/teardown
3. **Mock External APIs**: Don't consume real API quota in tests
4. **Clean Up**: Always clean database and Redis after tests
5. **Test Fast**: Keep unit tests fast (< 1 second each)
6. **Meaningful Names**: Use descriptive test function names
7. **One Assert Per Test**: Focus on one behavior per test
8. **Test Edge Cases**: Include boundary conditions and error cases
9. **Maintain Coverage**: Keep coverage above 80%
10. **Update Tests**: Update tests when code changes

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure >80% coverage
3. Add fixtures if needed
4. Update this README if adding new test categories
5. Run full test suite before committing

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/) (for load testing)

## License

Part of the USDCOP Trading System project.
