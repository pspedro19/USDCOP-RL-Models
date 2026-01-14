"""
Chaos Tests Package.

This package contains chaos engineering tests to verify
system resilience under failure conditions:

- test_circuit_breaker.py: Circuit breaker activation after losses
- test_db_disconnect.py: Database disconnection handling
- test_feed_disconnect.py: Price feed disconnection handling
- test_nan_handling.py: NaN/Inf value sanitization

Run with: pytest tests/chaos/ -v --timeout=30
"""
