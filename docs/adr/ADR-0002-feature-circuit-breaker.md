# ADR-0002: Feature Circuit Breaker for Data Quality Protection

**Status**: Accepted
**Date**: 2025-01-14
**Author(s)**: Trading Team
**Supersedes**: N/A
**Superseded by**: N/A

## Context

The USDCOP inference API must handle various data quality issues gracefully:

- Missing market data (NaN values)
- Stale data from upstream providers
- Data pipeline failures
- Macro data unavailability (weekends, holidays)

Without protection, the model could make predictions based on corrupted or incomplete observations, potentially leading to poor trading decisions.

## Decision

**Implement a Feature Circuit Breaker that halts inference when data quality falls below acceptable thresholds.**

Configuration:
- **NaN Threshold**: >20% of features are NaN triggers circuit breaker
- **Consecutive Failures**: 5 consecutive failures trigger circuit breaker
- **Cooldown Period**: 15 minutes before automatic retry
- **Default Action**: Return HOLD (action=0) when circuit breaker is open

## Rationale

### Why 20% NaN Threshold?

- 15 features total in observation space
- 20% = 3 features
- Missing 1-2 features may be acceptable (macro data on weekends)
- Missing 3+ features indicates systemic data issues

### Why 15-minute Cooldown?

- Most data pipeline issues resolve within 10-15 minutes
- Prevents rapid open/close cycling
- Aligned with typical market data refresh intervals

### Alternatives Considered

1. **No circuit breaker, use NaN imputation**
   - Pros: Always produces predictions
   - Cons: Imputed values may lead to poor decisions

2. **Stricter threshold (10%)**
   - Pros: Higher data quality requirement
   - Cons: Too many false positives, especially on weekends

3. **Model-level NaN handling**
   - Pros: Integrated solution
   - Cons: Requires model retraining, less transparent

## Consequences

### Positive

- Protection against corrupted predictions
- Clear system state (open/closed)
- Observable via Prometheus metrics
- Automatic recovery after cooldown

### Negative

- May miss trading opportunities during circuit breaker activation
- Additional latency for quality checks (~1ms)
- Requires monitoring and alerting setup

### Neutral

- HOLD action is safe default (no position change)
- Circuit breaker state visible in API response

## Implementation

### Files Changed

- `src/features/circuit_breaker.py` - NEW: FeatureCircuitBreaker class
- `services/inference_api/core/feature_adapter.py` - Integrated circuit breaker
- `config/prometheus/alerts/latency.yml` - Circuit breaker alerts

### Configuration

```python
@dataclass
class CircuitBreakerConfig:
    max_nan_ratio: float = 0.20      # 20% threshold
    max_consecutive_failures: int = 5
    cooldown_minutes: int = 15
    default_action: int = 0          # HOLD
```

### Prometheus Metrics

```yaml
- usdcop_circuit_breaker_state: 0=closed, 1=open
- usdcop_circuit_breaker_activations_total: Counter
- usdcop_feature_nan_ratio: Current NaN ratio
```

### API Response

When circuit breaker is active:
```json
{
  "signal": "HOLD",
  "confidence": 0.0,
  "circuit_breaker": {
    "active": true,
    "reason": "nan_threshold_exceeded",
    "cooldown_remaining_seconds": 420
  }
}
```

## Related Documents

- ADR-0001: Wilder's EMA for Technical Indicators
- `docs/SLA.md` - Error rate requirements
- Phase 13 Remediation Plan

## Review

- [x] Reviewed by: Trading Team
- [x] Approved by: Trading Team
- [x] Implementation complete
- [x] Tests added
