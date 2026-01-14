# ADR-0001: Use Wilder's EMA for RSI, ATR, and ADX Calculations

**Status**: Accepted
**Date**: 2025-01-14
**Author(s)**: Trading Team
**Supersedes**: N/A
**Superseded by**: N/A

## Context

The USDCOP trading system uses RSI (Relative Strength Index), ATR (Average True Range), and ADX (Average Directional Index) as key technical indicators for the reinforcement learning model.

During a code audit, we discovered an inconsistency between training and inference:

- **Training**: Used `src/feature_store/core.py` with Wilder's EMA (`alpha = 1/period`)
- **Inference**: Used `services/inference_api/core/observation_builder.py` with simple moving average (`np.mean()`)

This inconsistency could cause significant drift between training and production, as the same raw data would produce different feature values.

### Wilder's EMA vs Standard EMA

| Method | Alpha Formula | RSI-9 Alpha |
|--------|---------------|-------------|
| Wilder's EMA | `1/period` | 0.1111 |
| Standard EMA | `2/(period+1)` | 0.2 |
| Simple Mean | N/A | N/A |

J. Welles Wilder, who created RSI, ATR, and ADX, specified using the smoothing factor `alpha = 1/period` in his original 1978 book "New Concepts in Technical Trading Systems."

## Decision

**All RSI, ATR, and ADX calculations MUST use Wilder's EMA with `alpha = 1/period`.**

Implementation:
1. Create `InferenceFeatureAdapter` that delegates to SSOT calculators in `src/feature_store/core.py`
2. Modify `ObservationBuilder` to use the adapter instead of inline calculations
3. Add parity tests to ensure training and inference produce identical features

## Rationale

### Why Wilder's EMA?

1. **Historical Correctness**: Wilder's original specification used this smoothing method
2. **Industry Standard**: Most trading platforms (Bloomberg, Reuters, TradingView) use Wilder's method
3. **Training Alignment**: Our training pipeline already uses Wilder's EMA

### Alternatives Considered

1. **Change training to use simple mean**
   - Pros: Simpler implementation
   - Cons: Deviates from industry standard, requires retraining all models

2. **Use standard EMA (2/(period+1))**
   - Pros: Common in general statistics
   - Cons: Not what Wilder specified, not what most trading platforms use

3. **Keep both implementations (chosen against)**
   - Pros: No code changes
   - Cons: Training/inference parity violation, potential performance degradation

## Consequences

### Positive

- Feature parity between training and inference (tolerance < 1e-6)
- Alignment with industry-standard technical indicator calculations
- Single Source of Truth (SSOT) for all feature calculations
- Reduced risk of silent model drift

### Negative

- Existing trained models may need to be retrained if they were tested against inference with simple mean
- Additional abstraction layer (adapter) adds complexity
- Slightly higher inference latency (~1ms overhead)

### Neutral

- No change to model architecture or training process
- No change to database schema

## Implementation

### Files Changed

- `services/inference_api/core/feature_adapter.py` - NEW: Adapter delegating to SSOT
- `services/inference_api/core/observation_builder.py` - Modified to use adapter
- `tests/unit/test_feature_adapter.py` - NEW: Parity tests
- `tests/integration/test_feature_parity.py` - Enhanced parity validation

### Migration Steps

1. Deploy `InferenceFeatureAdapter` class
2. Update `ObservationBuilder` to delegate to adapter
3. Run parity tests to verify alignment
4. Monitor inference metrics for any anomalies
5. Deprecate old inline calculation methods

### Code Example

```python
# BEFORE (incorrect)
def _calculate_rsi(self, returns: np.ndarray, period: int = 9) -> float:
    gains = np.maximum(returns, 0)
    losses = np.abs(np.minimum(returns, 0))
    avg_gain = np.mean(gains[-period:])  # WRONG: Simple mean
    avg_loss = np.mean(losses[-period:])
    ...

# AFTER (correct)
from src.feature_store.core import RSICalculator

class InferenceFeatureAdapter:
    def __init__(self):
        self._calculators = {
            "rsi_9": RSICalculator(period=9),  # Uses Wilder's EMA
            ...
        }
```

## Related Documents

- `src/feature_store/core.py` - SSOT implementation
- `docs/SLA.md` - Latency requirements
- Phase 10 Remediation Plan - Original issue identification

## Review

- [x] Reviewed by: Trading Team
- [x] Approved by: Trading Team
- [x] Implementation complete
- [x] Tests added
