# Inference Module Architecture

## Overview

This module contains the **canonical implementation** of the InferenceEngine and all related inference components for the USDCOP-RL-Models trading system.

**Important**: As of 2026-01-17, `src/inference/` is the Single Source of Truth (SSOT) for inference operations. The previous implementation in `src/models/inference_engine.py` has been removed.

## Design Pattern: Facade

The `InferenceEngine` class implements the **Facade Pattern**, providing a unified interface that delegates to specialized components:

```
                    +------------------+
                    | InferenceEngine  |  <-- Facade
                    |     (SSOT)       |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v-------+  +--------v-------+  +--------v--------+
| ONNXModelLoader|  |  ONNXPredictor |  |EnsemblePredictor|
+----------------+  +----------------+  +-----------------+
```

## Components

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `InferenceEngine` | `inference_engine.py` | Unified facade, model management, health checks |
| `ONNXModelLoader` | `model_loader.py` | Load ONNX models, warmup, session management |
| `ONNXPredictor` | `predictor.py` | Single model inference, latency tracking |
| `EnsemblePredictor` | `ensemble_predictor.py` | Multi-model coordination, strategy execution |

### Extended Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ModelRouter` | `model_router.py` | Shadow mode execution for A/B testing |
| `ShadowPnLTracker` | `shadow_pnl.py` | Virtual PnL tracking for shadow models |
| `ValidatedPredictor` | `validated_predictor.py` | Contract-validated model wrapper |

## Usage Guide

### Basic Usage (PPO/ONNX Models)

```python
from src.inference import InferenceEngine

# Create engine with configuration
engine = InferenceEngine(config, ensemble_strategy="weighted_average")

# Load models from config
engine.load_models()

# Single model inference
result = engine.predict(observation)
print(f"Signal: {result.signal}, Confidence: {result.confidence}")

# Ensemble inference
ensemble_result = engine.predict_ensemble(observation)
print(f"Consensus: {ensemble_result.final_signal}")
```

### Loading Models Manually

```python
from src.inference import InferenceEngine

engine = InferenceEngine()

# Load individual ONNX model
engine.load_single_model(
    name="ppo_v3",
    onnx_path="/models/ppo_v3.onnx",
    observation_dim=45,
    weight=1.0
)

# Load another model
engine.load_single_model(
    name="ppo_v4",
    onnx_path="/models/ppo_v4.onnx",
    weight=0.8
)

# Now run ensemble
result = engine.predict_ensemble(observation)
```

### Shadow Mode (A/B Testing)

```python
from src.inference import ModelRouter, create_model_router

# Create router with primary and shadow models
router = create_model_router(
    primary_model=primary_engine,
    shadow_models=[shadow_engine],
    log_shadow_predictions=True
)

# Run inference - primary result returned, shadow logged
prediction = router.predict(observation)
```

### Health Checks

```python
# Quick health check
if engine.is_healthy:
    result = engine.predict(observation)

# Detailed health check
health = engine.health_check()
print(f"Status: {health['status']}")
for model, info in health['models'].items():
    print(f"  {model}: {info['status']}, latency={info['latency_ms']:.2f}ms")
```

## Migration Guide

### From Old Implementation (src/models/inference_engine.py)

The old implementation has been removed. Update your imports:

**Before:**
```python
from src.models import InferenceEngine
```

**After:**
```python
from src.inference import InferenceEngine
```

### API Changes

The new InferenceEngine has some API differences:

| Old Method | New Method | Notes |
|------------|------------|-------|
| `run_inference(model_id, features)` | `predict(observation, model_name)` | Simplified interface |
| `run_all_inferences(features)` | `predict_ensemble(observation)` | Returns EnsembleResult |
| `run_production_inference(features)` | `predict(observation)` | Uses first loaded model |
| `discretize_action(action)` | N/A | Handled internally |

### Result Objects

The new implementation uses standardized result objects from `src.core.interfaces.inference`:

```python
from src.core.interfaces.inference import InferenceResult, EnsembleResult, SignalType

# InferenceResult contains:
# - signal: SignalType (LONG, SHORT, HOLD)
# - confidence: float (0.0 to 1.0)
# - raw_output: np.ndarray
# - latency_ms: float
# - model_name: str
# - timestamp: datetime
```

## Ensemble Strategies

Available strategies via `EnsembleStrategyRegistry`:

| Strategy | Description |
|----------|-------------|
| `weighted_average` | Weight-based average of model outputs |
| `majority_vote` | Majority voting on discretized signals |
| `confidence_weighted` | Weight by model confidence scores |
| `best_performer` | Use highest confidence model |

Change strategy at runtime:
```python
engine.set_ensemble_strategy("majority_vote")
print(f"Available: {engine.get_available_strategies()}")
```

## Performance Considerations

1. **Model Warmup**: Call `load_models()` during startup, not per-request
2. **Session Reuse**: ONNX sessions are cached and reused
3. **Parallel Inference**: EnsemblePredictor supports parallel execution
4. **Latency Tracking**: Monitor via `engine.get_stats()`

## File Structure

```
src/inference/
├── __init__.py              # Module exports (SSOT marker)
├── ARCHITECTURE.md          # This file
├── inference_engine.py      # Main facade (SSOT)
├── model_loader.py          # ONNX model loading
├── predictor.py             # Single model inference
├── ensemble_predictor.py    # Multi-model ensemble
├── model_router.py          # Shadow mode routing
├── shadow_pnl.py            # Virtual PnL tracking
└── validated_predictor.py   # Contract validation
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.4.0 | 2026-01-17 | Consolidated SSOT, removed src/models duplicate |
| 2.3.0 | 2026-01-15 | Added ValidatedPredictor |
| 2.0.0 | 2025-01-14 | Refactored to Facade pattern |
| 1.0.0 | 2024-12-01 | Initial implementation |
