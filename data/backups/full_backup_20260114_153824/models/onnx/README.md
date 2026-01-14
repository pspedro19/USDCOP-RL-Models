# ONNX Models Directory

This directory stores ONNX-exported models for production inference.

## Expected Models

Based on `config/mlops.yaml`:

```
models/onnx/
├── ppo_usdcop_v3.onnx      # PPO model (weight: 1.0)
├── sac_usdcop_v2.onnx      # SAC model (weight: 0.8)
└── a2c_macro_v1.onnx       # A2C model (weight: 0.6)
```

## Exporting Models

Use the export script to convert trained models to ONNX format:

```bash
# Export a single model
python scripts/mlops/export_model_onnx.py \
    --model-path models/trained/ppo_usdcop_v3.zip \
    --output-path models/onnx/ppo_usdcop_v3.onnx \
    --algorithm PPO \
    --observation-dim 45

# Export with verification
python scripts/mlops/export_model_onnx.py \
    --model-path models/trained/sac_usdcop_v2.zip \
    --output-path models/onnx/sac_usdcop_v2.onnx \
    --algorithm SAC \
    --verify
```

## Model Requirements

- **Input shape**: `(batch_size, 45)` - 45 features
- **Output shape**: `(batch_size, 3)` - 3 actions (SELL, HOLD, BUY)
- **Data type**: `float32`

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference latency | < 5ms |
| Model load time | < 1s |
| Memory usage | < 100MB per model |

## Validation

After exporting, verify models with:

```python
from services.mlops import InferenceEngine
import numpy as np

engine = InferenceEngine()
engine.load_models()

# Test inference
obs = np.random.randn(45).astype(np.float32)
result = engine.predict(obs)

print(f"Signal: {result.signal}, Confidence: {result.confidence:.2%}")
print(f"Latency: {result.latency_ms:.2f}ms")
```
