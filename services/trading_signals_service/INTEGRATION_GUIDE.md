# Integration Guide - Trading Signals Service

## Overview

This guide explains how to integrate the Trading Signals Service into your USDCOP trading system.

## Architecture Integration

The Trading Signals Service fits into the USDCOP system as follows:

```
┌─────────────────────────────────────────────────────┐
│                 Trading Dashboard                    │
│              (Next.js Frontend)                      │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ WebSocket + REST
                   │
┌──────────────────▼──────────────────────────────────┐
│          Trading Signals Service :8003               │
│  - Signal Generation                                 │
│  - Model Inference (PPO-LSTM)                        │
│  - Position Management                               │
│  - Risk Calculation                                  │
└──────────────────┬──────────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│PostgreSQL│  │  Redis  │  │ Models  │
│TimescaleDB│  │ Cache   │  │ (ONNX)  │
└─────────┘  └─────────┘  └─────────┘
```

## 1. Docker Compose Integration

### Option A: Add to Existing docker-compose.yml

Copy the service definition from `docker-compose.snippet.yml` and add it to your main `docker-compose.yml`.

### Option B: Use docker-compose include (Recommended)

In your main `docker-compose.yml`:

```yaml
include:
  - services/trading_signals_service/docker-compose.snippet.yml
```

## 2. Model Setup

### Prepare Your PPO-LSTM Model

1. **Export your model to ONNX format:**

   ```python
   import torch
   import onnx

   # Load your PyTorch model
   model = YourPPOLSTMModel()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()

   # Create dummy input (adjust dimensions to your model)
   dummy_input = torch.randn(1, feature_dim)

   # Export to ONNX
   torch.onnx.export(
       model,
       dummy_input,
       'ppo_lstm_v3.2.onnx',
       export_params=True,
       opset_version=13,
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={
           'input': {0: 'batch_size'},
           'output': {0: 'batch_size'}
       }
   )
   ```

2. **Place model in models directory:**

   ```bash
   cp ppo_lstm_v3.2.onnx ./models/
   ```

3. **Verify model:**

   ```python
   import onnx
   model = onnx.load('models/ppo_lstm_v3.2.onnx')
   onnx.checker.check_model(model)
   print("Model is valid!")
   ```

### Placeholder Mode (Development)

If you don't have a model yet, the service runs in placeholder mode with simulated predictions. Perfect for development and testing!

## 3. Database Schema

The service uses the existing `usdcop_m5_ohlcv` table. Ensure your database has:

```sql
-- Verify table exists
SELECT COUNT(*) FROM usdcop_m5_ohlcv;

-- Verify recent data
SELECT MAX(time) as latest_data FROM usdcop_m5_ohlcv;
```

No additional tables are required - the service stores signals in memory.

## 4. Frontend Integration

### REST API Usage

```typescript
// trading-dashboard/lib/api/signals.ts

export async function getLatestSignal() {
  const response = await fetch('http://localhost:8003/api/signals/latest');
  return response.json();
}

export async function generateSignal(marketData: MarketData) {
  const response = await fetch('http://localhost:8003/api/signals/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(marketData)
  });
  return response.json();
}

export async function getSignalHistory(limit: number = 50) {
  const response = await fetch(
    `http://localhost:8003/api/signals/history?limit=${limit}`
  );
  return response.json();
}
```

### WebSocket Integration

```typescript
// trading-dashboard/hooks/useSignalWebSocket.ts

import { useEffect, useState } from 'react';

export function useSignalWebSocket() {
  const [signal, setSignal] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8003/ws/signals');

    ws.onopen = () => {
      console.log('Connected to signals WebSocket');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'signal') {
        setSignal(data.data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    // Heartbeat
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, []);

  return { signal, connected };
}
```

### Signal Display Component

```typescript
// trading-dashboard/components/SignalCard.tsx

import { useSignalWebSocket } from '@/hooks/useSignalWebSocket';

export function SignalCard() {
  const { signal, connected } = useSignalWebSocket();

  if (!signal) return <div>Waiting for signal...</div>;

  return (
    <div className="signal-card">
      <div className="signal-header">
        <span className={`action-badge ${signal.action}`}>
          {signal.action}
        </span>
        <span className="confidence">
          Confidence: {(signal.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="signal-prices">
        <div>Entry: ${signal.entry_price.toFixed(2)}</div>
        <div>SL: ${signal.stop_loss.toFixed(2)}</div>
        <div>TP: ${signal.take_profit.toFixed(2)}</div>
      </div>

      <div className="signal-metrics">
        <div>R/R: {signal.risk_reward_ratio.toFixed(2)}</div>
        <div>Size: {(signal.position_size * 100).toFixed(1)}%</div>
        <div>Latency: {signal.latency_ms.toFixed(1)}ms</div>
      </div>

      <div className="signal-reasoning">
        <strong>Reasoning:</strong>
        <ul>
          {signal.reasoning.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

## 5. Deployment

### Development

```bash
# Navigate to service directory
cd services/trading_signals_service

# Run locally
./run_local.sh   # Linux/Mac
run_local.bat    # Windows

# Or with Docker
docker-compose up trading-signals-service
```

### Production

```bash
# Build image
docker build -t trading-signals-service:latest .

# Run with docker-compose
docker-compose up -d trading-signals-service

# Verify service
curl http://localhost:8003/health
```

## 6. Monitoring

### Health Checks

```bash
# Simple health check
curl http://localhost:8003/health

# Detailed health check
curl http://localhost:8003/api/signals/health

# Statistics
curl http://localhost:8003/api/signals/statistics
```

### Logs

```bash
# Docker logs
docker logs -f usdcop-trading-signals

# Local logs
tail -f logs/trading_signals.log
```

### Metrics (Future)

The service exposes Prometheus metrics on port 9003 (when enabled):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'trading-signals'
    static_configs:
      - targets: ['trading-signals-service:9003']
```

## 7. Testing

### Run Test Suite

```bash
# Ensure service is running
curl http://localhost:8003/health

# Run tests
python test_service.py
```

### Manual Testing

```bash
# Generate signal from market data
curl -X POST http://localhost:8003/api/signals/generate \
  -H "Content-Type: application/json" \
  -d '{
    "close_price": 4250.50,
    "rsi": 35.5,
    "macd": -2.3
  }'

# Get latest signal
curl http://localhost:8003/api/signals/latest

# Get signal history
curl http://localhost:8003/api/signals/history?limit=10
```

## 8. Configuration

Key environment variables to configure:

```bash
# Service
SIGNAL_SERVICE_PORT=8003
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=usdcop-postgres-timescale
POSTGRES_DB=usdcop_trading

# Model
MODEL_PATH=/app/models/ppo_lstm_v3.2.onnx
MODEL_VERSION=ppo_lstm_v3.2

# Risk Management
CONFIDENCE_THRESHOLD=0.65      # Min confidence (0-1)
POSITION_SIZE_PCT=0.02         # Default 2% position size
ATR_MULTIPLIER_SL=2.0          # Stop loss = 2x ATR
ATR_MULTIPLIER_TP=3.0          # Take profit = 3x ATR
```

## 9. Troubleshooting

### Service won't start

1. Check logs: `docker logs usdcop-trading-signals`
2. Verify database connection: `curl http://localhost:8003/api/signals/health`
3. Check environment variables in docker-compose.yml

### Model not loading

1. Verify model file exists: `ls models/ppo_lstm_v3.2.onnx`
2. Check model format with ONNX checker
3. Service will run in placeholder mode if model not found

### WebSocket connection fails

1. Check if service is running: `curl http://localhost:8003/health`
2. Verify WebSocket endpoint: `ws://localhost:8003/ws/signals`
3. Check CORS settings in frontend

### Low signal confidence

1. Review technical indicators: Are they calculated correctly?
2. Check model performance: Generate test signals
3. Adjust `CONFIDENCE_THRESHOLD` if needed

## 10. Next Steps

1. **Integrate with Dashboard**: Add SignalCard component to your dashboard
2. **Enable Notifications**: Send alerts when high-confidence signals are generated
3. **Backtest Signals**: Compare signal performance against historical data
4. **Model Monitoring**: Track model drift and retrain as needed
5. **Scale Up**: Add load balancing for multiple instances

## Support

For issues or questions:
- Check logs: `logs/trading_signals.log`
- Review API docs: `http://localhost:8003/docs`
- Test endpoints: `python test_service.py`

---

**Author**: Pedro @ Lean Tech Solutions
**Created**: 2025-12-17
**Version**: 1.0.0
