# Trading Signals Service

Professional real-time trading signals backend service for the USDCOP trading system.

## Features

- **Real-time Signal Generation**: PPO-LSTM model inference for trading signals
- **WebSocket Broadcasting**: Real-time signal streaming to connected clients
- **Position Tracking**: Automatic position management with PnL tracking
- **Risk Management**: ATR-based stop loss/take profit calculation
- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA calculations
- **RESTful API**: Comprehensive endpoints for signal access and management

## Architecture

```
trading_signals_service/
├── main.py                    # FastAPI application
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image
├── models/                    # Data models
│   ├── signal_schema.py       # Pydantic schemas
│   └── model_loader.py        # ONNX model loading
├── services/                  # Business logic
│   ├── inference_service.py   # Model inference
│   ├── signal_generator.py    # Signal generation
│   └── position_manager.py    # Position tracking
├── api/                       # API layer
│   ├── routes.py              # REST endpoints
│   └── websocket.py           # WebSocket handler
└── utils/                     # Utilities
    ├── technical_indicators.py
    └── risk_calculator.py
```

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Run Service**
   ```bash
   python main.py
   ```

   The service will be available at `http://localhost:8003`

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t trading-signals-service:latest .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     --name trading-signals \
     -p 8003:8003 \
     -v $(pwd)/models:/app/models \
     --env-file .env \
     trading-signals-service:latest
   ```

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check |
| `/api/signals/latest` | GET | Get latest signal |
| `/api/signals/history` | GET | Get signal history |
| `/api/signals/generate` | POST | Generate signal from data |
| `/api/signals/generate-from-db` | POST | Generate signal from DB |
| `/api/signals/positions/active` | GET | Get active positions |
| `/api/signals/positions/closed` | GET | Get closed positions |
| `/api/signals/statistics` | GET | Get statistics |
| `/api/signals/model/info` | GET | Get model information |

### WebSocket

Connect to `/ws/signals` for real-time signal streaming.

**Message Types:**
- `signal`: New trading signal
- `market_update`: Market data update
- `position_update`: Position status update
- `heartbeat`: Connection keepalive
- `error`: Error notification

## Signal Response Format

```json
{
  "signal_id": "uuid-string",
  "timestamp": "2025-12-17T10:30:00Z",
  "symbol": "USDCOP",
  "action": "BUY",
  "confidence": 0.85,
  "entry_price": 4250.50,
  "stop_loss": 4200.00,
  "take_profit": 4325.00,
  "position_size": 0.02,
  "risk_reward_ratio": 1.5,
  "model_version": "ppo_lstm_v3.2",
  "reasoning": [
    "Model confidence: 85.0%",
    "RSI oversold (28.5)",
    "MACD bullish cross"
  ],
  "latency_ms": 12.5
}
```

## Configuration

Key configuration parameters (set via environment variables):

- `SIGNAL_SERVICE_PORT`: Service port (default: 8003)
- `POSTGRES_HOST`: Database host
- `MODEL_PATH`: Path to ONNX model file
- `CONFIDENCE_THRESHOLD`: Minimum confidence for signals (default: 0.65)
- `POSITION_SIZE_PCT`: Default position size (default: 0.02 = 2%)
- `ATR_MULTIPLIER_SL`: Stop loss ATR multiplier (default: 2.0)
- `ATR_MULTIPLIER_TP`: Take profit ATR multiplier (default: 3.0)

## Model Integration

The service expects a PPO-LSTM model in ONNX format. Place your model at the configured `MODEL_PATH`.

**Model Requirements:**
- Format: ONNX (`.onnx`)
- Input: Feature vector (shape depends on your model)
- Output: Action probabilities [HOLD, BUY, SELL]

If no model is found, the service runs in **PLACEHOLDER mode** with simulated predictions.

## Monitoring

Access comprehensive statistics via `/api/signals/statistics`:
- Inference performance
- Signal generation stats
- Position performance (win rate, PnL, etc.)

## Testing

### Test Signal Generation

```bash
curl -X POST "http://localhost:8003/api/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "USDCOP",
    "close_price": 4250.50,
    "rsi": 35.5,
    "macd": -2.3,
    "macd_signal": -1.8
  }'
```

### WebSocket Test (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8003/ws/signals');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.send(JSON.stringify({ type: 'ping' }));
```

## Performance

- **Inference Latency**: < 20ms (with ONNX model)
- **End-to-End Latency**: < 100ms (signal generation)
- **WebSocket Capacity**: 100+ concurrent connections
- **Throughput**: 1000+ signals/minute

## Dependencies

Core dependencies:
- **FastAPI**: Web framework
- **ONNX Runtime**: Model inference
- **pandas/numpy**: Data processing
- **psycopg2**: PostgreSQL connection
- **websockets**: Real-time communication

See `requirements.txt` for complete list.

## License

Copyright 2025 - USDCOP Trading System

## Author

Pedro @ Lean Tech Solutions
Created: 2025-12-17
