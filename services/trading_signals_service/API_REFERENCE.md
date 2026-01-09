# API Reference - Trading Signals Service

Complete API documentation for the Trading Signals Service.

**Base URL**: `http://localhost:8003`

**WebSocket**: `ws://localhost:8003/ws/signals`

---

## Table of Contents

1. [Authentication](#authentication)
2. [REST Endpoints](#rest-endpoints)
3. [WebSocket API](#websocket-api)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## Authentication

Currently, no authentication is required. In production, implement JWT tokens or API keys.

---

## REST Endpoints

### Service Information

#### GET `/`

Get service information and available endpoints.

**Response:**
```json
{
  "service": "trading-signals-service",
  "version": "1.0.0",
  "status": "active",
  "timestamp": "2025-12-17T10:30:00Z",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "signals": "/api/signals",
    "websocket": "/ws/signals"
  },
  "model": {
    "version": "ppo_lstm_v3.2",
    "type": "PPO-LSTM",
    "loaded": true
  }
}
```

---

### Health Checks

#### GET `/health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-17T10:30:00Z",
  "service": "trading-signals-service",
  "version": "1.0.0"
}
```

#### GET `/api/signals/health`

Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-17T10:30:00Z",
  "service": "trading-signals-service",
  "version": "1.0.0",
  "model_loaded": true,
  "database_connected": true,
  "redis_connected": true,
  "uptime_seconds": 3600.5,
  "total_signals_generated": 42,
  "last_signal_timestamp": "2025-12-17T10:29:45Z"
}
```

---

### Signal Operations

#### GET `/api/signals/latest`

Get the most recent trading signal.

**Response:**
```json
{
  "status": "success",
  "signal": {
    "signal_id": "123e4567-e89b-12d3-a456-426614174000",
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
    "model_type": "PPO-LSTM",
    "reasoning": [
      "Model confidence: 85.0%",
      "RSI oversold (28.5)",
      "MACD bullish cross",
      "Uptrend (EMA 20 > EMA 50)"
    ],
    "technical_factors": {
      "rsi": 28.5,
      "macd": 2.3,
      "macd_signal": 1.8,
      "ema_20": 4255.0,
      "ema_50": 4240.0,
      "atr": 25.0
    },
    "latency_ms": 12.5,
    "metadata": {
      "atr": 25.0,
      "volatility": 0.8,
      "trend_direction": "uptrend",
      "market_regime": "normal",
      "volume_profile": "normal"
    }
  },
  "message": "Latest signal retrieved successfully"
}
```

**Error Response (404):**
```json
{
  "detail": "No signals available"
}
```

---

#### GET `/api/signals/history`

Get historical signals with optional filtering.

**Query Parameters:**
- `limit` (integer, 1-1000): Maximum number of signals (default: 100)
- `action` (string): Filter by action type (BUY, SELL, HOLD)
- `start_date` (string): Start date in ISO format
- `end_date` (string): End date in ISO format

**Example Request:**
```bash
GET /api/signals/history?limit=50&action=BUY&start_date=2025-12-01T00:00:00Z
```

**Response:**
```json
{
  "status": "success",
  "signals": [
    { /* signal object */ },
    { /* signal object */ }
  ],
  "count": 50,
  "start_date": "2025-12-01T00:00:00Z",
  "end_date": null,
  "filters": {
    "action": "BUY",
    "limit": 50
  }
}
```

---

#### POST `/api/signals/generate`

Generate a new trading signal from provided market data.

**Request Body:**
```json
{
  "symbol": "USDCOP",
  "close_price": 4250.50,
  "open_price": 4245.00,
  "high_price": 4255.00,
  "low_price": 4240.00,
  "volume": 1500000,
  "rsi": 35.5,
  "macd": -2.3,
  "macd_signal": -1.8,
  "bb_upper": 4280.0,
  "bb_lower": 4220.0
}
```

**Required Fields:**
- `close_price` (float): Current close price

**Optional Fields:**
- `symbol` (string): Trading symbol (default: "USDCOP")
- `open_price` (float): Open price
- `high_price` (float): High price
- `low_price` (float): Low price
- `volume` (float): Volume
- `rsi` (float, 0-100): RSI indicator
- `macd` (float): MACD value
- `macd_signal` (float): MACD signal line
- `bb_upper` (float): Bollinger Band upper
- `bb_lower` (float): Bollinger Band lower
- `force_action` (string): Force specific action (testing only)
- `override_confidence` (float, 0-1): Override confidence (testing only)

**Response:**
```json
{
  "status": "success",
  "signal": { /* signal object */ },
  "message": "Signal generated successfully"
}
```

---

#### POST `/api/signals/generate-from-db`

Generate signal using latest data from database (with automatic indicator calculation).

**Query Parameters:**
- `symbol` (string): Trading symbol (default: "USDCOP")
- `lookback_bars` (integer, 50-500): Number of historical bars for indicators (default: 100)

**Example Request:**
```bash
POST /api/signals/generate-from-db?symbol=USDCOP&lookback_bars=100
```

**Response:**
```json
{
  "status": "success",
  "signal": { /* signal object */ },
  "message": "Signal generated from database successfully"
}
```

---

### Position Management

#### GET `/api/signals/positions/active`

Get all active trading positions.

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "positions": [
    {
      "position_id": "pos-123",
      "signal_id": "sig-456",
      "action": "BUY",
      "entry_time": "2025-12-17T10:00:00Z",
      "entry_price": 4250.50,
      "stop_loss": 4200.00,
      "take_profit": 4325.00,
      "position_size": 0.02,
      "current_pnl": 15.50,
      "current_pnl_pct": 0.36,
      "status": "OPEN"
    }
  ]
}
```

---

#### GET `/api/signals/positions/closed`

Get closed positions history.

**Query Parameters:**
- `limit` (integer, 1-500): Maximum number of positions (default: 50)

**Response:**
```json
{
  "status": "success",
  "count": 25,
  "positions": [
    {
      "position_id": "pos-789",
      "signal_id": "sig-101",
      "action": "SELL",
      "entry_time": "2025-12-16T14:00:00Z",
      "entry_price": 4240.00,
      "stop_loss": 4280.00,
      "take_profit": 4180.00,
      "position_size": 0.02,
      "current_pnl": 60.00,
      "current_pnl_pct": 1.42,
      "status": "CLOSED",
      "exit_time": "2025-12-16T16:30:00Z",
      "exit_price": 4180.00,
      "exit_reason": "TAKE_PROFIT"
    }
  ]
}
```

---

### Statistics & Monitoring

#### GET `/api/signals/statistics`

Get comprehensive service statistics.

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-12-17T10:30:00Z",
  "statistics": {
    "inference": {
      "total_inferences": 150,
      "total_latency_ms": 1875.5,
      "avg_latency_ms": 12.5,
      "model_loaded": true
    },
    "signals": {
      "signals_generated": 150,
      "confidence_threshold": 0.65,
      "min_risk_reward": 1.5,
      "position_size_pct": 0.02
    },
    "positions": {
      "total_signals": 150,
      "total_positions_opened": 45,
      "total_positions_closed": 40,
      "active_positions": 5,
      "total_pnl": 325.50,
      "avg_pnl": 8.14,
      "winning_positions": 28,
      "losing_positions": 12,
      "win_rate": 70.0
    }
  }
}
```

---

#### GET `/api/signals/model/info`

Get model information and status.

**Response:**
```json
{
  "status": "success",
  "model": {
    "loaded": true,
    "version": "ppo_lstm_v3.2",
    "path": "/app/models/ppo_lstm_v3.2.onnx",
    "mode": "onnx",
    "input_name": "input",
    "output_name": "output",
    "providers": ["CPUExecutionProvider"],
    "input_shape": [1, 15],
    "output_shape": [1, 3],
    "inference_count": 150,
    "avg_latency_ms": 12.5
  }
}
```

---

## WebSocket API

Connect to `ws://localhost:8003/ws/signals` for real-time updates.

### Connection Flow

1. Client connects to WebSocket
2. Server sends welcome message
3. Server sends recent signal history (last 5)
4. Client can subscribe/unsubscribe to channels
5. Server broadcasts signals and updates
6. Periodic heartbeats maintain connection

### Message Types

#### Server → Client

##### Connected
Sent immediately after connection.
```json
{
  "type": "connected",
  "timestamp": "2025-12-17T10:30:00Z",
  "message": "Connected to trading signals WebSocket",
  "version": "1.0.0"
}
```

##### Signal
New trading signal broadcast.
```json
{
  "type": "signal",
  "timestamp": "2025-12-17T10:30:00Z",
  "data": {
    /* full signal object */
  }
}
```

##### Market Update
Market data update.
```json
{
  "type": "market_update",
  "timestamp": "2025-12-17T10:30:00Z",
  "data": {
    "symbol": "USDCOP",
    "price": 4250.50,
    "volume": 1500000
  }
}
```

##### Position Update
Position status update.
```json
{
  "type": "position_update",
  "timestamp": "2025-12-17T10:30:00Z",
  "data": {
    "position_id": "pos-123",
    "status": "CLOSED",
    "pnl": 15.50
  }
}
```

##### Heartbeat
Connection keepalive.
```json
{
  "type": "heartbeat",
  "timestamp": "2025-12-17T10:30:00Z",
  "connections": 5
}
```

##### History
Recent signals (sent on connection).
```json
{
  "type": "history",
  "timestamp": "2025-12-17T10:30:00Z",
  "data": [
    { /* signal 1 */ },
    { /* signal 2 */ }
  ]
}
```

##### Error
Error notification.
```json
{
  "type": "error",
  "timestamp": "2025-12-17T10:30:00Z",
  "message": "Error description"
}
```

#### Client → Server

##### Ping
Request heartbeat response.
```json
{
  "type": "ping"
}
```
or simply:
```
ping
```

**Response:**
```json
{
  "type": "pong",
  "timestamp": "2025-12-17T10:30:00Z"
}
```

##### Subscribe
Subscribe to specific channels (future feature).
```json
{
  "type": "subscribe",
  "channels": ["signals", "positions"]
}
```

##### Unsubscribe
Unsubscribe from channels (future feature).
```json
{
  "type": "unsubscribe",
  "channels": ["positions"]
}
```

---

## Data Models

### TradingSignal

Complete trading signal with all metadata.

**Fields:**
- `signal_id` (string): Unique signal identifier (UUID)
- `timestamp` (datetime): Signal generation time (ISO 8601)
- `symbol` (string): Trading symbol
- `action` (enum): BUY | SELL | HOLD | CLOSE_LONG | CLOSE_SHORT
- `confidence` (float, 0-1): Model confidence score
- `entry_price` (float): Recommended entry price
- `stop_loss` (float): Stop loss price
- `take_profit` (float): Take profit price
- `position_size` (float, 0-1): Position size as % of capital
- `risk_reward_ratio` (float): Risk/Reward ratio
- `model_version` (string): Model version identifier
- `model_type` (string): Model architecture type
- `reasoning` (array): Human-readable reasoning strings
- `technical_factors` (object): Technical indicator values
- `latency_ms` (float): Signal generation latency
- `metadata` (object): Additional metadata

### SignalAction

Enumeration of possible actions:
- `BUY`: Open long position
- `SELL`: Open short position
- `HOLD`: No action recommended
- `CLOSE_LONG`: Close long position
- `CLOSE_SHORT`: Close short position

---

## Error Handling

All errors return a standard format:

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

### HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `425 Too Early`: Market closed (for real-time endpoints)
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service not initialized

---

## Rate Limiting

Current implementation: 100 requests per minute per IP.

Rate limit headers (future):
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

---

## Examples

### Python

```python
import requests

# Generate signal
response = requests.post(
    'http://localhost:8003/api/signals/generate',
    json={'close_price': 4250.50, 'rsi': 35.5}
)
signal = response.json()['signal']
print(f"Action: {signal['action']}, Confidence: {signal['confidence']}")
```

### JavaScript

```javascript
// REST API
const response = await fetch('http://localhost:8003/api/signals/latest');
const data = await response.json();
console.log(data.signal);

// WebSocket
const ws = new WebSocket('ws://localhost:8003/ws/signals');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'signal') {
    console.log('New signal:', data.data);
  }
};
```

### cURL

```bash
# Get latest signal
curl http://localhost:8003/api/signals/latest

# Generate signal
curl -X POST http://localhost:8003/api/signals/generate \
  -H "Content-Type: application/json" \
  -d '{"close_price": 4250.50, "rsi": 35.5}'

# Get statistics
curl http://localhost:8003/api/signals/statistics
```

---

**Interactive Documentation**: Visit `http://localhost:8003/docs` for Swagger UI

**Alternative Docs**: Visit `http://localhost:8003/redoc` for ReDoc

---

**Version**: 1.0.0
**Last Updated**: 2025-12-17
