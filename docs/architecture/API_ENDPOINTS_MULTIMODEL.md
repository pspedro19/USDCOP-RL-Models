# Multi-Model API Endpoints Reference

**Version:** 1.0.0
**Base URL:** `http://localhost:8006`
**Last Updated:** 2025-12-26

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Formats](#common-response-formats)
4. [Endpoints](#endpoints)
   - [Root & Health](#root--health)
   - [Signals](#signals)
   - [Performance](#performance)
   - [Equity Curves](#equity-curves)
   - [Positions](#positions)
   - [P&L Summary](#pl-summary)
   - [WebSocket](#websocket)
5. [Error Codes](#error-codes)
6. [Rate Limits](#rate-limits)

---

## Overview

The Multi-Model Trading API provides access to trading signals, performance metrics, and portfolio data from multiple trading strategies (RL, ML, LLM, Ensemble).

### Supported Strategies

| Code | Name | Type |
|------|------|------|
| `RL_PPO` | PPO Reinforcement Learning | RL |
| `ML_XGB` | XGBoost Classifier | ML |
| `ML_LGBM` | LightGBM Model | ML |
| `LLM_CLAUDE` | Claude LLM Signals | LLM |
| `ENSEMBLE` | Weighted Ensemble | ENSEMBLE |

---

## Authentication

Currently, the API does not require authentication. Future versions may implement JWT-based auth.

---

## Common Response Formats

### Success Response

```json
{
  "data": { ... },
  "timestamp": "2025-12-26T12:00:00.000Z"
}
```

### Error Response

```json
{
  "detail": "Error message description",
  "status_code": 500
}
```

---

## Endpoints

### Root & Health

#### GET `/`

Returns API information and available endpoints.

**Request:**
```bash
curl http://localhost:8006/
```

**Response:**
```json
{
  "message": "Multi-Model Trading Signals API",
  "version": "1.0.0",
  "endpoints": {
    "signals": "/api/models/signals/latest",
    "performance": "/api/models/performance/comparison",
    "equity_curves": "/api/models/equity-curves",
    "positions": "/api/models/positions/current",
    "pnl": "/api/models/pnl/summary",
    "websocket": "/ws/trading-signals"
  }
}
```

---

#### GET `/api/health`

Health check endpoint.

**Request:**
```bash
curl http://localhost:8006/api/health
```

**Response (Healthy):**
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-12-26T12:00:00.000Z"
}
```

**Response (Unhealthy):**
```json
{
  "status": "unhealthy",
  "database": "disconnected",
  "error": "Connection refused",
  "timestamp": "2025-12-26T12:00:00.000Z"
}
```

---

### Signals

#### GET `/api/models/signals/latest`

Returns the most recent trading signal from each active strategy.

**Request:**
```bash
curl http://localhost:8006/api/models/signals/latest
```

**Response Schema:**
```json
{
  "timestamp": "string (ISO 8601)",
  "market_price": "number",
  "market_status": "string (open|closed|pre_market)",
  "signals": [
    {
      "strategy_code": "string",
      "strategy_name": "string",
      "signal": "string (long|short|flat|close)",
      "side": "string (buy|sell|hold)",
      "confidence": "number (0.0-1.0)",
      "size": "number (0.0-1.0)",
      "entry_price": "number|null",
      "stop_loss": "number|null",
      "take_profit": "number|null",
      "risk_usd": "number",
      "reasoning": "string",
      "timestamp": "string (ISO 8601)",
      "age_seconds": "integer"
    }
  ]
}
```

**Example Response:**
```json
{
  "timestamp": "2025-12-26T12:00:00.000Z",
  "market_price": 4250.50,
  "market_status": "open",
  "signals": [
    {
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "signal": "long",
      "side": "buy",
      "confidence": 0.85,
      "size": 0.25,
      "entry_price": 4250.00,
      "stop_loss": 4200.00,
      "take_profit": 4350.00,
      "risk_usd": 50.00,
      "reasoning": "Bullish momentum detected with VIX < 20 and DXY declining",
      "timestamp": "2025-12-26T11:55:00.000Z",
      "age_seconds": 300
    },
    {
      "strategy_code": "ML_XGB",
      "strategy_name": "XGBoost Classifier",
      "signal": "long",
      "side": "buy",
      "confidence": 0.72,
      "size": 0.20,
      "entry_price": 4250.00,
      "stop_loss": 4210.00,
      "take_profit": 4320.00,
      "risk_usd": 40.00,
      "reasoning": "Feature importance indicates bullish setup",
      "timestamp": "2025-12-26T11:55:00.000Z",
      "age_seconds": 300
    }
  ]
}
```

---

### Performance

#### GET `/api/models/performance/comparison`

Compare performance metrics across all strategies.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `30d` | Time period: `24h`, `7d`, `30d`, `all` |

**Request:**
```bash
curl "http://localhost:8006/api/models/performance/comparison?period=30d"
```

**Response Schema:**
```json
{
  "period": "string",
  "start_date": "string (ISO 8601)",
  "end_date": "string (ISO 8601)",
  "strategies": [
    {
      "strategy_code": "string",
      "strategy_name": "string",
      "strategy_type": "string (RL|ML|LLM|ENSEMBLE)",
      "total_return_pct": "number",
      "daily_return_pct": "number",
      "sharpe_ratio": "number",
      "sortino_ratio": "number",
      "calmar_ratio": "number",
      "total_trades": "integer",
      "win_rate": "number (0.0-1.0)",
      "profit_factor": "number",
      "max_drawdown_pct": "number",
      "current_drawdown_pct": "number",
      "volatility_pct": "number",
      "avg_hold_time_minutes": "number",
      "current_equity": "number",
      "open_positions": "integer"
    }
  ]
}
```

**Example Response:**
```json
{
  "period": "30d",
  "start_date": "2025-11-26T00:00:00.000Z",
  "end_date": "2025-12-26T12:00:00.000Z",
  "strategies": [
    {
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "strategy_type": "RL",
      "total_return_pct": 8.5,
      "daily_return_pct": 0.28,
      "sharpe_ratio": 1.85,
      "sortino_ratio": 2.45,
      "calmar_ratio": 1.2,
      "total_trades": 145,
      "win_rate": 0.58,
      "profit_factor": 2.1,
      "max_drawdown_pct": 7.1,
      "current_drawdown_pct": 1.2,
      "volatility_pct": 8.0,
      "avg_hold_time_minutes": 45.5,
      "current_equity": 10850.00,
      "open_positions": 1
    }
  ]
}
```

---

### Equity Curves

#### GET `/api/models/equity-curves`

Get historical equity curves for charting.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hours` | integer | `24` | Hours of data (1-720) |
| `strategies` | string | all | Comma-separated strategy codes |
| `resolution` | string | `5m` | Data resolution: `5m`, `1h`, `1d` |

**Request:**
```bash
# Get 24 hours of 5-minute data for all strategies
curl "http://localhost:8006/api/models/equity-curves?hours=24&resolution=5m"

# Get 168 hours (1 week) of hourly data for specific strategies
curl "http://localhost:8006/api/models/equity-curves?hours=168&resolution=1h&strategies=RL_PPO,ML_XGB"
```

**Response Schema:**
```json
{
  "start_date": "string (ISO 8601)",
  "end_date": "string (ISO 8601)",
  "resolution": "string",
  "curves": [
    {
      "strategy_code": "string",
      "strategy_name": "string",
      "data": [
        {
          "timestamp": "string (ISO 8601)",
          "equity_value": "number",
          "return_pct": "number",
          "drawdown_pct": "number"
        }
      ],
      "summary": {
        "starting_equity": "number",
        "ending_equity": "number",
        "total_return_pct": "number"
      }
    }
  ]
}
```

**Example Response:**
```json
{
  "start_date": "2025-12-25T12:00:00.000Z",
  "end_date": "2025-12-26T12:00:00.000Z",
  "resolution": "1h",
  "curves": [
    {
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "data": [
        {
          "timestamp": "2025-12-25T13:00:00.000Z",
          "equity_value": 10000.00,
          "return_pct": 0.00,
          "drawdown_pct": 0.00
        },
        {
          "timestamp": "2025-12-25T14:00:00.000Z",
          "equity_value": 10050.00,
          "return_pct": 0.50,
          "drawdown_pct": 0.00
        }
      ],
      "summary": {
        "starting_equity": 10000.00,
        "ending_equity": 10850.00,
        "total_return_pct": 8.50
      }
    }
  ]
}
```

---

### Positions

#### GET `/api/models/positions/current`

Get all open positions across strategies.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | string | all | Filter by strategy code |

**Request:**
```bash
# Get all positions
curl http://localhost:8006/api/models/positions/current

# Get positions for specific strategy
curl "http://localhost:8006/api/models/positions/current?strategy=RL_PPO"
```

**Response Schema:**
```json
{
  "timestamp": "string (ISO 8601)",
  "total_positions": "integer",
  "total_notional": "number",
  "total_pnl": "number",
  "positions": [
    {
      "position_id": "integer",
      "strategy_code": "string",
      "strategy_name": "string",
      "side": "string (long|short)",
      "quantity": "number",
      "entry_price": "number",
      "current_price": "number",
      "stop_loss": "number|null",
      "take_profit": "number|null",
      "unrealized_pnl": "number",
      "unrealized_pnl_pct": "number",
      "entry_time": "string (ISO 8601)",
      "holding_time_minutes": "integer",
      "leverage": "integer"
    }
  ]
}
```

**Example Response:**
```json
{
  "timestamp": "2025-12-26T12:00:00.000Z",
  "total_positions": 3,
  "total_notional": 127500.00,
  "total_pnl": 425.00,
  "positions": [
    {
      "position_id": 1234,
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "side": "long",
      "quantity": 10.0,
      "entry_price": 4235.00,
      "current_price": 4250.50,
      "stop_loss": 4200.00,
      "take_profit": 4350.00,
      "unrealized_pnl": 155.00,
      "unrealized_pnl_pct": 0.37,
      "entry_time": "2025-12-26T10:30:00.000Z",
      "holding_time_minutes": 90,
      "leverage": 1
    }
  ]
}
```

---

### P&L Summary

#### GET `/api/models/pnl/summary`

Get profit and loss breakdown by strategy.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `today` | Time period: `today`, `week`, `month`, `all` |

**Request:**
```bash
curl "http://localhost:8006/api/models/pnl/summary?period=month"
```

**Response Schema:**
```json
{
  "period": "string",
  "start_date": "string (ISO 8601)",
  "end_date": "string (ISO 8601)",
  "strategies": [
    {
      "strategy_code": "string",
      "strategy_name": "string",
      "gross_profit": "number",
      "gross_loss": "number",
      "net_profit": "number",
      "total_fees": "number",
      "n_trades": "integer",
      "n_wins": "integer",
      "n_losses": "integer",
      "win_rate": "number (0.0-1.0)",
      "avg_win": "number",
      "avg_loss": "number",
      "avg_trade": "number",
      "profit_factor": "number"
    }
  ],
  "portfolio_total": "number"
}
```

**Example Response:**
```json
{
  "period": "month",
  "start_date": "2025-11-26",
  "end_date": "2025-12-26",
  "strategies": [
    {
      "strategy_code": "RL_PPO",
      "strategy_name": "PPO Reinforcement Learning",
      "gross_profit": 2500.00,
      "gross_loss": 1200.00,
      "net_profit": 1250.00,
      "total_fees": 50.00,
      "n_trades": 145,
      "n_wins": 84,
      "n_losses": 61,
      "win_rate": 0.58,
      "avg_win": 29.76,
      "avg_loss": 19.67,
      "avg_trade": 8.62,
      "profit_factor": 2.08
    }
  ],
  "portfolio_total": 3750.00
}
```

---

### WebSocket

#### WS `/ws/trading-signals`

Real-time WebSocket endpoint for trading signals and updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8006/ws/trading-signals');
```

**Message Types:**

1. **Connection Acknowledgment**
```json
{
  "type": "connection",
  "timestamp": "2025-12-26T12:00:00.000Z",
  "message": "Connected to trading signals stream"
}
```

2. **Heartbeat** (every 30 seconds)
```json
{
  "type": "heartbeat",
  "timestamp": "2025-12-26T12:00:30.000Z"
}
```

3. **Signal Update**
```json
{
  "type": "signal_update",
  "strategy_code": "RL_PPO",
  "signal": "long",
  "confidence": 0.85,
  "price": 4250.50,
  "timestamp": "2025-12-26T12:05:00.000Z"
}
```

4. **Position Change**
```json
{
  "type": "position_change",
  "strategy_code": "RL_PPO",
  "action": "open",
  "side": "long",
  "quantity": 10.0,
  "price": 4250.00,
  "timestamp": "2025-12-26T12:05:00.000Z"
}
```

**Keep-Alive:**
```javascript
// Send ping every 25 seconds
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 25000);

// Handle pong response
ws.onmessage = (event) => {
  if (event.data === 'pong') return;
  const data = JSON.parse(event.data);
  // Process data
};
```

**Full Example:**
```javascript
class TradingSignalsClient {
  constructor(url = 'ws://localhost:8006/ws/trading-signals') {
    this.url = url;
    this.ws = null;
    this.reconnectInterval = 5000;
    this.pingInterval = 25000;
    this.pingTimer = null;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected to trading signals');
      this.startPing();
    };

    this.ws.onmessage = (event) => {
      if (event.data === 'pong') return;

      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };

    this.ws.onclose = () => {
      console.log('Disconnected, reconnecting...');
      this.stopPing();
      setTimeout(() => this.connect(), this.reconnectInterval);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  handleMessage(data) {
    switch (data.type) {
      case 'connection':
        console.log('Connection established');
        break;
      case 'heartbeat':
        // Connection alive
        break;
      case 'signal_update':
        console.log('New signal:', data);
        this.onSignal(data);
        break;
      case 'position_change':
        console.log('Position change:', data);
        this.onPosition(data);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  }

  startPing() {
    this.pingTimer = setInterval(() => {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, this.pingInterval);
  }

  stopPing() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  // Override these methods
  onSignal(data) {}
  onPosition(data) {}
}

// Usage
const client = new TradingSignalsClient();
client.onSignal = (data) => {
  console.log(`New ${data.signal} signal from ${data.strategy_code}`);
};
client.connect();
```

---

## Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Resource or data not found |
| 422 | Validation Error | Parameter validation failed |
| 500 | Internal Server Error | Database connection, query errors |
| 503 | Service Unavailable | Database unavailable |

**Error Response Format:**
```json
{
  "detail": "Detailed error message",
  "status_code": 500
}
```

---

## Rate Limits

Currently, no rate limits are enforced. Future versions may implement:

| Endpoint Type | Limit |
|---------------|-------|
| REST API | 100 requests/minute |
| WebSocket | 1 connection/client |
| WebSocket Messages | 10 messages/second |

---

## Related Documentation

- [Main Multi-Model Backend Documentation](./MULTI_MODEL_BACKEND.md)
- [Database Schema V19](./DATABASE_SCHEMA_V19.md)
- [Trading API (Real-time)](./API_TRADING_REALTIME.md)
