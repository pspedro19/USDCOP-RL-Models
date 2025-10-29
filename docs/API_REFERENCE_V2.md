# USDCOP Trading System - API Reference V2.0
## Real-Time Orchestrator & WebSocket Protocol Documentation

**Version:** 2.0.0
**Date:** October 22, 2025
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [RT Orchestrator Service](#rt-orchestrator-service)
3. [WebSocket Protocol](#websocket-protocol)
4. [Authentication & Security](#authentication--security)
5. [Error Handling](#error-handling)
6. [Code Examples](#code-examples)
7. [Rate Limiting](#rate-limiting)

---

## Overview

This document covers the **Real-Time Orchestrator Service** and **WebSocket Protocol** introduced in Version 2.0 of the USDCOP Trading System. These components provide low-latency, real-time market data streaming to clients.

### Key Features

- **Market Hours Aware**: Only operates during Colombian market hours (8 AM - 12:55 PM COT)
- **L0 Pipeline Dependency**: Waits for L0 pipeline completion before starting RT collection
- **Redis Pub/Sub**: Enables multi-client broadcasting with horizontal scaling
- **WebSocket Streaming**: Sub-second latency for price updates
- **Automatic Reconnection**: Built-in resilience for connection failures

---

## RT Orchestrator Service

**Port:** 8085
**File:** `services/usdcop_realtime_orchestrator.py`
**Purpose:** Manage real-time data collection and orchestrate live streaming

### Base URL

```
http://localhost:8085
```

---

### Health Check

#### `GET /health`

Check if the RT Orchestrator service is running.

**Response:**

```json
{
  "status": "healthy",
  "service": "rt-orchestrator",
  "version": "2.0.0",
  "market_hours_active": true,
  "l0_pipeline_completed": true,
  "realtime_collection_active": true,
  "websocket_clients_connected": 5,
  "uptime_seconds": 3600,
  "last_data_timestamp": "2025-10-22T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is starting or degraded

---

### Market Status

#### `GET /api/market-status`

Get current market status and operational state.

**Response:**

```json
{
  "market_open": true,
  "current_time": "2025-10-22T10:30:00-05:00",
  "timezone": "America/Bogota",
  "market_hours": {
    "open": "08:00",
    "close": "12:55"
  },
  "l0_pipeline_status": {
    "completed": true,
    "completed_at": "2025-10-22T08:00:45-05:00",
    "records_processed": 1,
    "last_timestamp": "2025-10-22T08:00:00Z"
  },
  "realtime_collection": {
    "active": true,
    "started_at": "2025-10-22T08:01:00-05:00",
    "data_points_collected": 145,
    "last_update": "2025-10-22T10:30:00Z"
  }
}
```

---

### Pipeline Dependency Status

#### `GET /api/pipeline-dependency`

Check L0 pipeline completion status.

**Response:**

```json
{
  "l0_completed": true,
  "completed_at": "2025-10-22T08:00:45Z",
  "wait_duration_seconds": 45,
  "fallback_mode": false,
  "historical_data_available": true
}
```

**States:**
- `l0_completed: true`: L0 pipeline ran successfully, RT collection started
- `l0_completed: false`: Waiting for L0 pipeline (max 30 minutes)
- `fallback_mode: true`: Using historical data as L0 didn't complete

---

### Force Start (Emergency Use Only)

#### `POST /api/force-start`

Force RT Orchestrator to start without waiting for L0 pipeline.

**CAUTION:** This should only be used in emergencies. It may result in duplicate data or gaps.

**Request:**

```json
{
  "reason": "L0 pipeline stuck, emergency trading session"
}
```

**Response:**

```json
{
  "status": "force_started",
  "message": "RT collection started without L0 dependency check",
  "warning": "This may result in data inconsistencies",
  "started_at": "2025-10-22T10:35:00Z"
}
```

---

### Rotate API Keys

#### `POST /api/rotate-api-keys`

Manually trigger API key rotation (useful if current key is rate-limited).

**Response:**

```json
{
  "status": "rotated",
  "previous_key_index": 3,
  "current_key_index": 4,
  "remaining_credits": 5,
  "next_reset": "2025-10-22T18:00:00Z"
}
```

---

## WebSocket Protocol

**Port:** 8082
**Endpoint:** `ws://localhost:8082/ws/market-data`

The WebSocket service provides real-time streaming of market data via Redis Pub/Sub.

---

### Connection

#### JavaScript/TypeScript

```javascript
const ws = new WebSocket('ws://localhost:8082/ws/market-data');

ws.onopen = () => {
  console.log('Connected to market data stream');

  // Subscribe to symbols
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['USDCOP']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);

  // Handle different message types
  switch (data.type) {
    case 'market_data':
      updateChart(data);
      break;
    case 'ack':
      console.log('Subscription confirmed');
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from market data stream');
  // Implement reconnection logic
  setTimeout(reconnect, 5000);
};
```

#### Python

```python
import asyncio
import websockets
import json

async def connect_market_data():
    uri = "ws://localhost:8082/ws/market-data"

    async with websockets.connect(uri) as websocket:
        # Subscribe to symbols
        await websocket.send(json.dumps({
            "action": "subscribe",
            "symbols": ["USDCOP"]
        }))

        # Receive messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

            if data['type'] == 'market_data':
                process_market_data(data)

asyncio.run(connect_market_data())
```

---

### Message Types

#### 1. Subscribe Request (Client → Server)

Subscribe to one or more symbols.

**Format:**

```json
{
  "action": "subscribe",
  "symbols": ["USDCOP", "USDBRL"]
}
```

**Response:**

```json
{
  "type": "ack",
  "message": "Subscribed to 2 symbols",
  "symbols": ["USDCOP", "USDBRL"],
  "timestamp": "2025-10-22T10:30:00Z"
}
```

---

#### 2. Unsubscribe Request (Client → Server)

Unsubscribe from symbols.

**Format:**

```json
{
  "action": "unsubscribe",
  "symbols": ["USDBRL"]
}
```

**Response:**

```json
{
  "type": "ack",
  "message": "Unsubscribed from 1 symbol",
  "symbols": ["USDBRL"],
  "timestamp": "2025-10-22T10:30:00Z"
}
```

---

#### 3. Market Data Update (Server → Client)

Real-time market data pushed from server.

**Format:**

```json
{
  "type": "market_data",
  "timestamp": "2025-10-22T10:30:05Z",
  "symbol": "USDCOP",
  "data": {
    "price": 4350.50,
    "open": 4348.00,
    "high": 4352.00,
    "low": 4347.50,
    "close": 4350.50,
    "volume": 1234567,
    "bid": 4350.25,
    "ask": 4350.75,
    "spread": 0.50,
    "change": 2.50,
    "change_percent": 0.0575
  },
  "metadata": {
    "source": "realtime",
    "latency_ms": 45,
    "data_quality": "high"
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always "market_data" |
| `timestamp` | string | ISO 8601 timestamp (UTC) |
| `symbol` | string | Trading symbol |
| `data.price` | float | Current price |
| `data.open` | float | Open price of current bar |
| `data.high` | float | High price of current bar |
| `data.low` | float | Low price of current bar |
| `data.close` | float | Close/current price |
| `data.volume` | float | Trading volume |
| `data.bid` | float | Best bid price |
| `data.ask` | float | Best ask price |
| `data.spread` | float | Bid-ask spread |
| `data.change` | float | Price change from open |
| `data.change_percent` | float | Percentage change |
| `metadata.source` | string | Data source ("realtime", "L0", "backfill") |
| `metadata.latency_ms` | int | Latency from market to client (milliseconds) |
| `metadata.data_quality` | string | "high", "medium", "low" |

---

#### 4. Heartbeat (Server → Client)

Periodic heartbeat to keep connection alive (every 30 seconds).

**Format:**

```json
{
  "type": "heartbeat",
  "timestamp": "2025-10-22T10:30:00Z",
  "server_time": "2025-10-22T10:30:00Z",
  "clients_connected": 5
}
```

**Client Response:** Clients should respond with:

```json
{
  "action": "pong",
  "client_time": "2025-10-22T10:30:00Z"
}
```

If client doesn't respond to 3 consecutive heartbeats (90 seconds), server will disconnect.

---

#### 5. Error Message (Server → Client)

Error notifications from server.

**Format:**

```json
{
  "type": "error",
  "code": "INVALID_SYMBOL",
  "message": "Symbol 'INVALID' is not supported",
  "timestamp": "2025-10-22T10:30:00Z"
}
```

**Error Codes:**

| Code | Description | Action |
|------|-------------|--------|
| `INVALID_SYMBOL` | Symbol not supported | Check symbol list |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request rate |
| `MARKET_CLOSED` | Market is closed | Wait for market hours |
| `AUTHENTICATION_FAILED` | Invalid credentials | Re-authenticate |
| `INTERNAL_ERROR` | Server error | Retry with backoff |

---

#### 6. System Status (Server → Client)

System-wide notifications (market close, maintenance, etc.).

**Format:**

```json
{
  "type": "system_status",
  "status": "market_closing",
  "message": "Market closing in 5 minutes",
  "timestamp": "2025-10-22T12:50:00Z",
  "details": {
    "market_close_time": "2025-10-22T12:55:00Z",
    "seconds_until_close": 300
  }
}
```

**Status Values:**
- `market_opening`: Market about to open
- `market_closing`: Market about to close
- `market_closed`: Market closed, disconnecting
- `maintenance_mode`: System maintenance
- `degraded_service`: Reduced functionality

---

### Reconnection Strategy

Clients should implement exponential backoff for reconnections:

```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const baseDelay = 1000; // 1 second

function reconnect() {
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error('Max reconnection attempts reached');
    return;
  }

  const delay = Math.min(baseDelay * Math.pow(2, reconnectAttempts), 30000);
  console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1})`);

  setTimeout(() => {
    reconnectAttempts++;
    connectWebSocket();
  }, delay);
}

function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8082/ws/market-data');

  ws.onopen = () => {
    console.log('Connected');
    reconnectAttempts = 0; // Reset on successful connection
  };

  ws.onclose = () => {
    console.log('Disconnected');
    reconnect();
  };
}
```

---

## Authentication & Security

### Current Version (Development)

**Authentication:** None (open WebSocket)
**Suitable for:** Development, internal networks

### Future Version (Production)

**Authentication:** JWT token-based

#### Connection with JWT:

```javascript
const token = 'your_jwt_token_here';
const ws = new WebSocket(`ws://localhost:8082/ws/market-data?token=${token}`);
```

#### Token Format:

```json
{
  "sub": "user_123",
  "exp": 1698000000,
  "iat": 1697996400,
  "scopes": ["market_data:read", "trading:write"]
}
```

### Security Best Practices

1. **Use WSS (WebSocket Secure)** in production:
   ```javascript
   const ws = new WebSocket('wss://trading.example.com/ws/market-data');
   ```

2. **Validate all client messages** on server:
   ```python
   try:
       message = json.loads(data)
       if not validate_message(message):
           raise ValueError("Invalid message format")
   except Exception as e:
       await websocket.send(json.dumps({
           "type": "error",
           "message": str(e)
       }))
   ```

3. **Implement rate limiting** per connection:
   - Max 100 messages/minute per client
   - Max 10 subscriptions per client

4. **Monitor for abuse**:
   - Log connection attempts
   - Track subscription patterns
   - Alert on suspicious activity

---

## Error Handling

### Client-Side Error Handling

```typescript
// React hook example
import { useEffect, useState } from 'react';
import { useWebSocket } from 'react-use-websocket';

export function useMarketData(symbol: string) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    'ws://localhost:8082/ws/market-data',
    {
      onOpen: () => {
        console.log('Connected');
        sendMessage(JSON.stringify({
          action: 'subscribe',
          symbols: [symbol]
        }));
      },
      onError: (event) => {
        console.error('WebSocket error:', event);
        setError('Connection failed');
      },
      onClose: () => {
        console.log('Disconnected');
      },
      // Automatically reconnect
      shouldReconnect: (closeEvent) => true,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
    }
  );

  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const parsed = JSON.parse(lastMessage.data);

        if (parsed.type === 'market_data') {
          setData(parsed.data);
          setError(null);
        } else if (parsed.type === 'error') {
          setError(parsed.message);
        }
      } catch (err) {
        console.error('Failed to parse message:', err);
        setError('Invalid data received');
      }
    }
  }, [lastMessage]);

  return { data, error, readyState };
}
```

### Server-Side Error Handling

```python
# services/websocket_service.py
import logging
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.websocket("/ws/market-data")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Process subscription requests
            if data.get('action') == 'subscribe':
                symbols = data.get('symbols', [])
                # Add subscription logic
                await websocket.send_json({
                    "type": "ack",
                    "message": f"Subscribed to {len(symbols)} symbols"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
```

---

## Code Examples

### Complete React Component

```typescript
// components/RealTimePrice.tsx
'use client';

import { useEffect, useState } from 'react';
import { useWebSocket } from 'react-use-websocket';

interface PriceData {
  price: number;
  change: number;
  change_percent: number;
  timestamp: string;
}

export const RealTimePrice: React.FC<{ symbol: string }> = ({ symbol }) => {
  const [priceData, setPriceData] = useState<PriceData | null>(null);
  const [connected, setConnected] = useState(false);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    'ws://localhost:8082/ws/market-data',
    {
      onOpen: () => {
        console.log('WebSocket connected');
        setConnected(true);
        sendMessage(JSON.stringify({
          action: 'subscribe',
          symbols: [symbol]
        }));
      },
      onClose: () => {
        setConnected(false);
      },
      shouldReconnect: () => true,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
    }
  );

  useEffect(() => {
    if (lastMessage !== null) {
      const data = JSON.parse(lastMessage.data);

      if (data.type === 'market_data' && data.symbol === symbol) {
        setPriceData({
          price: data.data.price,
          change: data.data.change,
          change_percent: data.data.change_percent,
          timestamp: data.timestamp
        });
      }
    }
  }, [lastMessage, symbol]);

  if (!connected) {
    return <div className="text-yellow-500">Connecting...</div>;
  }

  if (!priceData) {
    return <div className="text-gray-500">Waiting for data...</div>;
  }

  const isPositive = priceData.change >= 0;

  return (
    <div className="p-4 border rounded">
      <div className="text-sm text-gray-500">{symbol}</div>
      <div className="text-3xl font-bold">
        ${priceData.price.toFixed(2)}
      </div>
      <div className={`text-sm ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
        {isPositive ? '▲' : '▼'} {Math.abs(priceData.change).toFixed(2)} (
        {priceData.change_percent.toFixed(2)}%)
      </div>
      <div className="text-xs text-gray-400 mt-2">
        Updated: {new Date(priceData.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
};
```

---

### Python Streaming Client

```python
# scripts/realtime_client.py
import asyncio
import websockets
import json
from datetime import datetime

class MarketDataClient:
    def __init__(self, uri: str, symbols: list[str]):
        self.uri = uri
        self.symbols = symbols
        self.websocket = None

    async def connect(self):
        """Connect to WebSocket and subscribe to symbols"""
        async with websockets.connect(self.uri) as websocket:
            self.websocket = websocket
            print(f"Connected to {self.uri}")

            # Subscribe
            await websocket.send(json.dumps({
                "action": "subscribe",
                "symbols": self.symbols
            }))

            # Listen for messages
            async for message in websocket:
                await self.handle_message(json.loads(message))

    async def handle_message(self, data: dict):
        """Process incoming messages"""
        msg_type = data.get('type')

        if msg_type == 'market_data':
            symbol = data['symbol']
            price = data['data']['price']
            change = data['data']['change_percent']
            timestamp = data['timestamp']

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"{symbol}: ${price:.2f} ({change:+.2f}%)")

        elif msg_type == 'ack':
            print(f"✓ {data['message']}")

        elif msg_type == 'heartbeat':
            print(f"♥ Heartbeat - {data['clients_connected']} clients")

        elif msg_type == 'error':
            print(f"✗ Error: {data['message']}")

        elif msg_type == 'system_status':
            print(f"ⓘ System: {data['message']}")

async def main():
    client = MarketDataClient(
        uri="ws://localhost:8082/ws/market-data",
        symbols=["USDCOP"]
    )

    try:
        await client.connect()
    except KeyboardInterrupt:
        print("\nDisconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run:**

```bash
python scripts/realtime_client.py
```

---

## Rate Limiting

### Current Limits

| Resource | Limit | Window | Consequence |
|----------|-------|--------|-------------|
| WebSocket connections | 100 per IP | - | Connection refused |
| Messages per connection | 100 | 1 minute | Temporary disconnect |
| Subscriptions per connection | 10 | - | Error message |
| API requests (RT Orchestrator) | 60 | 1 minute | HTTP 429 |

### Headers

Rate limit information is included in HTTP API responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1698000000
```

### Handling Rate Limits

```python
import time
import requests

def call_api_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 429:
            # Rate limit exceeded
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = reset_time - int(time.time())

            if wait_time > 0:
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time + 1)
            else:
                time.sleep(2 ** attempt)  # Exponential backoff

        else:
            raise Exception(f"API error: {response.status_code}")

    raise Exception("Max retries exceeded")
```

---

## Testing

### Test WebSocket Connection

```bash
# Using websocat (install: cargo install websocat)
websocat ws://localhost:8082/ws/market-data

# Send subscription
{"action": "subscribe", "symbols": ["USDCOP"]}

# You should see market data streaming
```

### Test RT Orchestrator Health

```bash
curl http://localhost:8085/health
curl http://localhost:8085/api/market-status
```

---

## Troubleshooting

### WebSocket Won't Connect

**Check:**
1. Service is running: `docker ps | grep websocket`
2. Port is accessible: `curl http://localhost:8082/health`
3. Firewall rules allow port 8082
4. Browser console for CORS errors

**Solution:**

```bash
# Restart WebSocket service
docker compose restart websocket-service

# Check logs
docker logs usdcop-websocket -f
```

---

### No Data Received

**Check:**
1. Market is open (8 AM - 12:55 PM COT Monday-Friday)
2. RT Orchestrator is collecting data
3. L0 pipeline completed
4. Subscription was acknowledged

**Debug:**

```bash
# Check RT Orchestrator status
curl http://localhost:8085/api/market-status

# Check Redis pub/sub
docker exec usdcop-redis redis-cli -a redis123 pubsub channels

# Check database for latest data
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE source = 'realtime';"
```

---

### High Latency

**Causes:**
1. Network congestion
2. Database connection pool exhausted
3. Too many WebSocket clients
4. Redis slow (memory full)

**Solutions:**

```bash
# Check Redis memory
docker exec usdcop-redis redis-cli -a redis123 info memory

# Check connection pool
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT count(*) FROM pg_stat_activity;"

# Restart services
docker compose restart websocket-service usdcop-realtime-orchestrator
```

---

## Migration from V1

If migrating from old WebSocket API:

**Old (V1):**
```javascript
ws://localhost:8000/ws  // Combined with Trading API
```

**New (V2):**
```javascript
ws://localhost:8082/ws/market-data  // Dedicated WebSocket service
```

**Changes:**
- Separate WebSocket service (better scalability)
- Redis pub/sub (supports multiple instances)
- Enhanced message format (includes metadata)
- Heartbeat mechanism (connection health)
- System status messages (market events)

---

## Additional Resources

- **Architecture Documentation:** `docs/ARCHITECTURE.md`
- **Development Guide:** `docs/DEVELOPMENT.md`
- **Operations Runbook:** `docs/RUNBOOK.md`
- **WebSocket RFC:** https://www.rfc-editor.org/rfc/rfc6455
- **Redis Pub/Sub:** https://redis.io/docs/manual/pubsub/

---

**Questions?** Contact the API team at api@trading.com
