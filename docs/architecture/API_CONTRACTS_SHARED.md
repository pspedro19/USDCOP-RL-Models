# API Contracts - Compartido Backend/Frontend

**Proyecto:** USDCOP RL Trading System V19
**Version:** 1.0.0
**Ultima Actualizacion:** 2026-01-07

---

## Proposito

Este documento define los **contratos de API** que deben ser respetados tanto por el backend (Python/FastAPI) como por el frontend (Next.js/TypeScript).

**IMPORTANTE:** Cualquier cambio en estos contratos debe ser comunicado y actualizado en ambos lados simultaneamente.

---

## Base URL

| Ambiente | URL |
|----------|-----|
| Development | `http://localhost:8000` |
| Docker Internal | `http://multi-model-api:8000` |
| Production | TBD |

---

## 1. Signals API

### GET /api/signals/latest

Obtiene las ultimas senales de trading de todos los modelos activos.

#### Request
```
GET /api/signals/latest
Headers:
  Content-Type: application/json
```

#### Response (200 OK)
```typescript
interface LatestSignalsResponse {
  timestamp: string;              // ISO 8601: "2026-01-07T16:30:00Z"
  market_price: number;           // Precio actual USD/COP: 4350.25
  market_status: MarketStatus;    // Estado del mercado
  signals: StrategySignal[];      // Array de senales por modelo
}

type MarketStatus = "open" | "closed" | "pre_market";

interface StrategySignal {
  strategy_code: string;          // ID unico: "ppo_v1", "sac_v2"
  strategy_name: string;          // Nombre display: "PPO Conservative"
  signal: SignalType;             // Tipo de senal
  side: TradeSide;                // Lado de la operacion
  confidence: number;             // 0.0 - 1.0
  size: number;                   // 0.0 - 1.0 (position sizing)
  entry_price?: number;           // Precio de entrada sugerido
  stop_loss?: number;             // Stop loss sugerido
  take_profit?: number;           // Take profit sugerido
  risk_usd: number;               // Riesgo en USD
  reasoning: string;              // Explicacion de la senal
  timestamp: string;              // ISO 8601: cuando se genero
  age_seconds: number;            // Antiguedad en segundos
}

type SignalType = "LONG" | "SHORT" | "HOLD" | "CLOSE";
type TradeSide = "buy" | "sell" | "hold";
```

#### Ejemplo Response
```json
{
  "timestamp": "2026-01-07T16:30:00Z",
  "market_price": 4350.25,
  "market_status": "open",
  "signals": [
    {
      "strategy_code": "ppo_v1",
      "strategy_name": "PPO Conservative",
      "signal": "LONG",
      "side": "buy",
      "confidence": 0.75,
      "size": 0.5,
      "entry_price": 4350.25,
      "stop_loss": 4320.00,
      "take_profit": 4410.00,
      "risk_usd": 150.00,
      "reasoning": "Bullish momentum detected with DXY weakness",
      "timestamp": "2026-01-07T16:29:55Z",
      "age_seconds": 5
    }
  ]
}
```

#### Pydantic Model (Backend)
```python
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime

class StrategySignal(BaseModel):
    strategy_code: str
    strategy_name: str
    signal: Literal["LONG", "SHORT", "HOLD", "CLOSE"]
    side: Literal["buy", "sell", "hold"]
    confidence: float
    size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_usd: float
    reasoning: str
    timestamp: datetime
    age_seconds: int

class LatestSignalsResponse(BaseModel):
    timestamp: datetime
    market_price: float
    market_status: Literal["open", "closed", "pre_market"]
    signals: List[StrategySignal]
```

---

## 2. Performance API

### GET /api/performance

Obtiene metricas de performance de todos los modelos.

#### Query Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| period | string | "7d" | Periodo: "1d", "7d", "30d", "90d", "all" |

#### Request
```
GET /api/performance?period=30d
```

#### Response (200 OK)
```typescript
interface PerformanceResponse {
  period: string;                     // "30d"
  start_date: string;                 // ISO 8601
  end_date: string;                   // ISO 8601
  strategies: StrategyPerformance[];
}

interface StrategyPerformance {
  strategy_code: string;              // "ppo_v1"
  strategy_name: string;              // "PPO Conservative"
  sharpe_ratio: number;               // 1.5
  sortino_ratio: number;              // 2.1
  calmar_ratio: number;               // 1.2
  total_return_pct: number;           // 5.5 (porcentaje)
  max_drawdown_pct: number;           // -8.2 (porcentaje, negativo)
  current_drawdown_pct: number;       // -2.1 (porcentaje)
  win_rate: number;                   // 0.58 (0-1)
  profit_factor: number;              // 1.8
  total_trades: number;               // 45
  winning_trades: number;             // 26
  losing_trades: number;              // 19
  avg_win_pct: number;                // 1.2
  avg_loss_pct: number;               // -0.8
  current_equity: number;             // 10550.00
  initial_equity: number;             // 10000.00
}
```

#### Ejemplo Response
```json
{
  "period": "30d",
  "start_date": "2025-12-08T00:00:00Z",
  "end_date": "2026-01-07T23:59:59Z",
  "strategies": [
    {
      "strategy_code": "ppo_v1",
      "strategy_name": "PPO Conservative",
      "sharpe_ratio": 1.52,
      "sortino_ratio": 2.14,
      "calmar_ratio": 1.18,
      "total_return_pct": 5.5,
      "max_drawdown_pct": -8.2,
      "current_drawdown_pct": -2.1,
      "win_rate": 0.58,
      "profit_factor": 1.82,
      "total_trades": 45,
      "winning_trades": 26,
      "losing_trades": 19,
      "avg_win_pct": 1.2,
      "avg_loss_pct": -0.8,
      "current_equity": 10550.00,
      "initial_equity": 10000.00
    }
  ]
}
```

#### Pydantic Model (Backend)
```python
class StrategyPerformance(BaseModel):
    strategy_code: str
    strategy_name: str
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    current_equity: float
    initial_equity: float

class PerformanceResponse(BaseModel):
    period: str
    start_date: datetime
    end_date: datetime
    strategies: List[StrategyPerformance]
```

---

## 3. Positions API

### GET /api/positions

Obtiene posiciones abiertas actuales de todos los modelos.

#### Response (200 OK)
```typescript
interface PositionsResponse {
  timestamp: string;                  // ISO 8601
  total_positions: number;            // Cantidad de posiciones abiertas
  total_unrealized_pnl: number;       // PnL total no realizado
  positions: Position[];
}

interface Position {
  position_id: number;                // ID unico
  strategy_code: string;              // "ppo_v1"
  strategy_name: string;              // "PPO Conservative"
  side: "buy" | "sell";               // Direccion
  quantity: number;                   // Tamano de posicion
  entry_price: number;                // Precio de entrada
  current_price: number;              // Precio actual
  unrealized_pnl: number;             // PnL en USD
  unrealized_pnl_pct: number;         // PnL en porcentaje
  entry_time: string;                 // ISO 8601
  holding_time_minutes: number;       // Tiempo en posicion
}
```

#### Ejemplo Response
```json
{
  "timestamp": "2026-01-07T16:30:00Z",
  "total_positions": 1,
  "total_unrealized_pnl": 25.50,
  "positions": [
    {
      "position_id": 123,
      "strategy_code": "ppo_v1",
      "strategy_name": "PPO Conservative",
      "side": "buy",
      "quantity": 1.0,
      "entry_price": 4340.00,
      "current_price": 4350.25,
      "unrealized_pnl": 25.50,
      "unrealized_pnl_pct": 0.24,
      "entry_time": "2026-01-07T14:15:00Z",
      "holding_time_minutes": 135
    }
  ]
}
```

---

## 4. Risk Status API (NUEVO)

### GET /api/risk/status

Obtiene el estado actual del RiskManager (safety layer).

#### Response (200 OK)
```typescript
interface RiskStatusResponse {
  kill_switch_active: boolean;        // true = trading detenido
  current_drawdown_pct: number;       // Drawdown actual
  daily_pnl_pct: number;              // PnL del dia
  trades_today: number;               // Trades ejecutados hoy
  consecutive_losses: number;         // Perdidas consecutivas
  cooldown_active: boolean;           // true = en cooldown
  cooldown_until?: string;            // ISO 8601: fin del cooldown
  limits: RiskLimits;
}

interface RiskLimits {
  max_drawdown_pct: number;           // 15.0 (kill switch)
  max_daily_loss_pct: number;         // 5.0 (stop diario)
  max_trades_per_day: number;         // 20
}
```

#### Ejemplo Response
```json
{
  "kill_switch_active": false,
  "current_drawdown_pct": 3.2,
  "daily_pnl_pct": -1.5,
  "trades_today": 8,
  "consecutive_losses": 1,
  "cooldown_active": false,
  "cooldown_until": null,
  "limits": {
    "max_drawdown_pct": 15.0,
    "max_daily_loss_pct": 5.0,
    "max_trades_per_day": 20
  }
}
```

#### Estados de Alerta UI
| Condicion | Color | Accion UI |
|-----------|-------|-----------|
| `kill_switch_active: true` | Rojo | Banner prominente, deshabilitar botones |
| `current_drawdown_pct > 10` | Naranja | Warning badge |
| `cooldown_active: true` | Amarillo | Mostrar countdown |
| `trades_today >= max_trades * 0.8` | Amarillo | Warning: cerca del limite |
| Todo OK | Verde | Normal operation |

---

## 5. Model Health API (NUEVO)

### GET /api/monitor/health

Obtiene estado de salud de los modelos (drift detection).

#### Response (200 OK)
```typescript
interface ModelHealthResponse {
  models: ModelHealth[];
  overall_status: "healthy" | "warning" | "critical";
}

interface ModelHealth {
  model_id: string;                   // "ppo_v1"
  action_drift_kl: number;            // KL divergence (0 = sin drift)
  stuck_behavior: boolean;            // true = posible problema
  rolling_sharpe: number;             // Sharpe ultimos N trades
  status: "healthy" | "warning" | "critical";
  last_inference: string;             // ISO 8601
}
```

#### Ejemplo Response
```json
{
  "models": [
    {
      "model_id": "ppo_v1",
      "action_drift_kl": 0.12,
      "stuck_behavior": false,
      "rolling_sharpe": 1.2,
      "status": "healthy",
      "last_inference": "2026-01-07T16:29:55Z"
    }
  ],
  "overall_status": "healthy"
}
```

---

## 6. WebSocket: Signals Stream

### WS /ws/trading-signals

Stream en tiempo real de senales de trading.

#### Connection
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/trading-signals");
```

#### Message Format (Server -> Client)
```typescript
interface SignalStreamMessage {
  type: "signal" | "heartbeat" | "market_status";
  timestamp: string;
  data: SignalData | HeartbeatData | MarketStatusData;
}

interface SignalData {
  strategy_code: string;
  signal: SignalType;
  confidence: number;
  price: number;
  reasoning: string;
}

interface HeartbeatData {
  server_time: string;
  connected_clients: number;
}

interface MarketStatusData {
  status: MarketStatus;
  next_open?: string;
  next_close?: string;
}
```

#### Ejemplo Messages
```json
// Signal
{
  "type": "signal",
  "timestamp": "2026-01-07T16:30:00Z",
  "data": {
    "strategy_code": "ppo_v1",
    "signal": "LONG",
    "confidence": 0.75,
    "price": 4350.25,
    "reasoning": "Bullish momentum"
  }
}

// Heartbeat (cada 30s)
{
  "type": "heartbeat",
  "timestamp": "2026-01-07T16:30:30Z",
  "data": {
    "server_time": "2026-01-07T16:30:30Z",
    "connected_clients": 3
  }
}

// Market Status Change
{
  "type": "market_status",
  "timestamp": "2026-01-07T16:00:00Z",
  "data": {
    "status": "closed",
    "next_open": "2026-01-08T08:00:00-05:00"
  }
}
```

---

## 7. Error Responses

### Standard Error Format
```typescript
interface ErrorResponse {
  error: string;                      // Codigo de error
  message: string;                    // Mensaje legible
  details?: Record<string, unknown>;  // Detalles adicionales
  timestamp: string;                  // ISO 8601
}
```

### HTTP Status Codes
| Code | Uso |
|------|-----|
| 200 | Success |
| 400 | Bad Request (parametros invalidos) |
| 401 | Unauthorized (futuro: auth) |
| 404 | Not Found |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable (DB down, etc) |

#### Ejemplo Error
```json
{
  "error": "INVALID_PERIOD",
  "message": "Period must be one of: 1d, 7d, 30d, 90d, all",
  "details": {
    "received": "invalid",
    "allowed": ["1d", "7d", "30d", "90d", "all"]
  },
  "timestamp": "2026-01-07T16:30:00Z"
}
```

---

## 8. TypeScript Types File

**Ubicacion:** `usdcop-trading-dashboard/types/contracts.ts`

> [!IMPORTANT]
> Reutilizar tipos existentes de `types/trading.ts` para evitar duplicación.

```typescript
// =============================================================================
// SHARED API CONTRACTS - DO NOT MODIFY WITHOUT BACKEND COORDINATION
// Version: 1.0.0
// =============================================================================

// -------------------- Enums --------------------
export type MarketStatus = "open" | "closed" | "pre_market";
export type SignalType = "LONG" | "SHORT" | "HOLD" | "CLOSE";
export type TradeSide = "buy" | "sell" | "hold";
export type HealthStatus = "healthy" | "warning" | "critical";

// -------------------- Signals --------------------
export interface StrategySignal {
  strategy_code: string;
  strategy_name: string;
  signal: SignalType;
  side: TradeSide;
  confidence: number;
  size: number;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
  risk_usd: number;
  reasoning: string;
  timestamp: string;
  age_seconds: number;
}

export interface LatestSignalsResponse {
  timestamp: string;
  market_price: number;
  market_status: MarketStatus;
  signals: StrategySignal[];
}

// -------------------- Performance --------------------
export interface StrategyPerformance {
  strategy_code: string;
  strategy_name: string;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  current_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  current_equity: number;
  initial_equity: number;
}

export interface PerformanceResponse {
  period: string;
  start_date: string;
  end_date: string;
  strategies: StrategyPerformance[];
}

// -------------------- Positions --------------------
export interface Position {
  position_id: number;
  strategy_code: string;
  strategy_name: string;
  side: "buy" | "sell";
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_time: string;
  holding_time_minutes: number;
}

export interface PositionsResponse {
  timestamp: string;
  total_positions: number;
  total_unrealized_pnl: number;
  positions: Position[];
}

// -------------------- Risk Status --------------------
export interface RiskLimits {
  max_drawdown_pct: number;
  max_daily_loss_pct: number;
  max_trades_per_day: number;
}

export interface RiskStatusResponse {
  kill_switch_active: boolean;
  current_drawdown_pct: number;
  daily_pnl_pct: number;
  trades_today: number;
  consecutive_losses: number;
  cooldown_active: boolean;
  cooldown_until?: string;
  limits: RiskLimits;
}

// -------------------- Model Health --------------------
export interface ModelHealth {
  model_id: string;
  action_drift_kl: number;
  stuck_behavior: boolean;
  rolling_sharpe: number;
  status: HealthStatus;
  last_inference: string;
}

export interface ModelHealthResponse {
  models: ModelHealth[];
  overall_status: HealthStatus;
}

// -------------------- WebSocket --------------------
export type WSMessageType = "signal" | "heartbeat" | "market_status";

export interface WSSignalData {
  strategy_code: string;
  signal: SignalType;
  confidence: number;
  price: number;
  reasoning: string;
}

export interface WSHeartbeatData {
  server_time: string;
  connected_clients: number;
}

export interface WSMarketStatusData {
  status: MarketStatus;
  next_open?: string;
  next_close?: string;
}

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  timestamp: string;
  data: T;
}

// -------------------- Errors --------------------
export interface APIError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}
```

---

## Tareas de Contratos

| ID | Tarea | Archivo | Owner |
|----|-------|---------|-------|
| CT-01 | Crear archivo tipos TypeScript | `types/contracts.ts` | Frontend |
| CT-02 | Alinear Pydantic con contratos | `services/multi_model_trading_api.py` | Backend |
| CT-03 | Implementar validacion de response | Ambos | Ambos |
| CT-04 | Documentar cambios en este archivo | `docs/API_CONTRACTS_SHARED.md` | Ambos |

---

## Versionamiento

Cuando se modifique un contrato:

1. Incrementar version en este documento
2. Agregar entrada en changelog abajo
3. Notificar al otro equipo
4. Actualizar tipos en ambos lados

### Changelog

| Version | Fecha | Cambios |
|---------|-------|---------|
| 1.0.0 | 2026-01-07 | Version inicial |
