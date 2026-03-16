# SDD Spec: SignalBridge OMS & Execution Layer

> **Responsibility**: Authoritative source for the SignalBridge order management system,
> exchange adapters, execution state machines, risk management, and the DAG-to-exchange bridge.
> Covers all three layers: DAG executors (stateless state machines), SignalBridge API
> (async OMS), and Dashboard execution module (operator UI).
>
> Contract: CTR-EXEC-001
> Version: 1.0.0
> Date: 2026-03-14
>
> Contracts:
> - Python (SignalBridge): `services/signalbridge_api/app/contracts/signal_bridge.py`
> - Python (DAG executors): `src/execution/smart_executor.py`, `src/execution/multiday_executor.py`
> - Python (Broker): `src/execution/broker_adapter.py`
> - TypeScript (Dashboard proxy): `usdcop-trading-dashboard/app/api/execution/`

---

## Architecture Overview

```
Layer 1: DAG Executors (Airflow)                    Layer 2: SignalBridge API (FastAPI)
┌─────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ H1-L7: SmartExecutor                │    │ SignalBridgeOrchestrator                 │
│   TrailingStopTracker               │    │   WebSocket Bridge (inference API)       │
│   PaperBroker (sync)                │    │   Redis Streams Bridge (L5 DAG)          │
│   → forecast_executions table       │    │   RiskBridgeService                      │
├─────────────────────────────────────┤    │   ExchangeAdapterFactory                 │
│ H5-L7: MultiDayExecutor            │    │     ├── MEXCAdapter (CCXT async)          │
│   TrailingStopTracker               │    │     ├── BinanceAdapter (CCXT async)       │
│   PaperBroker (sync)                │    │     └── MockAdapter (paper)               │
│   → forecast_h5_executions table    │    │   VaultService (AES-256-GCM)             │
└──────────────┬──────────────────────┘    └──────────────┬───────────────────────────┘
               │                                          │
               │ (GAP: DAGs use PaperBroker,              │
               │  not SignalBridge)                        │
               │                                          │
               ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Dashboard /execution (Next.js)                                         │
│   5 sub-pages: dashboard, exchanges, executions, settings, login                │
│   13 proxy API routes → SignalBridge backend                                    │
│   Kill switch control, risk limits, exchange management                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: DAG Executors (Stateless State Machines)

### Design Pattern

All DAG executors follow the **stateless functional pattern**: each method takes an
immutable state object in, performs an action, and returns a new state object out.
No database I/O happens inside the executor -- Airflow tasks handle persistence.
This makes executors purely testable.

```python
executor = SmartExecutor(config, broker)
state = executor.enter_position(signal_date, direction, leverage, price)
# persist state to DB
state = executor.monitor_bar(state, bar_high, bar_low, bar_close, bar_idx)
# persist again
```

### Shared Components

#### TrailingStopTracker (`src/execution/trailing_stop.py`)

Pure logic, no I/O. Monitors intraday bars and exits when profits reverse.

| State | Meaning | Transition |
|-------|---------|------------|
| `WAITING` | Price has not moved enough to activate trailing | Peak PnL >= `activation_pct` -> ACTIVE |
| `ACTIVE` | Trailing is armed, tracking peak price | Drawback from peak >= `trail_pct` -> TRIGGERED |
| `TRIGGERED` | Stop fired, position should close | Terminal |
| `EXPIRED` | Session ended without trigger | Terminal |

Direction handling:
- LONG: peak tracks `bar_high`, hard stop checks `bar_low`
- SHORT: peak tracks `bar_low` (inverted), hard stop checks `bar_high`

Hard stop is checked BEFORE trailing logic on every bar. Exit price is clamped to bar range
(LONG exit >= `bar_low`, SHORT exit <= `bar_high`).

#### BrokerAdapter (`src/execution/broker_adapter.py`)

Sync interface (NOT async). Abstract base class with one implementation.

| Method | Signature | Description |
|--------|-----------|-------------|
| `place_order` | `(side, price, quantity) -> OrderResult` | Place market order at given price |
| `close_position` | `(price) -> OrderResult` | Close current position |
| `get_position` | `() -> PositionInfo` | Return current position state |
| `cancel_all` | `() -> int` | Cancel all open orders |

**PaperBroker**: Immediate fills, configurable slippage (default 1.0 bps = MEXC assumption).
Slippage applied adversely: BUY fills at `price * (1 + bps/10000)`, SELL at `price * (1 - bps/10000)`.
Single position tracking. NOT the source of truth for position state -- that lives in DB tables.

Data classes:
- `OrderResult`: order_id, status (FILLED/REJECTED/CANCELLED), side, requested_price, fill_price, quantity, slippage_bps, timestamp
- `PositionInfo`: is_open, side, entry_price, quantity, unrealized_pnl

### SmartExecutor (`src/execution/smart_executor.py`) -- H1 Trailing Stop

Used by: `forecast_h1_l7_smart_executor.py` DAG

| Config Parameter | Default | Description |
|------------------|---------|-------------|
| `activation_pct` | 0.2% | Move to arm the trail |
| `trail_pct` | 0.3% | Drawback from peak triggers exit |
| `hard_stop_pct` | 1.5% | Adverse move = unconditional exit |
| `slippage_bps` | 1.0 | MEXC assumption |

States: `IDLE -> POSITIONED -> MONITORING -> CLOSED` (or `ERROR`)

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `enter_position` | signal_date, direction, leverage, entry_price | ExecutionState | Open via broker, status=POSITIONED |
| `monitor_bar` | state, bar_high, bar_low, bar_close, bar_idx | ExecutionState | Reconstruct tracker, feed bar, may close |
| `expire_session` | state, last_close | ExecutionState | Close at session end (12:55 COT), exit_reason=session_close |

PnL formula: `direction * leverage * (exit_price - entry_price) / entry_price`

### MultiDayExecutor (`src/execution/multiday_executor.py`) -- H5 TP/HS/Friday

Used by: `forecast_h5_l7_multiday_executor.py` DAG

| Config Parameter | Default | Description |
|------------------|---------|-------------|
| `activation_pct` | 0.2% | Tight trailing activation |
| `trail_pct` | 0.1% | Micro-profit capture (v2) |
| `hard_stop_pct` | 3.5% | Tighter hard stop (v2) |
| `cooldown_minutes` | 20 | 20 min = 4 bars at 5-min between re-entries |
| `slippage_bps` | 1.0 | MEXC assumption |

States: `PENDING -> POSITIONED -> MONITORING -> COOLDOWN -> CLOSED` (or `PAUSED`, `ERROR`)

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `enter` | signal_date, direction, leverage, price, timestamp | WeekExecutionState | Open first subtrade |
| `enter_subtrade` | state, price, timestamp | WeekExecutionState | Re-enter after cooldown (same direction) |
| `update` | state, bar_high, bar_low, bar_close, bar_ts | (state, event) | Process bar, returns event string or None |
| `close_week` | state, last_close, close_ts | WeekExecutionState | Friday 12:50 COT forced close |
| `close_circuit_breaker` | state, price, ts | WeekExecutionState | Emergency close |

Events returned by `update()`:
- `None`: No state change
- `"trailing_exit"`: Trailing stop fired, subtrade closed
- `"hard_stop"`: Hard stop fired, subtrade closed
- `"re_entry_ready"`: Cooldown expired, ready for re-entry

Subtrade tracking: Multiple entries/exits per week. Each entry/re-entry is a separate `SubtradeState`.
Week PnL is aggregated from all subtrades.

### Executor Config Files

| Executor | Config YAML | Used By |
|----------|-------------|---------|
| SmartExecutor (H1) | `config/execution/smart_executor_v1.yaml` | `forecast_h1_l7_smart_executor.py` |
| MultiDayExecutor (H5) | `config/execution/smart_simple_v1.yaml` | `forecast_h5_l7_multiday_executor.py` |

---

## Layer 2: SignalBridge API Service

### Service Identity

| Property | Value |
|----------|-------|
| Container | `usdcop-signalbridge` |
| Port | `8085:8000` |
| Framework | FastAPI + uvicorn |
| Database | PostgreSQL (asyncpg), shared `usdcop_trading` DB |
| Cache/Queue | Redis (streams, caching, Celery broker) |
| Location | `services/signalbridge_api/` |

### Trading Mode SSOT

The trading mode is the master switch for execution behavior. Set via `TRADING_MODE` env var.

| Mode | Priority | Behavior |
|------|----------|----------|
| `KILLED` | Highest | Emergency stop. All signals blocked. Kill switch active. |
| `DISABLED` | 2 | Trading disabled. All signals blocked. |
| `SHADOW` | 3 | Signal logging only. No execution. Returns simulated success. |
| `PAPER` | 4 | Simulated trading. Fills at current price, 0.1% commission. |
| `STAGING` | 5 | Pre-production with real data. Uses exchange testnet. |
| `LIVE` | 6 | Full production trading on real exchanges. |

Mode resolution order:
1. Kill switch override (if `_trading_mode_override` is set)
2. `TradingFlagsRedis` (if available): kill_switch -> KILLED, maintenance_mode -> DISABLED, paper_trading -> PAPER, else -> LIVE
3. `TRADING_MODE` env var (default: `PAPER`)

### Signal Ingestion (Two Bridges)

#### WebSocket Bridge (`app/services/websocket_bridge.py`)

Consumes predictions from the backtest/inference API via WebSocket.

| Config | Value |
|--------|-------|
| URL | `ws://usdcop-backtest-api:8000/api/v1/ws/predictions` |
| Reconnect | Exponential backoff: 1s initial, 60s max |
| Keepalive | Ping every 30s, pong timeout 10s |
| Pattern | Singleton via `WebSocketBridgeManager` |

#### Redis Streams Bridge (`app/services/redis_streams_bridge.py`)

Consumes signals from the L5 inference DAG via Redis Streams.

| Config | Value |
|--------|-------|
| Stream | `signals:ppo_primary:stream` |
| Consumer Group | `signalbridge` |
| Batch Size | 10 messages |
| Block Timeout | 5000ms |
| Acknowledgment | `XACK` after processing |

Both bridges forward signals to `SignalBridgeOrchestrator.process_signal()`.

### SignalBridgeOrchestrator (`app/services/signal_bridge_orchestrator.py`)

Central coordinator for the signal-to-execution flow.

```
Signal Received
    |
    v
Check Trading Mode ──→ KILLED/DISABLED → blocked (RiskReason.KILL_SWITCH)
    |
    v
Calculate Position Size (confidence-scaled, user limits)
    |
    v
Fetch Current Price (inference API → Redis cache → exchange fallback)
    |
    v
Risk Validation (RiskBridgeService) ──→ BLOCK → blocked result
    |                                   REDUCE → use adjusted_size
    v
Execute by Mode:
    ├── SHADOW  → log only, return simulated success
    ├── PAPER   → fill at current price, 0.1% commission, persist
    ├── STAGING → VaultService → credentials → adapter → testnet order
    └── LIVE    → VaultService → credentials → adapter → real order
    |
    v
Persist + Notify Listeners
```

Position sizing: `base_size * (0.5 + confidence * 0.5)`, capped at `max_position_size_usd`.

### Exchange Adapters (`app/adapters/`)

Strategy pattern for exchange integration. All adapters are async (CCXT).

| Adapter | File | Exchange | Library | Testnet |
|---------|------|----------|---------|---------|
| `MEXCAdapter` | `mexc.py` | MEXC | ccxt.async | Yes |
| `BinanceAdapter` | `binance.py` | Binance | ccxt.async | Yes (sandbox) |
| `MockAdapter` | `mock.py` | Mock | In-memory | N/A |

**ExchangeAdapter ABC** (`base.py`):

| Method | Signature | Description |
|--------|-----------|-------------|
| `validate_credentials` | `() -> bool` | Validate API key/secret |
| `get_balance` | `(asset?) -> List[BalanceInfo]` | Account balances |
| `get_ticker` | `(symbol) -> TickerInfo` | Current bid/ask/last/volume |
| `get_symbol_info` | `(symbol) -> SymbolInfo` | Min qty, step size, precision |
| `place_market_order` | `(symbol, side, quantity) -> OrderResult` | Market order |
| `place_limit_order` | `(symbol, side, quantity, price) -> OrderResult` | Limit order |
| `cancel_order` | `(symbol, order_id) -> OrderResult` | Cancel existing order |
| `get_order_status` | `(symbol, order_id) -> OrderResult` | Check order status |
| `get_open_orders` | `(symbol?) -> List[OrderResult]` | List open orders |

**Factory**: `ExchangeAdapterFactory.create(exchange, api_key, api_secret, passphrase?, testnet?)`

MEXC status mapping: `open -> SUBMITTED`, `closed -> FILLED`, `canceled -> CANCELLED`, `rejected -> REJECTED`

### Credential Security (VaultService)

Exchange API keys are encrypted at rest using AES-256-GCM.

| Config | Env Var | Default |
|--------|---------|---------|
| Encryption key | `VAULT_ENCRYPTION_KEY` | (must be 32 bytes) |
| Storage | PostgreSQL `exchange_credentials` table | Encrypted `api_key`, `api_secret` |

The `VaultService` decrypts credentials on demand when the orchestrator needs to place live orders.

### Risk Management (RiskBridgeService)

Risk validation runs BEFORE every execution. Returns `RiskCheckResult` with decision.

| Risk Check | Config | Default | Action |
|------------|--------|---------|--------|
| Kill switch | Global flag | false | BLOCK |
| Daily loss limit | `max_daily_loss_pct` | 2% | BLOCK |
| Trade count limit | `max_trades_per_day` | 10 | BLOCK |
| Position size | `max_position_size_usd` | $1,000 | BLOCK/REDUCE |
| Cooldown | `cooldown_minutes` | 15 min | BLOCK |
| Short permission | `enable_short` | false | BLOCK |
| Low confidence | Threshold check | -- | BLOCK |
| Exposure limit | Total open exposure | -- | BLOCK/REDUCE |
| Market closed | Trading calendar | -- | BLOCK |

Risk decisions: `ALLOW` (proceed), `BLOCK` (reject), `REDUCE` (allow but reduce size).

Per-user risk limits are stored in `user_risk_limits` table and override defaults.

### SignalBridge Contracts (`app/contracts/signal_bridge.py`)

#### Input Contracts

| Contract | Fields | Used By |
|----------|--------|---------|
| `InferenceSignalCreate` | signal_id (UUID), model_id, action (0/1/2), confidence (0-1), symbol, credential_id, timestamp, metadata | WebSocket bridge, Redis bridge |
| `ManualSignalCreate` | model_id (default "manual"), action, confidence, symbol, credential_id, quantity?, stop_loss?, take_profit?, metadata | REST API manual signals |
| `KillSwitchRequest` | activate (bool), reason, confirm (bool, required for deactivation) | Kill switch endpoint |

Action mapping: `0 = SELL`, `1 = HOLD`, `2 = BUY` (from `InferenceAction` IntEnum).

#### Output Contracts

| Contract | Fields | Used By |
|----------|--------|---------|
| `ExecutionResult` | success, execution_id, signal_id, status, exchange, symbol, side, requested/filled quantity, filled_price, commission, risk_check, error_message, processing_time_ms, metadata | All execution responses |
| `RiskCheckResult` | decision (ALLOW/BLOCK/REDUCE), reason (11 enum values), message, adjusted_size?, metadata | Risk validation output |
| `BridgeStatus` | is_active, kill_switch_active/reason, trading_mode, connected_exchanges, pending_executions, last_signal/execution_at, inference_ws_connected, uptime_seconds, stats | Status endpoint |
| `BridgeStatistics` | total_signals, executions (total/success/fail), blocked_by_risk, total_volume/pnl, avg_latency, by_exchange, by_model | Statistics endpoint |
| `BridgeHealthCheck` | status (healthy/degraded/unhealthy), database, redis, vault, inference_ws, exchanges, errors | Health endpoint |

#### WebSocket Notification Contracts

| Contract | Type Field | Purpose |
|----------|------------|---------|
| `InferencePredictionMessage` | `"prediction"` | Incoming from inference WS |
| `ExecutionNotification` | `"execution_update"` | Outgoing to dashboard WS |
| `RiskAlertNotification` | `"risk_alert"` | Risk warning to dashboard |
| `KillSwitchNotification` | `"kill_switch"` | Kill switch state change |

#### Audit Contract

| Contract | Fields | Purpose |
|----------|--------|---------|
| `ExecutionAuditEvent` | execution_id, event_type (11 types), event_data, created_at | Full execution audit trail |

Event types: `SIGNAL_RECEIVED`, `RISK_CHECK_STARTED/PASSED/FAILED`, `EXECUTION_STARTED/SUBMITTED/FILLED/FAILED/CANCELLED`, `KILL_SWITCH_ACTIVATED/DEACTIVATED`

### SignalBridge Database (Alembic Migrations)

| Migration | Table | Key Columns |
|-----------|-------|-------------|
| `001_user_risk_limits` | `user_risk_limits` | user_id, max_daily_loss_pct (2%), max_trades_per_day (10), max_position_size_usd (1000), cooldown_minutes (15), enable_short (false) |
| `002_execution_audit` | `executions` | symbol, side, order_type, qty, price, exchange, credential_id, exchange_order_id, status, filled_qty, avg_price, commission, SL, TP, error, raw_response (JSONB), metadata (JSONB) |
| `003_signal_execution_link` | `signals` | signal_id (UUID), model_id, action, confidence, symbol, timestamp, metadata (JSONB) |

Schema is managed via SQL init-scripts (`init-scripts/20-signalbridge-schema.sql`), NOT Alembic auto-create.

### SignalBridge API Routes

#### Signal Bridge Routes (`app/api/routes/signal_bridge.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/signal-bridge/status` | Current bridge status |
| GET | `/api/signal-bridge/health` | Component health check |
| POST | `/api/signal-bridge/process` | Process a signal |
| POST | `/api/signal-bridge/kill-switch` | Activate/deactivate kill switch |
| GET | `/api/signal-bridge/history` | Execution history (filtered) |
| GET | `/api/signal-bridge/statistics` | Bridge statistics |
| GET | `/api/signal-bridge/user/{id}/state` | User trading state |
| PUT | `/api/signal-bridge/user/{id}/limits` | Update user risk limits |

#### Exchange Routes (`app/api/routes/exchanges.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/exchanges` | List supported exchanges |
| POST | `/api/exchanges/{exchange}/connect` | Connect exchange credentials |
| GET | `/api/exchanges/{exchange}/balance` | Get exchange balance |
| GET | `/api/exchanges/balances` | Get all exchange balances |
| GET | `/api/exchanges/credentials` | List saved credentials |

#### Execution Routes (`app/api/routes/executions.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/executions` | List executions (filtered) |
| POST | `/api/executions` | Create manual execution |
| GET | `/api/executions/{id}` | Get execution details |
| PUT | `/api/executions/{id}` | Update execution |

#### Auth Routes (`app/api/routes/auth.py`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | JWT login (HS256, 30min access, 7d refresh) |
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/refresh` | Refresh access token |
| POST | `/api/auth/logout` | Invalidate session |

#### Trading Routes (`app/api/routes/trading.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/trading/positions` | Get open positions |
| GET | `/api/trading/performance` | Trading performance metrics |
| POST | `/api/trading/close-all` | Close all positions |

### SignalBridge Config (`app/core/config.py`)

| Category | Setting | Default |
|----------|---------|---------|
| Database | `database_url` | `postgresql+asyncpg://postgres:postgres@localhost:5432/signalbridge` |
| Database | `database_pool_size` / `max_overflow` | 10 / 20 |
| Redis | `redis_url` | `redis://localhost:6379/0` |
| Celery | `celery_broker_url` / `result_backend` | Redis DB 1 / DB 2 |
| JWT | `jwt_algorithm` / `access_token_expire_minutes` / `refresh_token_expire_days` | HS256 / 30 / 7 |
| Vault | `vault_encryption_key` | (32-byte AES key) |
| CORS | `cors_origins` | `localhost:5173`, `localhost:3000` |
| Trading | `trading_mode` | `PAPER` |
| Trading | `default_position_size_usd` | $100 |
| Trading | `position_size_pct` | 2% per trade |
| Trading | `max_position_size_usd` | $1,000 |
| Risk | `default_max_daily_loss_pct` | 2% |
| Risk | `default_max_trades_per_day` | 10 |
| Risk | `default_cooldown_minutes` | 15 |
| WebSocket | `inference_ws_url` | `ws://usdcop-backtest-api:8000/api/v1/ws/predictions` |
| Rate Limit | `rate_limit_per_minute` | 60 (disabled in dev) |

---

## Layer 3: Dashboard Execution Module

### Sub-Pages (`app/execution/`)

| Page | Route | Purpose |
|------|-------|---------|
| Root | `/execution` | Redirects to `/execution/dashboard` |
| Dashboard | `/execution/dashboard` | Kill switch control, status cards, health, statistics, auto-refresh 10s |
| Exchanges | `/execution/exchanges` | List exchanges, connect credentials, test connection, view balances |
| Executions | `/execution/executions` | Trade log with filters (exchange, symbol, status, date range) |
| Settings | `/execution/settings` | Risk limits, position sizing, trading mode, webhook configuration |
| Login | `/execution/login` | JWT authentication for SignalBridge API |

### Dashboard Proxy API Routes (13 routes)

All routes proxy to the SignalBridge backend at `SIGNALBRIDGE_BACKEND_URL` (default: `http://usdcop-signalbridge:8000`).

| Dashboard Route | Method | Proxies To |
|----------------|--------|------------|
| `/api/execution/signal-bridge/status` | GET | `/api/signal-bridge/status` |
| `/api/execution/signal-bridge/kill-switch` | POST | `/api/signal-bridge/kill-switch` |
| `/api/execution/signal-bridge/statistics` | GET | `/api/signal-bridge/statistics` |
| `/api/execution/signal-bridge/history` | GET | `/api/signal-bridge/history` |
| `/api/execution/signal-bridge/user/[userId]/limits` | GET/PUT | `/api/signal-bridge/user/{id}/limits` |
| `/api/execution/exchanges` | GET | `/api/exchanges` |
| `/api/execution/exchanges/[exchange]/connect` | POST | `/api/exchanges/{exchange}/connect` |
| `/api/execution/exchanges/[exchange]/balance` | GET | `/api/exchanges/{exchange}/balance` |
| `/api/execution/exchanges/balances` | GET | `/api/exchanges/balances` |
| `/api/execution/exchanges/credentials` | GET | `/api/exchanges/credentials` |
| `/api/execution/executions` | GET/POST | `/api/executions` |
| `/api/execution/auth/login` | POST | `/api/auth/login` |
| `/api/execution/auth/register` | POST | `/api/auth/register` |

---

## Docker Service Definition

```yaml
signalbridge-api:
  build:
    context: ./services/signalbridge_api
    dockerfile: Dockerfile
  container_name: usdcop-signalbridge
  ports:
    - "8085:8000"
  depends_on:
    - postgres
    - redis
  environment:
    DATABASE_URL: postgresql+asyncpg://admin:admin123@postgres:5432/usdcop_trading
    REDIS_URL: redis://redis:6379/0
    TRADING_MODE: PAPER
    APP_ENV: development
    DEBUG: true
    SIGNALBRIDGE_DEV_MODE: true
  networks:
    - usdcop-trading-network
```

Dashboard connects to SignalBridge via internal Docker network:
```
SIGNALBRIDGE_BACKEND_URL: http://usdcop-signalbridge:8000
```

---

## Code File Inventory

### SignalBridge API (`services/signalbridge_api/`)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `app/main.py` | 1 | FastAPI app, lifespan, WS + Redis bridge startup |
| `app/core/` | config.py, database.py, security.py, exceptions.py | Settings, DB engine, JWT, custom errors |
| `app/contracts/` | signal_bridge.py, signal.py, exchange.py, execution.py, auth.py, common.py, user.py, trading.py | Pydantic models for all API boundaries |
| `app/adapters/` | base.py, mexc.py, binance.py, mock.py, factory.py | Exchange adapter implementations (CCXT async) |
| `app/services/` | signal_bridge_orchestrator.py, websocket_bridge.py, redis_streams_bridge.py, risk_bridge.py, vault.py, exchange.py, execution.py, trading.py, user.py, signal.py, api_key_validator.py | Business logic services |
| `app/api/routes/` | signal_bridge.py, exchanges.py, executions.py, auth.py, trading.py, users.py, signals.py, webhooks.py, ws_notifications.py | API route handlers |
| `app/models/` | base.py, execution.py, exchange.py, signal.py, trading.py, user.py, audit.py | SQLAlchemy ORM models |
| `app/middleware/` | auth.py, rate_limit.py | JWT auth middleware, rate limiting |
| `app/tasks/` | celery_app.py, execution_tasks.py, signal_tasks.py | Celery async tasks |
| `alembic/versions/` | 001, 002, 003 | DB migrations (user_risk_limits, execution_audit, signal_execution_link) |

### DAG Executors (`src/execution/`)

| File | Class | Purpose |
|------|-------|---------|
| `trailing_stop.py` | `TrailingStopTracker`, `TrailingStopConfig`, `TrailingState` | Pure trailing stop logic |
| `broker_adapter.py` | `BrokerAdapter` (ABC), `PaperBroker`, `OrderResult`, `PositionInfo` | Sync broker interface |
| `smart_executor.py` | `SmartExecutor`, `SmartExecutorConfig`, `ExecutionState`, `ExecutionStatus` | H1 intraday trailing stop state machine |
| `multiday_executor.py` | `MultiDayExecutor`, `MultiDayConfig`, `WeekExecutionState`, `SubtradeState`, `WeekStatus` | H5 weekly TP/HS/re-entry state machine |

### Dashboard Execution (`usdcop-trading-dashboard/`)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `app/execution/` | page.tsx, layout.tsx, + 5 sub-pages | Execution module UI |
| `app/api/execution/` | 13 proxy route files | Next.js API routes proxying to SignalBridge |

---

## The Gap: DAG-to-SignalBridge Bridge

### Current State

H1-L7 (`forecast_h1_l7_smart_executor.py`) and H5-L7 (`forecast_h5_l7_multiday_executor.py`)
DAGs use `PaperBroker` directly for paper trading. They **never** call SignalBridge.
SignalBridge consumes signals from the RL inference pipeline (WebSocket + Redis Streams),
which is a different track entirely.

```
Current:
  H1-L7 DAG ──→ SmartExecutor(PaperBroker) ──→ forecast_executions table
  H5-L7 DAG ──→ MultiDayExecutor(PaperBroker) ──→ forecast_h5_executions table
  RL L5 DAG ──→ Redis Stream ──→ SignalBridge ──→ exchange adapter

Needed:
  H1-L7 DAG ──→ SmartExecutor(BrokerAdapter) ──→ {PaperBroker | SignalBridgeClient}
  H5-L7 DAG ──→ MultiDayExecutor(BrokerAdapter) ──→ {PaperBroker | SignalBridgeClient}
```

### Bridge Design (TODO)

To enable live execution from forecasting DAGs, the following is needed:

1. **`EXECUTION_MODE` env var** in Airflow: `paper` (default), `testnet`, `live`
2. **`signalbridge_client.py`** in `airflow/dags/utils/`: sync HTTP client that wraps `POST /api/executions` and implements the `BrokerAdapter` ABC
3. **Broker selection** in DAG entry point: `if mode == "paper": PaperBroker() else: SignalBridgeClient()`
4. **Credential ID mapping**: MEXC credential UUID stored in Airflow Variable or env var

The `BrokerAdapter` ABC already defines the interface. A `SignalBridgeClient` that implements
`place_order()` and `close_position()` by calling SignalBridge HTTP endpoints would complete
the bridge without changing any executor logic.

---

## Health Check Architecture

### SignalBridge Health (`GET /health`)

```json
{
  "status": "healthy|degraded|unhealthy",
  "app": "SignalBridge",
  "env": "development",
  "bridges": {
    "websocket": {"connected": true, "url": "ws://..."},
    "redis_streams": {"connected": true, "stream": "signals:ppo_primary:stream"}
  },
  "trading_mode": "PAPER"
}
```

Overall status is `"healthy"` if at least one bridge is connected, `"degraded"` if neither is.

### Component Health (`GET /api/signal-bridge/health`)

```json
{
  "status": "healthy|degraded|unhealthy",
  "database": true,
  "redis": true,
  "vault": true,
  "inference_ws": false,
  "exchanges": {"mexc": true, "binance": false},
  "errors": []
}
```

---

## Kill Switch Protocol

The kill switch is the highest-priority safety mechanism. When active, ALL signals are
blocked regardless of mode, risk status, or user permissions.

### Activation

```
POST /api/signal-bridge/kill-switch
{
  "activate": true,
  "reason": "Abnormal volatility detected",
  "confirm": false  (not needed for activation)
}
```

1. Sets `_trading_mode_override = TradingMode.KILLED`
2. If `TradingFlagsRedis` available: sets Redis kill_switch flag
3. Logs at CRITICAL level
4. Creates audit event (`KILL_SWITCH_ACTIVATED`)
5. Notifies all WebSocket listeners

### Deactivation

```
POST /api/signal-bridge/kill-switch
{
  "activate": false,
  "reason": "Market conditions normalized",
  "confirm": true  (REQUIRED for deactivation)
}
```

`confirm: true` is required to prevent accidental deactivation.

---

## Related Specs

| Spec | Relationship |
|------|-------------|
| `sdd-strategy-spec.md` | UniversalSignalRecord, execution strategies (WeeklyTPHSExecution, DailyTrailingStopExecution) |
| `sdd-pipeline-lifecycle.md` | H1-L7 and H5-L7 DAG schedules, monitoring stage |
| `h5-smart-simple-pipeline.md` | H5 executor architecture, TP/HS config, subtrade tracking |
| `sdd-approval-spec.md` | Strategy approval before production deploy |
| `l1-l5-inference-pipeline.md` | RL L5 signal generation (SignalBridge consumer) |
| `sdd-dashboard-integration.md` | Dashboard `/execution` page inventory |

---

## DO NOT

### SignalBridge
- Do NOT deploy with `TRADING_MODE=LIVE` without verifying exchange credentials and risk limits
- Do NOT deactivate the kill switch without `confirm: true` -- it is a safety gate
- Do NOT store exchange API keys in plaintext -- always use VaultService (AES-256-GCM)
- Do NOT bypass risk validation -- every signal must pass `RiskBridgeService` before execution
- Do NOT hard-code exchange credentials in config or env files -- use the credentials table
- Do NOT run with `rate_limit_per_minute` disabled in production (only disabled in dev)

### DAG Executors
- Do NOT perform database I/O inside executor methods -- Airflow tasks handle persistence
- Do NOT skip hard stop checks -- they run BEFORE trailing logic on every bar
- Do NOT modify executor state in place without returning it -- the pattern is functional (state in -> state out)
- Do NOT use async broker adapters in DAG executors -- they use synchronous `BrokerAdapter`
- Do NOT bypass cooldown in MultiDayExecutor -- 20-minute cooldown prevents churn in lateral markets

### Bridge Gap
- Do NOT call SignalBridge from forecasting DAGs until `signalbridge_client.py` is implemented
- Do NOT switch `EXECUTION_MODE` to `live` without the bridge client in place
- Do NOT modify `BrokerAdapter` ABC to add async methods -- create a separate `AsyncBrokerAdapter` if needed
- Do NOT route forecasting signals through Redis Streams -- use HTTP POST to `/api/executions` for the DAG bridge
