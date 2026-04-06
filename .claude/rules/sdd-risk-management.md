# SDD Spec: Risk Management System

> **Responsibility**: Authoritative source for the USDCOP trading system's risk management
> architecture, including pre-trade validation, kill switch, circuit breaker, cooldown,
> position sizing enforcement, and audit logging.
>
> Three complementary subsystems enforce risk at different layers:
> - **Chain of Responsibility** (`src/risk/checks/`): 9 modular pre-signal checks
> - **Command Pattern** (`src/risk/commands.py`): Risk operations with undo/redo + audit
> - **Enforcement Layer** (`src/trading/risk_enforcer.py`): 7 pluggable rules at execution time
>
> **v2.0 additions (2026-04-06):**
> - **Regime Gate** (`src/forecasting/regime_gate.py`): Hurst-based market regime classifier. Blocks trading in mean-reverting markets. Deployed in L5 vol-targeting DAG.
> - **Dynamic Leverage** (`src/forecasting/dynamic_leverage.py`): Scales leverage [0.25, 1.0] based on rolling WR and drawdown. Integrated in L5 DAG.
> - **Effective HS** (`smart_simple_v1.yaml`): `min(HS_base, 3.5%/leverage)` caps portfolio-level loss per trade.
> - DAG utilities: `airflow/dags/utils/regime_gate_live.py`, `dynamic_leverage_live.py`
>
> Interfaces: `src/core/interfaces/risk.py`
> Constants: `src/core/constants.py`
>
> Contract: CTR-RISK-001
> Version: 2.0.0
> Date: 2026-04-06

---

## Architecture Overview

```
Trading Signal (from L5 / H1-L5 / H5-L5)
    |
    v
┌────────────────────────────────────────────────┐
│  Layer 1: Chain of Responsibility              │
│  (src/risk/checks/risk_check_chain.py)         │
│                                                │
│  9 checks in order → first failure stops chain │
│  Input: RiskContext (signal, confidence, stats) │
│  Output: RiskCheckResult (approved/rejected)   │
└───────────────────┬────────────────────────────┘
                    │ APPROVED
                    v
┌────────────────────────────────────────────────┐
│  Layer 2: Risk Enforcer                        │
│  (src/trading/risk_enforcer.py)                │
│                                                │
│  7 pluggable rules → ALLOW / BLOCK / REDUCE   │
│  Input: signal, size, price, confidence        │
│  Output: RiskCheckResult (decision + adj size) │
└───────────────────┬────────────────────────────┘
                    │ ALLOW (or REDUCE)
                    v
┌────────────────────────────────────────────────┐
│  Layer 3: Risk Manager (legacy)                │
│  (src/risk/risk_manager.py)                    │
│                                                │
│  validate_signal() + record_trade_result()     │
│  Kill switch + daily loss + cooldown tracking  │
└───────────────────┬────────────────────────────┘
                    │ ALLOWED
                    v
              Trade Execution

Orthogonal: Command Pattern (src/risk/commands.py)
    → Encapsulates risk operations (trigger CB, set cooldown, reset kill switch)
    → Full undo/redo support for manual interventions
    → Audit trail via CommandInvoker history
```

### Relationship to H5/H1 Guardrails

Risk checks and pipeline guardrails operate at different granularities:

| Concern | Layer | Scope | Files |
|---------|-------|-------|-------|
| **Risk checks** | Tactical | Per-signal, per-bar | `src/risk/`, `src/trading/risk_enforcer.py` |
| **H5 guardrails** | Strategic | Per-week, rolling window | `config/execution/smart_simple_v1.yaml` |
| **H1 guardrails** | Strategic | Per-day, rolling window | `config/execution/smart_executor_v1.yaml` |

**Guardrails** ask: "Should we keep trading this direction?" (e.g., circuit breaker after 5 consecutive weekly losses, long insistence alarm after 60% LONGs in 8 weeks).

**Risk checks** ask: "Should we execute THIS specific trade right now?" (e.g., drawdown kill switch, daily loss limit, confidence threshold).

---

## Interfaces (`src/core/interfaces/risk.py`)

### Core Data Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `RiskStatus` | Enum of check outcomes | APPROVED, HOLD_SIGNAL, DAILY_LOSS_LIMIT, MAX_DRAWDOWN, CONSECUTIVE_LOSSES, LOW_CONFIDENCE, MAX_TRADES_REACHED, OUTSIDE_TRADING_HOURS, COOLDOWN_ACTIVE, CIRCUIT_BREAKER_ACTIVE, SYSTEM_ERROR |
| `RiskContext` | Input to the check chain | signal, confidence, daily_pnl_percent, current_drawdown, consecutive_losses, trades_today, enforce_trading_hours, extra |
| `RiskCheckResult` | Output from a check | approved (bool), status (RiskStatus), message, metadata |
| `DailyStats` | Daily trading statistics | date, pnl, drawdown, trades_count, consecutive_losses, circuit_breaker_triggered |
| `FullRiskCheckResult` | Complete check result | approved, status, original_signal, adjusted_signal, daily_stats, checks_passed, check_that_failed |

### Abstract Interfaces

| Interface | Purpose | Key Methods |
|-----------|---------|-------------|
| `IRiskCheck` | Individual check in chain | `check(context) -> RiskCheckResult`, `name`, `order` |
| `ITradingHoursChecker` | Trading hours validation | `is_trading_hours() -> (bool, str)`, `timezone` |
| `ICircuitBreaker` | Circuit breaker state | `is_active() -> (bool, reason)`, `trigger(reason)`, `reset()` |
| `ICooldownManager` | Cooldown management | `is_active() -> (bool, seconds)`, `set_cooldown(seconds, reason)`, `clear()` |
| `IPositionSizer` | Position sizing | `calculate_size(signal, confidence, context) -> float`, `max_position_size` |
| `IRiskManager` | Facade for risk ops | `check_signal(signal, confidence)`, `update_trade_result(pnl)`, `get_daily_stats()` |

---

## Layer 1: Chain of Responsibility (`src/risk/checks/`)

### Orchestrator: `RiskCheckChain`

**File**: `src/risk/checks/risk_check_chain.py`

The chain executes checks in order by their `order` property. It stops on the first failure
unless the check allows continuation (e.g., HOLD_SIGNAL short-circuits with approval).

```python
# Create chain with default checks
chain = RiskCheckChain.with_defaults(config)

# Run chain
context = RiskContext(signal="BUY", confidence=0.8, daily_pnl_percent=-1.5, ...)
result = chain.run(context)

if result.approved:
    execute_trade()
else:
    logger.warning(f"Blocked by {result.metadata['check_that_failed']}: {result.message}")
```

### 9 Risk Checks (Execution Order)

| Order | Check | File | Threshold | Action on Fail |
|-------|-------|------|-----------|----------------|
| 0 | HOLD Signal | `hold_signal_check.py` | Signal is HOLD | Short-circuit (approved, no trade) |
| 10 | Trading Hours | `trading_hours_check.py` | 8:00-16:00 COT Mon-Fri | Block or skip |
| 20 | Circuit Breaker | `circuit_breaker_check.py` | CB active | Block all trading |
| 30 | Cooldown | `cooldown_check.py` | Cooldown not expired | Block until expiry |
| 40 | Confidence | `confidence_check.py` | min_confidence = 0.6 | Reject low confidence |
| 50 | Daily Loss | `daily_loss_check.py` | -2% daily loss | Trigger circuit breaker |
| 55 | Drawdown | `drawdown_check.py` | -1% max drawdown | Trigger kill switch |
| 60 | Consecutive Losses | `consecutive_losses_check.py` | 3+ losses | Activate cooldown (300s) |
| 70 | Max Trades | `max_trades_check.py` | 10/day | Block further trades |

### Chain Factory

`RiskCheckChain.with_defaults(config)` accepts a config dict and optional dependency injections:

```python
chain = RiskCheckChain.with_defaults(
    config={
        "start_time": "08:00",
        "end_time": "16:00",
        "timezone": "America/Bogota",
        "min_confidence": 0.6,
        "max_daily_loss": -0.02,
        "max_drawdown": -0.01,
        "max_consecutive_losses": 3,
        "cooldown_after_loss": 300,
        "max_trades_per_day": 10,
    },
    circuit_breaker=my_circuit_breaker,       # ICircuitBreaker
    cooldown_manager=my_cooldown_manager,     # ICooldownManager
    trigger_circuit_breaker_fn=my_cb_trigger, # Callable
    set_cooldown_fn=my_cooldown_setter,       # Callable
)
```

### Chain Behavior

- Checks execute in ascending `order` value
- First check that returns `approved=False` stops the chain
- Exception in any check returns `SYSTEM_ERROR` (fail-safe)
- HOLD_SIGNAL short-circuits: approved=True but trade is not executed (no-op)
- Result metadata includes `checks_passed` list and `check_that_failed` name

### Adding Custom Checks

The chain is open for extension without modifying existing checks:

```python
chain = RiskCheckChain.with_defaults(config)
chain.add_check(MyCustomCheck())      # Inserted by order value
chain.remove_check("MaxTrades")       # Remove by name
```

---

## Layer 2: Risk Enforcer (`src/trading/risk_enforcer.py`)

### Main Interface

```python
enforcer = RiskEnforcer(limits=RiskLimits(...))

# Pre-trade check
result = enforcer.check_signal(
    signal="SHORT",
    size=100,
    price=4250.50,
    confidence=0.9
)

if result.is_allowed:
    final_size = result.adjusted_size or size  # May be reduced
    execute_trade(signal, final_size, price)

if result.is_blocked:
    logger.warning(f"Blocked: {result.reason.value} - {result.message}")

# Post-trade recording
enforcer.record_trade(pnl_pct=-0.5, signal="SHORT")
```

### 7 Pluggable Rules (Evaluation Order)

| # | Rule | Class | Decision on Fail | Description |
|---|------|-------|-------------------|-------------|
| 1 | Kill Switch | `KillSwitchRule` | BLOCK | Kill switch active OR drawdown >= limit |
| 2 | Daily Loss | `DailyLossRule` | BLOCK | Daily P&L <= negative limit |
| 3 | Trade Limit | `TradeLimitRule` | BLOCK | Trade count >= max per day |
| 4 | Cooldown | `CooldownRule` | BLOCK | Cooldown period active |
| 5 | Short | `ShortRule` | BLOCK | SHORT signal when shorts disabled |
| 6 | Position Size | `PositionSizeRule` | REDUCE | Size > max allowed (caps, does not block) |
| 7 | Confidence | `ConfidenceRule` | BLOCK | Confidence below threshold |

### Three-Way Decision

Unlike the chain (binary approve/reject), the enforcer supports three decisions:

| Decision | Meaning | Trade Outcome |
|----------|---------|---------------|
| `ALLOW` | All rules passed | Execute at requested size |
| `REDUCE` | Position size exceeded limit | Execute at reduced size (`adjusted_size`) |
| `BLOCK` | Critical rule failed | Do not execute |

Exit signals (`CLOSE`, `FLAT`, `HOLD`) always return `ALLOW` without evaluating rules.

### RiskLimits (Enforcer Configuration)

| Parameter | Default | Source Constant | Description |
|-----------|---------|----------------|-------------|
| `max_drawdown_pct` | 15.0% | `MAX_DRAWDOWN_PCT * 100` | Kill switch trigger |
| `max_daily_loss_pct` | 5.0% | `MAX_DAILY_LOSS_PCT * 100` | Daily loss block |
| `max_trades_per_day` | 20 | `MAX_DAILY_TRADES` | Trade count limit |
| `max_position_size` | 1.0 | `MAX_POSITION_SIZE` | Max position units |
| `max_position_pct` | 20.0% | Hardcoded | Max position as % of portfolio |
| `min_confidence` | 0.6 | `MIN_CONFIDENCE_THRESHOLD` | Minimum model confidence |
| `cooldown_after_losses` | 5 | `CONSECUTIVE_LOSS_LIMIT` | Consecutive losses before cooldown |
| `cooldown_minutes` | 60 | `COOLDOWN_MINUTES` | Cooldown duration |
| `enable_short` | true | Hardcoded | Whether SHORT signals allowed |
| `max_exposure_pct` | 100.0% | Hardcoded | Maximum total exposure |

### RiskState (Enforcer Runtime State)

| Field | Type | Reset | Description |
|-------|------|-------|-------------|
| `kill_switch_active` | bool | Manual only (`confirm=True`) | Halts ALL trading |
| `daily_blocked` | bool | Daily reset | Blocks trading for remainder of day |
| `cooldown_until` | datetime? | Daily reset or expiry | Blocks until timestamp |
| `trade_count_today` | int | Daily reset | Trades executed today |
| `daily_pnl_pct` | float | Daily reset | Cumulative daily P&L |
| `consecutive_losses` | int | Daily reset or first win | Current losing streak |
| `current_drawdown_pct` | float | Manual update | Portfolio drawdown |
| `total_exposure` | float | Manual update | Total position exposure |
| `current_day` | date | Auto-detected | For daily reset trigger |

### Daily Reset

The enforcer detects day changes automatically before each `check_signal()` call.
On day change, it resets: `trade_count_today`, `daily_pnl_pct`, `daily_blocked`,
`consecutive_losses`, `cooldown_until`. The kill switch is **NOT** reset (requires
manual `reset_kill_switch(confirm=True)`).

### Rule Management

```python
enforcer.add_rule(MyCustomRule())         # Add custom rule
enforcer.remove_rule("PositionSize")      # Remove by name
enforcer.get_rules()                      # List active rule names
enforcer.is_trading_allowed               # Quick property check
```

---

## Layer 3: Risk Manager (Legacy) (`src/risk/risk_manager.py`)

The original risk manager predates the chain and enforcer. It provides a simpler
interface used by the RL pipeline.

### RiskLimits (Manager Configuration)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_drawdown_pct` | 15.0 | Kill switch trigger (percentage) |
| `max_daily_loss_pct` | 5.0 | Daily loss block (percentage) |
| `max_trades_per_day` | 20 | Trade count limit |
| `cooldown_after_losses` | 5 | Consecutive losses before cooldown |
| `cooldown_minutes` | 60 | Cooldown duration (minutes) |

### Interface

```python
from src.risk import RiskManager, RiskLimits

rm = RiskManager(RiskLimits(max_drawdown_pct=15.0))

# Pre-trade validation
allowed, reason = rm.validate_signal("long", current_drawdown_pct=8.5)

# Post-trade recording
rm.record_trade_result(pnl_pct=-0.3, signal="long")

# Status monitoring
status = rm.get_status()  # Dict with all state + limits

# Manual resets
rm.reset_daily()                      # Reset daily counters (NOT kill switch)
rm.reset_kill_switch(confirm=True)    # Manual kill switch reset (requires confirm)
```

### Validation Order

1. **Exit signals** (`close`, `flat`) always allowed
2. **Kill switch** check (active? drawdown >= limit?)
3. **Daily loss** block check
4. **Trade count** limit check
5. **Cooldown** check (active? expired?)

---

## Command Pattern (`src/risk/commands.py`)

### 6 Commands

All commands implement `execute() -> CommandResult` and `undo() -> CommandResult`.

| Command | Purpose | Undo Action |
|---------|---------|-------------|
| `TriggerCircuitBreakerCommand` | Activate cooldown on risk manager | Restore previous cooldown state |
| `SetCooldownCommand` | Set specific cooldown duration | Restore previous cooldown state |
| `ClearCooldownCommand` | Remove active cooldown immediately | Re-activate previous cooldown |
| `ResetKillSwitchCommand` | Reset kill switch (requires `confirmed=True`) | Re-activate kill switch |
| `UpdateRiskLimitsCommand` | Modify risk limits (partial updates) | Restore previous limit values |
| `BlockTradingCommand` | Block trading for the day | Restore previous daily_blocked state |

### CommandInvoker

The invoker manages command execution, history, undo/redo, and batch operations.

```python
from src.risk.commands import (
    CommandInvoker, TriggerCircuitBreakerCommand,
    ResetKillSwitchCommand, UpdateRiskLimitsCommand
)

invoker = CommandInvoker(max_history=100)

# Execute a command
result = invoker.execute(
    TriggerCircuitBreakerCommand(rm, reason="5 consecutive losses", cooldown_minutes=30)
)

# Undo the last command
if invoker.can_undo():
    undo_result = invoker.undo()

# Redo
if invoker.can_redo():
    redo_result = invoker.redo()

# Batch execution (stops on first failure by default)
results = invoker.execute_batch([
    UpdateRiskLimitsCommand(rm, {"max_drawdown_pct": 10.0}),
    BlockTradingCommand(rm, reason="manual review"),
], stop_on_failure=True)

# Audit trail
history = invoker.get_history(limit=10)  # Returns list of dicts
```

### CommandResult

```python
@dataclass
class CommandResult:
    success: bool
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
```

### Safety: Kill Switch Reset

`ResetKillSwitchCommand` requires explicit `confirmed=True`. Without it, the command
returns `success=False` with message "Kill switch reset requires explicit confirmation".
This prevents accidental resets that could expose the portfolio to further losses.

---

## Database Tables

### Migration 014: Kill Switch Audit (`audit.kill_switch_audit`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `previous_state` | BOOLEAN | State before change |
| `new_state` | BOOLEAN | State after change |
| `action` | VARCHAR(20) | `ACTIVATED`, `DEACTIVATED`, `OVERRIDE` |
| `reason` | TEXT | Why the change was made |
| `severity` | VARCHAR(20) | `LOW`, `NORMAL`, `HIGH`, `CRITICAL` |
| `triggered_by` | VARCHAR(100) | Username, service, or `SYSTEM` |
| `trigger_source` | VARCHAR(50) | `API`, `DAG`, `MANUAL`, `AUTOMATED`, `ALERT` |
| `correlation_id` | UUID | Links to related request/event |
| `active_trades_count` | INTEGER | Open trades at time of change |
| `pending_orders_count` | INTEGER | Pending orders at time of change |
| `current_drawdown` | DECIMAL | Portfolio drawdown at time of change |
| `current_volatility` | DECIMAL | Market volatility at time of change |
| `market_status` | VARCHAR(20) | `OPEN`, `CLOSED`, `PRE_MARKET` |
| `alert_id` | VARCHAR(100) | Related alert reference |
| `incident_id` | VARCHAR(100) | Related incident reference |
| `environment` | VARCHAR(20) | `production` (default) |
| `hostname` | VARCHAR(100) | Server hostname |
| `metadata` | JSONB | Additional context |
| `created_at` | TIMESTAMPTZ | Timestamp of change |

### Helper Functions (Migration 014)

| Function | Purpose | Returns |
|----------|---------|---------|
| `audit.log_kill_switch_change(...)` | Log a kill switch state change with full context | Audit record ID |
| `audit.get_kill_switch_status()` | Get current status + total activations + duration | Table (is_active, last_change_at, ...) |

### Views (Migration 014)

| View | Purpose |
|------|---------|
| `audit.kill_switch_duration` | Shows each activation period with duration in minutes and `currently_active` flag |

### Alert Integration (`audit.kill_switch_alerts`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `kill_switch_audit_id` | INTEGER FK | References `kill_switch_audit(id)` |
| `alert_channel` | VARCHAR(50) | `SLACK`, `PAGERDUTY`, `EMAIL`, `SMS` |
| `alert_sent_at` | TIMESTAMPTZ | When alert was sent |
| `alert_status` | VARCHAR(20) | `SENT`, `FAILED`, `ACKNOWLEDGED` |
| `alert_response` | JSONB | Response from alert channel |
| `acknowledged_by` | VARCHAR(100) | Who acknowledged |
| `acknowledged_at` | TIMESTAMPTZ | When acknowledged |

### Migration 033: Circuit Breaker State (`circuit_breaker_state`)

| Column | Type | Description |
|--------|------|-------------|
| `circuit_id` | VARCHAR(100) PK | Circuit breaker identifier |
| `state` | VARCHAR(20) | `CLOSED`, `OPEN`, `HALF_OPEN` |
| `failure_count` | INTEGER | Consecutive failures |
| `success_count` | INTEGER | Consecutive successes (in HALF_OPEN) |
| `last_failure_at` | TIMESTAMPTZ | Last failure timestamp |
| `last_success_at` | TIMESTAMPTZ | Last success timestamp |
| `opened_at` | TIMESTAMPTZ | When circuit opened |
| `half_open_at` | TIMESTAMPTZ | When circuit went half-open |
| `reset_timeout_seconds` | INTEGER | Time before OPEN -> HALF_OPEN (default: 300) |
| `failure_threshold` | INTEGER | Failures before CLOSED -> OPEN (default: 3) |

### Related Tables (Migration 033)

| Table | Purpose |
|-------|---------|
| `event_dead_letter_queue` | Failed events for retry (DLQ pattern) |
| `event_processed_log` | Idempotency guarantee for event processing |
| `v_event_system_health` | View: DLQ pending/dead counts, processed 24h, open circuit breakers |

---

## Constants (`src/core/constants.py`)

| Constant | Value | Used By |
|----------|-------|---------|
| `MAX_DRAWDOWN_PCT` | 0.15 (15%) | Kill switch trigger |
| `MAX_DAILY_LOSS_PCT` | 0.05 (5%) | Daily loss block |
| `MAX_DAILY_TRADES` | 20 | Trade count limit |
| `MAX_POSITION_SIZE` | 1.0 | Position size cap |
| `MIN_CONFIDENCE_THRESHOLD` | 0.6 | Minimum model confidence |
| `CONSECUTIVE_LOSS_LIMIT` | 5 | Losses before cooldown |
| `COOLDOWN_MINUTES` | 60 | Cooldown duration |

These constants are the source of truth for default values. Both `RiskManager.RiskLimits`
and `RiskEnforcer.RiskLimits` derive defaults from these constants.

---

## File Inventory

### Core Risk Files

| File | Class(es) | Pattern |
|------|-----------|---------|
| `src/core/interfaces/risk.py` | `IRiskCheck`, `ITradingHoursChecker`, `ICircuitBreaker`, `ICooldownManager`, `IPositionSizer`, `IRiskManager`, `RiskStatus`, `RiskContext`, `RiskCheckResult`, `DailyStats`, `FullRiskCheckResult` | Interface definitions |
| `src/core/constants.py` | Risk limit constants | Default values |
| `src/risk/risk_manager.py` | `RiskManager`, `RiskLimits`, `TradeRecord` | Legacy manager |
| `src/risk/commands.py` | `Command`, `CommandInvoker`, 6 command classes, `CommandResult` | Command pattern |
| `src/risk/checks/risk_check_chain.py` | `RiskCheckChain` | Chain orchestrator |
| `src/trading/risk_enforcer.py` | `RiskEnforcer`, `RiskLimits`, `RiskState`, `RiskDecision`, `RiskReason`, `RiskCheckResult`, `IRiskRule`, 7 rule classes | Enforcement layer |

### Individual Check Files (`src/risk/checks/`)

| File | Class | Order |
|------|-------|-------|
| `hold_signal_check.py` | `HoldSignalCheck` | 0 |
| `trading_hours_check.py` | `TradingHoursCheck` | 10 |
| `circuit_breaker_check.py` | `CircuitBreakerCheck` | 20 |
| `cooldown_check.py` | `CooldownCheck` | 30 |
| `confidence_check.py` | `ConfidenceCheck` | 40 |
| `daily_loss_check.py` | `DailyLossLimitCheck` | 50 |
| `drawdown_check.py` | `DrawdownCheck` | 55 |
| `consecutive_losses_check.py` | `ConsecutiveLossesCheck` | 60 |
| `max_trades_check.py` | `MaxTradesCheck` | 70 |

### Database Migrations

| Migration | Tables/Objects Created |
|-----------|----------------------|
| `014_kill_switch_audit.sql` | `audit.kill_switch_audit`, `audit.kill_switch_alerts`, `audit.kill_switch_duration` view, `audit.log_kill_switch_change()`, `audit.get_kill_switch_status()` |
| `033_event_triggers.sql` | `circuit_breaker_state`, `event_dead_letter_queue`, `event_processed_log`, `v_event_system_health` view |

---

## Integration Examples

### RL Pipeline (5-min bars)

The RL pipeline uses the legacy `RiskManager` directly:

```python
from src.risk import RiskManager, RiskLimits

rm = RiskManager(RiskLimits(max_drawdown_pct=15.0))

# Before each RL action
allowed, reason = rm.validate_signal(signal, current_drawdown_pct=drawdown)
if not allowed:
    action = HOLD

# After each trade closes
rm.record_trade_result(pnl_pct=trade.pnl_pct, signal=trade.signal)
```

### Chain of Responsibility (Pre-Signal Validation)

```python
from src.risk.checks import RiskCheckChain
from src.core.interfaces.risk import RiskContext

chain = RiskCheckChain.with_defaults({
    "min_confidence": 0.6,
    "max_daily_loss": -0.02,
    "max_drawdown": -0.01,
    "max_consecutive_losses": 3,
    "max_trades_per_day": 10,
})

context = RiskContext(
    signal="SHORT",
    confidence=0.85,
    daily_pnl_percent=-0.5,
    current_drawdown=3.2,
    consecutive_losses=1,
    trades_today=4,
)

result = chain.run(context)
# result.approved, result.status, result.metadata["checks_passed"]
```

### Risk Enforcer (Pre-Execution Validation)

```python
from src.trading.risk_enforcer import RiskEnforcer, RiskLimits

enforcer = RiskEnforcer(limits=RiskLimits(
    max_drawdown_pct=15.0,
    max_daily_loss_pct=5.0,
    enable_short=True,
))

# Pre-trade
result = enforcer.check_signal(signal="SHORT", size=100, price=4250.5, confidence=0.9)
if result.decision == RiskDecision.REDUCE:
    size = result.adjusted_size
elif result.is_blocked:
    skip_trade()

# Post-trade
enforcer.record_trade(pnl_pct=-0.5, signal="SHORT")

# State update
enforcer.update_drawdown(drawdown_pct=4.2)
enforcer.update_exposure(exposure=150.0)

# Status
status = enforcer.get_status()  # Dict with full state + limits + rules
```

### Command Pattern (Risk Operations with Undo)

```python
from src.risk.commands import (
    CommandInvoker,
    TriggerCircuitBreakerCommand,
    ResetKillSwitchCommand,
    UpdateRiskLimitsCommand,
)

invoker = CommandInvoker()

# Trigger circuit breaker after consecutive losses
result = invoker.execute(
    TriggerCircuitBreakerCommand(rm, reason="5 consecutive losses", cooldown_minutes=30)
)

# Later: undo the circuit breaker (if deemed premature)
if invoker.can_undo():
    invoker.undo()

# Reset kill switch (requires explicit confirmation)
result = invoker.execute(
    ResetKillSwitchCommand(rm, confirmed=True, reason="drawdown recovered")
)

# Update limits dynamically
result = invoker.execute(
    UpdateRiskLimitsCommand(rm, {"max_drawdown_pct": 10.0}, reason="tighter limits for volatile period")
)

# Audit trail
history = invoker.get_history(limit=10)
# [{"description": "...", "executed_at": "...", "success": True, "message": "..."}, ...]
```

---

## Kill Switch Lifecycle

```
NORMAL TRADING ──→ drawdown >= 15% ──→ KILL SWITCH ACTIVE
       ^                                      |
       |                                      |
       └── reset_kill_switch(confirm=True) ───┘
                (manual only)

    While KILL SWITCH is ACTIVE:
    - ALL new trades BLOCKED (long, short)
    - Exit signals (close, flat) ALLOWED
    - Daily reset does NOT clear kill switch
    - Only manual reset with confirm=True
    - Audit log written to audit.kill_switch_audit
```

### Kill Switch Trigger Sources

| Source | How | Audit `trigger_source` |
|--------|-----|----------------------|
| RiskManager drawdown check | `validate_signal()` detects drawdown >= limit | `AUTOMATED` |
| RiskEnforcer KillSwitchRule | `check_signal()` detects drawdown >= limit | `AUTOMATED` |
| Command pattern | `TriggerCircuitBreakerCommand` (activates cooldown, not kill switch) | `API` / `MANUAL` |
| Direct API call | `reset_kill_switch(confirm=True)` | `API` |
| Airflow DAG | DAG task calls risk manager | `DAG` |

---

## Circuit Breaker States

The `circuit_breaker_state` table persists circuit breaker state across restarts:

```
CLOSED ──→ failure_count >= failure_threshold (3) ──→ OPEN
   ^                                                    |
   |                                                    |
   └── success in HALF_OPEN ──────────────────── HALF_OPEN
                                                    ^
                                                    |
                                    reset_timeout_seconds (300) elapsed
```

| State | Behavior | Transition |
|-------|----------|------------|
| `CLOSED` | Normal operation, requests pass through | -> OPEN after 3 failures |
| `OPEN` | All requests blocked | -> HALF_OPEN after 300s timeout |
| `HALF_OPEN` | Trial request allowed | -> CLOSED on success, -> OPEN on failure |

---

## Monitoring

### Status Queries

```python
# RiskManager status
status = rm.get_status()
# Returns: kill_switch_active, daily_blocked, cooldown_active, cooldown_remaining_minutes,
#          trade_count_today, trades_remaining, daily_pnl_pct, consecutive_losses, limits

# RiskEnforcer status
status = enforcer.get_status()
# Returns: all of the above + current_drawdown_pct, total_exposure, rules list

# Quick check
if enforcer.is_trading_allowed:
    proceed()
```

### Database Monitoring

```sql
-- Current kill switch status
SELECT * FROM audit.get_kill_switch_status();

-- Kill switch activation history with durations
SELECT * FROM audit.kill_switch_duration ORDER BY activated_at DESC LIMIT 10;

-- Circuit breaker state
SELECT circuit_id, state, failure_count, opened_at
FROM circuit_breaker_state WHERE state != 'CLOSED';

-- Event system health
SELECT * FROM v_event_system_health;
```

---

## Related Specs

- `h5-smart-simple-pipeline.md` -- H5 weekly guardrails (circuit breaker, long insistence, rolling DA)
- `sdd-pipeline-lifecycle.md` -- Pipeline guardrails overview
- `l1-l5-inference-pipeline.md` -- L5 inference where risk checks apply at runtime
- `sdd-strategy-spec.md` -- Universal signal record consumed by risk checks
- `data-freshness-enforcement.md` -- Data quality gates (complementary to risk gates)

---

## DO NOT

- Do NOT bypass the kill switch without `confirm=True` -- it exists to prevent catastrophic losses
- Do NOT reset the kill switch programmatically without human review -- use the Command pattern for audit
- Do NOT modify risk constants in `src/core/constants.py` without updating both `RiskManager` and `RiskEnforcer` defaults
- Do NOT add risk checks to the chain without assigning an `order` value in the correct range (see IRiskCheck docstring)
- Do NOT use the Chain checks and the Enforcer rules interchangeably -- they have different input/output types (`RiskContext` vs signal/size/price)
- Do NOT ignore `REDUCE` decisions from the Enforcer -- always use `adjusted_size` when returned
- Do NOT allow exit signals (`CLOSE`, `FLAT`) to be blocked by risk checks -- they must always pass for risk reduction
- Do NOT skip `record_trade_result()` / `record_trade()` after a trade closes -- state tracking depends on it
- Do NOT assume daily reset clears the kill switch -- kill switch persists until manual reset
- Do NOT remove the `HOLD_SIGNAL` short-circuit from the chain -- it prevents unnecessary risk evaluation on no-op signals
