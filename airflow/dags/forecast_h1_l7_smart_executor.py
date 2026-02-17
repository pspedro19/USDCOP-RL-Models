"""
DAG: forecast_h1_l7_smart_executor
===================================
Smart Executor for intraday execution of daily forecasting signals.

Bridges L5c vol-targeting signals -> TrailingStopTracker -> PaperBroker.
Uses a trailing stop to exit positions intra-session, protecting peaks.

Two modes (self-routing via BranchPythonOperator):

1. ENTRY (after L5c, ~13:35 COT on day T):
   - Detect new signal in forecast_vol_targeting_signals without execution
   - Enter position via PaperBroker at daily close price (last 5-min bar)
   - INSERT into forecast_executions (status='positioned')

2. MONITOR (every 5 min, 8:00-12:55 COT on day T+1):
   - Fetch unprocessed OHLCV bars from usdcop_m5_ohlcv
   - Feed each bar to SmartExecutor.monitor_bar()
   - If trailing stop triggers: status='closed', exit_reason
   - At session end (12:55 COT): SmartExecutor.expire_session()
   - UPDATE forecast_executions with new state

Schedule: */5 13-19 * * 1-5 (UTC = 8:00-14:55 COT, Mon-Fri)
  - 13:00-17:55 UTC (8:00-12:55 COT): monitoring window
  - 18:00-18:55 UTC (13:00-13:55 COT): entry window (L5c runs at 18:30 UTC)

Author: Trading Team
Version: 1.0.0
Date: 2026-02-15
Contract: SMART-EXECUTOR-V1
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import logging
import sys

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

from contracts.dag_registry import (
    FORECAST_H1_L7_SMART_EXECUTOR,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H1_L7_SMART_EXECUTOR
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

# Executor modules (may not be available during DAG parsing)
try:
    from src.execution.smart_executor import (
        SmartExecutor,
        SmartExecutorConfig,
        ExecutionState,
        ExecutionStatus,
    )
    from src.execution.broker_adapter import PaperBroker
    EXECUTOR_AVAILABLE = True
except ImportError as e:
    EXECUTOR_AVAILABLE = False
    logging.error(f"[L7] Executor module not available: {e}")

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

EXECUTOR_CONFIG = {
    "activation_pct": 0.002,
    "trail_pct": 0.003,
    "hard_stop_pct": 0.015,
    "slippage_bps": 1.0,
    "config_version": "smart_executor_v1",
}

# Direction filter: "all" (default) | "short_only" | "long_only"
# Loaded from config/execution/smart_executor_v1.yaml at runtime
DIRECTION_FILTER = "short_only"  # 2026-02-15: regime change, LONG WR=28%

# Session end: 12:55 COT = 17:55 UTC
SESSION_END_UTC_HOUR = 17
SESSION_END_UTC_MINUTE = 50  # use :50 to allow processing before :55


def _get_executor() -> "SmartExecutor":
    """Create a SmartExecutor instance from config."""
    cfg = SmartExecutorConfig(**EXECUTOR_CONFIG)
    broker = PaperBroker(slippage_bps=cfg.slippage_bps)
    return SmartExecutor(cfg, broker)


def _row_to_state(row) -> "ExecutionState":
    """Convert a DB row tuple to ExecutionState."""
    return ExecutionState(
        signal_date=str(row[0]),
        status=ExecutionStatus(row[1]),
        direction=int(row[2]),
        leverage=float(row[3]),
        entry_price=float(row[4]) if row[4] else None,
        entry_timestamp=row[5],
        peak_price=float(row[6]) if row[6] else None,
        trailing_state=row[7] or "waiting",
        bar_count=int(row[8]) if row[8] else 0,
        config_version=row[9] or EXECUTOR_CONFIG["config_version"],
    )


# =============================================================================
# TASK 1: ROUTE EXECUTION
# =============================================================================

def route_execution(**context) -> str:
    """
    Determine action: monitor open position, enter new position, or skip.
    Returns the task_id to execute next.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Priority 1: Monitor open positions
        cur.execute("""
            SELECT 1 FROM forecast_executions
            WHERE status IN ('positioned', 'monitoring')
            LIMIT 1
        """)
        if cur.fetchone():
            logger.info("[L7] Open position found, routing to monitor")
            return 'monitor_position'

        # Priority 2: Enter new position from pending signal
        cur.execute("""
            SELECT s.signal_date
            FROM forecast_vol_targeting_signals s
            LEFT JOIN forecast_executions e ON s.signal_date = e.signal_date
            WHERE e.id IS NULL
            ORDER BY s.signal_date DESC
            LIMIT 1
        """)
        pending = cur.fetchone()
        if pending:
            logger.info(f"[L7] Pending signal {pending[0]}, routing to entry")
            return 'enter_position'

        logger.debug("[L7] Nothing to do, skipping")
        return 'skip_run'

    finally:
        conn.close()


# =============================================================================
# TASK 2: ENTER POSITION
# =============================================================================

def enter_position(**context) -> None:
    """
    Enter a new position from a pending L5c signal.
    Entry price = close of the last 5-min bar on signal_date.
    """
    if not EXECUTOR_AVAILABLE:
        raise ImportError("[L7] Executor module not available")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Find pending signal
        cur.execute("""
            SELECT s.signal_date, s.forecast_direction, s.clipped_leverage
            FROM forecast_vol_targeting_signals s
            LEFT JOIN forecast_executions e ON s.signal_date = e.signal_date
            WHERE e.id IS NULL
            ORDER BY s.signal_date DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            logger.warning("[L7] No pending signal found (race condition?)")
            return

        signal_date, direction, leverage = str(row[0]), int(row[1]), float(row[2])

        # Direction filter: skip signals that don't match the allowed direction
        dir_str = "LONG" if direction == 1 else "SHORT"
        if DIRECTION_FILTER == "short_only" and direction == 1:
            logger.info(
                f"[L7] SKIPPED {dir_str} signal for {signal_date} "
                f"(direction_filter={DIRECTION_FILTER})"
            )
            # Record skipped signal so we don't re-process it
            cur.execute("""
                INSERT INTO forecast_executions (
                    signal_date, status, direction, leverage,
                    exit_reason, config_version
                ) VALUES (%s, 'closed', %s, %s, 'filtered_direction', %s)
                ON CONFLICT (signal_date) DO NOTHING
            """, (signal_date, direction, leverage, EXECUTOR_CONFIG["config_version"]))
            conn.commit()
            context['ti'].xcom_push(key='action', value={
                'type': 'filtered', 'signal_date': signal_date,
                'direction': dir_str, 'filter': DIRECTION_FILTER,
            })
            return
        elif DIRECTION_FILTER == "long_only" and direction == -1:
            logger.info(
                f"[L7] SKIPPED {dir_str} signal for {signal_date} "
                f"(direction_filter={DIRECTION_FILTER})"
            )
            cur.execute("""
                INSERT INTO forecast_executions (
                    signal_date, status, direction, leverage,
                    exit_reason, config_version
                ) VALUES (%s, 'closed', %s, %s, 'filtered_direction', %s)
                ON CONFLICT (signal_date) DO NOTHING
            """, (signal_date, direction, leverage, EXECUTOR_CONFIG["config_version"]))
            conn.commit()
            context['ti'].xcom_push(key='action', value={
                'type': 'filtered', 'signal_date': signal_date,
                'direction': dir_str, 'filter': DIRECTION_FILTER,
            })
            return

        # Entry price: last 5-min bar close on signal_date
        cur.execute("""
            SELECT close FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
              AND (time AT TIME ZONE 'America/Bogota')::date = %s
            ORDER BY time DESC
            LIMIT 1
        """, (signal_date,))
        price_row = cur.fetchone()

        if price_row is None:
            logger.error(f"[L7] No OHLCV close price for signal_date={signal_date}")
            return

        entry_price = float(price_row[0])

        # Execute entry
        executor = _get_executor()
        state = executor.enter_position(signal_date, direction, leverage, entry_price)

        if state.status == ExecutionStatus.ERROR:
            logger.error(f"[L7] Entry failed for {signal_date}")
            return

        # Persist
        cfg = EXECUTOR_CONFIG
        cur.execute("""
            INSERT INTO forecast_executions (
                signal_date, status, direction, leverage,
                entry_price, entry_timestamp,
                peak_price, trailing_state, bar_count,
                config_version, activation_pct, trail_pct, hard_stop_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_date) DO NOTHING
            RETURNING id
        """, (
            state.signal_date, state.status.value, state.direction, state.leverage,
            state.entry_price, state.entry_timestamp,
            state.peak_price, state.trailing_state, state.bar_count,
            state.config_version,
            cfg["activation_pct"], cfg["trail_pct"], cfg["hard_stop_pct"],
        ))
        conn.commit()

        result_row = cur.fetchone()
        exec_id = result_row[0] if result_row else "duplicate"
        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"[L7] ENTERED {dir_str} @ {state.entry_price:.2f}, "
            f"lev={leverage:.2f}x, id={exec_id}, signal={signal_date}"
        )

        context['ti'].xcom_push(key='action', value={
            'type': 'entry', 'signal_date': signal_date,
            'direction': dir_str, 'entry_price': state.entry_price,
        })

    finally:
        conn.close()


# =============================================================================
# TASK 3: MONITOR POSITION
# =============================================================================

def monitor_position(**context) -> None:
    """
    Monitor open position with unprocessed OHLCV bars.
    Processes all bars since last check. Expires at session end.
    """
    if not EXECUTOR_AVAILABLE:
        raise ImportError("[L7] Executor module not available")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Load open position
        cur.execute("""
            SELECT signal_date, status, direction, leverage,
                   entry_price, entry_timestamp,
                   peak_price, trailing_state, bar_count, config_version
            FROM forecast_executions
            WHERE status IN ('positioned', 'monitoring')
            ORDER BY signal_date DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            logger.info("[L7] No open position to monitor")
            return

        state = _row_to_state(row)

        # Get T+1 session bars (next trading day after signal_date)
        cur.execute("""
            WITH next_day AS (
                SELECT MIN((time AT TIME ZONE 'America/Bogota')::date) AS trade_date
                FROM usdcop_m5_ohlcv
                WHERE symbol = 'USD/COP'
                  AND (time AT TIME ZONE 'America/Bogota')::date > %s
            )
            SELECT time, high, low, close
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
              AND (time AT TIME ZONE 'America/Bogota')::date = (SELECT trade_date FROM next_day)
            ORDER BY time ASC
        """, (state.signal_date,))

        all_bars = cur.fetchall()
        if not all_bars:
            logger.info(f"[L7] No bars yet for T+1 session (signal={state.signal_date})")
            return

        # Skip already-processed bars
        unprocessed = all_bars[state.bar_count:]
        if not unprocessed:
            # No new bars — but check if we should expire
            now_utc = datetime.now(timezone.utc)
            if (now_utc.hour > SESSION_END_UTC_HOUR or
                    (now_utc.hour == SESSION_END_UTC_HOUR and now_utc.minute >= SESSION_END_UTC_MINUTE)):
                last_close = float(all_bars[-1][3])
                executor = _get_executor()
                state = executor.expire_session(state, last_close)
                _persist_closed(cur, state)
                conn.commit()
                logger.info(
                    f"[L7] EXPIRED at session end: pnl={state.pnl_pct:+.4%}, "
                    f"bars={state.bar_count}"
                )
                context['ti'].xcom_push(key='action', value={
                    'type': 'expire', 'pnl_pct': state.pnl_pct,
                    'bar_count': state.bar_count,
                })
            else:
                logger.debug(f"[L7] All {state.bar_count} bars processed, awaiting next bar")
            return

        # Process each unprocessed bar
        executor = _get_executor()
        for i, bar in enumerate(unprocessed):
            bar_time, bar_high, bar_low, bar_close = bar
            bar_idx = state.bar_count + i
            state = executor.monitor_bar(
                state, float(bar_high), float(bar_low), float(bar_close), bar_idx,
            )
            if state.status == ExecutionStatus.CLOSED:
                break

        # Check session end for expiry
        if SmartExecutor.should_monitor(state):
            now_utc = datetime.now(timezone.utc)
            if (now_utc.hour > SESSION_END_UTC_HOUR or
                    (now_utc.hour == SESSION_END_UTC_HOUR and now_utc.minute >= SESSION_END_UTC_MINUTE)):
                last_close = float(all_bars[-1][3]) if all_bars else float(unprocessed[-1][3])
                state = executor.expire_session(state, last_close)

        # Persist updated state
        if state.status == ExecutionStatus.CLOSED:
            _persist_closed(cur, state)
            logger.info(
                f"[L7] CLOSED: reason={state.exit_reason}, pnl={state.pnl_pct:+.4%}, "
                f"bars={state.bar_count}, peak={state.peak_price:.2f}"
            )
            context['ti'].xcom_push(key='action', value={
                'type': 'close', 'exit_reason': state.exit_reason,
                'pnl_pct': state.pnl_pct, 'bar_count': state.bar_count,
            })
        else:
            cur.execute("""
                UPDATE forecast_executions SET
                    status = %s, peak_price = %s, trailing_state = %s, bar_count = %s
                WHERE signal_date = %s
            """, (
                state.status.value, state.peak_price,
                state.trailing_state, state.bar_count,
                state.signal_date,
            ))
            logger.info(
                f"[L7] Monitoring: bars={state.bar_count}/{len(all_bars)}, "
                f"trail={state.trailing_state}, peak={state.peak_price:.2f}"
            )

        conn.commit()

    finally:
        conn.close()


def _persist_closed(cur, state: "ExecutionState") -> None:
    """Write closed execution state to DB."""
    cur.execute("""
        UPDATE forecast_executions SET
            status = %s, peak_price = %s, trailing_state = %s, bar_count = %s,
            exit_price = %s, exit_timestamp = %s, exit_reason = %s,
            pnl_pct = %s, pnl_unleveraged_pct = %s
        WHERE signal_date = %s
    """, (
        state.status.value, state.peak_price, state.trailing_state, state.bar_count,
        state.exit_price, state.exit_timestamp, state.exit_reason,
        state.pnl_pct, state.pnl_unleveraged_pct,
        state.signal_date,
    ))


# =============================================================================
# TASK 4: LOG RESULT
# =============================================================================

def log_result(**context) -> None:
    """Log a summary of what happened in this run."""
    ti = context['ti']

    # Try to get action from whichever task ran
    action = (
        ti.xcom_pull(key='action', task_ids='enter_position') or
        ti.xcom_pull(key='action', task_ids='monitor_position')
    )

    if not action:
        logger.debug("[L7] No action taken this run")
        return

    action_type = action.get('type', '?')
    if action_type == 'entry':
        logger.info(
            f"[L7] Summary: ENTRY {action['direction']} @ {action['entry_price']:.2f} "
            f"for signal {action['signal_date']}"
        )
    elif action_type == 'filtered':
        logger.info(
            f"[L7] Summary: FILTERED {action['direction']} for {action['signal_date']} "
            f"(filter={action['filter']})"
        )
    elif action_type in ('close', 'expire'):
        logger.info(
            f"[L7] Summary: {action.get('exit_reason', 'closed').upper()} "
            f"pnl={action['pnl_pct']:+.4%}, bars={action['bar_count']}"
        )


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Smart Executor: trailing stop intraday execution for forecasting signals',
    schedule_interval='*/5 13-19 * * 1-5',  # UTC = 8:00-14:55 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_route = BranchPythonOperator(
        task_id='route_execution',
        python_callable=route_execution,
        provide_context=True,
    )

    t_enter = PythonOperator(
        task_id='enter_position',
        python_callable=enter_position,
        provide_context=True,
    )

    t_monitor = PythonOperator(
        task_id='monitor_position',
        python_callable=monitor_position,
        provide_context=True,
    )

    t_skip = EmptyOperator(task_id='skip_run')

    t_log = PythonOperator(
        task_id='log_result',
        python_callable=log_result,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # DAG flow: route → [enter | monitor | skip] → log
    t_route >> [t_enter, t_monitor, t_skip]
    [t_enter, t_monitor, t_skip] >> t_log
