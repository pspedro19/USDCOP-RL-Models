"""
DAG: forecast_h5_l7_multiday_executor
=======================================
Smart Simple v1.0 — 5-day execution with TP/HS/Friday close.

Monday:   Enter position after L5c completes (09:00 COT)
Mon-Fri:  Monitor TP/HS every 30 min during session (08:00-12:55 COT)
Friday:   Close remaining position at 12:50 COT

No trailing stop, no re-entry after exit.
Adaptive TP/HS levels computed by L5c from realized vol.
Confidence-adjusted leverage from L5b + L5c.

Schedule: */30 13-17 * * 1-5 (Mon-Fri every 30min, 08:00-12:55 COT)

Config: smart_simple_v1.yaml
Persistence: forecast_h5_executions + forecast_h5_subtrades

Contract: FC-H5-L7-001
Version: 3.0.0 (Smart Simple v1.0)
Date: 2026-02-16
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import sys

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L7_MULTIDAY_EXECUTOR,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H5_L7_MULTIDAY_EXECUTOR
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')


# =============================================================================
# HELPERS
# =============================================================================

def _load_h5_config():
    """Load Smart Simple v1 config from YAML."""
    import yaml
    config_path = PROJECT_ROOT / 'config' / 'execution' / 'smart_simple_v1.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_current_week_signal(cur):
    """Get this week's H5 signal with confidence + stops."""
    cur.execute("""
        SELECT id, signal_date, direction, adjusted_leverage,
               inference_week, inference_year, collapse_flag,
               hard_stop_pct, take_profit_pct, skip_trade,
               confidence_tier
        FROM forecast_h5_signals
        WHERE signal_date <= CURRENT_DATE
        ORDER BY signal_date DESC
        LIMIT 1
    """)
    return cur.fetchone()


def _get_current_week_execution(cur, signal_date):
    """Get the execution record for this week."""
    cur.execute("""
        SELECT id, status, direction, leverage
        FROM forecast_h5_executions
        WHERE signal_date = %s
    """, (signal_date,))
    return cur.fetchone()


def _get_active_subtrade(cur, execution_id):
    """Get the open subtrade for an execution (simplified: no peak/trailing)."""
    cur.execute("""
        SELECT id, subtrade_index, direction, entry_price, entry_timestamp,
               bar_count
        FROM forecast_h5_subtrades
        WHERE execution_id = %s AND exit_price IS NULL
        ORDER BY subtrade_index DESC
        LIMIT 1
    """, (execution_id,))
    return cur.fetchone()


def _get_latest_bar(cur):
    """Get latest USDCOP 5-min bar."""
    cur.execute("""
        SELECT time, open, high, low, close
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time DESC
        LIMIT 1
    """)
    return cur.fetchone()


def _check_circuit_breaker(cur, h5_config) -> bool:
    """Check if circuit breaker should activate."""
    cb_config = h5_config.get("guardrails", {}).get("circuit_breaker", {})
    max_dd = -abs(cb_config.get("max_drawdown_pct", 12.0))
    max_losses = cb_config.get("max_consecutive_losses", 5)

    cur.execute("""
        SELECT cumulative_pnl_pct, consecutive_losses
        FROM forecast_h5_paper_trading
        ORDER BY signal_date DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        cum_pnl = row[0] or 0.0
        consec = row[1] or 0
        if cum_pnl <= max_dd:
            logger.warning(f"[H5-L7] CIRCUIT BREAKER: cum DD {cum_pnl:.2f}% <= {max_dd}%")
            return True
        if consec >= max_losses:
            logger.warning(f"[H5-L7] CIRCUIT BREAKER: {consec} consecutive losses >= {max_losses}")
            return True
    return False


# =============================================================================
# TASK: ROUTE — decide what action to take
# =============================================================================

def route_action(**context) -> str:
    """
    Smart Simple v1.0 routing (no re-entry, no trailing):
    - No signal / skip_trade / collapse -> skip
    - Circuit breaker -> close_position
    - Friday 12:50+ -> close_week
    - Monday + no execution -> enter_position
    - Has open subtrade -> monitor_position
    - Subtrade already closed by TP/HS -> skip (no re-entry)
    """
    now_utc = datetime.now(timezone.utc)
    day_of_week = now_utc.weekday()  # 0=Mon, 4=Fri

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get this week's signal
        signal = _get_current_week_signal(cur)
        if not signal:
            logger.info("[H5-L7] No signal this week, skipping")
            return "skip"

        (signal_id, signal_date, direction, leverage, week, year,
         collapse, hard_stop_pct, take_profit_pct, skip_trade,
         confidence_tier) = signal

        if skip_trade:
            logger.info(f"[H5-L7] Signal marked skip_trade (confidence={confidence_tier}), skipping")
            return "skip"

        if collapse:
            logger.warning("[H5-L7] Collapse flag set, skipping")
            return "skip"

        if leverage is None:
            logger.warning("[H5-L7] No leverage yet (L5c hasn't run), skipping")
            return "skip"

        # Check circuit breaker
        h5_config = _load_h5_config()
        if _check_circuit_breaker(cur, h5_config):
            return "close_position"

        # Check if we have an execution record
        exec_row = _get_current_week_execution(cur, signal_date)

        # Friday close check (17:50 UTC = 12:50 COT)
        if day_of_week == 4 and now_utc.hour >= 17 and now_utc.minute >= 50:
            if exec_row and exec_row[1] not in ('closed', 'paused', 'error'):
                return "close_week"
            return "skip"

        # No execution yet — enter (Monday only)
        if not exec_row:
            if day_of_week == 0:  # Monday
                return "enter_position"
            logger.info("[H5-L7] No execution and not Monday, skipping")
            return "skip"

        exec_id, status, _, _ = exec_row

        if status in ('closed', 'paused', 'error'):
            return "skip"

        # Check for active subtrade
        active_sub = _get_active_subtrade(cur, exec_id)
        if active_sub:
            return "monitor_position"

        # No active subtrade and not closed = already exited by TP/HS, no re-entry
        logger.info("[H5-L7] Subtrade closed by TP/HS, no re-entry in Smart Simple v1")
        return "skip"

    finally:
        conn.close()


# =============================================================================
# TASK: ENTER POSITION (Monday)
# =============================================================================

def enter_position(**context) -> Dict[str, Any]:
    """Enter initial position for the week with adaptive stops."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        signal = _get_current_week_signal(cur)
        if not signal:
            raise ValueError("[H5-L7] No signal to enter")

        (signal_id, signal_date, direction, leverage, week, year,
         _, hard_stop_pct, take_profit_pct, _, confidence_tier) = signal

        bar = _get_latest_bar(cur)
        if not bar:
            raise ValueError("[H5-L7] No bar data available")

        bar_time, bar_open, bar_high, bar_low, bar_close = bar
        entry_price = bar_close

        # Compute TP/HS price levels for logging
        if direction == 1:  # LONG
            tp_price = entry_price * (1 + take_profit_pct) if take_profit_pct else None
            hs_price = entry_price * (1 - hard_stop_pct) if hard_stop_pct else None
        else:  # SHORT
            tp_price = entry_price * (1 - take_profit_pct) if take_profit_pct else None
            hs_price = entry_price * (1 + hard_stop_pct) if hard_stop_pct else None

        # Create execution record (UPSERT for idempotent coexistence with DB seeding)
        cur.execute("""
            INSERT INTO forecast_h5_executions
            (signal_date, inference_week, inference_year, direction, leverage,
             entry_price, entry_timestamp, status, config_version,
             confidence_tier, hard_stop_pct, take_profit_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'positioned', 'smart_simple_v1',
                    %s, %s, %s)
            ON CONFLICT (signal_date) DO UPDATE SET
                entry_price = EXCLUDED.entry_price,
                entry_timestamp = EXCLUDED.entry_timestamp,
                status = EXCLUDED.status,
                confidence_tier = EXCLUDED.confidence_tier,
                hard_stop_pct = EXCLUDED.hard_stop_pct,
                take_profit_pct = EXCLUDED.take_profit_pct,
                updated_at = NOW()
            RETURNING id
        """, (signal_date, week, year, direction, leverage, entry_price, bar_time,
              confidence_tier, hard_stop_pct, take_profit_pct))
        exec_id = cur.fetchone()[0]

        # Ensure unique index exists for subtrade UPSERT
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                uq_h5_subtrades_exec_idx ON forecast_h5_subtrades(execution_id, subtrade_index)
        """)

        # Create single subtrade (UPSERT, no re-entry in Smart Simple)
        cur.execute("""
            INSERT INTO forecast_h5_subtrades
            (execution_id, subtrade_index, direction, entry_price,
             entry_timestamp, peak_price, trailing_state)
            VALUES (%s, 0, %s, %s, %s, %s, 'active')
            ON CONFLICT (execution_id, subtrade_index) DO UPDATE SET
                entry_price = EXCLUDED.entry_price,
                entry_timestamp = EXCLUDED.entry_timestamp,
                peak_price = EXCLUDED.peak_price,
                trailing_state = EXCLUDED.trailing_state,
                updated_at = NOW()
        """, (exec_id, direction, entry_price, bar_time, entry_price))

        conn.commit()

        dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"[H5-L7] ENTERED {dir_str}: price={entry_price:.2f}, "
            f"leverage={leverage:.3f}, confidence={confidence_tier}, "
            f"TP={tp_price:.2f} ({take_profit_pct*100:.2f}%), "
            f"HS={hs_price:.2f} ({hard_stop_pct*100:.2f}%), "
            f"exec_id={exec_id}"
        )

        return {
            "exec_id": exec_id,
            "direction": direction,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "hs_price": hs_price,
        }

    finally:
        conn.close()


# =============================================================================
# TASK: MONITOR POSITION (TP/HS check)
# =============================================================================

def monitor_position(**context) -> Dict[str, Any]:
    """Monitor active subtrade for take-profit and hard-stop."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.adaptive_stops import (
        check_hard_stop,
        check_take_profit,
        get_exit_price,
    )

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        signal = _get_current_week_signal(cur)
        if not signal:
            return {"action": "no_signal"}

        (signal_id, signal_date, direction, leverage, week, year,
         _, hard_stop_pct, take_profit_pct, _, _) = signal

        exec_row = _get_current_week_execution(cur, signal_date)
        if not exec_row:
            return {"action": "no_execution"}

        exec_id = exec_row[0]
        active_sub = _get_active_subtrade(cur, exec_id)
        if not active_sub:
            return {"action": "no_active_subtrade"}

        sub_id, sub_idx, sub_dir, entry_price, entry_ts, bar_count = active_sub

        # Get latest bar
        bar = _get_latest_bar(cur)
        if not bar:
            return {"action": "no_bar"}

        bar_time, bar_open, bar_high, bar_low, bar_close = bar

        # Check hard stop first (worst case)
        if check_hard_stop(sub_dir, entry_price, bar_high, bar_low, hard_stop_pct):
            exit_price = get_exit_price(
                sub_dir, entry_price, "hard_stop",
                hard_stop_pct, take_profit_pct, bar_close
            )
            raw_pnl = sub_dir * (exit_price - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'hard_stop',
                    trailing_state = 'triggered', bar_count = %s,
                    pnl_pct = %s, pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, bar_count + 1, lev_pnl, raw_pnl, sub_id))

            cur.execute("""
                UPDATE forecast_h5_executions
                SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                    exit_reason = 'hard_stop', week_pnl_pct = %s,
                    week_pnl_unleveraged_pct = %s, n_subtrades = 1,
                    updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, lev_pnl, raw_pnl, exec_id))

            conn.commit()
            logger.info(
                f"[H5-L7] HARD STOP: exit_price={exit_price:.2f}, "
                f"entry={entry_price:.2f}, pnl={lev_pnl:+.4f}%, "
                f"bars={bar_count + 1}"
            )
            return {"action": "hard_stop", "pnl_pct": lev_pnl, "exit_price": exit_price}

        # Check take profit
        if check_take_profit(sub_dir, entry_price, bar_high, bar_low, take_profit_pct):
            exit_price = get_exit_price(
                sub_dir, entry_price, "take_profit",
                hard_stop_pct, take_profit_pct, bar_close
            )
            raw_pnl = sub_dir * (exit_price - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'take_profit',
                    trailing_state = 'triggered', bar_count = %s,
                    pnl_pct = %s, pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, bar_count + 1, lev_pnl, raw_pnl, sub_id))

            cur.execute("""
                UPDATE forecast_h5_executions
                SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                    exit_reason = 'take_profit', week_pnl_pct = %s,
                    week_pnl_unleveraged_pct = %s, n_subtrades = 1,
                    updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, lev_pnl, raw_pnl, exec_id))

            conn.commit()
            logger.info(
                f"[H5-L7] TAKE PROFIT: exit_price={exit_price:.2f}, "
                f"entry={entry_price:.2f}, pnl={lev_pnl:+.4f}%, "
                f"bars={bar_count + 1}"
            )
            return {"action": "take_profit", "pnl_pct": lev_pnl, "exit_price": exit_price}

        # Still monitoring — compute distances for logging
        if sub_dir == 1:  # LONG
            dist_tp = (bar_close / entry_price - 1) / take_profit_pct * 100
            dist_hs = (1 - bar_close / entry_price) / hard_stop_pct * 100
        else:  # SHORT
            dist_tp = (1 - bar_close / entry_price) / take_profit_pct * 100
            dist_hs = (bar_close / entry_price - 1) / hard_stop_pct * 100

        # Update bar count
        cur.execute("""
            UPDATE forecast_h5_subtrades
            SET bar_count = %s, updated_at = NOW()
            WHERE id = %s
        """, (bar_count + 1, sub_id))
        conn.commit()

        logger.info(
            f"[H5-L7] MONITORING: bar={bar_time}, close={bar_close:.2f}, "
            f"entry={entry_price:.2f}, TP={dist_tp:.1f}%, HS={dist_hs:.1f}%, "
            f"bars={bar_count + 1}"
        )
        return {"action": "monitoring", "dist_tp_pct": dist_tp, "dist_hs_pct": dist_hs}

    finally:
        conn.close()


# =============================================================================
# TASK: CLOSE WEEK (Friday 12:50 COT)
# =============================================================================

def close_week(**context) -> Dict[str, Any]:
    """Close remaining position at Friday session end."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        signal = _get_current_week_signal(cur)
        if not signal:
            return {"action": "no_signal"}

        (signal_id, signal_date, direction, leverage, week, year,
         _, hard_stop_pct, take_profit_pct, _, _) = signal

        exec_row = _get_current_week_execution(cur, signal_date)
        if not exec_row:
            return {"action": "no_execution"}

        exec_id, status, _, _ = exec_row
        if status in ('closed', 'paused', 'error'):
            return {"action": "already_closed"}

        bar = _get_latest_bar(cur)
        if not bar:
            return {"action": "no_bar"}

        bar_time, _, _, _, bar_close = bar

        # Close active subtrade if any
        active_sub = _get_active_subtrade(cur, exec_id)
        if active_sub:
            sub_id = active_sub[0]
            entry_price = active_sub[3]
            sub_dir = active_sub[2]
            raw_pnl = sub_dir * (bar_close - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'week_end',
                    trailing_state = 'expired', pnl_pct = %s,
                    pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (bar_close, bar_time, lev_pnl, raw_pnl, sub_id))

        # Aggregate week PnL from all subtrades
        cur.execute("""
            SELECT COALESCE(SUM(pnl_pct), 0), COALESCE(SUM(pnl_unleveraged_pct), 0),
                   COUNT(*)
            FROM forecast_h5_subtrades
            WHERE execution_id = %s AND exit_price IS NOT NULL
        """, (exec_id,))
        total_pnl, total_unlev, n_subs = cur.fetchone()

        cur.execute("""
            UPDATE forecast_h5_executions
            SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                exit_reason = 'week_end', week_pnl_pct = %s,
                week_pnl_unleveraged_pct = %s, n_subtrades = %s,
                updated_at = NOW()
            WHERE id = %s
        """, (bar_close, bar_time, total_pnl, total_unlev, n_subs, exec_id))

        conn.commit()

        logger.info(
            f"[H5-L7] WEEK CLOSED: pnl={total_pnl:+.4f}%, "
            f"n_subtrades={n_subs}, exit_price={bar_close:.2f}"
        )

        return {"action": "closed", "week_pnl_pct": total_pnl, "n_subtrades": n_subs}

    finally:
        conn.close()


# =============================================================================
# TASK: CLOSE POSITION (circuit breaker)
# =============================================================================

def close_position(**context) -> Dict[str, Any]:
    """Emergency close due to circuit breaker."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        signal = _get_current_week_signal(cur)
        if not signal:
            return {"action": "no_signal"}

        (signal_id, signal_date, direction, leverage, week, year,
         _, _, _, _, _) = signal

        exec_row = _get_current_week_execution(cur, signal_date)
        if not exec_row or exec_row[1] in ('closed', 'paused'):
            return {"action": "nothing_to_close"}

        exec_id = exec_row[0]
        bar = _get_latest_bar(cur)
        bar_close = bar[4] if bar else 0
        bar_time = bar[0] if bar else datetime.now(timezone.utc)

        # Close active subtrade
        active_sub = _get_active_subtrade(cur, exec_id)
        if active_sub:
            sub_id = active_sub[0]
            entry_price = active_sub[3]
            sub_dir = active_sub[2]
            raw_pnl = sub_dir * (bar_close - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'circuit_breaker',
                    trailing_state = 'triggered', pnl_pct = %s,
                    pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (bar_close, bar_time, lev_pnl, raw_pnl, sub_id))

        # Aggregate and close execution
        cur.execute("""
            SELECT COALESCE(SUM(pnl_pct), 0), COUNT(*)
            FROM forecast_h5_subtrades
            WHERE execution_id = %s AND exit_price IS NOT NULL
        """, (exec_id,))
        total_pnl, n_subs = cur.fetchone()

        cur.execute("""
            UPDATE forecast_h5_executions
            SET status = 'paused', exit_price = %s, exit_timestamp = %s,
                exit_reason = 'circuit_breaker', week_pnl_pct = %s,
                n_subtrades = %s, updated_at = NOW()
            WHERE id = %s
        """, (bar_close, bar_time, total_pnl, n_subs, exec_id))

        conn.commit()
        logger.warning(f"[H5-L7] CIRCUIT BREAKER: closed, pnl={total_pnl:+.4f}%")

        return {"action": "circuit_breaker", "week_pnl_pct": total_pnl}

    finally:
        conn.close()


# =============================================================================
# TASK: SKIP
# =============================================================================

def skip(**context) -> None:
    """No action needed this cycle."""
    logger.info("[H5-L7] No action needed")


# =============================================================================
# TASK: LOG
# =============================================================================

def log_result(**context) -> None:
    """Log the result of this execution cycle."""
    ti = context['ti']
    for task_id in ['enter_position', 'monitor_position',
                     'close_week', 'close_position']:
        result = ti.xcom_pull(task_ids=task_id)
        if result:
            logger.info(f"[H5-L7] Result from {task_id}: {result}")
            return
    logger.info("[H5-L7] Cycle complete (skip)")


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Smart Simple v1.0: TP/HS/Friday close, no trailing (30-min monitor)',
    schedule_interval='*/30 13-17 * * 1-5',  # Mon-Fri every 30min, 08:00-12:55 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_route = BranchPythonOperator(
        task_id='route',
        python_callable=route_action,
    )

    t_enter = PythonOperator(
        task_id='enter_position',
        python_callable=enter_position,
    )

    t_monitor = PythonOperator(
        task_id='monitor_position',
        python_callable=monitor_position,
    )

    t_close_week = PythonOperator(
        task_id='close_week',
        python_callable=close_week,
    )

    t_close_cb = PythonOperator(
        task_id='close_position',
        python_callable=close_position,
    )

    t_skip = PythonOperator(
        task_id='skip',
        python_callable=skip,
    )

    t_log = PythonOperator(
        task_id='log_result',
        python_callable=log_result,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    t_route >> [t_enter, t_monitor, t_close_week, t_close_cb, t_skip]
    [t_enter, t_monitor, t_close_week, t_close_cb, t_skip] >> t_log
