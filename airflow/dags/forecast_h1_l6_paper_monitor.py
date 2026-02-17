"""
DAG: forecast_h1_l6_paper_monitor
=================================
USD/COP Trading System - Daily Paper Trading Monitor

Evaluates the PREVIOUS trading day's vol-targeting signal against actual
market returns. Tracks running statistics and enforces stop criteria.

Temporal Offset:
    L5c generates signal on day T at 13:30 COT, predicting T->T+1 return.
    L6 runs on day T+1 at 19:00 COT, evaluating signal(T):
      - entry_price = close[T]   (close on signal day)
      - close_price = close[T+1] (today's close)
      - actual_return = log(close[T+1] / close[T])

    Weekend handling:
      - Friday signal -> evaluated Monday (T=Friday, T+1=Monday)
      - No signal on Sat/Sun -> no evaluation

Schedule: Daily Mon-Fri at 00:00 UTC+1 (19:00 COT), after market close
Output: 1 row in forecast_paper_trading per evaluated day
Depends on: forecast_h1_l5_vol_targeting (generates signals)

Author: Trading Team
Version: 1.0.0
Date: 2026-02-15
Contract: FC-SIZE-001
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import logging
import math
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

from contracts.dag_registry import (
    FORECAST_H1_L6_PAPER_MONITOR,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H1_L6_PAPER_MONITOR
DAG_TAGS = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)


# =============================================================================
# STOP CRITERIA (from vol_target_v1.yaml paper_trading section)
# =============================================================================

STOP_CRITERIA = {
    "min_days_before_eval": 20,
    "continue_if_da_above": 50.0,
    "pause_if_da_below": 48.0,
    "pause_after_days": 40,
    "stop_if_da_below": 46.0,
    "stop_after_days": 60,
    "stop_if_maxdd_above": 20.0,
    "pause_if_maxdd_above": 15.0,
}


# =============================================================================
# TASK 1: FETCH YESTERDAY'S SIGNAL
# =============================================================================

def fetch_yesterdays_signal(**context) -> bool:
    """
    Read the most recent signal BEFORE today from forecast_vol_targeting_signals.
    Returns True to continue, False to short-circuit if no signal exists.
    """
    today = datetime.utcnow().date()
    today_str = today.strftime("%Y-%m-%d")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get the most recent signal before today
        cur.execute("""
            SELECT signal_date, forecast_direction, clipped_leverage,
                   forecast_return, position_size, ensemble_models
            FROM forecast_vol_targeting_signals
            WHERE signal_date < %s
            ORDER BY signal_date DESC
            LIMIT 1
        """, (today_str,))

        row = cur.fetchone()
        if row is None:
            logger.info("[L6] No signal found to evaluate. Short-circuiting.")
            return False

        signal_date = row[0]
        signal = {
            "signal_date": str(signal_date),
            "signal_direction": int(row[1]),
            "signal_leverage": float(row[2]),
            "forecast_return": float(row[3]) if row[3] else 0.0,
            "position_size": float(row[4]),
            "ensemble_models": row[5],
        }

        # Check if already evaluated
        cur.execute("""
            SELECT id FROM forecast_paper_trading
            WHERE signal_date = %s
        """, (str(signal_date),))

        if cur.fetchone():
            logger.info(f"[L6] Signal {signal_date} already evaluated. Short-circuiting.")
            return False

        logger.info(
            f"[L6] Evaluating signal from {signal_date}: "
            f"dir={signal['signal_direction']}, lev={signal['signal_leverage']:.3f}"
        )

        context['ti'].xcom_push(key='signal', value=signal)
        return True

    finally:
        conn.close()


# =============================================================================
# TASK 2: FETCH PRICES
# =============================================================================

def fetch_prices(**context) -> Dict[str, Any]:
    """
    Fetch entry_price (close on signal_date) and close_price (close on today)
    from bi.dim_daily_usdcop.
    """
    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='fetch_yesterdays_signal')

    if not signal:
        raise ValueError("[L6] No signal from upstream task")

    signal_date = signal["signal_date"]
    today = datetime.utcnow().date()
    today_str = today.strftime("%Y-%m-%d")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Entry price: close on signal_date (day T)
        cur.execute("""
            SELECT date, close FROM bi.dim_daily_usdcop
            WHERE date = %s AND close IS NOT NULL
        """, (signal_date,))

        entry_row = cur.fetchone()
        if entry_row is None:
            raise ValueError(f"[L6] No close price for signal_date={signal_date}")

        entry_price = float(entry_row[1])

        # Close price: today's close (day T+1) or most recent if today not available yet
        cur.execute("""
            SELECT date, close FROM bi.dim_daily_usdcop
            WHERE date > %s AND close IS NOT NULL
            ORDER BY date ASC
            LIMIT 1
        """, (signal_date,))

        close_row = cur.fetchone()
        if close_row is None:
            raise ValueError(
                f"[L6] No close price found after signal_date={signal_date}. "
                f"OHLCV may not be updated yet."
            )

        close_date = close_row[0]
        close_price = float(close_row[1])

        result = {
            "entry_price": entry_price,
            "entry_date": signal_date,
            "close_price": close_price,
            "close_date": str(close_date),
        }

        logger.info(
            f"[L6] Prices: entry={entry_price:.2f} ({signal_date}), "
            f"close={close_price:.2f} ({close_date})"
        )

        ti.xcom_push(key='prices', value=result)
        return result

    finally:
        conn.close()


# =============================================================================
# TASK 3: COMPUTE PAPER RESULT
# =============================================================================

def compute_paper_result(**context) -> Dict[str, Any]:
    """
    Compute actual return, strategy return, and correctness.
    actual_return = log(close[T+1] / close[T])
    strategy_return = direction * leverage * actual_return
    """
    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='fetch_yesterdays_signal')
    prices = ti.xcom_pull(key='prices', task_ids='fetch_prices')

    if not signal or not prices:
        raise ValueError("[L6] Missing signal or prices")

    entry_price = prices["entry_price"]
    close_price = prices["close_price"]

    # Log-return from signal_date close to next day close
    actual_return = math.log(close_price / entry_price)

    # Strategy return = direction * leverage * actual_return
    strategy_return = signal["signal_direction"] * signal["signal_leverage"] * actual_return

    # Direction correctness
    correct = (
        (signal["signal_direction"] == 1 and actual_return > 0) or
        (signal["signal_direction"] == -1 and actual_return < 0)
    )

    result = {
        "signal_date": signal["signal_date"],
        "signal_direction": signal["signal_direction"],
        "signal_leverage": signal["signal_leverage"],
        "entry_price": entry_price,
        "close_price": close_price,
        "actual_return_1d": actual_return,
        "strategy_return": strategy_return,
        "correct": correct,
    }

    direction_str = "LONG" if signal["signal_direction"] == 1 else "SHORT"
    logger.info(
        f"[L6] Paper result: {direction_str} @ {signal['signal_leverage']:.2f}x, "
        f"actual={actual_return:+.4f}, strategy={strategy_return:+.4f}, "
        f"correct={correct}"
    )

    ti.xcom_push(key='paper_result', value=result)
    return result


# =============================================================================
# TASK 4: PERSIST RESULT
# =============================================================================

def persist_result(**context) -> None:
    """UPSERT paper trading result into forecast_paper_trading."""
    ti = context['ti']
    result = ti.xcom_pull(key='paper_result', task_ids='compute_paper_result')

    if not result:
        raise ValueError("[L6] No paper result to persist")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO forecast_paper_trading (
                signal_date, signal_direction, signal_leverage,
                entry_price, close_price,
                actual_return_1d, strategy_return
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_date) DO UPDATE SET
                signal_direction = EXCLUDED.signal_direction,
                signal_leverage = EXCLUDED.signal_leverage,
                entry_price = EXCLUDED.entry_price,
                close_price = EXCLUDED.close_price,
                actual_return_1d = EXCLUDED.actual_return_1d,
                strategy_return = EXCLUDED.strategy_return,
                created_at = NOW()
            RETURNING id
        """, (
            result["signal_date"],
            result["signal_direction"],
            result["signal_leverage"],
            result["entry_price"],
            result["close_price"],
            result["actual_return_1d"],
            result["strategy_return"],
        ))

        row = cur.fetchone()
        conn.commit()
        logger.info(f"[L6] Persisted paper result id={row[0] if row else '?'}")

    finally:
        conn.close()


# =============================================================================
# TASK 5: COMPUTE RUNNING STATS
# =============================================================================

def compute_running_stats(**context) -> Dict[str, Any]:
    """
    Query all paper trades, compute running DA, Sharpe, MaxDD, cumulative return.
    Update the current row with running stats.
    """
    ti = context['ti']
    result = ti.xcom_pull(key='paper_result', task_ids='compute_paper_result')

    if not result:
        raise ValueError("[L6] No paper result")

    signal_date = result["signal_date"]

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get all paper trades ordered chronologically
        cur.execute("""
            SELECT signal_date, signal_direction, actual_return_1d, strategy_return
            FROM forecast_paper_trading
            WHERE actual_return_1d IS NOT NULL
            ORDER BY signal_date ASC
        """)

        rows = cur.fetchall()
        if not rows:
            logger.warning("[L6] No paper trades to compute stats")
            return {}

        n_days = len(rows)
        strategy_returns = [r[3] for r in rows]
        directions = [r[1] for r in rows]
        actual_returns = [r[2] for r in rows]

        # Direction accuracy
        correct = sum(
            1 for d, a in zip(directions, actual_returns)
            if (d == 1 and a > 0) or (d == -1 and a < 0)
        )
        da_pct = (correct / n_days) * 100.0 if n_days > 0 else 0.0

        # Cumulative return: product of (1 + strategy_return)
        cumulative = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for ret in strategy_returns:
            cumulative *= (1.0 + ret)
            peak = max(peak, cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)

        # Annualized Sharpe (using daily returns)
        if n_days >= 2:
            mean_ret = sum(strategy_returns) / n_days
            variance = sum((r - mean_ret) ** 2 for r in strategy_returns) / (n_days - 1)
            std_ret = math.sqrt(variance) if variance > 0 else 0.0001
            sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        stats = {
            "n_days": n_days,
            "da_pct": round(da_pct, 2),
            "cumulative_return": round(cumulative, 6),
            "running_sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 6),
        }

        # Update the current row with running stats
        cur.execute("""
            UPDATE forecast_paper_trading SET
                cumulative_return = %s,
                running_da_pct = %s,
                running_sharpe = %s,
                running_max_drawdown = %s,
                n_days_traded = %s
            WHERE signal_date = %s
        """, (
            stats["cumulative_return"],
            stats["da_pct"],
            stats["running_sharpe"],
            stats["max_drawdown"],
            stats["n_days"],
            signal_date,
        ))
        conn.commit()

        logger.info(
            f"[L6] Running stats (n={n_days}): DA={da_pct:.1f}%, "
            f"Cumulative={cumulative:.4f}, Sharpe={sharpe:.3f}, "
            f"MaxDD={max_drawdown:.4f}"
        )

        ti.xcom_push(key='running_stats', value=stats)
        return stats

    finally:
        conn.close()


# =============================================================================
# TASK 5b: UPDATE EXECUTION ALPHA (L7 cross-reference)
# =============================================================================

def update_execution_alpha(**context) -> None:
    """
    Fill comparison fields in forecast_executions for closed L7 positions.
    Uses the paper trading result (hold-to-close) as baseline to compute
    execution_alpha_pct = smart executor PnL - hold strategy PnL.

    Only updates rows where daily_close_price IS NULL (not yet filled).
    Safe to run even if L7 is not active (no-op if no closed executions).
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE forecast_executions fe
            SET
                daily_close_price = pt.close_price,
                hold_to_close_pnl = pt.strategy_return,
                execution_alpha_pct = fe.pnl_pct - pt.strategy_return
            FROM forecast_paper_trading pt
            WHERE fe.signal_date = pt.signal_date
              AND fe.status = 'closed'
              AND fe.daily_close_price IS NULL
        """)
        updated = cur.rowcount
        conn.commit()

        if updated > 0:
            logger.info(f"[L6] Updated execution alpha for {updated} closed L7 position(s)")
        else:
            logger.debug("[L6] No L7 executions to update (none pending or L7 not active)")

    finally:
        conn.close()


# =============================================================================
# TASK 6: CHECK STOP CRITERIA
# =============================================================================

def check_stop_criteria(**context) -> None:
    """
    Evaluate paper trading stop criteria from vol_target_v1.yaml.
    CONTINUE if DA > 50% and cumulative_return > 0.
    PAUSE if DA < 48% after 40 days OR MaxDD > 15%.
    STOP if DA < 46% after 60 days OR MaxDD > 20%.
    """
    ti = context['ti']
    stats = ti.xcom_pull(key='running_stats', task_ids='compute_running_stats')

    if not stats:
        logger.warning("[L6] No running stats to check")
        return

    n = stats["n_days"]
    da = stats["da_pct"]
    cum = stats["cumulative_return"]
    sharpe = stats["running_sharpe"]
    maxdd = abs(stats["max_drawdown"]) * 100  # Convert to percentage

    criteria = STOP_CRITERIA

    # Skip criteria check if too few days
    if n < criteria["min_days_before_eval"]:
        logger.info(
            f"[L6] Too early to evaluate ({n}/{criteria['min_days_before_eval']} days). "
            f"Current: DA={da:.1f}%, MaxDD={maxdd:.1f}%"
        )
        return

    # STOP criteria (most severe first)
    if maxdd > criteria["stop_if_maxdd_above"]:
        logger.error(
            f"[L6] STOP: MaxDD {maxdd:.1f}% > {criteria['stop_if_maxdd_above']}% threshold. "
            f"Paper trading should be STOPPED. n={n}, DA={da:.1f}%"
        )
        return

    if n >= criteria["stop_after_days"] and da < criteria["stop_if_da_below"]:
        logger.error(
            f"[L6] STOP: DA {da:.1f}% < {criteria['stop_if_da_below']}% "
            f"after {n} days (threshold={criteria['stop_after_days']}). "
            f"Paper trading should be STOPPED."
        )
        return

    # PAUSE criteria
    if maxdd > criteria["pause_if_maxdd_above"]:
        logger.warning(
            f"[L6] PAUSE: MaxDD {maxdd:.1f}% > {criteria['pause_if_maxdd_above']}% threshold. "
            f"Review recommended. n={n}, DA={da:.1f}%"
        )
        return

    if n >= criteria["pause_after_days"] and da < criteria["pause_if_da_below"]:
        logger.warning(
            f"[L6] PAUSE: DA {da:.1f}% < {criteria['pause_if_da_below']}% "
            f"after {n} days (threshold={criteria['pause_after_days']}). "
            f"Review recommended."
        )
        return

    # CONTINUE criteria
    if da > criteria["continue_if_da_above"] and cum > 1.0:
        logger.info(
            f"[L6] CONTINUE: DA {da:.1f}% > {criteria['continue_if_da_above']}%, "
            f"cumulative={cum:.4f} (positive). n={n}, Sharpe={sharpe:.3f}"
        )
    else:
        logger.info(
            f"[L6] MONITORING: DA={da:.1f}%, cum={cum:.4f}, MaxDD={maxdd:.1f}%, "
            f"Sharpe={sharpe:.3f}, n={n}. No stop/pause criteria triggered."
        )


# =============================================================================
# TASK 7: DAILY SUMMARY
# =============================================================================

def daily_summary(**context) -> None:
    """Log all metrics in structured format. Runs even if upstream fails."""
    ti = context['ti']
    result = ti.xcom_pull(key='paper_result', task_ids='compute_paper_result')
    stats = ti.xcom_pull(key='running_stats', task_ids='compute_running_stats')

    logger.info("=" * 60)
    logger.info("[L6] PAPER TRADING DAILY SUMMARY")
    logger.info("=" * 60)

    if result:
        direction_str = "LONG" if result["signal_direction"] == 1 else "SHORT"
        correct_str = "CORRECT" if result.get("correct") else "WRONG"
        logger.info(f"  Signal Date:    {result['signal_date']}")
        logger.info(f"  Direction:      {direction_str} @ {result['signal_leverage']:.2f}x")
        logger.info(f"  Entry Price:    {result['entry_price']:.2f}")
        logger.info(f"  Close Price:    {result['close_price']:.2f}")
        logger.info(f"  Actual Return:  {result['actual_return_1d']:+.4f}")
        logger.info(f"  Strategy Ret:   {result['strategy_return']:+.4f}")
        logger.info(f"  Prediction:     {correct_str}")
    else:
        logger.info("  No paper result today (no signal to evaluate)")

    if stats:
        logger.info("-" * 40)
        logger.info(f"  Days Traded:    {stats['n_days']}")
        logger.info(f"  DA:             {stats['da_pct']:.1f}%")
        logger.info(f"  Cumulative:     {stats['cumulative_return']:.4f}")
        logger.info(f"  Sharpe:         {stats['running_sharpe']:.3f}")
        logger.info(f"  MaxDD:          {stats['max_drawdown']:.4f}")

    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 15),  # Creation date
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Daily paper trading monitor for vol-targeting signals',
    schedule_interval='0 0 * * 2-6',  # 00:00 UTC Tue-Sat = 19:00 COT Mon-Fri
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS,
) as dag:

    t_fetch_signal = ShortCircuitOperator(
        task_id='fetch_yesterdays_signal',
        python_callable=fetch_yesterdays_signal,
        provide_context=True,
    )

    t_fetch_prices = PythonOperator(
        task_id='fetch_prices',
        python_callable=fetch_prices,
        provide_context=True,
    )

    t_compute_result = PythonOperator(
        task_id='compute_paper_result',
        python_callable=compute_paper_result,
        provide_context=True,
    )

    t_persist = PythonOperator(
        task_id='persist_result',
        python_callable=persist_result,
        provide_context=True,
    )

    t_running_stats = PythonOperator(
        task_id='compute_running_stats',
        python_callable=compute_running_stats,
        provide_context=True,
    )

    t_exec_alpha = PythonOperator(
        task_id='update_execution_alpha',
        python_callable=update_execution_alpha,
        provide_context=True,
    )

    t_stop_check = PythonOperator(
        task_id='check_stop_criteria',
        python_callable=check_stop_criteria,
        provide_context=True,
    )

    t_summary = PythonOperator(
        task_id='daily_summary',
        python_callable=daily_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG flow: sequential evaluation pipeline
    (
        t_fetch_signal
        >> t_fetch_prices
        >> t_compute_result
        >> t_persist
        >> t_running_stats
        >> t_exec_alpha
        >> t_stop_check
        >> t_summary
    )
