"""
DAG: forecast_h5_l6_weekly_monitor
=====================================
Track B — Friday weekly evaluation + decision gate check.

Schedule: Friday 19:30 UTC = 14:30 COT (post-market)
Input: forecast_h5_executions (closed week)
Output: forecast_h5_paper_trading (running metrics + gate decision)

Decision Gates (evaluated at week >= 15):
    DA > 55% -> promote to production
    DA_SHORT > 60% AND DA_LONG > 45% -> keep bidirectional
    DA_SHORT > 60% AND DA_LONG < 40% -> switch SHORT-only
    DA < 50% -> discard

Alarms:
    LONG% > 60% in last 8 weeks -> alert
    Cumulative DD > -12% -> circuit breaker
    5 consecutive losses -> circuit breaker

Contract: FC-H5-L6-001
Version: 1.1.0 (Smart Simple v1.0 — updated guardrail windows)
Date: 2026-02-16
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L6_WEEKLY_MONITOR,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H5_L6_WEEKLY_MONITOR
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')


# =============================================================================
# TASK 1: LOAD COMPLETED WEEK RESULTS
# =============================================================================

def load_results(**context) -> Dict[str, Any]:
    """Load this week's closed execution from forecast_h5_executions."""
    import pandas as pd

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get most recent closed execution
        cur.execute("""
            SELECT id, signal_date, inference_week, inference_year,
                   direction, leverage, week_pnl_pct, week_pnl_unleveraged_pct,
                   n_subtrades, entry_price, exit_price
            FROM forecast_h5_executions
            WHERE status = 'closed'
            ORDER BY signal_date DESC
            LIMIT 1
        """)
        row = cur.fetchone()

        if not row:
            logger.warning("[H5-L6] No closed execution found")
            return {"found": False}

        exec_id, signal_date, week, year, direction, leverage, pnl, pnl_unlev, n_subs, entry_p, exit_p = row

        # Check if already evaluated
        cur.execute("""
            SELECT 1 FROM forecast_h5_paper_trading
            WHERE signal_date = %s
        """, (signal_date,))
        if cur.fetchone():
            logger.info(f"[H5-L6] Week {signal_date} already evaluated")
            return {"found": False, "reason": "already_evaluated"}

        result = {
            "found": True,
            "exec_id": exec_id,
            "signal_date": str(signal_date),
            "inference_week": week,
            "inference_year": year,
            "direction": direction,
            "leverage": leverage,
            "week_pnl_pct": float(pnl or 0),
            "n_subtrades": n_subs or 0,
        }
        context['ti'].xcom_push(key='results', value=result)
        return result

    finally:
        conn.close()


# =============================================================================
# TASK 2: COMPUTE RUNNING METRICS
# =============================================================================

def compute_metrics(**context) -> Dict[str, Any]:
    """Compute running cumulative PnL, DA, Sharpe, drawdown."""
    import numpy as np

    ti = context['ti']
    results = ti.xcom_pull(key='results', task_ids='load_results')

    if not results or not results.get("found"):
        return {"computed": False}

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Load all historical week PnLs
        cur.execute("""
            SELECT signal_date, direction, week_pnl_pct
            FROM forecast_h5_executions
            WHERE status = 'closed'
            ORDER BY signal_date ASC
        """)
        rows = cur.fetchall()

        dates = [r[0] for r in rows]
        directions = [r[1] for r in rows]
        pnls = [float(r[2] or 0) for r in rows]
        n_weeks = len(pnls)

        # Cumulative PnL (compounding)
        cumulative = 0.0
        for p in pnls:
            cumulative += p  # Simple sum for paper trading
        cumulative_pnl = cumulative

        # Direction accuracy (actual return direction matches signal direction)
        correct = sum(1 for d, p in zip(directions, pnls) if (d * p) > 0)
        running_da = (correct / n_weeks * 100) if n_weeks > 0 else 0.0

        # DA by direction
        short_weeks = [(d, p) for d, p in zip(directions, pnls) if d == -1]
        long_weeks = [(d, p) for d, p in zip(directions, pnls) if d == 1]

        short_correct = sum(1 for d, p in short_weeks if p > 0) if short_weeks else 0
        long_correct = sum(1 for d, p in long_weeks if p > 0) if long_weeks else 0
        da_short = (short_correct / len(short_weeks) * 100) if short_weeks else None
        da_long = (long_correct / len(long_weeks) * 100) if long_weeks else None

        # Running Sharpe (annualized from weekly returns)
        if n_weeks >= 4 and np.std(pnls) > 0:
            running_sharpe = float(np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(52))
        else:
            running_sharpe = None

        # Max drawdown
        equity = [0.0]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            dd = e - peak
            max_dd = min(max_dd, dd)

        # L/S ratio in last 8 weeks
        recent_directions = directions[-8:] if len(directions) >= 8 else directions
        n_long_recent = sum(1 for d in recent_directions if d == 1)
        long_pct_8w = (n_long_recent / len(recent_directions) * 100) if recent_directions else 0

        # Consecutive losses
        consecutive_losses = 0
        for p in reversed(pnls):
            if p < 0:
                consecutive_losses += 1
            else:
                break

        # L/S totals
        n_long = sum(1 for d in directions if d == 1)
        n_short = sum(1 for d in directions if d == -1)

        metrics = {
            "n_weeks": n_weeks,
            "cumulative_pnl_pct": round(cumulative_pnl, 4),
            "running_da_pct": round(running_da, 1),
            "running_da_short_pct": round(da_short, 1) if da_short is not None else None,
            "running_da_long_pct": round(da_long, 1) if da_long is not None else None,
            "running_sharpe": round(running_sharpe, 3) if running_sharpe is not None else None,
            "running_max_dd_pct": round(max_dd, 4),
            "n_long": n_long,
            "n_short": n_short,
            "long_pct_8w": round(long_pct_8w, 1),
            "consecutive_losses": consecutive_losses,
        }
        context['ti'].xcom_push(key='metrics', value=metrics)
        return metrics

    finally:
        conn.close()


# =============================================================================
# TASK 3: CHECK DECISION GATES
# =============================================================================

def check_gates(**context) -> Dict[str, Any]:
    """
    Evaluate decision gates at week >= 15.
    Returns gate_status and any alarms.
    """
    import yaml

    ti = context['ti']
    results = ti.xcom_pull(key='results', task_ids='load_results')
    metrics = ti.xcom_pull(key='metrics', task_ids='compute_metrics')

    if not results or not results.get("found") or not metrics:
        return {"evaluated": False}

    # Load config for thresholds
    config_path = PROJECT_ROOT / 'config' / 'execution' / 'smart_simple_v1.yaml'
    with open(config_path) as f:
        h5_config = yaml.safe_load(f)

    gates = h5_config.get("gates", {})
    min_weeks = gates.get("evaluation_week", 15)
    n_weeks = metrics["n_weeks"]

    alarms = []
    gate_status = None
    circuit_breaker = False

    # L/S ratio alarm (Smart Simple: 8 weeks window)
    guardrails = h5_config.get("guardrails", {})
    ls_alarm = guardrails.get("long_insistence_alarm", {})
    ls_window = ls_alarm.get("window_weeks", 8)
    ls_threshold = ls_alarm.get("threshold_pct", 60)
    if metrics["long_pct_8w"] > ls_threshold:
        alarms.append(f"LONG% in last {ls_window} weeks = {metrics['long_pct_8w']:.0f}% > {ls_threshold}%")
        logger.warning(f"[H5-L6] L/S ALARM: {alarms[-1]}")

    # Circuit breaker check
    cb = guardrails.get("circuit_breaker", {})
    max_dd = -abs(cb.get("max_drawdown_pct", 12.0))
    if metrics["running_max_dd_pct"] <= max_dd:
        circuit_breaker = True
        alarms.append(f"Cumulative DD {metrics['running_max_dd_pct']:.2f}% <= {max_dd}%")
        logger.warning(f"[H5-L6] CIRCUIT BREAKER: {alarms[-1]}")

    if metrics["consecutive_losses"] >= cb.get("max_consecutive_losses", 5):
        circuit_breaker = True
        alarms.append(f"{metrics['consecutive_losses']} consecutive losses >= {cb['max_consecutive_losses']}")
        logger.warning(f"[H5-L6] CIRCUIT BREAKER: {alarms[-1]}")

    # Decision gates (only at week >= min_weeks)
    if n_weeks >= min_weeks:
        da = metrics["running_da_pct"]
        da_short = metrics.get("running_da_short_pct")
        da_long = metrics.get("running_da_long_pct")

        promote_da = gates.get("promote_threshold", {}).get("da_overall", 55.0)
        discard_da = gates.get("discard_threshold", {}).get("da_overall", 50.0)
        keep_short_da = gates.get("keep_conditions", {}).get("da_short_min", 60.0)
        keep_long_da = gates.get("keep_conditions", {}).get("da_long_min", 45.0)
        switch_short_da = keep_short_da   # Same threshold for switch
        switch_long_max = keep_long_da    # Below this → switch to SHORT-only

        if da >= promote_da:
            gate_status = "promote"
        elif da_short and da_long and da_short >= keep_short_da and da_long >= keep_long_da:
            gate_status = "keep"
        elif da_short and da_long and da_short >= switch_short_da and da_long < switch_long_max:
            gate_status = "switch_short"
        elif da < discard_da:
            gate_status = "discard"
        else:
            gate_status = "continue"

        logger.info(
            f"[H5-L6] GATE CHECK (week {n_weeks}): DA={da:.1f}%, "
            f"DA_SHORT={da_short}, DA_LONG={da_long} -> {gate_status}"
        )
    else:
        logger.info(f"[H5-L6] Week {n_weeks} < {min_weeks}, gates not evaluated yet")

    gate_result = {
        "evaluated": n_weeks >= min_weeks,
        "gate_status": gate_status,
        "circuit_breaker": circuit_breaker,
        "alarms": alarms,
    }
    context['ti'].xcom_push(key='gates', value=gate_result)
    return gate_result


# =============================================================================
# TASK 4: PERSIST TO PAPER TRADING TABLE
# =============================================================================

def persist_evaluation(**context) -> Dict[str, Any]:
    """Persist weekly evaluation to forecast_h5_paper_trading."""
    ti = context['ti']
    results = ti.xcom_pull(key='results', task_ids='load_results')
    metrics = ti.xcom_pull(key='metrics', task_ids='compute_metrics')
    gates = ti.xcom_pull(key='gates', task_ids='check_gates')

    if not results or not results.get("found"):
        return {"persisted": False}

    notes_parts = []
    if gates and gates.get("alarms"):
        notes_parts.extend(gates["alarms"])
    if gates and gates.get("gate_status"):
        notes_parts.append(f"Gate: {gates['gate_status']}")
    notes = "; ".join(notes_parts) if notes_parts else None

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO forecast_h5_paper_trading
            (signal_date, inference_week, inference_year, direction, leverage,
             week_pnl_pct, n_subtrades, cumulative_pnl_pct,
             running_da_pct, running_da_short_pct, running_da_long_pct,
             running_sharpe, running_max_dd_pct, n_weeks, n_long, n_short,
             long_pct_8w, consecutive_losses, circuit_breaker, gate_status, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_date) DO UPDATE SET
                week_pnl_pct = EXCLUDED.week_pnl_pct,
                cumulative_pnl_pct = EXCLUDED.cumulative_pnl_pct,
                running_da_pct = EXCLUDED.running_da_pct,
                gate_status = EXCLUDED.gate_status,
                notes = EXCLUDED.notes
        """, (
            results["signal_date"],
            results["inference_week"],
            results["inference_year"],
            results["direction"],
            results["leverage"],
            results["week_pnl_pct"],
            results["n_subtrades"],
            metrics.get("cumulative_pnl_pct", 0) if metrics else 0,
            metrics.get("running_da_pct") if metrics else None,
            metrics.get("running_da_short_pct") if metrics else None,
            metrics.get("running_da_long_pct") if metrics else None,
            metrics.get("running_sharpe") if metrics else None,
            metrics.get("running_max_dd_pct") if metrics else None,
            metrics.get("n_weeks", 0) if metrics else 0,
            metrics.get("n_long", 0) if metrics else 0,
            metrics.get("n_short", 0) if metrics else 0,
            metrics.get("long_pct_8w") if metrics else None,
            metrics.get("consecutive_losses", 0) if metrics else 0,
            gates.get("circuit_breaker", False) if gates else False,
            gates.get("gate_status") if gates else None,
            notes,
        ))
        conn.commit()
        logger.info(f"[H5-L6] Persisted evaluation for {results['signal_date']}")
        return {"persisted": True}
    finally:
        conn.close()


# =============================================================================
# TASK 5: ALERT SUMMARY
# =============================================================================

def alert_summary(**context) -> None:
    """Log weekly summary with alarms."""
    ti = context['ti']
    results = ti.xcom_pull(key='results', task_ids='load_results')
    metrics = ti.xcom_pull(key='metrics', task_ids='compute_metrics')
    gates = ti.xcom_pull(key='gates', task_ids='check_gates')

    logger.info("=" * 60)
    logger.info("[H5-L6] WEEKLY EVALUATION SUMMARY")
    logger.info("=" * 60)

    if not results or not results.get("found"):
        logger.info("  No completed week to evaluate")
        logger.info("=" * 60)
        return

    dir_str = "LONG" if results["direction"] == 1 else "SHORT"
    logger.info(f"  Week:             {results['signal_date']}")
    logger.info(f"  Direction:        {dir_str}")
    logger.info(f"  Week PnL:         {results['week_pnl_pct']:+.4f}%")
    logger.info(f"  Subtrades:        {results['n_subtrades']}")

    if metrics:
        logger.info(f"  Cumulative PnL:   {metrics.get('cumulative_pnl_pct', 0):+.4f}%")
        logger.info(f"  Running DA:       {metrics.get('running_da_pct', 0):.1f}%")
        logger.info(f"  DA SHORT:         {metrics.get('running_da_short_pct', '-')}")
        logger.info(f"  DA LONG:          {metrics.get('running_da_long_pct', '-')}")
        logger.info(f"  Sharpe:           {metrics.get('running_sharpe', '-')}")
        logger.info(f"  Max DD:           {metrics.get('running_max_dd_pct', 0):.4f}%")
        logger.info(f"  Weeks:            {metrics.get('n_weeks', 0)} ({metrics.get('n_long', 0)}L/{metrics.get('n_short', 0)}S)")
        logger.info(f"  LONG% (8w):       {metrics.get('long_pct_8w', 0):.0f}%")
        logger.info(f"  Consec losses:    {metrics.get('consecutive_losses', 0)}")

    if gates:
        if gates.get("gate_status"):
            logger.info(f"  GATE DECISION:    {gates['gate_status']}")
        if gates.get("circuit_breaker"):
            logger.warning("  *** CIRCUIT BREAKER TRIGGERED ***")
        for alarm in gates.get("alarms", []):
            logger.warning(f"  ALARM: {alarm}")

    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 16),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Track B: Friday H=5 weekly evaluation + decision gates (14:30 COT)',
    schedule_interval='30 19 * * 5',  # Vie 19:30 UTC = 14:30 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_load = PythonOperator(
        task_id='load_results',
        python_callable=load_results,
    )

    t_metrics = PythonOperator(
        task_id='compute_metrics',
        python_callable=compute_metrics,
    )

    t_gates = PythonOperator(
        task_id='check_gates',
        python_callable=check_gates,
    )

    t_persist = PythonOperator(
        task_id='persist_evaluation',
        python_callable=persist_evaluation,
    )

    t_alert = PythonOperator(
        task_id='alert_summary',
        python_callable=alert_summary,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_load >> t_metrics >> t_gates >> t_persist >> t_alert
