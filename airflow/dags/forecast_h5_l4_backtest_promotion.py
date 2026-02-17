"""
DAG: forecast_h5_l4_backtest_promotion
=======================================
H5 Weekly Smart Simple — L4 Backtest Promotion DAG.

Automates the OOS backtest + dashboard export + approval gates workflow
that was previously manual (train_and_export_smart_simple.py --phase backtest).

Architecture:
    Manual trigger / Airflow UI
        |
        v
    validate_data       — Check OHLCV + macro freshness, config exists
        |
        v
    run_backtest        — Call train_and_export_smart_simple.py --phase backtest
        |
        v
    validate_output     — Check summary_2025.json + approval_state.json exist
        |
        v
    report_metrics      — Log results, push metrics to XCom
        |
        v
    notify              — Log that strategy is ready for review on /dashboard

Schedule: None (manual trigger)
Output: Dashboard files in public/data/production/ (summary, approval_state, trades)
Downstream: Operator reviews on /dashboard -> clicks Approve (Vote 2/2)

Author: Trading Team
Version: 1.0.0
Date: 2026-02-17
Contract: FC-H5-L4-001
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import json
import logging
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L4_BACKTEST_PROMOTION,
    get_dag_tags,
)

DAG_ID = FORECAST_H5_L4_BACKTEST_PROMOTION
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')
DASHBOARD_DATA_DIR = PROJECT_ROOT / 'usdcop-trading-dashboard' / 'public' / 'data' / 'production'
BACKTEST_SCRIPT = PROJECT_ROOT / 'scripts' / 'train_and_export_smart_simple.py'
CONFIG_PATH = PROJECT_ROOT / 'config' / 'execution' / 'smart_simple_v1.yaml'
OHLCV_PATH = PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_daily_ohlcv.parquet'
MACRO_PATH = PROJECT_ROOT / 'data' / 'pipeline' / '04_cleaning' / 'output' / 'MACRO_DAILY_CLEAN.parquet'


# =============================================================================
# TASK 1: VALIDATE DATA
# =============================================================================

def validate_data(**context) -> Dict[str, Any]:
    """
    Pre-flight checks: verify that all required inputs exist and are fresh.
    """
    errors = []

    # Check OHLCV seed
    if not OHLCV_PATH.exists():
        errors.append(f"OHLCV seed not found: {OHLCV_PATH}")
    else:
        import os
        age_hours = (datetime.now().timestamp() - os.path.getmtime(OHLCV_PATH)) / 3600
        logger.info(f"[H5-L4] OHLCV seed age: {age_hours:.1f} hours")
        if age_hours > 7 * 24:  # Warn if older than 1 week
            logger.warning(f"[H5-L4] OHLCV seed is {age_hours / 24:.1f} days old")

    # Check macro data
    if not MACRO_PATH.exists():
        errors.append(f"Macro data not found: {MACRO_PATH}")
    else:
        import os
        age_hours = (datetime.now().timestamp() - os.path.getmtime(MACRO_PATH)) / 3600
        logger.info(f"[H5-L4] Macro data age: {age_hours:.1f} hours")

    # Check config
    if not CONFIG_PATH.exists():
        errors.append(f"Smart Simple config not found: {CONFIG_PATH}")
    else:
        logger.info(f"[H5-L4] Config found: {CONFIG_PATH}")

    # Check backtest script
    if not BACKTEST_SCRIPT.exists():
        errors.append(f"Backtest script not found: {BACKTEST_SCRIPT}")

    if errors:
        raise ValueError(f"[H5-L4] Pre-flight validation failed:\n" + "\n".join(errors))

    result = {
        "ohlcv_path": str(OHLCV_PATH),
        "macro_path": str(MACRO_PATH),
        "config_path": str(CONFIG_PATH),
        "script_path": str(BACKTEST_SCRIPT),
    }
    context['ti'].xcom_push(key='validation', value=result)
    logger.info("[H5-L4] Pre-flight validation PASSED")
    return result


# =============================================================================
# TASK 2: RUN BACKTEST
# =============================================================================

def run_backtest(**context) -> Dict[str, Any]:
    """
    Execute train_and_export_smart_simple.py --phase backtest.
    This trains models, runs OOS backtest, evaluates gates, and exports to dashboard.
    """
    cmd = [
        sys.executable,
        str(BACKTEST_SCRIPT),
        "--phase", "backtest",
    ]

    logger.info(f"[H5-L4] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
        cwd=str(PROJECT_ROOT),
    )

    # Log output
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-20:]:  # Last 20 lines
            logger.info(f"[H5-L4] stdout: {line}")
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-10:]:  # Last 10 lines
            logger.warning(f"[H5-L4] stderr: {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"[H5-L4] Backtest script failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr[-500:] if result.stderr else 'none'}"
        )

    run_result = {
        "exit_code": result.returncode,
        "stdout_lines": len(result.stdout.split('\n')) if result.stdout else 0,
    }
    context['ti'].xcom_push(key='backtest_run', value=run_result)
    logger.info("[H5-L4] Backtest script completed successfully")
    return run_result


# =============================================================================
# TASK 3: VALIDATE OUTPUT
# =============================================================================

def validate_output(**context) -> Dict[str, Any]:
    """
    Verify that the backtest script produced valid output files.
    """
    errors = []
    files_checked = {}

    # Check summary_2025.json
    summary_path = DASHBOARD_DATA_DIR / 'summary_2025.json'
    if not summary_path.exists():
        errors.append(f"summary_2025.json not found: {summary_path}")
    else:
        with open(summary_path) as f:
            content = f.read()
            if 'Infinity' in content or 'NaN' in content:
                errors.append("summary_2025.json contains Infinity or NaN (JSON safety violation)")
            summary = json.loads(content)
            files_checked['summary_2025'] = True
            logger.info(f"[H5-L4] summary_2025.json: strategy={summary.get('strategy_id')}")

    # Check approval_state.json
    approval_path = DASHBOARD_DATA_DIR / 'approval_state.json'
    if not approval_path.exists():
        errors.append(f"approval_state.json not found: {approval_path}")
    else:
        with open(approval_path) as f:
            approval = json.loads(f.read())
            status = approval.get('status')
            if status != 'PENDING_APPROVAL':
                errors.append(f"Expected status=PENDING_APPROVAL, got {status}")
            files_checked['approval_state'] = True
            logger.info(f"[H5-L4] approval_state.json: status={status}")

    # Check trade file
    # Infer strategy_id from summary
    strategy_id = summary.get('strategy_id', 'smart_simple_v11') if 'summary' in dir() else 'smart_simple_v11'
    trades_path = DASHBOARD_DATA_DIR / 'trades' / f'{strategy_id}_2025.json'
    if not trades_path.exists():
        errors.append(f"Trade file not found: {trades_path}")
    else:
        with open(trades_path) as f:
            trades_data = json.loads(f.read())
            n_trades = len(trades_data.get('trades', []))
            files_checked['trades'] = True
            logger.info(f"[H5-L4] trades: {n_trades} trades for {strategy_id}")

    if errors:
        raise ValueError(f"[H5-L4] Output validation failed:\n" + "\n".join(errors))

    result = {
        "files_checked": files_checked,
        "strategy_id": strategy_id,
        "all_valid": True,
    }
    context['ti'].xcom_push(key='output_validation', value=result)
    logger.info("[H5-L4] Output validation PASSED")
    return result


# =============================================================================
# TASK 4: REPORT METRICS
# =============================================================================

def report_metrics(**context) -> Dict[str, Any]:
    """
    Extract and log backtest metrics from summary_2025.json.
    Push to XCom for downstream use.
    """
    summary_path = DASHBOARD_DATA_DIR / 'summary_2025.json'
    approval_path = DASHBOARD_DATA_DIR / 'approval_state.json'

    with open(summary_path) as f:
        summary = json.loads(f.read())

    with open(approval_path) as f:
        approval = json.loads(f.read())

    strategy_id = summary.get('strategy_id', 'unknown')
    stats = summary.get('strategies', {}).get(strategy_id, {})
    tests = summary.get('statistical_tests', {})
    gates = approval.get('gates', [])
    recommendation = approval.get('backtest_recommendation', 'UNKNOWN')

    metrics = {
        "strategy_id": strategy_id,
        "strategy_name": summary.get('strategy_name', 'Unknown'),
        "return_pct": stats.get('total_return_pct'),
        "sharpe": stats.get('sharpe'),
        "max_dd_pct": stats.get('max_dd_pct'),
        "win_rate_pct": stats.get('win_rate_pct'),
        "trades": stats.get('trading_days'),
        "p_value": tests.get('p_value'),
        "significant": tests.get('significant', False),
        "recommendation": recommendation,
        "gates_passed": sum(1 for g in gates if g.get('passed')),
        "gates_total": len(gates),
    }

    # Log structured report
    logger.info("=" * 60)
    logger.info("[H5-L4] BACKTEST PROMOTION REPORT")
    logger.info("=" * 60)
    logger.info(f"  Strategy:     {metrics['strategy_name']} ({strategy_id})")
    logger.info(f"  Return:       {metrics['return_pct']}%")
    logger.info(f"  Sharpe:       {metrics['sharpe']}")
    logger.info(f"  MaxDD:        {metrics['max_dd_pct']}%")
    logger.info(f"  Win Rate:     {metrics['win_rate_pct']}%")
    logger.info(f"  p-value:      {metrics['p_value']} {'(SIGNIFICANT)' if metrics['significant'] else '(not significant)'}")
    logger.info(f"  Trades:       {metrics['trades']}")
    logger.info(f"  Gates:        {metrics['gates_passed']}/{metrics['gates_total']}")
    logger.info(f"  Recommendation: {recommendation}")
    for g in gates:
        status = "PASS" if g.get('passed') else "FAIL"
        logger.info(f"    [{status}] {g.get('label', g.get('gate'))}: {g.get('value')} (threshold: {g.get('threshold')})")
    logger.info("=" * 60)

    context['ti'].xcom_push(key='backtest_metrics', value=metrics)
    context['ti'].xcom_push(key='gate_results', value=gates)
    context['ti'].xcom_push(key='recommendation', value=recommendation)
    return metrics


# =============================================================================
# TASK 5: NOTIFY
# =============================================================================

def notify(**context) -> None:
    """
    Log notification that strategy is ready for human review.
    """
    ti = context['ti']
    metrics = ti.xcom_pull(key='backtest_metrics', task_ids='report_metrics')
    recommendation = ti.xcom_pull(key='recommendation', task_ids='report_metrics')

    strategy_name = metrics.get('strategy_name', 'Unknown') if metrics else 'Unknown'

    logger.info("=" * 60)
    logger.info("[H5-L4] NOTIFICATION: Strategy ready for review")
    logger.info(f"  Strategy: {strategy_name}")
    logger.info(f"  Recommendation: {recommendation}")
    logger.info(f"  Review at: /dashboard (Vote 2/2 required)")
    logger.info(f"  Action: Navigate to /dashboard and click Approve or Reject")
    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=20),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='H5 Smart Simple — OOS backtest + dashboard export + approval gates (Vote 1/2)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )

    t_run_backtest = PythonOperator(
        task_id='run_backtest',
        python_callable=run_backtest,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )

    t_validate_output = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        provide_context=True,
    )

    t_report_metrics = PythonOperator(
        task_id='report_metrics',
        python_callable=report_metrics,
        provide_context=True,
    )

    t_notify = PythonOperator(
        task_id='notify',
        python_callable=notify,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG flow: validate -> backtest -> validate output -> report -> notify
    t_validate_data >> t_run_backtest >> t_validate_output >> t_report_metrics >> t_notify
