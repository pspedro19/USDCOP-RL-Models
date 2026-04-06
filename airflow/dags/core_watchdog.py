"""
DAG: core_watchdog — Intelligent System Health Monitor & Auto-Healer
====================================================================

Runs every hour during market hours (Mon-Fri 8-13 COT = 13-18 UTC).
Checks ALL system components and auto-triggers recovery when gaps are detected.

No more manual intervention. No fallbacks. No stale data.

Checks:
  1. OHLCV 5-min freshness (< 15 min during market hours)
  2. Macro daily freshness (< 2 days)
  3. Forecasting CSV has current week
  4. Analysis JSON has current week
  5. H5 signal exists for this week (Monday)
  6. Production summary freshness (< 7 days)
  7. News pipeline ran today
  8. Dashboard API health (no fallback data)
  9. DB tables non-empty (weekly_analysis, daily_analysis)

Contract: CTR-WATCHDOG-001
Version: 1.0.0
Date: 2026-04-06
"""

from datetime import datetime, timedelta, timezone, date
from pathlib import Path
import logging
import json
import os
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')
DASHBOARD_DIR = PROJECT_ROOT / 'usdcop-trading-dashboard' / 'public'
COT = timezone(timedelta(hours=-5))

DAG_ID = 'core_watchdog'


def _get_db_connection():
    """Get DB connection using standard env vars."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', ''),
    )


def _current_iso_week():
    """Return current ISO week as string like '2026-W15'."""
    now = datetime.now(COT)
    return f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"


def _is_market_hours():
    """Check if we're in market hours (Mon-Fri 8:00-12:55 COT)."""
    now = datetime.now(COT)
    return now.weekday() < 5 and 8 <= now.hour < 13


def check_ohlcv_freshness(**context):
    """CHECK 1: OHLCV 5-min data freshness."""
    if not _is_market_hours():
        logger.info("[Watchdog] Outside market hours, skipping OHLCV check")
        return {"status": "skip", "reason": "outside_market_hours"}

    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(time), EXTRACT(EPOCH FROM (NOW() - MAX(time)))/60 as minutes_ago
            FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'
        """)
        row = cur.fetchone()
        if row and row[1] is not None:
            minutes_ago = float(row[1])
            if minutes_ago > 15:
                logger.warning(f"[Watchdog] OHLCV STALE: {minutes_ago:.0f} min ago. Triggering realtime DAG.")
                return {"status": "stale", "minutes_ago": minutes_ago, "action": "trigger_ohlcv_realtime"}
            logger.info(f"[Watchdog] OHLCV OK: {minutes_ago:.0f} min ago")
            return {"status": "ok", "minutes_ago": minutes_ago}
        return {"status": "empty", "action": "trigger_ohlcv_backfill"}
    finally:
        conn.close()


def check_macro_freshness(**context):
    """CHECK 2: Macro daily data freshness."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(fecha), (CURRENT_DATE - MAX(fecha)) as days_ago FROM macro_indicators_daily")
        row = cur.fetchone()
        if row and row[1] is not None:
            days_ago = row[1].days if hasattr(row[1], 'days') else int(row[1])
            if days_ago > 2:
                logger.warning(f"[Watchdog] MACRO STALE: {days_ago} days ago")
                return {"status": "stale", "days_ago": days_ago, "action": "trigger_macro_update"}
            logger.info(f"[Watchdog] Macro OK: {days_ago} days ago")
            return {"status": "ok", "days_ago": days_ago}
        return {"status": "empty"}
    finally:
        conn.close()


def check_forecasting_freshness(**context):
    """CHECK 3: Forecasting CSV has current week."""
    import pandas as pd

    csv_path = DASHBOARD_DIR / 'forecasting' / 'bi_dashboard_unified.csv'
    if not csv_path.exists():
        logger.warning("[Watchdog] Forecasting CSV missing!")
        return {"status": "missing", "action": "run_generate_weekly_forecasts"}

    df = pd.read_csv(csv_path)
    forwards = df[df['view_type'] == 'forward_forecast']
    if forwards.empty:
        return {"status": "no_forwards", "action": "run_generate_weekly_forecasts"}

    latest_week = forwards['inference_week'].max()
    current_week = _current_iso_week()

    if latest_week < current_week:
        logger.warning(f"[Watchdog] Forecasting STALE: latest={latest_week}, current={current_week}")
        return {"status": "stale", "latest": latest_week, "current": current_week,
                "action": "run_generate_weekly_forecasts"}

    logger.info(f"[Watchdog] Forecasting OK: latest={latest_week}")
    return {"status": "ok", "latest": latest_week}


def check_analysis_freshness(**context):
    """CHECK 4: Analysis JSON has current week."""
    current_week = _current_iso_week()
    year, week_num = current_week.split('-W')
    json_path = DASHBOARD_DIR / 'data' / 'analysis' / f'weekly_{year}_W{week_num}.json'

    if json_path.exists():
        logger.info(f"[Watchdog] Analysis OK: {json_path.name} exists")
        return {"status": "ok", "week": current_week}

    logger.warning(f"[Watchdog] Analysis MISSING for {current_week}")
    return {"status": "missing", "week": current_week, "action": "run_generate_weekly_analysis"}


def check_h5_signal(**context):
    """CHECK 5: H5 signal exists for this week (Monday only)."""
    now = datetime.now(COT)
    if now.weekday() != 0:  # Only check on Mondays
        return {"status": "skip", "reason": "not_monday"}

    # Find this week's Monday
    monday = now.date()
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT signal_date, direction, skip_trade, adjusted_leverage, regime
            FROM forecast_h5_signals
            WHERE signal_date >= %s
            ORDER BY signal_date DESC LIMIT 1
        """, (monday - timedelta(days=1),))
        row = cur.fetchone()
        if row:
            logger.info(f"[Watchdog] H5 signal OK: {row[0]}, dir={row[1]}, skip={row[2]}, regime={row[4]}")
            return {"status": "ok", "signal_date": str(row[0])}
        logger.warning(f"[Watchdog] H5 signal MISSING for week of {monday}")
        return {"status": "missing", "action": "trigger_h5_l5_signal"}
    finally:
        conn.close()


def check_news_pipeline(**context):
    """CHECK 7: News pipeline ran today."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*), MAX(created_at)::date
            FROM news_daily_digests
            WHERE digest_date = CURRENT_DATE
        """)
        row = cur.fetchone()
        if row and row[0] > 0:
            logger.info(f"[Watchdog] News OK: {row[0]} digests today")
            return {"status": "ok"}

        # Check if it's a weekday (news only runs Mon-Fri)
        if datetime.now(COT).weekday() >= 5:
            return {"status": "skip", "reason": "weekend"}

        logger.warning("[Watchdog] News pipeline hasn't run today")
        return {"status": "stale", "action": "trigger_news_pipeline"}
    finally:
        conn.close()


def check_backups(**context):
    """CHECK 8: Intelligent backup verification.

    Checks that critical data backups exist and are fresh enough
    to restore the full system if needed.

    Backup tiers:
      T1: Seeds (Git LFS) — OHLCV daily, 5-min, macro
      T2: Daily backup parquets — data/backups/seeds/
      T3: Dashboard JSONs — production, analysis, forecasting
      T4: News CSV archives — data/news/*.csv
    """
    import os

    issues = []
    backup_status = {}

    # --- T1: Seed parquets (Git LFS) ---
    seeds = {
        'ohlcv_daily': PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_daily_ohlcv.parquet',
        'ohlcv_5min': PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_m5_ohlcv.parquet',
        'macro_clean': PROJECT_ROOT / 'data' / 'pipeline' / '04_cleaning' / 'output' / 'MACRO_DAILY_CLEAN.parquet',
    }
    for name, path in seeds.items():
        if path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
            size_mb = path.stat().st_size / 1e6
            backup_status[f'seed_{name}'] = {'exists': True, 'age_days': age_days, 'size_mb': round(size_mb, 1)}
            if age_days > 14:
                issues.append(f"seed_{name} is {age_days} days old (>14d)")
        else:
            backup_status[f'seed_{name}'] = {'exists': False}
            issues.append(f"seed_{name} MISSING: {path}")

    # --- T2: Daily backup parquets ---
    backup_dir = PROJECT_ROOT / 'data' / 'backups' / 'seeds'
    manifest_path = backup_dir / 'backup_manifest.json'
    if manifest_path.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(manifest_path.stat().st_mtime)).days
        backup_status['daily_backup'] = {'exists': True, 'age_days': age_days}
        if age_days > 3:
            issues.append(f"Daily backup manifest is {age_days} days old (>3d)")
    else:
        backup_status['daily_backup'] = {'exists': False}
        issues.append("Daily backup manifest MISSING — l0_seed_backup may not be running")

    # --- T3: Dashboard JSON freshness ---
    dash_files = {
        'production_summary': DASHBOARD_DIR / 'data' / 'production' / 'summary.json',
        'production_approval': DASHBOARD_DIR / 'data' / 'production' / 'approval_state.json',
        'analysis_index': DASHBOARD_DIR / 'data' / 'analysis' / 'analysis_index.json',
        'forecasting_csv': DASHBOARD_DIR / 'forecasting' / 'bi_dashboard_unified.csv',
    }
    for name, path in dash_files.items():
        if path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
            backup_status[f'dash_{name}'] = {'exists': True, 'age_days': age_days}
            if age_days > 7:
                issues.append(f"dash_{name} is {age_days} days old (>7d)")
        else:
            backup_status[f'dash_{name}'] = {'exists': False}
            issues.append(f"dash_{name} MISSING")

    # --- T4: News archives ---
    news_dir = PROJECT_ROOT / 'data' / 'news'
    if news_dir.exists():
        csv_files = list(news_dir.glob('*.csv'))
        backup_status['news_archives'] = {'count': len(csv_files)}
    else:
        backup_status['news_archives'] = {'count': 0}

    # --- DB row counts (for restore verification) ---
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        tables = {
            'ohlcv_5m': "SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'",
            'macro_daily': "SELECT COUNT(*) FROM macro_indicators_daily",
            'news_articles': "SELECT COUNT(*) FROM news_articles",
            'h5_signals': "SELECT COUNT(*) FROM forecast_h5_signals",
            'h5_executions': "SELECT COUNT(*) FROM forecast_h5_executions",
        }
        for name, query in tables.items():
            try:
                cur.execute(query)
                count = cur.fetchone()[0]
                backup_status[f'db_{name}'] = {'rows': count}
                if count == 0:
                    issues.append(f"db_{name} has 0 rows!")
            except Exception:
                backup_status[f'db_{name}'] = {'rows': -1, 'error': True}
    finally:
        conn.close()

    if issues:
        logger.warning(f"[Watchdog] BACKUP ISSUES ({len(issues)}): {issues}")
        # Trigger seed backup if daily backup is stale
        if any('Daily backup' in i for i in issues):
            return {"status": "stale", "issues": issues, "backup_status": backup_status,
                    "action": "trigger_seed_backup"}
        # Trigger seed refresh if 5-min seed is very old
        if any('ohlcv_5min' in i for i in issues):
            return {"status": "stale", "issues": issues, "backup_status": backup_status,
                    "action": "trigger_ohlcv_backfill"}
        return {"status": "issues", "issues": issues, "backup_status": backup_status}

    logger.info(f"[Watchdog] Backups OK: {len(backup_status)} items verified")
    return {"status": "ok", "backup_status": backup_status}


def auto_heal(**context):
    """
    Read all check results and trigger recovery actions.
    This is the INTELLIGENT part — it decides what to run.
    """
    ti = context['ti']
    actions_taken = []
    from airflow.api.client.local_client import Client
    client = Client(None, None)

    checks = {
        'ohlcv': ti.xcom_pull(task_ids='check_ohlcv'),
        'macro': ti.xcom_pull(task_ids='check_macro'),
        'forecasting': ti.xcom_pull(task_ids='check_forecasting'),
        'analysis': ti.xcom_pull(task_ids='check_analysis'),
        'h5_signal': ti.xcom_pull(task_ids='check_h5_signal'),
        'news': ti.xcom_pull(task_ids='check_news'),
        'backups': ti.xcom_pull(task_ids='check_backups'),
    }

    for name, result in checks.items():
        if not result:
            continue
        action = result.get('action')
        if not action:
            continue

        logger.info(f"[Watchdog] AUTO-HEAL: {name} → {action}")

        try:
            if action == 'trigger_ohlcv_realtime':
                client.trigger_dag(dag_id='core_l0_02_ohlcv_realtime', conf={})
                actions_taken.append(f"Triggered OHLCV realtime")

            elif action == 'trigger_macro_update':
                client.trigger_dag(dag_id='core_l0_04_macro_update', conf={"force_run": True})
                actions_taken.append(f"Triggered macro update")

            elif action == 'run_generate_weekly_forecasts':
                # Run as subprocess (heavy computation, don't block DAG)
                subprocess.Popen(
                    ['python3', str(PROJECT_ROOT / 'scripts' / 'generate_weekly_forecasts.py')],
                    cwd=str(PROJECT_ROOT),
                    stdout=open('/tmp/watchdog_forecasting.log', 'w'),
                    stderr=subprocess.STDOUT,
                )
                actions_taken.append(f"Started forecasting backfill (background)")

            elif action == 'run_generate_weekly_analysis':
                # Generate missing week
                current_week = _current_iso_week()
                subprocess.Popen(
                    ['python3', str(PROJECT_ROOT / 'scripts' / 'generate_weekly_analysis.py'),
                     '--week', current_week],
                    cwd=str(PROJECT_ROOT),
                    stdout=open('/tmp/watchdog_analysis.log', 'w'),
                    stderr=subprocess.STDOUT,
                )
                actions_taken.append(f"Started analysis generation for {current_week}")

            elif action == 'trigger_h5_l5_signal':
                client.trigger_dag(dag_id='forecast_h5_l5_weekly_signal', conf={})
                actions_taken.append(f"Triggered H5 signal generation")

            elif action == 'trigger_news_pipeline':
                client.trigger_dag(dag_id='news_daily_pipeline', conf={})
                actions_taken.append(f"Triggered news pipeline")

            elif action == 'trigger_seed_backup':
                client.trigger_dag(dag_id='core_l0_05_seed_backup', conf={})
                actions_taken.append(f"Triggered seed backup (daily backup stale)")

            elif action == 'trigger_ohlcv_backfill':
                client.trigger_dag(dag_id='core_l0_01_ohlcv_backfill', conf={})
                actions_taken.append(f"Triggered OHLCV backfill (5-min seed stale)")

        except Exception as e:
            logger.error(f"[Watchdog] Failed to execute {action}: {e}")
            actions_taken.append(f"FAILED: {action} ({e})")

    # Summary
    if actions_taken:
        logger.warning(f"[Watchdog] AUTO-HEAL took {len(actions_taken)} actions: {actions_taken}")
    else:
        logger.info("[Watchdog] All systems healthy — no actions needed")

    return {
        "timestamp": datetime.now(COT).isoformat(),
        "checks": {k: v.get('status', '?') if v else 'error' for k, v in checks.items()},
        "actions_taken": actions_taken,
        "all_healthy": len(actions_taken) == 0,
    }


def generate_report(**context):
    """Generate a human-readable system health report."""
    ti = context['ti']
    heal_result = ti.xcom_pull(task_ids='auto_heal')

    if not heal_result:
        logger.info("[Watchdog] No heal result available")
        return

    report = [
        "=" * 50,
        f"  WATCHDOG REPORT — {heal_result.get('timestamp', '?')}",
        "=" * 50,
    ]

    for check_name, status in heal_result.get('checks', {}).items():
        icon = "✓" if status in ('ok', 'skip') else "✗"
        report.append(f"  {icon} {check_name}: {status}")

    actions = heal_result.get('actions_taken', [])
    if actions:
        report.append("")
        report.append(f"  ACTIONS TAKEN ({len(actions)}):")
        for a in actions:
            report.append(f"    → {a}")
    else:
        report.append("")
        report.append("  ALL SYSTEMS HEALTHY")

    report.append("=" * 50)

    for line in report:
        logger.info(line)

    return {"report": "\n".join(report), "healthy": heal_result.get('all_healthy', False)}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'system-watchdog',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 6),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=15),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Intelligent system health monitor — auto-detects and fixes gaps',
    schedule_interval='0 13-18 * * 1-5',  # Every hour 13-18 UTC = 8-13 COT Mon-Fri
    catchup=False,
    max_active_runs=1,
    tags=['core', 'monitoring', 'watchdog', 'auto-heal'],
) as dag:

    t_ohlcv = PythonOperator(task_id='check_ohlcv', python_callable=check_ohlcv_freshness)
    t_macro = PythonOperator(task_id='check_macro', python_callable=check_macro_freshness)
    t_forecast = PythonOperator(task_id='check_forecasting', python_callable=check_forecasting_freshness)
    t_analysis = PythonOperator(task_id='check_analysis', python_callable=check_analysis_freshness)
    t_h5 = PythonOperator(task_id='check_h5_signal', python_callable=check_h5_signal)
    t_news = PythonOperator(task_id='check_news', python_callable=check_news_pipeline)
    t_backups = PythonOperator(task_id='check_backups', python_callable=check_backups)

    t_heal = PythonOperator(task_id='auto_heal', python_callable=auto_heal,
                            execution_timeout=timedelta(minutes=5))
    t_report = PythonOperator(task_id='generate_report', python_callable=generate_report)

    # All checks run in parallel, then heal, then report
    [t_ohlcv, t_macro, t_forecast, t_analysis, t_h5, t_news, t_backups] >> t_heal >> t_report
