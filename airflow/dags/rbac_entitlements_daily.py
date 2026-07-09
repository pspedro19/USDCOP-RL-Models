"""
RBAC L8 — Daily entitlements degradation (CTR-RBAC-001 R6)
==========================================================
Persists what `effectiveEntitlements` already enforces lazily per-request: any
`sb_users.entitlements` whose `expires_at` is in the past degrades to the free plan
(assets [usdcop], delays free-tier, execution disabled). Idempotent; audits each
degradation. Schedule: daily 05:00 UTC (00:00 COT), off trading hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import json
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

FREE_ENTITLEMENTS = {
    "plan": "free",
    "assets": ["usdcop"],
    "forecast_delay_hours": 168,
    "analysis_delay_days": 7,
    "signals_realtime": False,
    "execution": {"enabled": False, "mode": "paper", "paper_weeks_required": 4,
                  "max_notional_usd": 0, "max_daily_loss_pct": 0, "max_open_positions": 0},
}


def degrade_expired(**context):
    from utils.dag_common import get_db_connection

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Find expired, not-already-free entitlements.
        cur.execute("""
            SELECT id, email, entitlements->>'plan' AS plan, entitlements->>'expires_at' AS exp
            FROM sb_users
            WHERE entitlements ? 'expires_at'
              AND (entitlements->>'expires_at')::timestamptz < NOW()
              AND COALESCE(entitlements->>'plan', 'free') <> 'free'
        """)
        rows = cur.fetchall()
        degraded = 0
        for uid, email, plan, exp in rows:
            cur.execute(
                "UPDATE sb_users SET entitlements = %s::jsonb WHERE id = %s",
                (json.dumps(FREE_ENTITLEMENTS), uid))
            cur.execute(
                "INSERT INTO audit_log (user_id, action, object_type, detail) "
                "VALUES (%s, 'entitlement_degraded', 'billing', %s::jsonb)",
                (str(uid), json.dumps({"from_plan": plan, "expired_at": exp, "by": "rbac_entitlements_daily"})))
            degraded += 1
            logger.info(f"[RBAC-R6] degraded {email}: {plan} (expired {exp}) -> free")
        conn.commit()
        logger.info(f"[RBAC-R6] degraded {degraded} expired entitlement(s)")
        return {"degraded": degraded}
    finally:
        conn.close()


with DAG(
    "rbac_entitlements_daily",
    default_args={"owner": "rbac", "retries": 1, "retry_delay": timedelta(minutes=10),
                  "start_date": datetime(2026, 7, 1)},
    description="RBAC R6: persist expiry degradation of sb_users.entitlements (daily)",
    schedule_interval="0 5 * * *",  # 00:00 COT
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,  # safe maintenance job — must survive cold boot unpaused
    tags=["rbac", "billing", "r6"],
) as dag:
    PythonOperator(task_id="degrade_expired", python_callable=degrade_expired)
