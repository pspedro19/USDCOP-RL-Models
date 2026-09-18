"""
RBAC L8 — Daily entitlements degradation (CTR-RBAC-001 R6)
==========================================================
Persists what `effectiveEntitlements` already enforces lazily per-request: any
`sb_users.entitlements` whose `expires_at` is in the past degrades to the free plan
(assets [usdcop, xauusd], delays free-tier, execution disabled). Idempotent; audits each
degradation. Schedule: daily 05:00 UTC (00:00 COT), off trading hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import json
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Template for a degraded row. `assets` and `expires_at` are overridden per user in
# degrade_expired(): the purchased asset list is retained as a record and the past expiry
# is carried over so `effectiveEntitlements` keeps resolving the row to the free plan.
FREE_ENTITLEMENTS = {
    "plan": "free",
    "assets": ["usdcop", "xauusd"],  # mirrors PLAN_DEFAULTS.free (Gold ships with every tier)
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
            SELECT id, email, entitlements->>'plan' AS plan, entitlements->>'expires_at' AS exp,
                   entitlements->'assets' AS assets
            FROM sb_users
            WHERE entitlements ? 'expires_at'
              AND (entitlements->>'expires_at')::timestamptz < NOW()
              AND COALESCE(entitlements->>'plan', 'free') <> 'free'
        """)
        rows = cur.fetchall()
        degraded = 0
        for uid, email, plan, exp, assets in rows:
            prev_assets = assets if isinstance(assets, list) else []
            # Degrade the PLAN, keep the PURCHASE RECORD. Writing FREE_ENTITLEMENTS verbatim
            # used to overwrite `assets` with ["usdcop"], so a customer who let the plan lapse
            # and renewed lost every per-asset add-on they had paid for, permanently and with
            # no way to tell what was lost.
            #
            # The expired timestamp is deliberately CARRIED OVER rather than dropped: every
            # reader resolves through `effectiveEntitlements`, which returns the free plan for
            # any row whose `expires_at` is in the past. Keeping it in the past is what makes
            # the retained asset list a record instead of a free upgrade — dropping the field
            # (or nulling it) would mark the row as never-expiring and hand the paid add-ons
            # out for free. Restoring them on reactivation is the admin path
            # (PATCH /api/admin/users/:id), which unions the stored list.
            degraded_row = {**FREE_ENTITLEMENTS, "assets": prev_assets, "expires_at": exp}
            cur.execute(
                "UPDATE sb_users SET entitlements = %s::jsonb WHERE id = %s",
                (json.dumps(degraded_row), uid))
            cur.execute(
                "INSERT INTO audit_log (user_id, action, object_type, detail) "
                "VALUES (%s, 'entitlement_degraded', 'billing', %s::jsonb)",
                (str(uid), json.dumps({"from_plan": plan, "expired_at": exp,
                                       "retained_assets": prev_assets,
                                       "by": "rbac_entitlements_daily"})))
            degraded += 1
            logger.info(f"[RBAC-R6] degraded {email}: {plan} (expired {exp}) -> free "
                        f"(retained {len(prev_assets)} asset record(s))")
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
