"""L0 DAG for official USD/COP forward-looking PIT releases.

This is a sibling of ``l0_macro_update``.  It deliberately targets the long
``macro_indicators_pit`` table because the legacy wide tables cannot retain
publication vintages.  The same scraper module is used by the standalone
backfill command and by tests.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DAGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAGS_DIR.parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src", DAGS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.data.usdcop_forward_macro import ForwardMacroScraper, upsert_pit_rows

try:
    from contracts.dag_registry import (
        CORE_L0_USDCOP_FORWARD_MACRO_PIT,
        get_dag_tags,
    )
except ImportError:  # pragma: no cover - standalone parse fallback
    CORE_L0_USDCOP_FORWARD_MACRO_PIT = "core_l0_04_usdcop_forward_macro_pit"

    def get_dag_tags(_dag_id):
        return ["core", "l0", "data", "macro", "pit", "forward"]

LOGGER = logging.getLogger(__name__)


def ingest_forward_macro(**context):
    conf = (context.get("dag_run").conf if context.get("dag_run") else {}) or {}
    end = pd.Timestamp(conf.get("end_date", pd.Timestamp.now(tz="America/Bogota").date()))
    full_backfill = bool(conf.get("full_backfill", False))
    start = pd.Timestamp(conf.get("start_date", "2015-01-01" if full_backfill else end - pd.Timedelta(days=120)))
    sources = conf.get(
        "sources",
        ["daily_forward", "forward_history", "monthly_derivatives", "eme", "sfc"],
    )
    if isinstance(sources, str):
        sources = [item.strip() for item in sources.split(",") if item.strip()]

    scraper = ForwardMacroScraper(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config" / "usdcop_forward_macro_sources.yaml",
        raw_dir=PROJECT_ROOT / "data" / "pipeline" / "01_sources" / "17_forward_looking",
        output_path=(
            PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
            / "USDCOP_FORWARD_MACRO_PIT.parquet"
        ),
        manifest_dir=(
            PROJECT_ROOT / "data" / "pipeline" / "02_scrapers" / "storage" / "manifests"
        ),
        force=bool(conf.get("force", False)),
        offline=bool(conf.get("offline", False)),
    )
    result = scraper.run(start, end, sources)
    if result.combined.empty:
        raise RuntimeError(f"Forward-macro ingestion has no durable rows: {result.errors[:5]}")

    from airflow.providers.postgres.hooks.postgres import PostgresHook

    connection = PostgresHook(postgres_conn_id="timescale_conn").get_conn()
    try:
        rows_upserted = upsert_pit_rows(connection, result.extracted)
    finally:
        connection.close()
    summary = result.summary()
    summary["database_rows_upserted"] = rows_upserted
    LOGGER.info("Forward-macro PIT ingestion: %s", summary)
    return summary


def validate_forward_macro(**context):
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    connection = PostgresHook(postgres_conn_id="timescale_conn").get_conn()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT series_id) AS series,
                    MIN(observation_date),
                    MAX(observation_date),
                    COUNT(*) FILTER (WHERE available_at < observation_date::timestamptz) AS pit_violations,
                    COUNT(*) FILTER (
                        WHERE promotion_eligible AND NOT pit_vintage
                    ) AS eligibility_violations,
                    COUNT(*) FILTER (
                        WHERE source = 'banrep_forward_history'
                          AND availability_policy LIKE '%reconstructed%'
                          AND promotion_eligible
                    ) AS reconstructed_promotion_violations
                FROM macro_indicators_pit
                """
            )
            (
                rows,
                series,
                first_date,
                last_date,
                violations,
                eligibility_violations,
                reconstructed_promotion_violations,
            ) = cursor.fetchone()
    finally:
        connection.close()
    if rows == 0 or series == 0:
        raise RuntimeError("macro_indicators_pit is empty after ingestion")
    if violations:
        raise RuntimeError(f"PIT contract violations detected: {violations}")
    if eligibility_violations or reconstructed_promotion_violations:
        raise RuntimeError(
            "Forward-history evidence-class violation: "
            f"eligibility={eligibility_violations}, "
            f"reconstructed={reconstructed_promotion_violations}"
        )
    return {
        "rows": rows,
        "series": series,
        "first_observation": str(first_date),
        "last_observation": str(last_date),
        "pit_violations": violations,
        "eligibility_violations": eligibility_violations,
        "reconstructed_promotion_violations": reconstructed_promotion_violations,
    }


default_args = {
    "owner": "usdcop-data-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    CORE_L0_USDCOP_FORWARD_MACRO_PIT,
    default_args=default_args,
    description="Official BanRep/SFC forward-looking data with PIT availability",
    schedule_interval="30 22 * * 1-5",  # 17:30 America/Bogota, after official refresh
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=get_dag_tags(CORE_L0_USDCOP_FORWARD_MACRO_PIT),
    params={
        "full_backfill": False,
        "start_date": None,
        "end_date": None,
        "sources": "daily_forward,forward_history,monthly_derivatives,eme,sfc",
        "force": False,
        "offline": False,
    },
) as dag:
    ingest = PythonOperator(
        task_id="ingest_official_forward_macro",
        python_callable=ingest_forward_macro,
    )
    validate = PythonOperator(
        task_id="validate_pit_contract",
        python_callable=validate_forward_macro,
    )
    ingest >> validate
