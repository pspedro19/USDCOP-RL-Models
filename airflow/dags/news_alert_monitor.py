"""
NewsEngine Alert Monitor (SDD-10)
===================================
Every 30 min during market hours: GDELT crisis scan for breaking news.

DAG ID: news_alert_monitor
Schedule: */30 7-22 * * 1-5  (UTC = every 30min 02:00-17:00 COT)
"""

from __future__ import annotations

import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "usdcop",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}


def _scan_for_alerts(**context):
    """Quick GDELT scan for breaking news + volume spikes."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.ingestion.gdelt_adapter import GDELTDocAdapter
    from src.news_engine.enrichment.pipeline import EnrichmentPipeline
    from src.news_engine.output.alert_system import AlertSystem
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)

    # Quick GDELT fetch (last 30 min worth)
    adapter = GDELTDocAdapter(config.gdelt)
    try:
        articles = adapter.fetch_latest()
    except Exception as e:
        logger.warning(f"GDELT fetch failed: {e}")
        return 0

    # Enrich
    pipeline = EnrichmentPipeline(config.enrichment)
    enriched = pipeline.enrich_batch(articles)

    # Store new articles
    count = db.upsert_articles_batch(enriched)

    # Check for alerts
    alert_system = AlertSystem(config.alert)
    alerts = alert_system.check_articles(enriched)

    # Check volume spike
    recent_count = db.get_article_count(hours=1)
    volume_alert = alert_system.check_volume_spike(recent_count)
    if volume_alert:
        alerts.append(volume_alert)

    if alerts:
        for alert in alerts:
            alert_system.send_alert(alert)
        logger.info(f"Sent {len(alerts)} alert(s)")

    logger.info(f"Alert scan: {count} new articles, {len(alerts)} alerts")
    return len(alerts)


with DAG(
    dag_id="news_alert_monitor",
    default_args=DEFAULT_ARGS,
    description="NewsEngine: GDELT breaking news scan (every 30min)",
    schedule="*/30 7-22 * * 1-5",  # UTC = 02:00-17:00 COT range
    start_date=days_ago(1),
    catchup=False,
    tags=["news", "alerts", "sdd-10"],
    max_active_runs=1,
) as dag:

    scan = PythonOperator(
        task_id="scan_for_alerts",
        python_callable=_scan_for_alerts,
    )
