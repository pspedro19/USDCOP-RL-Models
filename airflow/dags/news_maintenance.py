"""
NewsEngine Maintenance (SDD-10)
==================================
Sunday 22:00 COT: Cleanup old articles, vacuum tables, update statistics.

DAG ID: news_maintenance
Schedule: 0 3 * * 0  (UTC Sunday 03:00 = COT 22:00)
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
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

RETENTION_DAYS = 180  # Keep 6 months of articles


def _cleanup_old_articles(**context):
    """Delete articles older than retention period."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)

    deleted = db.execute_maintenance(
        f"""
        DELETE FROM news_articles
        WHERE published_at < NOW() - INTERVAL '{RETENTION_DAYS} days'
        AND relevance_score < 0.3
        """
    )
    logger.info(f"Deleted {deleted} old low-relevance articles")
    return deleted


def _vacuum_tables(**context):
    """Run VACUUM ANALYZE on NewsEngine tables."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)

    tables = [
        "news_articles",
        "news_cross_references",
        "news_cross_reference_articles",
        "news_daily_digests",
        "news_feature_snapshots",
        "news_ingestion_log",
    ]

    for table in tables:
        try:
            db.execute_maintenance(f"VACUUM ANALYZE {table}", autocommit=True)
            logger.info(f"VACUUM ANALYZE {table} completed")
        except Exception as e:
            logger.warning(f"VACUUM {table} failed: {e}")


def _cleanup_ingestion_log(**context):
    """Clean old ingestion log entries."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)

    deleted = db.execute_maintenance(
        """
        DELETE FROM news_ingestion_log
        WHERE started_at < NOW() - INTERVAL '90 days'
        """
    )
    logger.info(f"Cleaned {deleted} old ingestion log entries")


def _report_stats(**context):
    """Log database statistics."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)

    stats = db.get_source_stats()
    for source, count in stats.items():
        logger.info(f"Source {source}: {count} articles")


with DAG(
    dag_id="news_maintenance",
    default_args=DEFAULT_ARGS,
    description="NewsEngine: cleanup + vacuum (Sunday night)",
    schedule="0 3 * * 0",  # UTC Sunday 03:00 = COT 22:00
    start_date=days_ago(1),
    catchup=False,
    tags=["news", "maintenance", "sdd-10"],
    max_active_runs=1,
) as dag:

    cleanup = PythonOperator(
        task_id="cleanup_old_articles",
        python_callable=_cleanup_old_articles,
    )

    cleanup_log = PythonOperator(
        task_id="cleanup_ingestion_log",
        python_callable=_cleanup_ingestion_log,
    )

    vacuum = PythonOperator(
        task_id="vacuum_tables",
        python_callable=_vacuum_tables,
    )

    report = PythonOperator(
        task_id="report_stats",
        python_callable=_report_stats,
    )

    [cleanup, cleanup_log] >> vacuum >> report
