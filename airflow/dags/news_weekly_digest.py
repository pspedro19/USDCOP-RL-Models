"""
NewsEngine Weekly Digest (SDD-10)
====================================
Monday 03:00 COT: Generate weekly text digest summarizing all news from the past week.

DAG ID: news_weekly_digest
Schedule: 0 8 * * 1  (UTC = 03:00 COT Monday)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "usdcop",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
}


def _generate_weekly_digest(**context):
    """Generate weekly news digest from the past 7 days."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig
    from src.news_engine.output.digest_generator import DigestGenerator
    from src.news_engine.storage.database import NewsDatabase

    config = NewsEngineConfig.from_env()
    db = NewsDatabase(config.database)
    generator = DigestGenerator()

    today = date.today()
    week_start = today - timedelta(days=7)

    articles = db.get_articles_for_week(week_start, today)
    digest = generator.generate_weekly(articles, week_start, today)

    if digest:
        db.upsert_digest(digest)
        logger.info(f"Generated weekly digest: {digest.article_count} articles, "
                     f"{digest.source_breakdown}")
    else:
        logger.warning("No articles found for weekly digest")


with DAG(
    dag_id="news_weekly_digest",
    default_args=DEFAULT_ARGS,
    description="NewsEngine: weekly text digest (Monday)",
    schedule="0 8 * * 1",  # UTC Monday 08:00 = COT 03:00
    start_date=days_ago(1),
    catchup=False,
    tags=["news", "digest", "sdd-10"],
    max_active_runs=1,
) as dag:

    digest = PythonOperator(
        task_id="generate_weekly_digest",
        python_callable=_generate_weekly_digest,
    )
