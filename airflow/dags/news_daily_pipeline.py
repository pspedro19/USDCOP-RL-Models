"""
NewsEngine Daily Pipeline (SDD-10)
====================================
3x/day (02:00, 07:00, 13:00 COT): Ingest articles from all sources,
enrich, cross-reference, and export daily feature snapshots.

DAG ID: news_daily_pipeline
Schedule: 0 7,12,18 * * 1-5  (UTC = 02:00, 07:00, 13:00 COT)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)


def _today_cot():
    """Return today's date in America/Bogota timezone (COT = UTC-5).

    The DB query uses DATE(published_at AT TIME ZONE 'America/Bogota'),
    so we must use the same timezone for date.today() to match.
    """
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Bogota")).date()

DEFAULT_ARGS = {
    "owner": "usdcop",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


def _get_news_config():
    """Build NewsEngineConfig using the standard POSTGRES_* env vars."""
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.config import NewsEngineConfig, DatabaseConfig

    db_cfg = DatabaseConfig(
        host=os.environ.get("POSTGRES_HOST", "timescaledb"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    config = NewsEngineConfig.from_env()
    # Override database with standard Airflow env vars
    config.database = db_cfg
    return config


def _rows_to_enriched(rows):
    """Convert DB dict rows to EnrichedArticle objects."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.news_engine.models import RawArticle, EnrichedArticle

    result = []
    for row in rows:
        raw = RawArticle(
            url=row["url"],
            title=row["title"],
            source_id=row["source_id"],
            published_at=row["published_at"],
            content=row.get("content"),
            summary=row.get("summary"),
            gdelt_tone=row.get("gdelt_tone"),
            language=row.get("language", "es"),
            country_focus=row.get("country_focus", "CO"),
        )
        ea = EnrichedArticle(
            raw=raw,
            category=row.get("category"),
            relevance_score=row.get("relevance_score", 0.0),
            sentiment_score=row.get("sentiment_score"),
            sentiment_label=row.get("sentiment_label"),
            keywords=row.get("keywords") or [],
            entities=[],
            is_breaking=row.get("is_breaking", False),
        )
        result.append(ea)
    return result


def _ingest_all_sources(**context):
    """Fetch articles from all enabled news sources."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.ingestion.registry import SourceRegistry
    from src.news_engine.storage.database import NewsDatabase

    config = _get_news_config()
    registry = SourceRegistry.from_config(config)
    db = NewsDatabase(config.database)

    total_articles = 0
    source_results = {}

    for adapter in registry.enabled_adapters():
        name = adapter.source_id
        log_id = db.log_ingestion_start(name)
        try:
            articles = adapter.fetch_latest()
            count = db.upsert_raw_articles_batch(articles)
            db.log_ingestion_end(log_id, articles_fetched=len(articles), articles_new=count, status="success")
            source_results[name] = count
            total_articles += count
            logger.info(f"[{name}] Ingested {count} articles")
        except Exception as e:
            db.log_ingestion_end(log_id, status="failed", error_details=str(e), errors=1)
            source_results[name] = 0
            logger.error(f"[{name}] Ingestion failed: {e}")

    context["ti"].xcom_push(key="total_articles", value=total_articles)
    context["ti"].xcom_push(key="source_results", value=source_results)
    return total_articles


def _enrich_articles(**context):
    """Run enrichment pipeline on today's articles (COT date)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.enrichment.pipeline import EnrichmentPipeline
    from src.news_engine.models import RawArticle
    from src.news_engine.storage.database import NewsDatabase

    config = _get_news_config()
    db = NewsDatabase(config.database)
    pipeline = EnrichmentPipeline(config.enrichment.min_relevance_score)

    today = _today_cot()
    rows = db.get_articles_for_date(today)
    logger.info(f"Found {len(rows)} articles to enrich for {today}")

    enriched_count = 0
    for row in rows:
        try:
            raw = RawArticle(
                url=row["url"],
                title=row["title"],
                source_id=row["source_id"],
                published_at=row["published_at"],
                content=row.get("content"),
                summary=row.get("summary"),
                gdelt_tone=row.get("gdelt_tone"),
                language=row.get("language", "es"),
                country_focus=row.get("country_focus", "CO"),
            )
            enriched = pipeline.enrich(raw)
            db.upsert_article(enriched)
            enriched_count += 1
        except Exception as e:
            logger.warning(f"Failed to enrich article {row.get('id')}: {e}")

    logger.info(f"Enriched {enriched_count}/{len(rows)} articles")
    return enriched_count


def _cross_reference(**context):
    """Find cross-source topic clusters using dict-based articles."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.cross_reference.engine import CrossReferenceEngine
    from src.news_engine.storage.database import NewsDatabase

    config = _get_news_config()
    db = NewsDatabase(config.database)
    engine = CrossReferenceEngine(config.cross_reference)

    rows = db.get_articles_for_date(_today_cot())
    enriched_articles = _rows_to_enriched(rows)
    clusters = engine.find_clusters(enriched_articles)

    for cluster in clusters:
        # cluster.articles contains indices, not DB IDs — pass empty for now
        db.insert_cross_reference(cluster, article_ids=[])

    logger.info(f"Found {len(clusters)} cross-reference clusters from {len(rows)} articles")
    return len(clusters)


def _export_features(**context):
    """Export daily news feature snapshot from dict-based articles."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.output.feature_exporter import FeatureExporter
    from src.news_engine.storage.database import NewsDatabase

    config = _get_news_config()
    db = NewsDatabase(config.database)
    exporter = FeatureExporter()

    today = _today_cot()
    rows = db.get_articles_for_date(today)
    enriched = _rows_to_enriched(rows)
    features = exporter.export_daily(enriched, today)

    if features:
        snap_date = features.snapshot_date
        if hasattr(snap_date, 'date'):
            snap_date = snap_date.date()
        db.upsert_feature_snapshot(
            snapshot_date=snap_date,
            features=features.features,
            article_count=features.article_count,
            source_counts=features.source_counts,
        )
        logger.info(f"Exported feature snapshot: {features.feature_count} features")
    else:
        logger.warning("No features exported (insufficient articles)")


def _generate_digest(**context):
    """Generate daily text digest from dict-based articles."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.news_engine.output.digest_generator import DigestGenerator
    from src.news_engine.storage.database import NewsDatabase

    config = _get_news_config()
    db = NewsDatabase(config.database)
    generator = DigestGenerator()

    today = _today_cot()
    rows = db.get_articles_for_date(today)
    enriched = _rows_to_enriched(rows)
    digest = generator.generate_daily(enriched, today)

    if digest:
        db.upsert_digest(digest)
        logger.info(f"Generated daily digest: {digest.total_articles} articles")


with DAG(
    dag_id="news_daily_pipeline",
    default_args=DEFAULT_ARGS,
    description="NewsEngine: ingest + enrich + cross-ref + features (3x/day)",
    schedule="0 7,12,18 * * 1-5",  # UTC = 02:00, 07:00, 13:00 COT
    start_date=days_ago(1),
    catchup=False,
    tags=["news", "l0", "sdd-10"],
    max_active_runs=1,
) as dag:

    ingest = PythonOperator(
        task_id="ingest_all_sources",
        python_callable=_ingest_all_sources,
    )

    enrich = PythonOperator(
        task_id="enrich_articles",
        python_callable=_enrich_articles,
    )

    cross_ref = PythonOperator(
        task_id="cross_reference",
        python_callable=_cross_reference,
    )

    features = PythonOperator(
        task_id="export_features",
        python_callable=_export_features,
    )

    digest = PythonOperator(
        task_id="generate_digest",
        python_callable=_generate_digest,
    )

    ingest >> enrich >> cross_ref >> [features, digest]
