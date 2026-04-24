"""
NewsEngine Database Layer (SDD-03)
====================================
PostgreSQL storage operations for articles, digests, cross-references,
ingestion logs, and feature snapshots.

Uses raw psycopg2 for compatibility with existing project patterns
(no SQLAlchemy ORM to keep it simple).
"""

from __future__ import annotations

import json
import logging
from datetime import date

import psycopg2
import psycopg2.extras

from src.news_engine.config import DatabaseConfig
from src.news_engine.models import CrossReference, EnrichedArticle, NewsDigest

logger = logging.getLogger(__name__)


class NewsDatabase:
    """PostgreSQL operations for NewsEngine tables (migration 045)."""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.config.url)
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def upsert_article(self, article: EnrichedArticle) -> int:
        """Insert or update an article. Returns article ID."""
        raw = article.raw
        sql = """
            INSERT INTO news_articles (
                source_id, url, url_hash, title, content, summary,
                published_at, category, subcategory, relevance_score,
                sentiment_score, sentiment_label, gdelt_tone,
                keywords, entities, language, country_focus,
                is_breaking, is_weekly_relevant, image_url, author, raw_json
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (source_id, url_hash) DO UPDATE SET
                category = EXCLUDED.category,
                relevance_score = EXCLUDED.relevance_score,
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                keywords = EXCLUDED.keywords,
                entities = EXCLUDED.entities,
                is_breaking = EXCLUDED.is_breaking,
                is_weekly_relevant = EXCLUDED.is_weekly_relevant,
                updated_at = NOW()
            RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                raw.source_id, raw.url, raw.url_hash, raw.title,
                raw.content, raw.summary, raw.published_at,
                article.category, article.subcategory, article.relevance_score,
                article.sentiment_score, article.sentiment_label, raw.gdelt_tone,
                article.keywords or None, article.entities or None,
                raw.language, raw.country_focus,
                article.is_breaking, article.is_weekly_relevant,
                raw.image_url, raw.author,
                json.dumps(raw.raw_json) if raw.raw_json else None,
            ))
            row = cur.fetchone()
            self.conn.commit()
            return row[0] if row else 0

    def upsert_articles_batch(self, articles: list[EnrichedArticle]) -> int:
        """Batch upsert articles. Returns count of upserted rows."""
        if not articles:
            return 0

        values = []
        for article in articles:
            raw = article.raw
            values.append((
                raw.source_id, raw.url, raw.url_hash, raw.title,
                raw.content, raw.summary, raw.published_at,
                article.category, article.subcategory, article.relevance_score,
                article.sentiment_score, article.sentiment_label, raw.gdelt_tone,
                article.keywords or None, article.entities or None,
                raw.language, raw.country_focus,
                article.is_breaking, article.is_weekly_relevant,
                raw.image_url, raw.author,
                json.dumps(raw.raw_json) if raw.raw_json else None,
            ))

        sql = """
            INSERT INTO news_articles (
                source_id, url, url_hash, title, content, summary,
                published_at, category, subcategory, relevance_score,
                sentiment_score, sentiment_label, gdelt_tone,
                keywords, entities, language, country_focus,
                is_breaking, is_weekly_relevant, image_url, author, raw_json
            ) VALUES %s
            ON CONFLICT (source_id, url_hash) DO UPDATE SET
                category = EXCLUDED.category,
                relevance_score = EXCLUDED.relevance_score,
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                updated_at = NOW()
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=100)
            count = cur.rowcount
            self.conn.commit()
        logger.info(f"Upserted {count} articles")
        return count

    def upsert_raw_articles_batch(self, articles: list) -> int:
        """Batch upsert raw (un-enriched) articles. Returns count."""
        if not articles:
            return 0

        values = []
        for raw in articles:
            values.append((
                raw.source_id, raw.url, raw.url_hash, raw.title,
                raw.content, raw.summary, raw.published_at,
                None, None, 0.0,  # category, subcategory, relevance
                None, None, raw.gdelt_tone,  # sentiment_score, label, gdelt_tone
                None, None,  # keywords, entities
                raw.language, raw.country_focus,
                False, False,  # is_breaking, is_weekly_relevant
                raw.image_url, raw.author,
                json.dumps(raw.raw_json) if raw.raw_json else None,
            ))

        sql = """
            INSERT INTO news_articles (
                source_id, url, url_hash, title, content, summary,
                published_at, category, subcategory, relevance_score,
                sentiment_score, sentiment_label, gdelt_tone,
                keywords, entities, language, country_focus,
                is_breaking, is_weekly_relevant, image_url, author, raw_json
            ) VALUES %s
            ON CONFLICT (source_id, url_hash) DO NOTHING
        """
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=100)
            count = cur.rowcount
            self.conn.commit()
        logger.info(f"Upserted {count} raw articles")
        return count

    def get_articles_for_date(
        self,
        target_date: date,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """Get articles for a specific date."""
        sql = """
            SELECT id, source_id, url, title, content, summary, published_at,
                   category, relevance_score, sentiment_score, sentiment_label,
                   gdelt_tone, keywords, is_breaking, language, country_focus
            FROM news_articles
            WHERE DATE(published_at AT TIME ZONE 'America/Bogota') = %s
              AND relevance_score >= %s
            ORDER BY relevance_score DESC, published_at DESC
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (target_date, min_relevance))
            return [dict(row) for row in cur.fetchall()]

    def get_articles_for_week(
        self,
        start_date: date,
        end_date: date,
        min_relevance: float = 0.0,
        limit: int = 500,
    ) -> list[dict]:
        """Get articles for a week range."""
        sql = """
            SELECT id, source_id, url, title, summary, published_at,
                   category, relevance_score, sentiment_score, sentiment_label,
                   keywords, is_breaking
            FROM news_articles
            WHERE DATE(published_at AT TIME ZONE 'America/Bogota') BETWEEN %s AND %s
              AND relevance_score >= %s
            ORDER BY relevance_score DESC, published_at DESC
            LIMIT %s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (start_date, end_date, min_relevance, limit))
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Digests
    # ------------------------------------------------------------------

    def upsert_digest(self, digest: NewsDigest) -> None:
        """Upsert a daily/weekly digest."""
        sql = """
            INSERT INTO news_daily_digests (
                digest_date, digest_type, total_articles, by_source,
                by_category, avg_sentiment, top_keywords, top_articles,
                cross_ref_count, summary_text
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (digest_date) DO UPDATE SET
                total_articles = EXCLUDED.total_articles,
                by_source = EXCLUDED.by_source,
                by_category = EXCLUDED.by_category,
                avg_sentiment = EXCLUDED.avg_sentiment,
                top_keywords = EXCLUDED.top_keywords,
                top_articles = EXCLUDED.top_articles,
                cross_ref_count = EXCLUDED.cross_ref_count,
                summary_text = EXCLUDED.summary_text
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                digest.digest_date, digest.digest_type,
                digest.total_articles,
                json.dumps(digest.by_source),
                json.dumps(digest.by_category),
                digest.avg_sentiment,
                json.dumps(digest.top_keywords),
                json.dumps(digest.top_articles),
                digest.cross_ref_count,
                digest.summary_text,
            ))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Cross-references
    # ------------------------------------------------------------------

    def insert_cross_reference(
        self,
        cross_ref: CrossReference,
        article_ids: list[int],
    ) -> int:
        """Insert a cross-reference cluster with article links."""
        sql = """
            INSERT INTO news_cross_references (
                topic, cluster_date, article_count, avg_sentiment,
                dominant_category, sources_involved, summary
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                cross_ref.topic, cross_ref.cluster_date,
                len(article_ids), cross_ref.avg_sentiment,
                cross_ref.dominant_category, cross_ref.sources_involved,
                cross_ref.summary,
            ))
            xref_id = cur.fetchone()[0]

            # Link articles
            if article_ids:
                link_sql = """
                    INSERT INTO news_cross_reference_articles
                        (cross_reference_id, article_id, similarity_score)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """
                link_values = [(xref_id, aid, 1.0) for aid in article_ids]
                psycopg2.extras.execute_values(cur, link_sql, link_values)

            self.conn.commit()
            return xref_id

    # ------------------------------------------------------------------
    # Feature snapshots
    # ------------------------------------------------------------------

    def upsert_feature_snapshot(
        self,
        snapshot_date: date,
        features: dict,
        article_count: int = 0,
        source_counts: dict | None = None,
    ) -> None:
        """Upsert daily feature snapshot."""
        sql = """
            INSERT INTO news_feature_snapshots (
                snapshot_date, features, article_count, source_counts
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                features = EXCLUDED.features,
                article_count = EXCLUDED.article_count,
                source_counts = EXCLUDED.source_counts
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                snapshot_date,
                json.dumps(features),
                article_count,
                json.dumps(source_counts or {}),
            ))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Ingestion log
    # ------------------------------------------------------------------

    def log_ingestion_start(self, source_id: str, run_type: str = "scheduled") -> int:
        """Log the start of an ingestion run. Returns log ID."""
        sql = """
            INSERT INTO news_ingestion_log (source_id, run_type, started_at, status)
            VALUES (%s, %s, NOW(), 'running')
            RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (source_id, run_type))
            log_id = cur.fetchone()[0]
            self.conn.commit()
            return log_id

    def log_ingestion_end(
        self,
        log_id: int,
        articles_fetched: int = 0,
        articles_new: int = 0,
        errors: int = 0,
        error_details: str | None = None,
        status: str = "success",
    ) -> None:
        """Log the end of an ingestion run."""
        sql = """
            UPDATE news_ingestion_log SET
                finished_at = NOW(),
                articles_fetched = %s,
                articles_new = %s,
                errors = %s,
                error_details = %s,
                status = %s
            WHERE id = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                articles_fetched, articles_new, errors,
                error_details, status, log_id,
            ))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_article_count(self, days_back: int = 7) -> int:
        """Get total article count for recent days."""
        sql = """
            SELECT COUNT(*) FROM news_articles
            WHERE published_at > NOW() - INTERVAL '%s days'
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (days_back,))
            return cur.fetchone()[0]

    def get_source_stats(self, days_back: int = 7) -> dict:
        """Get per-source article counts for recent days."""
        sql = """
            SELECT source_id, COUNT(*) as cnt
            FROM news_articles
            WHERE published_at > NOW() - INTERVAL '%s days'
            GROUP BY source_id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (days_back,))
            return {row[0]: row[1] for row in cur.fetchall()}
