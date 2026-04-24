"""
Digest Generator (SDD-06 §3)
===============================
Generates daily and weekly text digests from enriched articles.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime

from src.news_engine.models import EnrichedArticle, NewsDigest

logger = logging.getLogger(__name__)


class DigestGenerator:
    """Generates statistical digests from enriched articles."""

    def generate_daily(
        self,
        articles: list[EnrichedArticle],
        digest_date: date | None = None,
    ) -> NewsDigest:
        """Generate a daily digest."""
        digest_date = digest_date or date.today()

        # Source breakdown
        by_source = Counter(a.raw.source_id for a in articles)
        by_category = Counter(a.category for a in articles if a.category)

        # Sentiment
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None

        # Top keywords (aggregate from all articles)
        kw_counter = Counter()
        for a in articles:
            for kw in a.keywords:
                kw_counter[kw] += 1
        top_keywords = [
            {"keyword": kw, "count": cnt}
            for kw, cnt in kw_counter.most_common(10)
        ]

        # Top articles by relevance
        sorted_articles = sorted(articles, key=lambda a: a.relevance_score, reverse=True)
        top_articles = [
            {
                "title": a.raw.title,
                "source": a.raw.source_id,
                "relevance": round(a.relevance_score, 3),
                "sentiment": a.sentiment_label,
            }
            for a in sorted_articles[:10]
        ]

        # Summary text
        summary = self._build_summary(
            len(articles), by_source, by_category, avg_sentiment, top_keywords,
        )

        return NewsDigest(
            digest_date=datetime.combine(digest_date, datetime.min.time()),
            digest_type="daily",
            total_articles=len(articles),
            by_source=dict(by_source),
            by_category=dict(by_category),
            avg_sentiment=avg_sentiment,
            top_keywords=top_keywords,
            top_articles=top_articles,
            summary_text=summary,
        )

    def generate_weekly(
        self,
        articles: list[EnrichedArticle],
        week_start: date,
    ) -> NewsDigest:
        """Generate a weekly digest."""
        digest = self.generate_daily(articles, week_start)
        digest.digest_type = "weekly"
        return digest

    @staticmethod
    def _build_summary(
        total: int,
        by_source: dict,
        by_category: dict,
        avg_sentiment: float | None,
        top_keywords: list,
    ) -> str:
        """Build a textual summary for the digest."""
        parts = [f"Total de articulos: {total}."]

        # Top sources
        top_sources = sorted(by_source.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_sources:
            src_str = ", ".join(f"{s}: {c}" for s, c in top_sources)
            parts.append(f"Fuentes principales: {src_str}.")

        # Top categories
        top_cats = sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_cats:
            cat_str = ", ".join(f"{c}: {n}" for c, n in top_cats)
            parts.append(f"Categorias: {cat_str}.")

        # Sentiment
        if avg_sentiment is not None:
            sentiment_word = "positivo" if avg_sentiment > 0.15 else (
                "negativo" if avg_sentiment < -0.15 else "neutral"
            )
            parts.append(f"Sentimiento promedio: {sentiment_word} ({avg_sentiment:.2f}).")

        # Top keywords
        if top_keywords:
            kw_str = ", ".join(kw["keyword"] for kw in top_keywords[:5])
            parts.append(f"Temas clave: {kw_str}.")

        return " ".join(parts)
