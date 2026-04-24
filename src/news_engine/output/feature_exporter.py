"""
Feature Exporter (SDD-06 §5)
===============================
Exports ~60 news-derived features for potential ML integration.
Feature groups: volume, keywords, sentiment, cross-ref, relevance.
No macro features here — macro comes from existing L0 pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, datetime

from src.news_engine.models import EnrichedArticle, NewsFeatureVector

logger = logging.getLogger(__name__)

# Categories for volume features
CATEGORIES = [
    "monetary_policy", "fx_market", "commodities", "inflation",
    "fiscal_policy", "risk_premium", "capital_flows", "balance_payments",
    "political",
]

# Top keywords for mention count features
TOP_KEYWORDS = [
    "dolar", "tasa de cambio", "banco de la republica",
    "tasa de interes", "fed", "petroleo", "inflacion",
    "embi", "devaluacion", "remesas", "tes", "riesgo pais",
    "inversion extranjera", "reforma tributaria",
]

# Sources for per-source features
SOURCES = ["gdelt_doc", "newsapi", "investing", "larepublica", "portafolio"]


class FeatureExporter:
    """Exports daily news feature vectors."""

    def export_daily(
        self,
        articles: list[EnrichedArticle],
        snapshot_date: date | None = None,
    ) -> NewsFeatureVector:
        """Compute ~60 features from a day's articles.

        Feature groups:
        A. Volume per category (9 features)
        B. Keyword mention counts (14 features)
        C. Sentiment per category (9 features)
        D. Overall sentiment stats (6 features)
        E. Per-source volume (5 features)
        F. Cross-reference features (5 features — filled later)
        G. Relevance features (4 features)
        H. Flags (4 features)
        """
        snapshot_date = snapshot_date or date.today()
        features = {}

        # A. Volume per category
        cat_counts = Counter(a.category for a in articles if a.category)
        for cat in CATEGORIES:
            features[f"vol_{cat}"] = cat_counts.get(cat, 0)

        # B. Keyword mentions
        all_text = " ".join(
            (a.raw.title or "").lower() + " " + (a.raw.summary or "").lower()
            for a in articles
        )
        for kw in TOP_KEYWORDS:
            features[f"kw_{kw.replace(' ', '_')}"] = all_text.count(kw.lower())

        # C. Sentiment per category
        cat_sentiments = defaultdict(list)
        for a in articles:
            if a.category and a.sentiment_score is not None:
                cat_sentiments[a.category].append(a.sentiment_score)
        for cat in CATEGORIES:
            vals = cat_sentiments.get(cat, [])
            features[f"sent_{cat}"] = sum(vals) / len(vals) if vals else 0.0

        # D. Overall sentiment stats
        all_sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        features["sent_mean"] = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
        features["sent_std"] = _std(all_sentiments) if len(all_sentiments) > 1 else 0.0
        features["sent_min"] = min(all_sentiments) if all_sentiments else 0.0
        features["sent_max"] = max(all_sentiments) if all_sentiments else 0.0
        features["sent_positive_ratio"] = (
            sum(1 for s in all_sentiments if s > 0.15) / len(all_sentiments)
            if all_sentiments else 0.0
        )
        features["sent_negative_ratio"] = (
            sum(1 for s in all_sentiments if s < -0.15) / len(all_sentiments)
            if all_sentiments else 0.0
        )

        # E. Per-source volume
        source_counts = Counter(a.raw.source_id for a in articles)
        for src in SOURCES:
            features[f"src_{src}"] = source_counts.get(src, 0)

        # F. Cross-reference placeholder (filled by pipeline after clustering)
        features["xref_cluster_count"] = 0
        features["xref_max_cluster_size"] = 0
        features["xref_avg_cluster_size"] = 0.0
        features["xref_multi_source_ratio"] = 0.0
        features["xref_coverage_pct"] = 0.0

        # G. Relevance features
        all_relevance = [a.relevance_score for a in articles]
        features["rel_mean"] = sum(all_relevance) / len(all_relevance) if all_relevance else 0.0
        features["rel_max"] = max(all_relevance) if all_relevance else 0.0
        features["rel_high_count"] = sum(1 for r in all_relevance if r >= 0.7)
        features["rel_total_articles"] = len(articles)

        # H. Flags
        features["flag_breaking_count"] = sum(1 for a in articles if a.is_breaking)
        features["flag_weekly_relevant"] = sum(1 for a in articles if a.is_weekly_relevant)
        features["flag_es_articles"] = sum(1 for a in articles if a.raw.language == "es")
        features["flag_co_focus"] = sum(1 for a in articles if a.raw.country_focus == "CO")

        return NewsFeatureVector(
            snapshot_date=datetime.combine(snapshot_date, datetime.min.time()),
            features=features,
            article_count=len(articles),
            source_counts=dict(source_counts),
        )


def _std(values: list) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
