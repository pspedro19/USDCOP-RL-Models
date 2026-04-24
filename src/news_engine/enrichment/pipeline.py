"""
Enrichment Pipeline Orchestrator (SDD-04)
============================================
Takes RawArticle list → enriches each → returns EnrichedArticle list.
"""

from __future__ import annotations

import logging

from src.news_engine.enrichment.categorizer import categorize_article
from src.news_engine.enrichment.relevance import score_relevance
from src.news_engine.enrichment.sentiment import analyze_sentiment
from src.news_engine.enrichment.tagger import extract_entities, extract_keywords
from src.news_engine.models import EnrichedArticle, RawArticle

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """Orchestrates the enrichment of raw articles.

    Pipeline stages:
    1. Categorize (assign category from 9 options)
    2. Score relevance (0.0-1.0)
    3. Analyze sentiment (GDELT tone primary, VADER fallback)
    4. Extract keywords & entities
    5. Flag breaking/weekly-relevant
    """

    def __init__(self, min_relevance: float = 0.1):
        self.min_relevance = min_relevance

    def enrich(self, article: RawArticle) -> EnrichedArticle:
        """Enrich a single raw article."""
        # 1. Categorize
        category, subcategory = categorize_article(
            article.title, article.content, article.summary,
        )

        # 2. Score relevance
        relevance = score_relevance(
            article.title, article.content, article.summary,
            article.source_id, article.published_at,
        )

        # 3. Sentiment
        sentiment_score, sentiment_label = analyze_sentiment(
            article.title, article.content, article.gdelt_tone,
        )

        # 4. Keywords & entities
        keywords = extract_keywords(article.title, article.content)
        entities = extract_entities(article.title, article.content)

        # 5. Flags
        is_breaking = self._detect_breaking(article, relevance, sentiment_score)
        is_weekly = relevance >= 0.5 or category in (
            "monetary_policy", "fx_market", "commodities",
        )

        return EnrichedArticle(
            raw=article,
            category=category,
            subcategory=subcategory,
            relevance_score=relevance,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            keywords=keywords,
            entities=entities,
            is_breaking=is_breaking,
            is_weekly_relevant=is_weekly,
        )

    def enrich_batch(self, articles: list[RawArticle]) -> list[EnrichedArticle]:
        """Enrich a batch of articles."""
        enriched = []
        for article in articles:
            try:
                result = self.enrich(article)
                if result.relevance_score >= self.min_relevance:
                    enriched.append(result)
            except Exception as e:
                logger.warning(f"Enrichment failed for {article.url}: {e}")
        logger.info(
            f"Enriched {len(enriched)}/{len(articles)} articles "
            f"(min_relevance={self.min_relevance})"
        )
        return enriched

    @staticmethod
    def _detect_breaking(
        article: RawArticle,
        relevance: float,
        sentiment_score: float | None,
    ) -> bool:
        """Detect if an article is breaking news."""
        # High relevance + extreme sentiment
        if relevance >= 0.8 and sentiment_score is not None:
            if abs(sentiment_score) >= 0.6:
                return True
        # GDELT extreme tone
        if article.gdelt_tone is not None and abs(article.gdelt_tone) >= 15:
            return True
        return False
