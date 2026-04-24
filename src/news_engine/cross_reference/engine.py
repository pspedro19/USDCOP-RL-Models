"""
Cross-Reference Engine (SDD-05)
=================================
Clusters related articles across sources by title/entity/category similarity.
Weights: title 40% + entity 30% + summary 20% + category 10%.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from src.news_engine.config import CrossReferenceConfig
from src.news_engine.models import CrossReference, EnrichedArticle

logger = logging.getLogger(__name__)


class CrossReferenceEngine:
    """Clusters related articles across sources."""

    def __init__(self, config: CrossReferenceConfig | None = None):
        self.cfg = config or CrossReferenceConfig()

    def find_clusters(
        self,
        articles: list[EnrichedArticle],
        cluster_date: datetime | None = None,
    ) -> list[CrossReference]:
        """Find topic clusters in a list of enriched articles.

        Uses simple token overlap similarity (no ML dependencies).

        Returns:
            List of CrossReference clusters.
        """
        cluster_date = cluster_date or datetime.utcnow()
        if len(articles) < self.cfg.min_cluster_size:
            return []

        # Build similarity matrix
        n = len(articles)
        clusters = []
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            cluster_members = [i]
            for j in range(i + 1, n):
                if j in assigned:
                    continue

                sim = self._compute_similarity(articles[i], articles[j])
                if sim >= self.cfg.similarity_threshold:
                    cluster_members.append(j)

            if len(cluster_members) >= self.cfg.min_cluster_size:
                # Limit cluster size
                cluster_members = cluster_members[:self.cfg.max_cluster_size]
                assigned.update(cluster_members)

                cluster_articles = [articles[idx] for idx in cluster_members]
                cluster = self._build_cluster(cluster_articles, cluster_date)
                clusters.append(cluster)

        logger.info(
            f"Found {len(clusters)} clusters from {n} articles "
            f"(threshold={self.cfg.similarity_threshold})"
        )
        return clusters

    def _compute_similarity(
        self,
        a: EnrichedArticle,
        b: EnrichedArticle,
    ) -> float:
        """Compute weighted similarity between two articles."""
        # Title similarity (Jaccard on tokens)
        title_sim = self._jaccard(
            self._tokenize(a.raw.title),
            self._tokenize(b.raw.title),
        )

        # Entity overlap
        entity_sim = self._jaccard(
            set(a.entities),
            set(b.entities),
        ) if a.entities and b.entities else 0.0

        # Summary similarity
        summary_sim = 0.0
        if a.raw.summary and b.raw.summary:
            summary_sim = self._jaccard(
                self._tokenize(a.raw.summary),
                self._tokenize(b.raw.summary),
            )

        # Category match
        category_sim = 1.0 if (a.category and a.category == b.category) else 0.0

        # Weighted combination
        return (
            title_sim * self.cfg.title_weight +
            entity_sim * self.cfg.entity_weight +
            summary_sim * self.cfg.summary_weight +
            category_sim * self.cfg.category_weight
        )

    def _build_cluster(
        self,
        articles: list[EnrichedArticle],
        cluster_date: datetime,
    ) -> CrossReference:
        """Build a CrossReference from a cluster of articles."""
        # Topic = most common title words
        all_words = []
        for a in articles:
            all_words.extend(self._tokenize(a.raw.title))
        word_counts = defaultdict(int)
        for w in all_words:
            word_counts[w] += 1
        top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
        topic = " ".join(top_words)

        # Aggregate sentiment
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None

        # Dominant category
        cat_counts = defaultdict(int)
        for a in articles:
            if a.category:
                cat_counts[a.category] += 1
        dominant_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None

        # Sources involved
        sources = list(set(a.raw.source_id for a in articles))

        return CrossReference(
            topic=topic,
            cluster_date=cluster_date,
            avg_sentiment=avg_sentiment,
            dominant_category=dominant_cat,
            sources_involved=sources,
            summary=f"Cluster of {len(articles)} articles about: {topic}",
        )

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize text to lowercase word set, removing stopwords."""
        if not text:
            return set()
        stopwords = {
            "el", "la", "los", "las", "de", "del", "en", "con", "por", "para",
            "un", "una", "que", "se", "su", "al", "es", "no", "lo", "y", "a",
            "the", "an", "in", "of", "to", "for", "and", "is", "on", "at",
        }
        words = set(text.lower().split())
        return words - stopwords

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
