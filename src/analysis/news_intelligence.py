"""
News Intelligence Engine (Multi-Agent System — Phase 2)
========================================================
Advanced news processing with:
- Transformer-based sentiment (FinBERT/multilingual, lazy-loaded)
- Semantic clustering (FAISS in-memory + HDBSCAN)
- pgvector RAG for historical context (persistent, SQL-filterable)
- Media bias lookup
- Reuses existing enrichment functions from src/news_engine/enrichment/
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArticleEnriched:
    """An article after enrichment with sentiment + category + relevance."""
    title: str = ""
    url: str = ""
    source: str = ""
    date: str = ""
    language: str = "en"
    tone: float = 0.0          # Sentiment score -1 to +1
    category: str = ""         # e.g., monetary_policy, fx_market
    relevance: float = 0.0     # 0 to 1
    keywords: list[str] = field(default_factory=list)
    bias_label: str = ""       # center, left-center, right-center
    factuality: str = ""       # high, mixed, low
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)  # Don't include 384-dim vector in JSON
        return d


@dataclass
class NewsClusterEnriched:
    """A cluster of related articles."""
    cluster_id: int = 0
    label: str = ""             # LLM-assigned 2-4 word topic
    article_count: int = 0
    avg_sentiment: float = 0.0
    dominant_category: str = ""
    bias_distribution: dict = field(default_factory=dict)
    representative_titles: list[str] = field(default_factory=list)
    narrative_summary: str = ""  # LLM-generated 2-3 sentences
    articles: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsIntelligenceReport:
    """Full news intelligence output for a period."""
    total_articles: int = 0
    relevant_articles: int = 0
    avg_sentiment: float = 0.0
    sentiment_distribution: dict = field(default_factory=dict)
    clusters: list[NewsClusterEnriched] = field(default_factory=list)
    top_stories: list[dict] = field(default_factory=list)
    source_diversity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_articles": self.total_articles,
            "relevant_articles": self.relevant_articles,
            "avg_sentiment": self.avg_sentiment,
            "sentiment_distribution": self.sentiment_distribution,
            "clusters": [c.to_dict() for c in self.clusters],
            "top_stories": self.top_stories,
            "source_diversity": self.source_diversity,
        }


# ---------------------------------------------------------------------------
# Media bias lookup (~30 known outlets)
# ---------------------------------------------------------------------------

MEDIA_BIAS: dict[str, tuple[str, str]] = {
    # (bias_label, factuality)
    # 5-tier scale: left, center-left, center, center-right, right
    # Factuality: high, mixed, low
    # --- International (English) ---
    "reuters.com": ("center", "high"),
    "apnews.com": ("center", "high"),
    "bloomberg.com": ("center-right", "high"),
    "ft.com": ("center", "high"),
    "wsj.com": ("center-right", "high"),
    "nytimes.com": ("center-left", "high"),
    "washingtonpost.com": ("center-left", "high"),
    "bbc.com": ("center", "high"),
    "bbc.co.uk": ("center", "high"),
    "cnbc.com": ("center-right", "high"),
    "cnn.com": ("center-left", "mixed"),
    "foxnews.com": ("right", "mixed"),
    "foxbusiness.com": ("right", "mixed"),
    "economist.com": ("center", "high"),
    "theguardian.com": ("center-left", "high"),
    "aljazeera.com": ("center", "mixed"),
    "france24.com": ("center", "high"),
    "dw.com": ("center", "high"),
    "investing.com": ("center", "high"),
    "marketwatch.com": ("center", "high"),
    "yahoo.com": ("center", "mixed"),
    "finance.yahoo.com": ("center", "mixed"),
    "barrons.com": ("center-right", "high"),
    "seekingalpha.com": ("center-right", "mixed"),
    "forbes.com": ("center-right", "high"),
    # --- Colombia ---
    "eltiempo.com": ("center", "mixed"),
    "portafolio.co": ("center", "high"),
    "larepublica.co": ("center", "high"),
    "semana.com": ("center-right", "mixed"),
    "elespectador.com": ("center-left", "high"),
    "dinero.com": ("center", "high"),
    "valora.com": ("center", "mixed"),
    "pulzo.com": ("center", "mixed"),
    "elcolombiano.com": ("center", "mixed"),
    "caracol.com.co": ("center", "mixed"),
    "caracoltv.com": ("center", "mixed"),
    "rcnradio.com": ("center-right", "mixed"),
    "noticiasrcn.com": ("center-right", "mixed"),
    "bluradio.com": ("center", "mixed"),
    "lafm.com.co": ("center", "mixed"),
    "wradio.com.co": ("center", "mixed"),
    "elheraldo.co": ("center", "mixed"),
    "eluniversal.com.co": ("center", "mixed"),
    "elpais.com.co": ("center", "mixed"),
    "lasillavacia.com": ("center-left", "high"),
    "razonpublica.com": ("center-left", "high"),
    "cambio.com.co": ("center-left", "mixed"),
    # --- LATAM ---
    "elfinanciero.com.mx": ("center", "high"),
    "expansion.mx": ("center", "mixed"),
    "eleconomista.com.mx": ("center", "high"),
    "americaeconomia.com": ("center", "high"),
    "infobae.com": ("center", "mixed"),
    # --- Official / Institutional ---
    "banrep.gov.co": ("center", "high"),
    "dane.gov.co": ("center", "high"),
    "minhacienda.gov.co": ("center", "high"),
    "imf.org": ("center", "high"),
    "worldbank.org": ("center", "high"),
    "federalreserve.gov": ("center", "high"),
}


def get_media_bias(source: str) -> tuple[str, str]:
    """Look up media bias for a source domain.

    Returns:
        (bias_label, factuality). Defaults to ("unknown", "unknown").
    """
    source_lower = source.lower().strip()
    # Try exact match first
    if source_lower in MEDIA_BIAS:
        return MEDIA_BIAS[source_lower]
    # Try domain extraction
    for domain, bias in MEDIA_BIAS.items():
        if domain in source_lower:
            return bias
    return ("unknown", "unknown")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NewsIntelligenceEngine:
    """Advanced news processing with NLP sentiment, clustering, and RAG."""

    def __init__(self) -> None:
        self._sentiment_model = None
        self._embedding_model = None
        self._models_loaded = False

    def _load_models(self) -> None:
        """Lazy-load transformer models (heavy, only when needed).

        Sentiment is delegated to SentimentAnalyzer (hybrid ensemble).
        Only the embedding model is loaded here for clustering.
        """
        if self._models_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.info("Loaded sentence-transformer model (384d)")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformer: {e}")

        # Sentiment is now handled by SentimentAnalyzer (src/analysis/sentiment_analyzer.py)
        # which loads multilingual + FinBERT models internally.
        try:
            from src.analysis.sentiment_analyzer import get_analyzer
            self._sentiment_analyzer = get_analyzer()
            logger.info("Using hybrid SentimentAnalyzer for sentiment scoring")
        except Exception as e:
            logger.warning(f"Failed to load SentimentAnalyzer: {e}. Will use fallback.")
            self._sentiment_analyzer = None

        self._models_loaded = True

    # ------------------------------------------------------------------
    # Article Processing
    # ------------------------------------------------------------------

    def process_articles(
        self,
        articles: list[dict],
        min_relevance: float = 0.3,
    ) -> list[ArticleEnriched]:
        """Enrich raw articles with sentiment, category, relevance, bias.

        Args:
            articles: List of dicts with at least {title, source, date}.
            min_relevance: Minimum relevance score to keep.

        Returns:
            List of enriched articles above relevance threshold.
        """
        enriched = []

        for art in articles:
            title = str(art.get("title", ""))
            if not title or len(title) < 10:
                continue

            source = str(art.get("source", art.get("domain", "")))
            url = str(art.get("url", ""))

            # Relevance scoring (reuse existing logic if available)
            relevance = self._score_relevance(title, source)
            if relevance < min_relevance:
                continue

            # Category
            category = self._categorize(title)

            # Sentiment
            tone = self._compute_sentiment(title, art.get("tone"))

            # Media bias
            bias_label, factuality = get_media_bias(source)

            # Keywords
            keywords = self._extract_keywords(title)

            enriched.append(ArticleEnriched(
                title=title[:300],
                url=url,
                source=source,
                date=str(art.get("date", ""))[:10],
                language=str(art.get("language", "en")),
                tone=round(tone, 3),
                category=category,
                relevance=round(relevance, 3),
                keywords=keywords,
                bias_label=bias_label,
                factuality=factuality,
            ))

        # Sort by relevance
        enriched.sort(key=lambda x: x.relevance, reverse=True)
        return enriched

    def _score_relevance(self, title: str, source: str) -> float:
        """Score article relevance to USDCOP trading (0-1)."""
        try:
            from src.news_engine.enrichment.relevance import score_relevance
            return score_relevance(title, source)
        except ImportError:
            pass

        # Inline fallback
        title_lower = title.lower()
        score = 0.0

        high_keywords = ["usd/cop", "dolar colombiano", "peso colombiano", "banrep",
                        "tasa de cambio", "divisa colombia"]
        medium_keywords = ["colombia", "petroleo", "fed", "dxy", "embi", "emergentes",
                          "latam", "oil price", "dollar", "treasury", "vix"]
        low_keywords = ["forex", "cambio", "economia", "inflacion", "tasas de interes"]

        for kw in high_keywords:
            if kw in title_lower:
                score += 0.4
        for kw in medium_keywords:
            if kw in title_lower:
                score += 0.15
        for kw in low_keywords:
            if kw in title_lower:
                score += 0.05

        # Source quality bonus
        quality_sources = {"reuters", "bloomberg", "portafolio", "larepublica", "banrep"}
        if any(s in source.lower() for s in quality_sources):
            score += 0.1

        return min(score, 1.0)

    def _categorize(self, title: str) -> str:
        """Categorize article into one of 9 financial categories."""
        try:
            from src.news_engine.enrichment.categorizer import categorize_article
            result = categorize_article(title)
            # categorize_article returns (category, subcategory) tuple
            if isinstance(result, tuple):
                return result[0] or "general"
            return result or "general"
        except ImportError:
            pass

        # Inline fallback
        title_lower = title.lower()
        categories = {
            "monetary_policy": ["tasa", "fed", "banrep", "interest rate", "rate hike", "rate cut", "monetary"],
            "fx_market": ["dolar", "peso", "usd/cop", "forex", "exchange rate", "divisa", "cambio"],
            "commodities": ["petroleo", "oil", "crude", "wti", "brent", "gold", "oro", "commodity"],
            "inflation": ["inflacion", "cpi", "ipc", "consumer price", "inflation"],
            "fiscal_policy": ["fiscal", "presupuesto", "budget", "spending", "deficit", "debt"],
            "risk_premium": ["embi", "riesgo pais", "spread", "credit default", "country risk"],
            "capital_flows": ["inversion extranjera", "foreign investment", "capital flow", "fdi"],
            "political": ["politica", "congreso", "presidente", "election", "gobierno"],
        }

        for cat, keywords in categories.items():
            if any(kw in title_lower for kw in keywords):
                return cat
        return "general"

    def _compute_sentiment(self, title: str, gdelt_tone: Optional[float] = None) -> float:
        """Compute sentiment score (-1 to +1).

        Delegates to SentimentAnalyzer (hybrid ensemble) when available.
        Falls back to legacy GDELT tone + VADER.
        """
        # Try hybrid SentimentAnalyzer
        analyzer = getattr(self, "_sentiment_analyzer", None)
        if analyzer is not None:
            try:
                result = analyzer.analyze_single(
                    title=title,
                    gdelt_tone=gdelt_tone,
                )
                return result.fx_adjusted_score
            except Exception:
                pass

        # GDELT tone fallback (already -1 to +1 range approximately)
        if gdelt_tone is not None and gdelt_tone != 0:
            return max(-1.0, min(1.0, gdelt_tone / 10.0))

        # VADER / keyword fallback
        try:
            from src.news_engine.enrichment.sentiment import analyze_sentiment
            score, _label = analyze_sentiment(title)
            return max(-1.0, min(1.0, float(score)))
        except (ImportError, Exception):
            pass

        return 0.0

    def _extract_keywords(self, title: str) -> list[str]:
        """Extract keywords from title."""
        try:
            from src.news_engine.enrichment.tagger import extract_keywords
            return extract_keywords(title)
        except ImportError:
            pass

        # Simple fallback
        important_words = [
            "dolar", "peso", "petroleo", "oil", "banrep", "fed", "inflacion",
            "tasa", "embi", "vix", "treasury", "colombia", "recesion", "crisis",
        ]
        title_lower = title.lower()
        return [w for w in important_words if w in title_lower][:5]

    # ------------------------------------------------------------------
    # Embedding + Clustering (FAISS in-memory)
    # ------------------------------------------------------------------

    def embed_titles(self, titles: list[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a list of titles.

        Returns:
            np.ndarray of shape (n, 384) or None if model unavailable.
        """
        self._load_models()
        if self._embedding_model is None:
            return None

        try:
            embeddings = self._embedding_model.encode(
                titles,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def cluster_articles(
        self,
        articles: list[ArticleEnriched],
        min_cluster_size: int = 3,
    ) -> list[NewsClusterEnriched]:
        """Cluster articles using FAISS + HDBSCAN.

        Args:
            articles: Enriched articles to cluster.
            min_cluster_size: Minimum articles per cluster.

        Returns:
            List of clusters with aggregate stats.
        """
        if len(articles) < min_cluster_size:
            return []

        titles = [a.title for a in articles]
        embeddings = self.embed_titles(titles)

        if embeddings is None:
            # Fallback: simple keyword-based grouping by category
            return self._cluster_by_category(articles)

        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(embeddings)

            # Group articles by cluster
            cluster_map: dict[int, list[ArticleEnriched]] = {}
            for i, label in enumerate(labels):
                if label == -1:  # noise
                    continue
                cluster_map.setdefault(label, []).append(articles[i])

            clusters = []
            for cluster_id, arts in sorted(cluster_map.items()):
                sentiments = [a.tone for a in arts]
                categories = [a.category for a in arts if a.category]
                sources = [a.source for a in arts if a.source]
                biases = [a.bias_label for a in arts if a.bias_label and a.bias_label != "unknown"]

                dominant_cat = max(set(categories), key=categories.count) if categories else ""
                bias_dist = {}
                for b in biases:
                    bias_dist[b] = bias_dist.get(b, 0) + 1

                clusters.append(NewsClusterEnriched(
                    cluster_id=int(cluster_id),
                    article_count=len(arts),
                    avg_sentiment=round(float(np.mean(sentiments)), 3) if sentiments else 0,
                    dominant_category=dominant_cat,
                    bias_distribution=bias_dist,
                    representative_titles=[a.title for a in arts[:5]],
                    articles=[a.to_dict() for a in arts],
                ))

            # Sort by article count
            clusters.sort(key=lambda c: c.article_count, reverse=True)
            return clusters

        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}")
            return self._cluster_by_category(articles)

    def _cluster_by_category(
        self, articles: list[ArticleEnriched]
    ) -> list[NewsClusterEnriched]:
        """Fallback clustering by category."""
        cat_map: dict[str, list[ArticleEnriched]] = {}
        for art in articles:
            cat = art.category or "general"
            cat_map.setdefault(cat, []).append(art)

        clusters = []
        for i, (cat, arts) in enumerate(sorted(cat_map.items(), key=lambda x: -len(x[1]))):
            if len(arts) < 2:
                continue
            sentiments = [a.tone for a in arts]
            clusters.append(NewsClusterEnriched(
                cluster_id=i,
                label=cat.replace("_", " ").title(),
                article_count=len(arts),
                avg_sentiment=round(float(np.mean(sentiments)), 3) if sentiments else 0,
                dominant_category=cat,
                representative_titles=[a.title for a in arts[:5]],
                articles=[a.to_dict() for a in arts],
            ))

        return clusters[:10]

    async def label_clusters_with_llm(
        self,
        clusters: list[NewsClusterEnriched],
        llm_client: object,
    ) -> list[NewsClusterEnriched]:
        """Use LLM (Haiku) to label each cluster with a 2-4 word topic.

        Args:
            clusters: Clusters to label.
            llm_client: LLMClient instance with generate() method.

        Returns:
            Same clusters with label and narrative_summary filled.
        """
        for cluster in clusters:
            titles_str = "\n".join(f"- {t}" for t in cluster.representative_titles[:5])
            prompt = (
                f"Dado estos {cluster.article_count} articulos financieros:\n{titles_str}\n\n"
                f"1. Genera un titulo de 2-4 palabras que resuma el tema principal.\n"
                f"2. Escribe un resumen de 2-3 oraciones en español.\n\n"
                f"Formato:\nTITULO: <titulo>\nRESUMEN: <resumen>"
            )

            try:
                result = llm_client.generate(
                    system="Eres un editor financiero. Responde solo el formato pedido.",
                    prompt=prompt,
                    max_tokens=200,
                    cache_key=f"cluster_{cluster.cluster_id}_{hashlib.md5(titles_str.encode()).hexdigest()[:8]}",
                )
                content = result.get("content", "")

                # Parse response
                for line in content.split("\n"):
                    if line.startswith("TITULO:"):
                        cluster.label = line.replace("TITULO:", "").strip()[:50]
                    elif line.startswith("RESUMEN:"):
                        cluster.narrative_summary = line.replace("RESUMEN:", "").strip()[:500]

                if not cluster.label:
                    cluster.label = cluster.dominant_category.replace("_", " ").title()

            except Exception as e:
                logger.warning(f"LLM cluster labeling failed: {e}")
                cluster.label = cluster.dominant_category.replace("_", " ").title()

        return clusters

    # ------------------------------------------------------------------
    # pgvector RAG (persistent)
    # ------------------------------------------------------------------

    def search_historical_context(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[dict] = None,
        db_conn: object = None,
    ) -> list[dict]:
        """Search historical articles via pgvector with SQL metadata filtering.

        Args:
            query: Search query text.
            n_results: Number of results.
            filters: Optional dict with keys: date_from, date_to, category, language, bias_label.
            db_conn: psycopg2 connection (or None to skip).

        Returns:
            List of matching article dicts with title, source, date, similarity.
        """
        if db_conn is None:
            logger.debug("No DB connection for pgvector search")
            return []

        self._load_models()
        if self._embedding_model is None:
            return []

        try:
            # Generate query embedding
            query_emb = self._embedding_model.encode([query], normalize_embeddings=True)[0]
            emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"

            # Build SQL with filters
            where_clauses = []
            params = [emb_str, n_results]

            if filters:
                if filters.get("date_from"):
                    where_clauses.append("fecha >= %s")
                    params.insert(-1, filters["date_from"])
                if filters.get("date_to"):
                    where_clauses.append("fecha <= %s")
                    params.insert(-1, filters["date_to"])
                if filters.get("category"):
                    where_clauses.append("category = %s")
                    params.insert(-1, filters["category"])
                if filters.get("language"):
                    where_clauses.append("language = %s")
                    params.insert(-1, filters["language"])

            where_str = " AND ".join(where_clauses) if where_clauses else "1=1"

            sql = f"""
                SELECT title, url, source, fecha, category, tone,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM gdelt_embeddings
                WHERE {where_str}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            # We need the embedding param twice (for WHERE and ORDER BY)
            params_final = [emb_str] + params[:-1] + [emb_str, n_results]

            with db_conn.cursor() as cur:
                cur.execute(sql, params_final)
                rows = cur.fetchall()

            return [
                {
                    "title": row[0],
                    "url": row[1],
                    "source": row[2],
                    "date": str(row[3]),
                    "category": row[4],
                    "tone": row[5],
                    "similarity": round(float(row[6]), 3),
                }
                for row in rows
            ]

        except Exception as e:
            logger.warning(f"pgvector search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        articles: list[dict],
        week_start: str,
        week_end: str,
        min_relevance: float = 0.3,
    ) -> NewsIntelligenceReport:
        """Run full news intelligence analysis.

        Args:
            articles: Raw article dicts.
            week_start: ISO date.
            week_end: ISO date.
            min_relevance: Min relevance score.

        Returns:
            NewsIntelligenceReport.
        """
        # Enrich
        enriched = self.process_articles(articles, min_relevance=min_relevance)

        # Cluster
        clusters = self.cluster_articles(enriched)

        # Compute aggregate stats
        sentiments = [a.tone for a in enriched]
        positive = sum(1 for s in sentiments if s > 0.15)
        negative = sum(1 for s in sentiments if s < -0.15)
        neutral = len(sentiments) - positive - negative

        # Source diversity
        sources = [a.source for a in enriched if a.source]
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        # Top stories (highest relevance)
        top_stories = [a.to_dict() for a in enriched[:10]]

        return NewsIntelligenceReport(
            total_articles=len(articles),
            relevant_articles=len(enriched),
            avg_sentiment=round(float(np.mean(sentiments)), 3) if sentiments else 0,
            sentiment_distribution={
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            },
            clusters=clusters,
            top_stories=top_stories,
            source_diversity=source_counts,
        )
