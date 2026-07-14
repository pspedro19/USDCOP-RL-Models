"""
News Enrichment (shared, LangGraph-independent)
===============================================
Single source of truth for turning a list of raw article dicts into the two
dashboard-facing news blocks:

  - ``news_intelligence``      → NewsClusterCard  (clusters, sentiment, sources)
  - ``political_bias_analysis`` → BiasDistributionCard (source bias + factuality)

Both the USD/COP weekly generator (``weekly_generator.py``) and the Gold/BTC
asset generator (``asset_analysis_generator.py``) call ``enrich_news`` so the
two tracks emit an identical, contract-conforming shape. This deliberately does
**not** depend on the LangGraph multi-agent pipeline (which silently no-ops when
its heavy deps are absent, e.g. in the Airflow/Docker generation env) — it reuses
the already-complete ``NewsIntelligenceEngine`` + ``PoliticalBiasDetector``, both
of which degrade gracefully (HDBSCAN → category clustering; sentence-transformer /
FinBERT sentiment → GDELT tone → 0). Mirrors the ``_compute_regime_direct``
pattern that keeps ``macro_regime`` populated regardless of LangGraph.

Business-logic alignment (see ``.claude/rules/quant-constitution.md``):
  - News is classified/summarized only — it never touches a trading decision.
  - LLM polish (cluster topic labels + bias narrative) is **budget-gated and
    opt-in** via ``allow_llm``; the deterministic path (category labels +
    source-based bias) always runs at zero cost.
  - GDELT tone is preferred over VADER for GDELT articles (data-governance rule),
    which the underlying engine already honours.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def enrich_news(
    articles: list[dict],
    week_start: str,
    week_end: str,
    *,
    min_relevance: float = 0.3,
    min_kept: int = 5,
    llm_client: object | None = None,
    allow_llm: bool = False,
) -> tuple[dict | None, dict | None]:
    """Compute (news_intelligence, political_bias_analysis) from raw articles.

    Args:
        articles: Raw article dicts with at least {title, source}. Optional
            keys used when present: url, date, tone, language.
        week_start / week_end: ISO dates (for context / provenance only).
        min_relevance: Preferred USD/COP relevance floor. Pass 0.0 for
            pre-filtered asset feeds (Gold/BTC already query asset terms).
        min_kept: If fewer than this many articles clear ``min_relevance``, fall
            back to the full set so a genuinely-covered week never renders an
            empty panel (adaptive floor). Set 0 to disable the fallback.
        llm_client: Optional LLMClient (``.generate(...)``) for topic labels +
            bias narrative. Only used when ``allow_llm`` is True.
        allow_llm: When True *and* ``llm_client`` is provided, enrich cluster
            labels + bias narrative via budget-gated LLM calls. Defaults to the
            deterministic (zero-cost) path.

    Returns:
        (news_intelligence_dict, political_bias_dict). Either may be None when
        there are no usable articles. Both are JSON-safe (floats rounded, no
        Infinity/NaN) — the caller still runs ``_sanitize_for_json`` on export.
    """
    if not articles:
        return None, None

    try:
        import numpy as np

        from src.analysis.news_intelligence import (
            NewsClusterEnriched,
            NewsIntelligenceEngine,
            NewsIntelligenceReport,
        )
    except Exception as e:  # pragma: no cover - import guard
        logger.warning(f"NewsIntelligenceEngine unavailable: {e}")
        return None, None

    engine = NewsIntelligenceEngine()

    # Single enrichment pass (sentiment is the expensive step) — score every
    # article, then choose the floor adaptively.
    enriched_all = engine.process_articles(articles, min_relevance=0.0)
    if not enriched_all:
        return None, None

    high = [a for a in enriched_all if a.relevance >= min_relevance]
    enriched = high if len(high) >= min_kept else enriched_all
    if not enriched:
        return None, None

    clusters = engine.cluster_articles(enriched)

    # Deterministic floor: every cluster carries a human label. HDBSCAN clusters
    # come back label-less; fall back to the dominant category (Title Cased).
    for i, cluster in enumerate(clusters):
        if not cluster.label:
            cluster.label = (cluster.dominant_category or f"tema {i + 1}").replace("_", " ").title()

    # Optional LLM polish — topic labels + narrative summaries (budget-gated).
    use_llm = bool(allow_llm and llm_client is not None)
    if use_llm and clusters:
        try:
            import asyncio

            clusters = asyncio.run(engine.label_clusters_with_llm(clusters, llm_client))
        except Exception as e:
            logger.warning(f"LLM cluster labeling skipped: {e}")

    # Aggregate stats over the chosen article set.
    sentiments = [a.tone for a in enriched]
    positive = sum(1 for s in sentiments if s > 0.15)
    negative = sum(1 for s in sentiments if s < -0.15)
    source_counts: dict[str, int] = {}
    for a in enriched:
        if a.source:
            source_counts[a.source] = source_counts.get(a.source, 0) + 1

    report = NewsIntelligenceReport(
        total_articles=len(articles),
        relevant_articles=len(enriched),
        avg_sentiment=round(float(np.mean(sentiments)), 3) if sentiments else 0.0,
        sentiment_distribution={
            "positive": positive,
            "negative": negative,
            "neutral": len(sentiments) - positive - negative,
        },
        clusters=clusters,
        top_stories=[a.to_dict() for a in enriched[:10]],
        source_diversity=source_counts,
    )
    news_intelligence = report.to_dict()

    # Political bias — Layer 1 (source lookup) is always free; Layer 2 (cluster
    # narrative bias) only fires when LLM is allowed. Detector is defensive.
    political_bias: dict | None = None
    try:
        from src.analysis.bias_detector import PoliticalBiasDetector

        political_bias = PoliticalBiasDetector().analyze(
            articles=[a.to_dict() for a in enriched],
            clusters=news_intelligence.get("clusters", []),
            llm_client=llm_client if use_llm else None,
        )
    except Exception as e:
        logger.warning(f"Political bias analysis skipped: {e}")

    return news_intelligence, political_bias
