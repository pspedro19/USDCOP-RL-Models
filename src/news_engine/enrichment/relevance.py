"""
Relevance Scorer (SDD-04 §3)
===============================
Scores articles 0.0-1.0 based on keyword matches, source priority, and recency.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


# High-priority keywords with weights
KEYWORD_WEIGHTS = {
    "usdcop": 1.0, "usd cop": 1.0,
    "dolar colombia": 0.9, "tasa de cambio": 0.9,
    "banco de la republica": 0.8, "banrep": 0.8,
    "tasa de interes": 0.7, "fed rate": 0.7,
    "petroleo": 0.6, "embi": 0.7,
    "inflacion colombia": 0.7, "devaluacion": 0.8,
    "remesas": 0.5, "inversion extranjera": 0.5,
    "reforma tributaria": 0.5, "tes": 0.5,
    "riesgo pais": 0.6, "peso colombiano": 0.8,
}

# Source quality weights
SOURCE_WEIGHTS = {
    "gdelt_doc": 0.6,
    "gdelt_context": 0.5,
    "newsapi": 0.7,
    "investing": 0.8,
    "larepublica": 0.9,
    "portafolio": 0.9,
}


def score_relevance(
    title: str,
    content: Optional[str] = None,
    summary: Optional[str] = None,
    source_id: str = "",
    published_at: Optional[datetime] = None,
) -> float:
    """Compute relevance score for an article.

    Components (weighted average):
    - Keyword match score (60%)
    - Source quality (20%)
    - Recency bonus (20%)

    Returns:
        float between 0.0 and 1.0
    """
    text = (title or "") + " " + (content or "") + " " + (summary or "")
    text_lower = text.lower()

    # 1. Keyword score (0-1)
    keyword_score = 0.0
    matched = 0
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if keyword in text_lower:
            keyword_score = max(keyword_score, weight)
            matched += 1

    # Bonus for multiple keyword matches
    if matched >= 3:
        keyword_score = min(1.0, keyword_score + 0.1)

    # 2. Source quality (0-1)
    source_score = SOURCE_WEIGHTS.get(source_id, 0.5)

    # 3. Recency bonus (0-1)
    recency_score = 0.5
    if published_at:
        hours_ago = (datetime.utcnow() - published_at.replace(tzinfo=None)).total_seconds() / 3600
        if hours_ago < 6:
            recency_score = 1.0
        elif hours_ago < 24:
            recency_score = 0.8
        elif hours_ago < 72:
            recency_score = 0.6
        else:
            recency_score = 0.3

    # Weighted combination
    score = (
        keyword_score * 0.60 +
        source_score * 0.20 +
        recency_score * 0.20
    )
    return round(min(1.0, max(0.0, score)), 4)
