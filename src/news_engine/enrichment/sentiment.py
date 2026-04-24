"""
Sentiment Analyzer (SDD-04 §4)
=================================
Delegates to the hybrid SentimentAnalyzer for multilingual, FX-aware scoring.
Falls back to legacy GDELT tone + VADER if the hybrid analyzer is unavailable.

Output: score (-1.0 to 1.0) + label (positive/negative/neutral).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# VADER lazy-loaded (legacy fallback)
_vader = None


def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed — VADER fallback disabled")
            _vader = False  # Sentinel: tried but failed
    return _vader if _vader is not False else None


def analyze_sentiment(
    title: str,
    content: str | None = None,
    gdelt_tone: float | None = None,
) -> tuple[float | None, str | None]:
    """Analyze sentiment of an article.

    Delegates to hybrid SentimentAnalyzer when available.
    Falls back to legacy GDELT tone + VADER if import fails.

    Returns:
        (score, label) where score is [-1, 1] and label is pos/neg/neutral
    """
    try:
        from src.analysis.sentiment_analyzer import get_analyzer
        analyzer = get_analyzer()
        result = analyzer.analyze_single(
            title=title,
            content=content,
            gdelt_tone=gdelt_tone,
            language=_detect_language(title),
        )
        score = result.fx_adjusted_score
        label = result.label
        if score is not None:
            return score, label
    except Exception as e:
        logger.debug(f"Hybrid analyzer unavailable, using legacy: {e}")

    # Legacy fallback
    return _legacy_analyze_sentiment(title, content, gdelt_tone)


def _legacy_analyze_sentiment(
    title: str,
    content: str | None = None,
    gdelt_tone: float | None = None,
) -> tuple[float | None, str | None]:
    """Legacy sentiment: GDELT tone (primary) + VADER fallback."""
    # Primary: GDELT tone (range: -100 to +100, practical: -20 to +20)
    if gdelt_tone is not None:
        score = _normalize_gdelt_tone(gdelt_tone)
        label = _score_to_label(score)
        return score, label

    # Fallback: VADER
    vader = _get_vader()
    if vader is not None:
        text = (title or "") + ". " + (content or "")[:500]
        try:
            vs = vader.polarity_scores(text)
            score = vs["compound"]  # Already [-1, 1]
            label = _score_to_label(score)
            return score, label
        except Exception as e:
            logger.warning(f"VADER failed: {e}")

    # No sentiment available
    return None, None


def _detect_language(text: str) -> str:
    """Simple language detection for routing."""
    if not text:
        return "en"
    es_markers = {"el", "la", "los", "las", "de", "del", "en", "por", "para", "con", "que", "se"}
    words = set(text.lower().split())
    return "es" if len(words & es_markers) >= 3 else "en"


def _normalize_gdelt_tone(tone: float) -> float:
    """Normalize GDELT tone (-100 to +100) to [-1, 1].

    Practical range is -20 to +20, so we use clamp + scale.
    """
    # Clamp to practical range
    clamped = max(-20.0, min(20.0, tone))
    # Scale to [-1, 1]
    return round(clamped / 20.0, 4)


def _score_to_label(score: float, threshold: float = 0.15) -> str:
    """Convert score to sentiment label."""
    if score >= threshold:
        return "positive"
    elif score <= -threshold:
        return "negative"
    return "neutral"
