"""
Outlook Accuracy Tracker (SDD)
==============================
Compares W{N-1} outlook prediction against W{N} actual results.
Called at the START of generate_for_week() to evaluate prior predictions.

100% deterministic — no LLM calls.
"""

from __future__ import annotations

ACCURACY_LABELS = {
    "hit": "✅ Cumplido",
    "partial": "〜 Parcial",
    "miss": "❌ No cumplido",
    "n/a": "— Sin outlook previo",
}


def evaluate_outlook(prior_outlook: dict | None, actual: dict) -> str:
    """
    Compare prior week's outlook against actual results.

    Args:
        prior_outlook: The 'outlook_next_week' dict from W{N-1} JSON, or None
        actual: Dict with 'close' and 'change_pct' from current week

    Returns:
        One of: "hit", "partial", "miss", "n/a"
    """
    if not prior_outlook:
        return "n/a"

    bias = prior_outlook.get("bias", "neutral")
    chg = actual.get("change_pct", 0)
    close = actual.get("close", 0)
    low = prior_outlook.get("range_low", 0)
    high = prior_outlook.get("range_high", float("inf"))

    # Bias check:
    # bearish_cop = USD sube (chg > 0)
    # bullish_cop = USD baja (chg < 0)
    bias_hit = (
        (bias == "bearish_cop" and chg > 0.2)
        or (bias == "bullish_cop" and chg < -0.2)
        or (bias == "neutral" and abs(chg) <= 0.3)
    )

    range_hit = low <= close <= high

    if bias_hit and range_hit:
        return "hit"
    if bias_hit or range_hit:
        return "partial"
    return "miss"
