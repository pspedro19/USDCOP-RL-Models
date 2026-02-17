"""
Confidence Scorer — Dynamic sizing based on model agreement + magnitude.
========================================================================

Scores ensemble predictions from Ridge + BayesianRidge into 3 tiers:
    HIGH:   Both models agree on direction AND prediction is large
    MEDIUM: Partial agreement or moderate prediction
    LOW:    Large divergence or weak prediction

Each tier maps to a direction-dependent sizing multiplier:
    SHORT: HIGH=2.0x, MEDIUM=1.5x, LOW=1.0x
    LONG:  HIGH=1.0x, MEDIUM=0.5x, LOW=SKIP (0.0x)

Rationale for asymmetric sizing:
    SHORT DA is stable (62-72% across 2022-2025)
    LONG DA is volatile (25-69%), N=13 in best year (2025)
    LONGs are "exploratory" trades — capture alpha when present, minimize damage when absent

@version 1.0.0
@contract FC-H5-CONF-001
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConfidenceTier(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ConfidenceConfig:
    """Thresholds for confidence classification."""
    # Agreement: |ridge_pred - br_pred| threshold
    agreement_tight: float = 0.001    # < 0.1% divergence = strong agreement
    agreement_loose: float = 0.005    # < 0.5% divergence = moderate agreement

    # Magnitude: |ensemble_return| threshold
    magnitude_high: float = 0.010     # > 1.0% predicted move = strong signal
    magnitude_medium: float = 0.005   # > 0.5% predicted move = moderate signal

    # Sizing multipliers by direction
    short_high: float = 2.0
    short_medium: float = 1.5
    short_low: float = 1.0

    long_high: float = 1.0       # Conservative: LONGs capped at 1.0x
    long_medium: float = 0.5     # Exploratory: half size
    long_low: float = 0.0        # SKIP: no trade


@dataclass
class ConfidenceScore:
    """Result of confidence scoring."""
    tier: ConfidenceTier
    agreement: float              # |ridge - br| divergence
    magnitude: float              # |ensemble_return|
    sizing_multiplier: float      # Direction-adjusted multiplier
    skip_trade: bool              # True if LOW confidence LONG → don't trade


def score_confidence(
    ridge_pred: float,
    br_pred: float,
    direction: int,
    config: ConfidenceConfig = ConfidenceConfig(),
) -> ConfidenceScore:
    """
    Score the confidence of an H=5 ensemble prediction.

    Args:
        ridge_pred: Ridge model prediction (log-return).
        br_pred: BayesianRidge model prediction (log-return).
        direction: +1 for LONG, -1 for SHORT.
        config: Confidence thresholds and sizing multipliers.

    Returns:
        ConfidenceScore with tier, multiplier, and skip flag.
    """
    agreement = abs(ridge_pred - br_pred)
    ensemble = (ridge_pred + br_pred) / 2.0
    magnitude = abs(ensemble)

    # Classify tier
    if agreement < config.agreement_tight and magnitude > config.magnitude_high:
        tier = ConfidenceTier.HIGH
    elif agreement < config.agreement_loose and magnitude > config.magnitude_medium:
        tier = ConfidenceTier.MEDIUM
    else:
        tier = ConfidenceTier.LOW

    # Direction-dependent sizing
    if direction == -1:  # SHORT
        mult_map = {
            ConfidenceTier.HIGH: config.short_high,
            ConfidenceTier.MEDIUM: config.short_medium,
            ConfidenceTier.LOW: config.short_low,
        }
    else:  # LONG
        mult_map = {
            ConfidenceTier.HIGH: config.long_high,
            ConfidenceTier.MEDIUM: config.long_medium,
            ConfidenceTier.LOW: config.long_low,
        }

    sizing_multiplier = mult_map[tier]
    skip_trade = sizing_multiplier == 0.0

    return ConfidenceScore(
        tier=tier,
        agreement=agreement,
        magnitude=magnitude,
        sizing_multiplier=sizing_multiplier,
        skip_trade=skip_trade,
    )
