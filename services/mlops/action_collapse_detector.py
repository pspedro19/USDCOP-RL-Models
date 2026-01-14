"""
Action Collapse Detector
========================

Detects when an RL model is exhibiting "mode collapse" behavior,
i.e., predicting the same action repeatedly regardless of market conditions.

This is a critical MLOps monitor for production systems.

Signals:
- Low entropy in action distribution indicates collapse
- Single action dominating > 80% indicates potential collapse
- Sudden shift in action distribution indicates drift

Author: Trading Team
Date: 2026-01-14
Version: 1.0.0
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Trading action types."""
    HOLD = "HOLD"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class ActionCollapseResult:
    """Result of action collapse detection analysis."""

    # Entropy (0 = collapsed, log2(3) ≈ 1.58 = uniform)
    entropy: float

    # Action distribution {HOLD: %, LONG: %, SHORT: %}
    distribution: Dict[str, float]

    # Whether collapse is detected
    is_collapsed: bool

    # Most frequent action
    dominant_action: str

    # Percentage of dominant action
    dominant_pct: float

    # Warning message if any
    warning: Optional[str] = None

    # Severity level (0=OK, 1=WARNING, 2=CRITICAL)
    severity: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "entropy": round(self.entropy, 4),
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "is_collapsed": self.is_collapsed,
            "dominant_action": self.dominant_action,
            "dominant_pct": round(self.dominant_pct, 4),
            "warning": self.warning,
            "severity": self.severity,
        }


@dataclass
class ActionCollapseConfig:
    """Configuration for action collapse detection."""

    # Entropy threshold below which collapse is detected
    # log2(3) ≈ 1.58 for uniform distribution
    # 0.5 is very low entropy (near-collapse)
    entropy_threshold: float = 0.5

    # Percentage threshold for dominant action (0-1)
    # 0.8 means 80% same action triggers warning
    dominance_threshold: float = 0.80

    # Minimum samples required for analysis
    min_samples: int = 50

    # Window size for rolling analysis
    window_size: int = 200

    # Expected healthy HOLD percentage range
    expected_hold_min: float = 0.30  # At least 30% HOLD
    expected_hold_max: float = 0.80  # At most 80% HOLD


class ActionCollapseDetector:
    """
    Detects action collapse (mode collapse) in RL model predictions.

    Action collapse occurs when the model learns to always output the same
    action regardless of the observation. This is a critical failure mode.

    Usage:
        detector = ActionCollapseDetector()

        # During inference loop
        for action in model_actions:
            detector.record(action)

        # Check for collapse
        result = detector.check()
        if result.is_collapsed:
            logger.critical(f"Mode collapse detected: {result.warning}")
    """

    def __init__(self, config: Optional[ActionCollapseConfig] = None):
        """
        Initialize the action collapse detector.

        Args:
            config: Detection configuration. Uses defaults if not provided.
        """
        self.config = config or ActionCollapseConfig()
        self._action_history: deque = deque(maxlen=self.config.window_size)
        self._total_actions = 0

    def record(self, action: str) -> None:
        """
        Record a new action.

        Args:
            action: Action string ("HOLD", "LONG", "SHORT")
        """
        # Normalize action name
        action_upper = action.upper()
        if action_upper not in ["HOLD", "LONG", "SHORT"]:
            logger.warning(f"Unknown action type: {action}, treating as HOLD")
            action_upper = "HOLD"

        self._action_history.append(action_upper)
        self._total_actions += 1

    def record_batch(self, actions: List[str]) -> None:
        """
        Record multiple actions at once.

        Args:
            actions: List of action strings
        """
        for action in actions:
            self.record(action)

    def check(self, action_history: Optional[List[str]] = None) -> ActionCollapseResult:
        """
        Check for action collapse.

        Args:
            action_history: Optional explicit action history to analyze.
                          If not provided, uses internal history.

        Returns:
            ActionCollapseResult with analysis details
        """
        # Use provided history or internal history
        history = action_history if action_history is not None else list(self._action_history)

        if len(history) < self.config.min_samples:
            return ActionCollapseResult(
                entropy=1.58,  # Assume uniform
                distribution={"HOLD": 0.33, "LONG": 0.33, "SHORT": 0.33},
                is_collapsed=False,
                dominant_action="UNKNOWN",
                dominant_pct=0.33,
                warning=f"Insufficient samples ({len(history)}/{self.config.min_samples})",
                severity=0,
            )

        # Calculate distribution
        counts = {"HOLD": 0, "LONG": 0, "SHORT": 0}
        for action in history:
            action_upper = action.upper()
            if action_upper in counts:
                counts[action_upper] += 1

        total = len(history)
        distribution = {k: v / total for k, v in counts.items()}

        # Calculate entropy
        entropy = self._calculate_entropy(distribution)

        # Find dominant action
        dominant_action = max(distribution, key=distribution.get)
        dominant_pct = distribution[dominant_action]

        # Determine collapse status
        is_collapsed = False
        severity = 0
        warnings = []

        # Check entropy threshold
        if entropy < self.config.entropy_threshold:
            is_collapsed = True
            severity = 2
            warnings.append(f"Low entropy ({entropy:.3f} < {self.config.entropy_threshold})")

        # Check dominance threshold
        if dominant_pct > self.config.dominance_threshold:
            is_collapsed = True
            severity = max(severity, 2)
            warnings.append(f"High dominance ({dominant_pct:.1%} {dominant_action})")

        # Check HOLD percentage in healthy range
        hold_pct = distribution.get("HOLD", 0)
        if hold_pct < self.config.expected_hold_min:
            severity = max(severity, 1)
            warnings.append(f"Low HOLD ({hold_pct:.1%} < {self.config.expected_hold_min:.1%})")
        elif hold_pct > self.config.expected_hold_max:
            severity = max(severity, 1)
            warnings.append(f"High HOLD ({hold_pct:.1%} > {self.config.expected_hold_max:.1%})")

        # Check for zero trading (all HOLD)
        if distribution.get("LONG", 0) == 0 and distribution.get("SHORT", 0) == 0:
            is_collapsed = True
            severity = 2
            warnings.append("No trading actions (100% HOLD)")

        warning_msg = "; ".join(warnings) if warnings else None

        return ActionCollapseResult(
            entropy=entropy,
            distribution=distribution,
            is_collapsed=is_collapsed,
            dominant_action=dominant_action,
            dominant_pct=dominant_pct,
            warning=warning_msg,
            severity=severity,
        )

    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of action distribution.

        H = -Σ p(x) * log2(p(x))

        Args:
            distribution: Action probability distribution

        Returns:
            Entropy value (0 to log2(3) ≈ 1.58)
        """
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy

    def get_recent_distribution(self, n_samples: int = 50) -> Dict[str, float]:
        """
        Get action distribution for the most recent N samples.

        Args:
            n_samples: Number of recent samples to analyze

        Returns:
            Distribution dictionary
        """
        recent = list(self._action_history)[-n_samples:]
        if not recent:
            return {"HOLD": 0.33, "LONG": 0.33, "SHORT": 0.33}

        counts = {"HOLD": 0, "LONG": 0, "SHORT": 0}
        for action in recent:
            if action in counts:
                counts[action] += 1

        total = len(recent)
        return {k: v / total for k, v in counts.items()}

    def detect_distribution_shift(
        self,
        window_a: int = 100,
        window_b: int = 100
    ) -> Tuple[bool, float]:
        """
        Detect if action distribution has shifted between two windows.

        Uses Jensen-Shannon divergence to measure distribution shift.

        Args:
            window_a: Size of first window (older)
            window_b: Size of second window (newer)

        Returns:
            Tuple of (shift_detected, divergence_score)
        """
        history = list(self._action_history)
        if len(history) < window_a + window_b:
            return False, 0.0

        # Get distributions for both windows
        old_window = history[-(window_a + window_b):-window_b]
        new_window = history[-window_b:]

        dist_old = self._count_to_dist(old_window)
        dist_new = self._count_to_dist(new_window)

        # Calculate JS divergence
        js_div = self._js_divergence(dist_old, dist_new)

        # Threshold for significant shift
        shift_threshold = 0.1
        return js_div > shift_threshold, js_div

    def _count_to_dist(self, actions: List[str]) -> Dict[str, float]:
        """Convert action list to distribution."""
        counts = {"HOLD": 0, "LONG": 0, "SHORT": 0}
        for action in actions:
            if action in counts:
                counts[action] += 1
        total = len(actions) if actions else 1
        return {k: v / total for k, v in counts.items()}

    def _js_divergence(
        self,
        p: Dict[str, float],
        q: Dict[str, float]
    ) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        # Calculate midpoint distribution
        m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in set(p) | set(q)}

        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        def kl_div(dist1, dist2):
            kl = 0.0
            for k, v in dist1.items():
                if v > 0 and dist2.get(k, 0) > 0:
                    kl += v * math.log2(v / dist2[k])
            return kl

        return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

    def reset(self) -> None:
        """Reset the detector state."""
        self._action_history.clear()
        self._total_actions = 0

    @property
    def sample_count(self) -> int:
        """Get current sample count."""
        return len(self._action_history)

    @property
    def total_recorded(self) -> int:
        """Get total actions ever recorded."""
        return self._total_actions

    def __repr__(self) -> str:
        return (
            f"ActionCollapseDetector("
            f"samples={self.sample_count}, "
            f"entropy_threshold={self.config.entropy_threshold})"
        )


# Convenience function
def check_action_collapse(
    actions: List[str],
    entropy_threshold: float = 0.5,
    dominance_threshold: float = 0.80
) -> ActionCollapseResult:
    """
    Quick check for action collapse in a list of actions.

    Args:
        actions: List of action strings
        entropy_threshold: Entropy below this triggers collapse
        dominance_threshold: Single action above this triggers collapse

    Returns:
        ActionCollapseResult
    """
    config = ActionCollapseConfig(
        entropy_threshold=entropy_threshold,
        dominance_threshold=dominance_threshold,
        min_samples=10,  # Lower for quick checks
    )
    detector = ActionCollapseDetector(config)
    return detector.check(actions)
