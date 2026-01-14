"""
Regime Detector Module for USD/COP Trading System
==================================================

This module implements a market regime detection system that classifies market
conditions into NORMAL, VOLATILE, and CRISIS regimes based on macroeconomic
indicators and realized volatility.

Author: USD/COP RL Trading System
Date: 2025-12-25
Version: 1.0.0

Key Features:
- Multi-factor regime detection using VIX, EMBI, and realized volatility
- Probabilistic regime classification
- Position sizing adjustments based on market regime
- Historical validation and regime distribution analysis
- Integration with RL trading environment

Usage Example:
    >>> detector = RegimeDetector()
    >>> regime = detector.detect_regime(vix_z=1.5, embi_z=0.8, vol_pct=72.0)
    >>> print(regime)  # "VOLATILE"
    >>> multiplier = detector.get_position_multiplier(regime)
    >>> print(multiplier)  # 0.5
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import warnings


class MarketRegime(Enum):
    """Market regime classifications."""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds."""

    # VIX z-score thresholds
    vix_crisis_threshold: float = 2.0
    vix_volatile_threshold: float = 1.0

    # EMBI z-score thresholds
    embi_crisis_threshold: float = 2.0
    embi_volatile_threshold: float = 1.0

    # Realized volatility percentile thresholds
    vol_crisis_percentile: float = 95.0
    vol_volatile_percentile: float = 75.0

    # Position multipliers by regime
    crisis_multiplier: float = 0.0
    volatile_multiplier: float = 0.5
    normal_multiplier: float = 1.0

    # Probabilities weighting
    vix_weight: float = 0.4
    embi_weight: float = 0.3
    vol_weight: float = 0.3

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 <= self.crisis_multiplier <= 1.0
        assert 0 <= self.volatile_multiplier <= 1.0
        assert 0 <= self.normal_multiplier <= 1.0
        assert abs(self.vix_weight + self.embi_weight + self.vol_weight - 1.0) < 1e-6
        assert self.vix_crisis_threshold > self.vix_volatile_threshold
        assert self.embi_crisis_threshold > self.embi_volatile_threshold
        assert self.vol_crisis_percentile > self.vol_volatile_percentile


class RegimeDetector:
    """
    Market Regime Detector for USD/COP Trading System.

    Detects market regimes (NORMAL, VOLATILE, CRISIS) based on multiple
    macroeconomic indicators and realized volatility measures.

    The detector uses z-scores for VIX and EMBI, and percentiles for
    realized volatility to classify market conditions and adjust position
    sizing accordingly.

    Attributes:
        config (RegimeConfig): Configuration parameters for regime detection
        regime_history (list): Historical regime classifications

    Example:
        >>> config = RegimeConfig()
        >>> detector = RegimeDetector(config)
        >>> regime = detector.detect_regime(vix_z=2.5, embi_z=1.8, vol_pct=96.0)
        >>> print(f"Regime: {regime}")
        >>> print(f"Position multiplier: {detector.get_position_multiplier(regime)}")
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize the Regime Detector.

        Args:
            config: Optional custom configuration. Uses defaults if not provided.
        """
        self.config = config or RegimeConfig()
        self.regime_history: list = []
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.crisis_multiplier > self.config.volatile_multiplier:
            warnings.warn(
                "Crisis multiplier is greater than volatile multiplier. "
                "This may lead to unexpected behavior."
            )

    def detect_regime(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float,
        return_probabilities: bool = False
    ) -> str | Tuple[str, Dict[str, float]]:
        """
        Detect market regime based on macroeconomic indicators.

        Detection Logic:
        - CRISIS: VIX_z > 2.0 OR EMBI_z > 2.0 OR vol > 95th percentile
        - VOLATILE: VIX_z > 1.0 OR EMBI_z > 1.0 OR vol > 75th percentile
        - NORMAL: Otherwise

        Args:
            vix_z: VIX z-score (standardized)
            embi_z: EMBI+ z-score (standardized)
            vol_pct: Realized volatility percentile (0-100)
            return_probabilities: If True, also return regime probabilities

        Returns:
            Regime classification ("NORMAL", "VOLATILE", or "CRISIS")
            If return_probabilities=True, returns (regime, probabilities_dict)

        Example:
            >>> detector = RegimeDetector()
            >>> regime, probs = detector.detect_regime(
            ...     vix_z=2.5, embi_z=1.2, vol_pct=88.0, return_probabilities=True
            ... )
            >>> print(regime)  # "CRISIS"
            >>> print(probs)   # {"CRISIS": 0.85, "VOLATILE": 0.12, "NORMAL": 0.03}
        """
        # Calculate probabilities for each regime
        probs = self.get_regime_probs(vix_z, embi_z, vol_pct)

        # Determine regime using hard thresholds (most conservative)
        if self._is_crisis(vix_z, embi_z, vol_pct):
            regime = MarketRegime.CRISIS.value
        elif self._is_volatile(vix_z, embi_z, vol_pct):
            regime = MarketRegime.VOLATILE.value
        else:
            regime = MarketRegime.NORMAL.value

        # Store in history
        self.regime_history.append({
            'regime': regime,
            'vix_z': vix_z,
            'embi_z': embi_z,
            'vol_pct': vol_pct,
            'probabilities': probs
        })

        if return_probabilities:
            return regime, probs
        return regime

    def _is_crisis(self, vix_z: float, embi_z: float, vol_pct: float) -> bool:
        """Check if conditions meet CRISIS threshold."""
        return (
            vix_z > self.config.vix_crisis_threshold or
            embi_z > self.config.embi_crisis_threshold or
            vol_pct > self.config.vol_crisis_percentile
        )

    def _is_volatile(self, vix_z: float, embi_z: float, vol_pct: float) -> bool:
        """Check if conditions meet VOLATILE threshold."""
        return (
            vix_z > self.config.vix_volatile_threshold or
            embi_z > self.config.embi_volatile_threshold or
            vol_pct > self.config.vol_volatile_percentile
        )

    def get_regime_probs(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float
    ) -> Dict[str, float]:
        """
        Calculate probabilistic regime classification.

        Uses weighted combination of VIX, EMBI, and volatility signals
        to compute soft probabilities for each regime.

        Args:
            vix_z: VIX z-score
            embi_z: EMBI+ z-score
            vol_pct: Realized volatility percentile

        Returns:
            Dictionary with probabilities for each regime

        Example:
            >>> detector = RegimeDetector()
            >>> probs = detector.get_regime_probs(vix_z=1.5, embi_z=0.8, vol_pct=72.0)
            >>> print(probs)
            {'CRISIS': 0.12, 'VOLATILE': 0.58, 'NORMAL': 0.30}
        """
        # Calculate individual signal scores (0-1 for each regime)
        vix_scores = self._score_indicator(
            vix_z,
            self.config.vix_volatile_threshold,
            self.config.vix_crisis_threshold
        )

        embi_scores = self._score_indicator(
            embi_z,
            self.config.embi_volatile_threshold,
            self.config.embi_crisis_threshold
        )

        vol_scores = self._score_indicator(
            vol_pct,
            self.config.vol_volatile_percentile,
            self.config.vol_crisis_percentile
        )

        # Weighted combination
        crisis_prob = (
            self.config.vix_weight * vix_scores['crisis'] +
            self.config.embi_weight * embi_scores['crisis'] +
            self.config.vol_weight * vol_scores['crisis']
        )

        volatile_prob = (
            self.config.vix_weight * vix_scores['volatile'] +
            self.config.embi_weight * embi_scores['volatile'] +
            self.config.vol_weight * vol_scores['volatile']
        )

        normal_prob = (
            self.config.vix_weight * vix_scores['normal'] +
            self.config.embi_weight * embi_scores['normal'] +
            self.config.vol_weight * vol_scores['normal']
        )

        # Normalize to sum to 1.0
        total = crisis_prob + volatile_prob + normal_prob
        if total > 0:
            crisis_prob /= total
            volatile_prob /= total
            normal_prob /= total
        else:
            # Default to normal if all scores are zero
            normal_prob = 1.0

        return {
            MarketRegime.CRISIS.value: round(crisis_prob, 4),
            MarketRegime.VOLATILE.value: round(volatile_prob, 4),
            MarketRegime.NORMAL.value: round(normal_prob, 4)
        }

    def _score_indicator(
        self,
        value: float,
        volatile_threshold: float,
        crisis_threshold: float
    ) -> Dict[str, float]:
        """
        Score a single indicator for regime probabilities.

        Uses sigmoid-like transitions between regimes.

        Args:
            value: Indicator value (z-score or percentile)
            volatile_threshold: Threshold for volatile regime
            crisis_threshold: Threshold for crisis regime

        Returns:
            Dictionary with scores for each regime
        """
        if value >= crisis_threshold:
            # Strong crisis signal
            return {'crisis': 1.0, 'volatile': 0.0, 'normal': 0.0}
        elif value >= volatile_threshold:
            # Transition between volatile and crisis
            ratio = (value - volatile_threshold) / (crisis_threshold - volatile_threshold)
            return {
                'crisis': ratio,
                'volatile': 1.0 - ratio,
                'normal': 0.0
            }
        elif value >= 0:
            # Transition between normal and volatile
            ratio = value / volatile_threshold
            return {
                'crisis': 0.0,
                'volatile': ratio,
                'normal': 1.0 - ratio
            }
        else:
            # Strong normal signal
            return {'crisis': 0.0, 'volatile': 0.0, 'normal': 1.0}

    def get_position_multiplier(self, regime: str) -> float:
        """
        Get position size multiplier for given regime.

        Position Multipliers:
        - CRISIS: 0.0 (no trading)
        - VOLATILE: 0.5 (50% position)
        - NORMAL: 1.0 (100% position)

        Args:
            regime: Market regime ("NORMAL", "VOLATILE", or "CRISIS")

        Returns:
            Position multiplier between 0.0 and 1.0

        Example:
            >>> detector = RegimeDetector()
            >>> detector.get_position_multiplier("CRISIS")
            0.0
            >>> detector.get_position_multiplier("VOLATILE")
            0.5
            >>> detector.get_position_multiplier("NORMAL")
            1.0
        """
        multipliers = {
            MarketRegime.CRISIS.value: self.config.crisis_multiplier,
            MarketRegime.VOLATILE.value: self.config.volatile_multiplier,
            MarketRegime.NORMAL.value: self.config.normal_multiplier
        }

        if regime not in multipliers:
            warnings.warn(f"Unknown regime '{regime}'. Defaulting to CRISIS multiplier.")
            return self.config.crisis_multiplier

        return multipliers[regime]

    def validate_on_historical(
        self,
        df: pd.DataFrame,
        vix_col: str = 'VIX_z',
        embi_col: str = 'EMBI_z',
        vol_col: str = 'vol_percentile'
    ) -> pd.DataFrame:
        """
        Validate regime detection on historical data.

        Applies regime detection to entire historical dataset and returns
        statistics about regime distribution and transitions.

        Args:
            df: DataFrame with historical data
            vix_col: Column name for VIX z-scores
            embi_col: Column name for EMBI z-scores
            vol_col: Column name for volatility percentiles

        Returns:
            DataFrame with regime classifications and statistics

        Example:
            >>> detector = RegimeDetector()
            >>> results = detector.validate_on_historical(historical_df)
            >>> print(results['regime'].value_counts())
            NORMAL      1250
            VOLATILE     380
            CRISIS        70
        """
        # Detect regime for each row
        regimes = []
        probs_list = []

        for idx, row in df.iterrows():
            vix_z = row[vix_col] if vix_col in df.columns else 0.0
            embi_z = row[embi_col] if embi_col in df.columns else 0.0
            vol_pct = row[vol_col] if vol_col in df.columns else 50.0

            regime, probs = self.detect_regime(
                vix_z, embi_z, vol_pct, return_probabilities=True
            )
            regimes.append(regime)
            probs_list.append(probs)

        # Add to dataframe
        result_df = df.copy()
        result_df['regime'] = regimes
        result_df['regime_crisis_prob'] = [p['CRISIS'] for p in probs_list]
        result_df['regime_volatile_prob'] = [p['VOLATILE'] for p in probs_list]
        result_df['regime_normal_prob'] = [p['NORMAL'] for p in probs_list]
        result_df['position_multiplier'] = result_df['regime'].apply(
            self.get_position_multiplier
        )

        return result_df

    def get_regime_statistics(self) -> Dict[str, any]:
        """
        Get statistics about detected regimes from history.

        Returns:
            Dictionary with regime distribution and transition statistics

        Example:
            >>> detector = RegimeDetector()
            >>> # ... after detecting many regimes ...
            >>> stats = detector.get_regime_statistics()
            >>> print(stats['distribution'])
            {'NORMAL': 0.72, 'VOLATILE': 0.23, 'CRISIS': 0.05}
        """
        if not self.regime_history:
            return {
                'total_observations': 0,
                'distribution': {},
                'avg_indicators': {},
                'transitions': {}
            }

        df = pd.DataFrame(self.regime_history)

        # Regime distribution
        distribution = df['regime'].value_counts(normalize=True).to_dict()

        # Average indicators by regime
        avg_indicators = df.groupby('regime').agg({
            'vix_z': 'mean',
            'embi_z': 'mean',
            'vol_pct': 'mean'
        }).to_dict('index')

        # Regime transitions
        transitions = {}
        for i in range(1, len(df)):
            prev = df.iloc[i-1]['regime']
            curr = df.iloc[i]['regime']
            key = f"{prev}_to_{curr}"
            transitions[key] = transitions.get(key, 0) + 1

        return {
            'total_observations': len(df),
            'distribution': distribution,
            'avg_indicators': avg_indicators,
            'transitions': transitions
        }

    def reset_history(self) -> None:
        """Clear regime detection history."""
        self.regime_history = []

    def get_regime_for_observation(
        self,
        observation: np.ndarray,
        vix_idx: int,
        embi_idx: int,
        vol_idx: int
    ) -> Tuple[str, float]:
        """
        Extract regime and position multiplier from observation array.

        Utility method for integration with RL environment.

        Args:
            observation: Observation array from environment
            vix_idx: Index of VIX z-score in observation
            embi_idx: Index of EMBI z-score in observation
            vol_idx: Index of volatility percentile in observation

        Returns:
            Tuple of (regime, position_multiplier)

        Example:
            >>> detector = RegimeDetector()
            >>> obs = env.reset()
            >>> regime, multiplier = detector.get_regime_for_observation(
            ...     obs, vix_idx=45, embi_idx=46, vol_idx=47
            ... )
        """
        vix_z = observation[vix_idx]
        embi_z = observation[embi_idx]
        vol_pct = observation[vol_idx]

        regime = self.detect_regime(vix_z, embi_z, vol_pct)
        multiplier = self.get_position_multiplier(regime)

        return regime, multiplier

    def __repr__(self) -> str:
        """String representation of detector."""
        return (
            f"RegimeDetector(\n"
            f"  VIX thresholds: {self.config.vix_volatile_threshold} / "
            f"{self.config.vix_crisis_threshold}\n"
            f"  EMBI thresholds: {self.config.embi_volatile_threshold} / "
            f"{self.config.embi_crisis_threshold}\n"
            f"  Vol percentiles: {self.config.vol_volatile_percentile} / "
            f"{self.config.vol_crisis_percentile}\n"
            f"  History size: {len(self.regime_history)}\n"
            f")"
        )


# ============================================================================
# Integration with RL Environment
# ============================================================================

class RegimeAwareEnvironmentWrapper:
    """
    Wrapper to integrate regime detection with RL trading environment.

    This wrapper augments the observation space with regime information
    and automatically adjusts position sizes based on market regime.

    Example:
        >>> from environment_v19 import TradingEnvironment
        >>> base_env = TradingEnvironment(df)
        >>> regime_env = RegimeAwareEnvironmentWrapper(base_env)
        >>> obs = regime_env.reset()
        >>> # obs now includes regime features
    """

    def __init__(
        self,
        base_env,
        detector: Optional[RegimeDetector] = None,
        vix_col: str = 'VIX_z',
        embi_col: str = 'EMBI_z',
        vol_col: str = 'vol_percentile'
    ):
        """
        Initialize regime-aware environment wrapper.

        Args:
            base_env: Base trading environment (e.g., from environment_v19.py)
            detector: Optional custom RegimeDetector instance
            vix_col: Column name for VIX z-scores in environment data
            embi_col: Column name for EMBI z-scores in environment data
            vol_col: Column name for volatility percentiles in environment data
        """
        self.base_env = base_env
        self.detector = detector or RegimeDetector()
        self.vix_col = vix_col
        self.embi_col = embi_col
        self.vol_col = vol_col

        # Augment observation space (add 3 features: regime one-hot)
        self.original_obs_dim = base_env.observation_space.shape[0]
        self.new_obs_dim = self.original_obs_dim + 3  # +3 for regime one-hot

    def _get_regime_features(self) -> np.ndarray:
        """Get current regime as one-hot encoded features."""
        current_row = self.base_env.df.iloc[self.base_env.current_step]

        vix_z = current_row.get(self.vix_col, 0.0)
        embi_z = current_row.get(self.embi_col, 0.0)
        vol_pct = current_row.get(self.vol_col, 50.0)

        regime = self.detector.detect_regime(vix_z, embi_z, vol_pct)

        # One-hot encoding: [is_crisis, is_volatile, is_normal]
        regime_features = np.array([
            1.0 if regime == MarketRegime.CRISIS.value else 0.0,
            1.0 if regime == MarketRegime.VOLATILE.value else 0.0,
            1.0 if regime == MarketRegime.NORMAL.value else 0.0
        ])

        return regime_features

    def reset(self):
        """Reset environment with regime features."""
        base_obs = self.base_env.reset()
        regime_features = self._get_regime_features()
        return np.concatenate([base_obs, regime_features])

    def step(self, action):
        """Take step with regime-adjusted position sizing."""
        # Get current regime
        current_row = self.base_env.df.iloc[self.base_env.current_step]
        vix_z = current_row.get(self.vix_col, 0.0)
        embi_z = current_row.get(self.embi_col, 0.0)
        vol_pct = current_row.get(self.vol_col, 50.0)
        regime = self.detector.detect_regime(vix_z, embi_z, vol_pct)

        # Adjust action based on regime
        multiplier = self.detector.get_position_multiplier(regime)
        adjusted_action = action * multiplier

        # Take step in base environment
        base_obs, reward, done, info = self.base_env.step(adjusted_action)

        # Add regime info
        regime_features = self._get_regime_features()
        augmented_obs = np.concatenate([base_obs, regime_features])
        info['regime'] = regime
        info['regime_multiplier'] = multiplier
        info['original_action'] = action
        info['adjusted_action'] = adjusted_action

        return augmented_obs, reward, done, info

    def __getattr__(self, name):
        """Delegate unknown attributes to base environment."""
        return getattr(self.base_env, name)


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_regime_distribution(
    df: pd.DataFrame,
    regime_col: str = 'regime',
    date_col: str = 'timestamp'
) -> None:
    """
    Visualize regime distribution over time.

    Args:
        df: DataFrame with regime classifications
        regime_col: Column name for regime
        date_col: Column name for dates
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Regime over time
        regime_numeric = df[regime_col].map({
            'CRISIS': 2, 'VOLATILE': 1, 'NORMAL': 0
        })

        axes[0].fill_between(
            range(len(df)),
            regime_numeric,
            alpha=0.3,
            step='post'
        )
        axes[0].set_ylabel('Regime')
        axes[0].set_title('Market Regime Over Time')
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(['NORMAL', 'VOLATILE', 'CRISIS'])
        axes[0].grid(True, alpha=0.3)

        # Regime distribution
        regime_counts = df[regime_col].value_counts()
        axes[1].bar(regime_counts.index, regime_counts.values, alpha=0.7)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Regime Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
        print("\nRegime Distribution:")
        print(df[regime_col].value_counts())


def create_regime_report(
    detector: RegimeDetector,
    df: pd.DataFrame
) -> str:
    """
    Create detailed text report of regime detection results.

    Args:
        detector: RegimeDetector instance with history
        df: DataFrame with regime classifications

    Returns:
        Formatted text report
    """
    stats = detector.get_regime_statistics()

    report = []
    report.append("=" * 70)
    report.append("MARKET REGIME DETECTION REPORT")
    report.append("=" * 70)
    report.append(f"\nTotal Observations: {stats['total_observations']}")

    report.append("\n\nREGIME DISTRIBUTION:")
    report.append("-" * 70)
    for regime, pct in sorted(stats['distribution'].items()):
        count = int(pct * stats['total_observations'])
        report.append(f"  {regime:12s}: {count:5d} ({pct*100:5.2f}%)")

    report.append("\n\nAVERAGE INDICATORS BY REGIME:")
    report.append("-" * 70)
    for regime, indicators in sorted(stats['avg_indicators'].items()):
        report.append(f"\n  {regime}:")
        report.append(f"    VIX z-score:    {indicators['vix_z']:7.3f}")
        report.append(f"    EMBI z-score:   {indicators['embi_z']:7.3f}")
        report.append(f"    Vol percentile: {indicators['vol_pct']:7.2f}")

    if stats['transitions']:
        report.append("\n\nREGIME TRANSITIONS (Top 10):")
        report.append("-" * 70)
        sorted_transitions = sorted(
            stats['transitions'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for transition, count in sorted_transitions:
            report.append(f"  {transition:25s}: {count:5d}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)


if __name__ == "__main__":
    """
    Example usage and testing of RegimeDetector.
    """
    print("Regime Detector Module - Example Usage")
    print("=" * 70)

    # Create detector with default config
    detector = RegimeDetector()
    print(f"\n{detector}")

    # Test regime detection
    print("\n\nTEST 1: Normal Market Conditions")
    print("-" * 70)
    regime = detector.detect_regime(vix_z=0.5, embi_z=0.3, vol_pct=45.0)
    probs = detector.get_regime_probs(vix_z=0.5, embi_z=0.3, vol_pct=45.0)
    multiplier = detector.get_position_multiplier(regime)
    print(f"Regime: {regime}")
    print(f"Probabilities: {probs}")
    print(f"Position Multiplier: {multiplier}")

    print("\n\nTEST 2: Volatile Market Conditions")
    print("-" * 70)
    regime = detector.detect_regime(vix_z=1.5, embi_z=0.8, vol_pct=78.0)
    probs = detector.get_regime_probs(vix_z=1.5, embi_z=0.8, vol_pct=78.0)
    multiplier = detector.get_position_multiplier(regime)
    print(f"Regime: {regime}")
    print(f"Probabilities: {probs}")
    print(f"Position Multiplier: {multiplier}")

    print("\n\nTEST 3: Crisis Market Conditions")
    print("-" * 70)
    regime = detector.detect_regime(vix_z=2.8, embi_z=2.2, vol_pct=97.0)
    probs = detector.get_regime_probs(vix_z=2.8, embi_z=2.2, vol_pct=97.0)
    multiplier = detector.get_position_multiplier(regime)
    print(f"Regime: {regime}")
    print(f"Probabilities: {probs}")
    print(f"Position Multiplier: {multiplier}")

    # Show statistics
    print("\n\nRegime Statistics:")
    print("-" * 70)
    stats = detector.get_regime_statistics()
    print(f"Total observations: {stats['total_observations']}")
    print(f"Distribution: {stats['distribution']}")

    print("\n" + "=" * 70)
    print("Module loaded successfully!")
    print("=" * 70)
