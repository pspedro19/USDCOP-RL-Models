"""
Multi-Timeframe Analyzer (Multi-Agent System — Phase 1)
========================================================
Aggregates 5-min OHLCV into 15m/1h/4h/1d bars, then runs
TechnicalAnalysisEngine on each. Computes alignment score
and confluent S/R levels across timeframes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.analysis.technical_engine import TechnicalAnalysisEngine, TechnicalAnalysisReport

logger = logging.getLogger(__name__)

# Timeframe weights for alignment scoring (higher = more weight)
TIMEFRAME_WEIGHTS = {
    "1d": 4.0,
    "4h": 3.0,
    "1h": 2.0,
    "15m": 1.0,
}


@dataclass
class MultiTimeframeReport:
    """Aggregated analysis across multiple timeframes."""
    reports: dict = field(default_factory=dict)  # timeframe -> TechnicalAnalysisReport.to_dict()
    alignment_score: float = 0.0        # -1 (all bearish) to +1 (all bullish)
    alignment_label: str = "neutral"    # strongly_bullish, bullish, neutral, bearish, strongly_bearish
    confluent_supports: list[float] = field(default_factory=list)
    confluent_resistances: list[float] = field(default_factory=list)
    dominant_timeframe: str = "1d"      # which timeframe has strongest signal

    def to_dict(self) -> dict:
        return asdict(self)


class MultiTimeframeAnalyzer:
    """Aggregate 5-min OHLCV into multiple timeframes and analyze each."""

    def __init__(self) -> None:
        self.engine = TechnicalAnalysisEngine()

    def aggregate_timeframes(self, m5_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Resample 5-min OHLCV into higher timeframes.

        Args:
            m5_df: DataFrame with 5-min OHLCV, DatetimeIndex.

        Returns:
            Dict mapping timeframe label to resampled DataFrame.
        """
        if m5_df.empty:
            return {}

        df = m5_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("time", "timestamp", "date"):
                if col in df.columns:
                    df = df.set_index(col)
                    break
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        resample_map = {
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
        }

        result = {}
        for label, rule in resample_map.items():
            try:
                resampled = df.resample(rule).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }).dropna()

                if len(resampled) >= 30:
                    result[label] = resampled
                else:
                    logger.debug(f"Timeframe {label}: only {len(resampled)} bars (need 30)")
            except Exception as e:
                logger.warning(f"Resampling to {label} failed: {e}")

        return result

    def analyze_all_timeframes(
        self,
        data: dict[str, pd.DataFrame],
        daily_df: pd.DataFrame | None = None,
    ) -> MultiTimeframeReport:
        """Run TA on each timeframe and compute alignment.

        Args:
            data: Dict from aggregate_timeframes().
            daily_df: Optional daily OHLCV to use for 1d instead of resampled.

        Returns:
            MultiTimeframeReport with alignment score and confluent levels.
        """
        reports = {}

        # Use provided daily OHLCV if available (more accurate than resampled 5m -> 1d)
        if daily_df is not None and not daily_df.empty and len(daily_df) >= 30:
            data["1d"] = daily_df

        for tf, df in data.items():
            try:
                report = self.engine.analyze(df, timeframe=tf)
                reports[tf] = report
            except Exception as e:
                logger.warning(f"TA analysis failed for {tf}: {e}")

        if not reports:
            return MultiTimeframeReport()

        # Compute alignment
        alignment_score, alignment_label = self._compute_alignment(reports)

        # Find confluent S/R levels
        confluent_s, confluent_r = self._find_confluent_levels(reports)

        # Dominant timeframe (strongest bias_confidence)
        dominant = max(
            reports.items(),
            key=lambda x: x[1].bias_confidence * TIMEFRAME_WEIGHTS.get(x[0], 1),
        )

        return MultiTimeframeReport(
            reports={tf: r.to_dict() for tf, r in reports.items()},
            alignment_score=alignment_score,
            alignment_label=alignment_label,
            confluent_supports=confluent_s,
            confluent_resistances=confluent_r,
            dominant_timeframe=dominant[0],
        )

    def _compute_alignment(
        self, reports: dict[str, TechnicalAnalysisReport]
    ) -> tuple[float, str]:
        """Compute weighted alignment score across timeframes.

        Returns:
            (score, label): score in [-1, +1], label like "strongly_bullish".
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, report in reports.items():
            weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
            if report.dominant_bias == "bullish":
                weighted_sum += weight * report.bias_confidence
            elif report.dominant_bias == "bearish":
                weighted_sum -= weight * report.bias_confidence
            total_weight += weight

        if total_weight < 0.1:
            return 0.0, "neutral"

        score = round(weighted_sum / total_weight, 3)

        if score > 0.6:
            label = "strongly_bullish"
        elif score > 0.2:
            label = "bullish"
        elif score < -0.6:
            label = "strongly_bearish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"

        return score, label

    def _find_confluent_levels(
        self, reports: dict[str, TechnicalAnalysisReport]
    ) -> tuple[list[float], list[float]]:
        """Find S/R levels that appear in 2+ timeframes.

        Returns:
            (confluent_supports, confluent_resistances): Sorted lists.
        """
        all_supports = []
        all_resistances = []

        for report in reports.values():
            all_supports.extend(report.support_resistance.key_supports)
            all_resistances.extend(report.support_resistance.key_resistances)

            # Also include Fibonacci levels as potential S/R
            if report.fibonacci.nearest_support:
                all_supports.append(report.fibonacci.nearest_support)
            if report.fibonacci.nearest_resistance:
                all_resistances.append(report.fibonacci.nearest_resistance)

        confluent_s = self._find_clusters(all_supports, min_count=2, threshold_pct=0.15)
        confluent_r = self._find_clusters(all_resistances, min_count=2, threshold_pct=0.15)

        return confluent_s[:5], confluent_r[:5]

    @staticmethod
    def _find_clusters(
        levels: list[float], min_count: int = 2, threshold_pct: float = 0.15
    ) -> list[float]:
        """Find price level clusters (levels within threshold_pct of each other)."""
        if not levels:
            return []

        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for lvl in levels[1:]:
            if current_cluster and abs(lvl - current_cluster[-1]) / max(current_cluster[-1], 1) * 100 < threshold_pct:
                current_cluster.append(lvl)
            else:
                if len(current_cluster) >= min_count:
                    clusters.append(round(np.mean(current_cluster), 2))
                current_cluster = [lvl]

        if len(current_cluster) >= min_count:
            clusters.append(round(np.mean(current_cluster), 2))

        return clusters
