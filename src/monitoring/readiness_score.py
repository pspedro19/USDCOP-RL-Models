"""
Data Readiness Score Module
===========================

Computes and exposes a readiness score metric for the trading system.
This metric indicates the overall health of data pipelines and feature freshness.

P1: Readiness Score Metric for Production Monitoring

Components:
- Feature freshness scoring
- Data completeness validation
- Pipeline health aggregation
- Prometheus metric exposure

Usage:
    from src.monitoring.readiness_score import (
        DataReadinessScorer,
        DailyDataReadinessReport,
        compute_readiness_score,
        get_readiness_metrics,
    )

    scorer = DataReadinessScorer()
    report = scorer.generate_report()
    print(f"Readiness Score: {report.overall_score:.2%}")

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ReadinessLevel(Enum):
    """Readiness level thresholds."""
    CRITICAL = 0.0    # < 50% - System should not trade
    WARNING = 0.5     # 50-80% - Degraded, monitor closely
    HEALTHY = 0.8     # 80-95% - Normal operation
    OPTIMAL = 0.95    # > 95% - All systems optimal


@dataclass
class FeatureReadiness:
    """Readiness status for a single feature."""
    feature_name: str
    last_updated: Optional[datetime]
    staleness_days: float
    max_allowed_staleness: float
    is_fresh: bool
    score: float  # 0.0 to 1.0

    @property
    def status(self) -> str:
        """Human-readable status."""
        if self.score >= 0.95:
            return "optimal"
        elif self.score >= 0.8:
            return "healthy"
        elif self.score >= 0.5:
            return "warning"
        else:
            return "critical"


@dataclass
class PipelineReadiness:
    """Readiness status for a data pipeline."""
    pipeline_name: str
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    success_rate_24h: float
    avg_duration_seconds: float
    is_healthy: bool
    score: float


@dataclass
class DailyDataReadinessReport:
    """
    Complete daily data readiness report.

    This is the main report class used to assess system readiness
    for trading operations.
    """
    report_date: datetime
    overall_score: float  # 0.0 to 1.0
    readiness_level: ReadinessLevel

    # Component scores
    feature_readiness_score: float
    pipeline_health_score: float
    data_quality_score: float

    # Detailed breakdowns
    feature_details: List[FeatureReadiness] = field(default_factory=list)
    pipeline_details: List[PipelineReadiness] = field(default_factory=list)

    # Blocking indicators
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    computation_time_ms: float = 0.0

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading should be allowed based on readiness."""
        return (
            self.overall_score >= ReadinessLevel.WARNING.value
            and len(self.blocking_issues) == 0
        )

    @property
    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"Readiness: {self.overall_score:.1%} ({self.readiness_level.name}) | "
            f"Features: {self.feature_readiness_score:.1%} | "
            f"Pipelines: {self.pipeline_health_score:.1%} | "
            f"Quality: {self.data_quality_score:.1%} | "
            f"Blocking: {len(self.blocking_issues)} | "
            f"Warnings: {len(self.warnings)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_date": self.report_date.isoformat(),
            "overall_score": self.overall_score,
            "readiness_level": self.readiness_level.name,
            "feature_readiness_score": self.feature_readiness_score,
            "pipeline_health_score": self.pipeline_health_score,
            "data_quality_score": self.data_quality_score,
            "is_trading_allowed": self.is_trading_allowed,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "feature_count": len(self.feature_details),
            "pipeline_count": len(self.pipeline_details),
            "computation_time_ms": self.computation_time_ms,
        }


class DataReadinessScorer:
    """
    Computes data readiness scores for the trading system.

    This class aggregates multiple signals to produce an overall
    readiness score that indicates whether the system has fresh,
    complete, and high-quality data for trading decisions.
    """

    # Default staleness limits (in days)
    DEFAULT_STALENESS_LIMITS = {
        "daily": 5,
        "hourly": 1,
        "realtime": 0.1,  # ~2.4 hours
        "monthly": 35,
        "quarterly": 95,
    }

    # Feature frequency mapping
    FEATURE_FREQUENCIES = {
        # Realtime/Hourly features
        "log_ret_5m": "realtime",
        "log_ret_15m": "realtime",
        "log_ret_1h": "hourly",
        "rsi_14": "hourly",
        "macd_signal": "hourly",
        "bb_position": "hourly",
        "volatility_1h": "hourly",

        # Daily features
        "overnight_gap": "daily",
        "prev_close": "daily",
        "daily_range": "daily",
        "volume_ratio": "daily",

        # Macro features
        "fed_funds_rate": "monthly",
        "cpi_yoy": "monthly",
        "gdp_growth": "quarterly",
        "unemployment_rate": "monthly",
        "ppi_yoy": "monthly",
    }

    def __init__(
        self,
        staleness_limits: Optional[Dict[str, float]] = None,
        critical_features: Optional[List[str]] = None,
    ):
        """
        Initialize the readiness scorer.

        Args:
            staleness_limits: Custom staleness limits by frequency
            critical_features: Features that block trading if stale
        """
        self.staleness_limits = {
            **self.DEFAULT_STALENESS_LIMITS,
            **(staleness_limits or {})
        }
        self.critical_features = critical_features or [
            "log_ret_5m", "log_ret_15m", "rsi_14", "macd_signal"
        ]

    def compute_feature_score(
        self,
        feature_name: str,
        last_updated: Optional[datetime],
        now: Optional[datetime] = None,
    ) -> FeatureReadiness:
        """
        Compute readiness score for a single feature.

        Args:
            feature_name: Name of the feature
            last_updated: When the feature was last updated
            now: Current time (defaults to now)

        Returns:
            FeatureReadiness with score and status
        """
        now = now or datetime.utcnow()

        # Get frequency and staleness limit
        frequency = self.FEATURE_FREQUENCIES.get(feature_name, "daily")
        max_staleness = self.staleness_limits.get(frequency, 5)

        # Compute staleness
        if last_updated is None:
            staleness_days = float('inf')
        else:
            staleness_days = (now - last_updated).total_seconds() / 86400

        # Compute score (linear decay from 1.0 to 0.0)
        if staleness_days >= max_staleness:
            score = 0.0
        else:
            score = 1.0 - (staleness_days / max_staleness)

        is_fresh = staleness_days < max_staleness

        return FeatureReadiness(
            feature_name=feature_name,
            last_updated=last_updated,
            staleness_days=staleness_days,
            max_allowed_staleness=max_staleness,
            is_fresh=is_fresh,
            score=score,
        )

    def compute_pipeline_score(
        self,
        pipeline_name: str,
        last_success: Optional[datetime],
        last_failure: Optional[datetime],
        success_rate_24h: float,
        avg_duration_seconds: float,
    ) -> PipelineReadiness:
        """
        Compute readiness score for a pipeline.

        Args:
            pipeline_name: Name of the pipeline
            last_success: When pipeline last succeeded
            last_failure: When pipeline last failed
            success_rate_24h: Success rate in last 24 hours
            avg_duration_seconds: Average run duration

        Returns:
            PipelineReadiness with score and status
        """
        # Score based on success rate (70% weight) and recency (30% weight)
        success_score = success_rate_24h

        # Recency score - penalize if no success in 24h
        now = datetime.utcnow()
        if last_success is None:
            recency_score = 0.0
        else:
            hours_since_success = (now - last_success).total_seconds() / 3600
            if hours_since_success < 6:
                recency_score = 1.0
            elif hours_since_success < 24:
                recency_score = 0.8
            elif hours_since_success < 48:
                recency_score = 0.5
            else:
                recency_score = 0.2

        score = 0.7 * success_score + 0.3 * recency_score
        is_healthy = score >= 0.7

        return PipelineReadiness(
            pipeline_name=pipeline_name,
            last_success=last_success,
            last_failure=last_failure,
            success_rate_24h=success_rate_24h,
            avg_duration_seconds=avg_duration_seconds,
            is_healthy=is_healthy,
            score=score,
        )

    def generate_report(
        self,
        feature_timestamps: Optional[Dict[str, datetime]] = None,
        pipeline_stats: Optional[List[Dict]] = None,
        data_quality_score: float = 1.0,
    ) -> DailyDataReadinessReport:
        """
        Generate a complete readiness report.

        Args:
            feature_timestamps: Dict of feature_name -> last_updated
            pipeline_stats: List of pipeline statistics
            data_quality_score: Pre-computed data quality score

        Returns:
            DailyDataReadinessReport with all scores and details
        """
        import time
        start_time = time.time()

        now = datetime.utcnow()
        feature_timestamps = feature_timestamps or {}
        pipeline_stats = pipeline_stats or []

        # Compute feature readiness
        feature_details = []
        for feature_name in self.FEATURE_FREQUENCIES.keys():
            last_updated = feature_timestamps.get(feature_name)
            readiness = self.compute_feature_score(feature_name, last_updated, now)
            feature_details.append(readiness)

        # Compute pipeline readiness
        pipeline_details = []
        for stats in pipeline_stats:
            readiness = self.compute_pipeline_score(
                pipeline_name=stats.get("name", "unknown"),
                last_success=stats.get("last_success"),
                last_failure=stats.get("last_failure"),
                success_rate_24h=stats.get("success_rate_24h", 0.0),
                avg_duration_seconds=stats.get("avg_duration_seconds", 0.0),
            )
            pipeline_details.append(readiness)

        # Calculate aggregate scores
        if feature_details:
            feature_readiness_score = sum(f.score for f in feature_details) / len(feature_details)
        else:
            feature_readiness_score = 0.0

        if pipeline_details:
            pipeline_health_score = sum(p.score for p in pipeline_details) / len(pipeline_details)
        else:
            pipeline_health_score = 1.0  # No pipelines to check

        # Overall score (weighted average)
        overall_score = (
            0.5 * feature_readiness_score +
            0.3 * pipeline_health_score +
            0.2 * data_quality_score
        )

        # Determine readiness level
        if overall_score >= ReadinessLevel.OPTIMAL.value:
            level = ReadinessLevel.OPTIMAL
        elif overall_score >= ReadinessLevel.HEALTHY.value:
            level = ReadinessLevel.HEALTHY
        elif overall_score >= ReadinessLevel.WARNING.value:
            level = ReadinessLevel.WARNING
        else:
            level = ReadinessLevel.CRITICAL

        # Identify blocking issues and warnings
        blocking_issues = []
        warnings = []

        for feature in feature_details:
            if feature.feature_name in self.critical_features and not feature.is_fresh:
                blocking_issues.append(
                    f"Critical feature '{feature.feature_name}' is stale "
                    f"({feature.staleness_days:.1f} days, max {feature.max_allowed_staleness})"
                )
            elif not feature.is_fresh:
                warnings.append(
                    f"Feature '{feature.feature_name}' is stale "
                    f"({feature.staleness_days:.1f} days)"
                )

        for pipeline in pipeline_details:
            if not pipeline.is_healthy:
                if pipeline.score < 0.3:
                    blocking_issues.append(
                        f"Pipeline '{pipeline.pipeline_name}' is failing "
                        f"(success rate: {pipeline.success_rate_24h:.1%})"
                    )
                else:
                    warnings.append(
                        f"Pipeline '{pipeline.pipeline_name}' is degraded "
                        f"(success rate: {pipeline.success_rate_24h:.1%})"
                    )

        computation_time_ms = (time.time() - start_time) * 1000

        return DailyDataReadinessReport(
            report_date=now,
            overall_score=overall_score,
            readiness_level=level,
            feature_readiness_score=feature_readiness_score,
            pipeline_health_score=pipeline_health_score,
            data_quality_score=data_quality_score,
            feature_details=feature_details,
            pipeline_details=pipeline_details,
            blocking_issues=blocking_issues,
            warnings=warnings,
            computation_time_ms=computation_time_ms,
        )


# Prometheus metrics (lazy initialization)
_metrics_initialized = False
_readiness_gauge = None
_feature_staleness_gauge = None
_pipeline_health_gauge = None


def _init_prometheus_metrics():
    """Initialize Prometheus metrics (lazy)."""
    global _metrics_initialized, _readiness_gauge, _feature_staleness_gauge, _pipeline_health_gauge

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Gauge

        _readiness_gauge = Gauge(
            'data_readiness_score',
            'Overall data readiness score (0.0 to 1.0)',
            ['level']
        )

        _feature_staleness_gauge = Gauge(
            'feature_staleness_days',
            'Feature staleness in days',
            ['feature_name', 'frequency']
        )

        _pipeline_health_gauge = Gauge(
            'pipeline_health_score',
            'Pipeline health score (0.0 to 1.0)',
            ['pipeline_name']
        )

        _metrics_initialized = True
        logger.info("Prometheus readiness metrics initialized")

    except ImportError:
        logger.warning("prometheus_client not available, metrics disabled")


def update_prometheus_metrics(report: DailyDataReadinessReport) -> None:
    """
    Update Prometheus metrics from a readiness report.

    Args:
        report: The readiness report to export
    """
    _init_prometheus_metrics()

    if not _metrics_initialized:
        return

    # Update overall readiness
    _readiness_gauge.labels(level=report.readiness_level.name).set(report.overall_score)

    # Update feature staleness
    scorer = DataReadinessScorer()
    for feature in report.feature_details:
        frequency = scorer.FEATURE_FREQUENCIES.get(feature.feature_name, "daily")
        _feature_staleness_gauge.labels(
            feature_name=feature.feature_name,
            frequency=frequency
        ).set(feature.staleness_days if feature.staleness_days != float('inf') else -1)

    # Update pipeline health
    for pipeline in report.pipeline_details:
        _pipeline_health_gauge.labels(pipeline_name=pipeline.pipeline_name).set(pipeline.score)


def compute_readiness_score(
    feature_timestamps: Optional[Dict[str, datetime]] = None,
    pipeline_stats: Optional[List[Dict]] = None,
    update_metrics: bool = True,
) -> DailyDataReadinessReport:
    """
    Convenience function to compute readiness score.

    Args:
        feature_timestamps: Dict of feature_name -> last_updated
        pipeline_stats: List of pipeline statistics
        update_metrics: Whether to update Prometheus metrics

    Returns:
        DailyDataReadinessReport
    """
    scorer = DataReadinessScorer()
    report = scorer.generate_report(
        feature_timestamps=feature_timestamps,
        pipeline_stats=pipeline_stats,
    )

    if update_metrics:
        update_prometheus_metrics(report)

    logger.info(f"Readiness computed: {report.summary}")

    return report


def get_readiness_metrics() -> Dict[str, Any]:
    """
    Get current readiness metrics as a dictionary.

    Returns:
        Dict with readiness metrics for API exposure
    """
    # Generate a quick report with no data (will show all stale)
    # In production, this would query the database for timestamps
    report = compute_readiness_score(update_metrics=False)
    return report.to_dict()


__all__ = [
    "ReadinessLevel",
    "FeatureReadiness",
    "PipelineReadiness",
    "DailyDataReadinessReport",
    "DataReadinessScorer",
    "compute_readiness_score",
    "get_readiness_metrics",
    "update_prometheus_metrics",
]
