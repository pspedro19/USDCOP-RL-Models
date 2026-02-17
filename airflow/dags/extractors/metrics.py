# -*- coding: utf-8 -*-
"""
Extraction Metrics Module
=========================
Metrics collection for L0 macro data extraction pipeline.

Contract: CTR-L0-METRICS-001

Provides:
- ExtractionMetrics: Per-extraction metrics dataclass
- MetricsCollector: Aggregates metrics across pipeline run
- Prometheus-compatible output format

Usage:
    collector = MetricsCollector()

    # Record individual extraction
    collector.record(ExtractionMetrics(
        source='fred',
        variable='polr_fed_funds_usa_m_fedfunds',
        records_extracted=100,
        duration_ms=1500,
    ))

    # Get summary
    summary = collector.get_summary()
    print(f"Total records: {summary['total_records']}")

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """
    Metrics for a single extraction operation.

    Designed to be Prometheus-compatible for future integration.
    """
    # Identity
    source: str
    variable: str

    # Counts
    records_extracted: int = 0
    records_validated: int = 0
    records_inserted: int = 0
    records_skipped: int = 0

    # Timing (milliseconds)
    extraction_duration_ms: int = 0
    validation_duration_ms: int = 0
    upsert_duration_ms: int = 0
    total_duration_ms: int = 0

    # Retry and fallback
    retry_count: int = 0
    fallback_used: bool = False
    fallback_source: Optional[str] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source,
            'variable': self.variable,
            'records_extracted': self.records_extracted,
            'records_validated': self.records_validated,
            'records_inserted': self.records_inserted,
            'records_skipped': self.records_skipped,
            'extraction_duration_ms': self.extraction_duration_ms,
            'validation_duration_ms': self.validation_duration_ms,
            'upsert_duration_ms': self.upsert_duration_ms,
            'total_duration_ms': self.total_duration_ms,
            'retry_count': self.retry_count,
            'fallback_used': self.fallback_used,
            'fallback_source': self.fallback_source,
            'success': self.success,
            'error_message': self.error_message,
            'validation_errors': self.validation_errors,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class SourceMetrics:
    """Aggregated metrics for a single source."""
    source: str
    variables_count: int = 0
    variables_success: int = 0
    variables_failed: int = 0
    total_records: int = 0
    total_duration_ms: int = 0
    retry_total: int = 0
    fallback_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.variables_count == 0:
            return 0.0
        return self.variables_success / self.variables_count

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration per variable."""
        if self.variables_count == 0:
            return 0.0
        return self.total_duration_ms / self.variables_count


class MetricsCollector:
    """
    Collector for extraction pipeline metrics.

    Aggregates ExtractionMetrics across the entire pipeline run
    and provides summary statistics.
    """

    def __init__(self):
        self._metrics: List[ExtractionMetrics] = []
        self._start_time: datetime = datetime.utcnow()

    def record(self, metrics: ExtractionMetrics) -> None:
        """
        Record metrics for a single extraction.

        Args:
            metrics: ExtractionMetrics instance
        """
        self._metrics.append(metrics)
        logger.debug(
            f"[METRICS] {metrics.source}.{metrics.variable}: "
            f"{metrics.records_extracted} records, "
            f"{metrics.total_duration_ms}ms, "
            f"success={metrics.success}"
        )

    def get_metrics_for_source(self, source: str) -> List[ExtractionMetrics]:
        """Get all metrics for a specific source."""
        return [m for m in self._metrics if m.source == source]

    def get_source_summary(self, source: str) -> SourceMetrics:
        """Get aggregated metrics for a source."""
        source_metrics = self.get_metrics_for_source(source)

        summary = SourceMetrics(source=source)
        for m in source_metrics:
            summary.variables_count += 1
            if m.success:
                summary.variables_success += 1
            else:
                summary.variables_failed += 1
                if m.error_message:
                    summary.errors.append(f"{m.variable}: {m.error_message}")

            summary.total_records += m.records_extracted
            summary.total_duration_ms += m.total_duration_ms
            summary.retry_total += m.retry_count
            if m.fallback_used:
                summary.fallback_count += 1

        return summary

    def get_all_sources(self) -> List[str]:
        """Get list of all sources with recorded metrics."""
        return list(set(m.source for m in self._metrics))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregated summary across all extractions.

        Returns:
            Dictionary with summary statistics
        """
        end_time = datetime.utcnow()

        total_records = sum(m.records_extracted for m in self._metrics)
        total_validated = sum(m.records_validated for m in self._metrics)
        total_inserted = sum(m.records_inserted for m in self._metrics)
        total_skipped = sum(m.records_skipped for m in self._metrics)

        success_count = sum(1 for m in self._metrics if m.success)
        failed_count = sum(1 for m in self._metrics if not m.success)
        total_count = len(self._metrics)

        total_retries = sum(m.retry_count for m in self._metrics)
        fallback_count = sum(1 for m in self._metrics if m.fallback_used)

        total_duration_ms = sum(m.total_duration_ms for m in self._metrics)
        pipeline_duration_ms = int((end_time - self._start_time).total_seconds() * 1000)

        # Per-source breakdown
        by_source = {}
        for source in self.get_all_sources():
            source_summary = self.get_source_summary(source)
            by_source[source] = {
                'variables': source_summary.variables_count,
                'success': source_summary.variables_success,
                'failed': source_summary.variables_failed,
                'success_rate': source_summary.success_rate,
                'records': source_summary.total_records,
                'duration_ms': source_summary.total_duration_ms,
                'retries': source_summary.retry_total,
                'fallbacks': source_summary.fallback_count,
            }

        # Collect all errors
        all_errors = [
            {'source': m.source, 'variable': m.variable, 'error': m.error_message}
            for m in self._metrics if not m.success and m.error_message
        ]

        return {
            # Counts
            'total_variables': total_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / total_count if total_count > 0 else 0.0,

            # Records
            'total_records_extracted': total_records,
            'total_records_validated': total_validated,
            'total_records_inserted': total_inserted,
            'total_records_skipped': total_skipped,

            # Reliability
            'total_retries': total_retries,
            'fallback_count': fallback_count,

            # Timing
            'total_extraction_duration_ms': total_duration_ms,
            'pipeline_duration_ms': pipeline_duration_ms,
            'avg_duration_per_variable_ms': total_duration_ms / total_count if total_count > 0 else 0,

            # Timestamps
            'start_time': self._start_time.isoformat(),
            'end_time': end_time.isoformat(),

            # Breakdown
            'by_source': by_source,

            # Errors
            'errors': all_errors,
        }

    def to_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus exposition format.

        Returns:
            String in Prometheus text format
        """
        lines = []
        lines.append("# HELP l0_extraction_records_total Total records extracted")
        lines.append("# TYPE l0_extraction_records_total counter")

        lines.append("# HELP l0_extraction_duration_ms Extraction duration in milliseconds")
        lines.append("# TYPE l0_extraction_duration_ms gauge")

        lines.append("# HELP l0_extraction_success Extraction success (1=success, 0=failure)")
        lines.append("# TYPE l0_extraction_success gauge")

        for m in self._metrics:
            labels = f'source="{m.source}",variable="{m.variable}"'
            lines.append(f"l0_extraction_records_total{{{labels}}} {m.records_extracted}")
            lines.append(f"l0_extraction_duration_ms{{{labels}}} {m.total_duration_ms}")
            lines.append(f"l0_extraction_success{{{labels}}} {1 if m.success else 0}")

        # Aggregates
        summary = self.get_summary()
        lines.append("")
        lines.append("# HELP l0_pipeline_total_records Total records across all sources")
        lines.append("# TYPE l0_pipeline_total_records gauge")
        lines.append(f"l0_pipeline_total_records {summary['total_records_extracted']}")

        lines.append("# HELP l0_pipeline_success_rate Pipeline success rate")
        lines.append("# TYPE l0_pipeline_success_rate gauge")
        lines.append(f"l0_pipeline_success_rate {summary['success_rate']:.4f}")

        lines.append("# HELP l0_pipeline_duration_ms Total pipeline duration")
        lines.append("# TYPE l0_pipeline_duration_ms gauge")
        lines.append(f"l0_pipeline_duration_ms {summary['pipeline_duration_ms']}")

        return "\n".join(lines)

    def log_summary(self) -> None:
        """Log summary to standard logger."""
        summary = self.get_summary()

        logger.info("=" * 60)
        logger.info("EXTRACTION METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Variables: {summary['success_count']}/{summary['total_variables']} success ({summary['success_rate']:.1%})")
        logger.info(f"Records: {summary['total_records_extracted']} extracted, {summary['total_records_inserted']} inserted")
        logger.info(f"Duration: {summary['pipeline_duration_ms']}ms total, {summary['avg_duration_per_variable_ms']:.0f}ms avg/variable")
        logger.info(f"Retries: {summary['total_retries']}, Fallbacks: {summary['fallback_count']}")

        if summary['failed_count'] > 0:
            logger.warning(f"Failures: {summary['failed_count']}")
            for error in summary['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error['source']}.{error['variable']}: {error['error']}")

        logger.info("By source:")
        for source, stats in summary['by_source'].items():
            status = 'OK' if stats['failed'] == 0 else 'PARTIAL'
            logger.info(
                f"  [{status}] {source}: {stats['success']}/{stats['variables']} vars, "
                f"{stats['records']} records, {stats['duration_ms']}ms"
            )

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._metrics = []
        self._start_time = datetime.utcnow()


# Global collector instance for convenience
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector() -> MetricsCollector:
    """Reset and return new global metrics collector."""
    global _global_collector
    _global_collector = MetricsCollector()
    return _global_collector
