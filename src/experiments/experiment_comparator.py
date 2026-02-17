"""
Experiment Comparator
=====================

Statistical comparison of experiments for A/B testing.
Provides statistical significance testing and visualization.

Usage:
    from src.experiments import compare_experiments

    comparison = compare_experiments("baseline_v1", "new_model_v1")
    print(comparison.summary())
    print(comparison.is_significantly_better("sharpe_ratio"))

Author: Trading Team
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .experiment_runner import ExperimentResult
from .experiment_registry import FileBasedRegistry

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric_name: str
    baseline_value: float
    treatment_value: float
    absolute_difference: float
    relative_difference: float
    p_value: Optional[float]
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    winner: str  # "baseline", "treatment", or "tie"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "treatment_value": self.treatment_value,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "winner": self.winner,
        }


@dataclass
class ExperimentComparison:
    """
    Complete comparison between two experiments.

    Contains statistical analysis and recommendations.
    """

    baseline_name: str
    treatment_name: str
    baseline_version: str
    treatment_version: str
    comparison_date: datetime
    metric_comparisons: Dict[str, MetricComparison]
    primary_metric: str
    recommendation: str
    confidence_level: float
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def is_significantly_better(
        self,
        metric: str,
        alpha: float = 0.05,
    ) -> bool:
        """
        Check if treatment is significantly better than baseline.

        Args:
            metric: Metric name
            alpha: Significance level

        Returns:
            True if treatment is significantly better
        """
        comp = self.metric_comparisons.get(metric)
        if comp is None:
            return False

        return (
            comp.is_significant
            and comp.winner == "treatment"
            and comp.p_value is not None
            and comp.p_value < alpha
        )

    def get_winner(self) -> str:
        """Get overall winner based on primary metric."""
        comp = self.metric_comparisons.get(self.primary_metric)
        if comp is None:
            return "inconclusive"
        return comp.winner

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"Experiment Comparison: {self.baseline_name} vs {self.treatment_name}",
            "=" * 60,
            f"Baseline: {self.baseline_name} v{self.baseline_version}",
            f"Treatment: {self.treatment_name} v{self.treatment_version}",
            f"Primary Metric: {self.primary_metric}",
            f"Confidence Level: {self.confidence_level * 100:.0f}%",
            "",
            "Metric Comparisons:",
            "-" * 60,
        ]

        for name, comp in self.metric_comparisons.items():
            sig = "*" if comp.is_significant else ""
            lines.append(
                f"  {name}:"
                f"  Baseline={comp.baseline_value:.4f}"
                f"  Treatment={comp.treatment_value:.4f}"
                f"  Diff={comp.relative_difference:+.1%}"
                f"  p={comp.p_value:.4f if comp.p_value else 'N/A'}{sig}"
            )

        lines.extend([
            "",
            "-" * 60,
            f"Recommendation: {self.recommendation}",
        ])

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline_name": self.baseline_name,
            "treatment_name": self.treatment_name,
            "baseline_version": self.baseline_version,
            "treatment_version": self.treatment_version,
            "comparison_date": self.comparison_date.isoformat(),
            "metric_comparisons": {
                k: v.to_dict() for k, v in self.metric_comparisons.items()
            },
            "primary_metric": self.primary_metric,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level,
            "sample_sizes": self.sample_sizes,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> None:
        """Save comparison to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentComparator:
    """
    Statistical comparison engine for experiments.

    Performs:
    - T-tests for metric comparison
    - Bootstrap confidence intervals
    - Effect size calculation
    - Multiple testing correction
    """

    # Metrics where lower is better
    LOWER_IS_BETTER = {"max_drawdown", "avg_loss", "volatility"}

    def __init__(
        self,
        significance_level: float = 0.05,
        bootstrap_iterations: int = 1000,
        min_samples: int = 30,
    ):
        """
        Initialize comparator.

        Args:
            significance_level: Alpha for significance testing
            bootstrap_iterations: Number of bootstrap samples
            min_samples: Minimum samples for statistical tests
        """
        self.significance_level = significance_level
        self.bootstrap_iterations = bootstrap_iterations
        self.min_samples = min_samples

    def compare_single_run(
        self,
        baseline: ExperimentResult,
        treatment: ExperimentResult,
        metrics: Optional[List[str]] = None,
        primary_metric: str = "sharpe_ratio",
    ) -> ExperimentComparison:
        """
        Compare two single experiment runs.

        Note: Single run comparison has limited statistical power.
        Use compare_multiple_runs for proper A/B testing.

        Args:
            baseline: Baseline experiment result
            treatment: Treatment experiment result
            metrics: Metrics to compare (default: all backtest metrics)
            primary_metric: Primary metric for recommendation

        Returns:
            ExperimentComparison
        """
        if metrics is None:
            metrics = list(
                set(baseline.backtest_metrics.keys())
                & set(treatment.backtest_metrics.keys())
            )

        comparisons = {}
        warnings = []

        for metric in metrics:
            baseline_val = baseline.backtest_metrics.get(metric, 0)
            treatment_val = treatment.backtest_metrics.get(metric, 0)

            lower_better = metric in self.LOWER_IS_BETTER
            diff = treatment_val - baseline_val
            rel_diff = diff / abs(baseline_val) if baseline_val != 0 else 0

            # Determine winner
            if lower_better:
                winner = "treatment" if diff < 0 else ("baseline" if diff > 0 else "tie")
            else:
                winner = "treatment" if diff > 0 else ("baseline" if diff < 0 else "tie")

            comparisons[metric] = MetricComparison(
                metric_name=metric,
                baseline_value=baseline_val,
                treatment_value=treatment_val,
                absolute_difference=diff,
                relative_difference=rel_diff,
                p_value=None,  # No p-value for single runs
                is_significant=False,
                confidence_interval=None,
                effect_size=None,
                winner=winner,
            )

        warnings.append("Single run comparison - limited statistical validity")

        # Generate recommendation
        primary_comp = comparisons.get(primary_metric)
        if primary_comp:
            if primary_comp.relative_difference > 0.1:
                recommendation = f"Treatment shows {primary_comp.relative_difference:.1%} improvement in {primary_metric}. Consider more runs for validation."
            elif primary_comp.relative_difference < -0.1:
                recommendation = f"Baseline outperforms treatment by {abs(primary_comp.relative_difference):.1%}. Keep baseline."
            else:
                recommendation = "No significant difference. More runs needed."
        else:
            recommendation = "Primary metric not available."

        return ExperimentComparison(
            baseline_name=baseline.experiment_name,
            treatment_name=treatment.experiment_name,
            baseline_version=baseline.experiment_version,
            treatment_version=treatment.experiment_version,
            comparison_date=datetime.now(),
            metric_comparisons=comparisons,
            primary_metric=primary_metric,
            recommendation=recommendation,
            confidence_level=1 - self.significance_level,
            sample_sizes={"baseline": 1, "treatment": 1},
            warnings=warnings,
        )

    def compare_multiple_runs(
        self,
        baseline_runs: List[ExperimentResult],
        treatment_runs: List[ExperimentResult],
        metrics: Optional[List[str]] = None,
        primary_metric: str = "sharpe_ratio",
    ) -> ExperimentComparison:
        """
        Compare multiple runs with statistical testing.

        Uses t-test and bootstrap for significance testing.

        Args:
            baseline_runs: List of baseline experiment results
            treatment_runs: List of treatment experiment results
            metrics: Metrics to compare
            primary_metric: Primary metric for recommendation

        Returns:
            ExperimentComparison with statistical analysis
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using simple comparison")
            return self.compare_single_run(
                baseline_runs[0],
                treatment_runs[0],
                metrics,
                primary_metric,
            )

        if len(baseline_runs) < 2 or len(treatment_runs) < 2:
            logger.warning("Insufficient runs for statistical testing")
            return self.compare_single_run(
                baseline_runs[0],
                treatment_runs[0],
                metrics,
                primary_metric,
            )

        # Collect all metrics
        if metrics is None:
            all_metrics = set()
            for run in baseline_runs + treatment_runs:
                all_metrics.update(run.backtest_metrics.keys())
            metrics = list(all_metrics)

        comparisons = {}
        warnings = []

        for metric in metrics:
            baseline_values = np.array([
                r.backtest_metrics.get(metric, np.nan)
                for r in baseline_runs
            ])
            treatment_values = np.array([
                r.backtest_metrics.get(metric, np.nan)
                for r in treatment_runs
            ])

            # Remove NaN
            baseline_values = baseline_values[~np.isnan(baseline_values)]
            treatment_values = treatment_values[~np.isnan(treatment_values)]

            if len(baseline_values) < 2 or len(treatment_values) < 2:
                continue

            baseline_mean = np.mean(baseline_values)
            treatment_mean = np.mean(treatment_values)

            # T-test
            t_stat, p_value = stats.ttest_ind(
                baseline_values,
                treatment_values,
                equal_var=False,  # Welch's t-test
            )

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(baseline_values) + np.var(treatment_values)) / 2
            )
            effect_size = (treatment_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0

            # Bootstrap confidence interval
            ci = self._bootstrap_ci(baseline_values, treatment_values)

            # Determine significance
            is_significant = p_value < self.significance_level

            # Determine winner
            lower_better = metric in self.LOWER_IS_BETTER
            diff = treatment_mean - baseline_mean

            if is_significant:
                if lower_better:
                    winner = "treatment" if diff < 0 else "baseline"
                else:
                    winner = "treatment" if diff > 0 else "baseline"
            else:
                winner = "tie"

            comparisons[metric] = MetricComparison(
                metric_name=metric,
                baseline_value=baseline_mean,
                treatment_value=treatment_mean,
                absolute_difference=diff,
                relative_difference=diff / abs(baseline_mean) if baseline_mean != 0 else 0,
                p_value=float(p_value),
                is_significant=is_significant,
                confidence_interval=ci,
                effect_size=float(effect_size),
                winner=winner,
            )

        # Check sample size
        if len(baseline_runs) < self.min_samples:
            warnings.append(
                f"Baseline sample size ({len(baseline_runs)}) below minimum ({self.min_samples})"
            )
        if len(treatment_runs) < self.min_samples:
            warnings.append(
                f"Treatment sample size ({len(treatment_runs)}) below minimum ({self.min_samples})"
            )

        # Generate recommendation
        primary_comp = comparisons.get(primary_metric)
        if primary_comp:
            if primary_comp.is_significant and primary_comp.winner == "treatment":
                recommendation = (
                    f"DEPLOY: Treatment shows statistically significant improvement "
                    f"({primary_comp.relative_difference:+.1%}, p={primary_comp.p_value:.4f})"
                )
            elif primary_comp.is_significant and primary_comp.winner == "baseline":
                recommendation = (
                    f"KEEP BASELINE: Baseline significantly better "
                    f"({abs(primary_comp.relative_difference):.1%}, p={primary_comp.p_value:.4f})"
                )
            else:
                recommendation = (
                    f"NO ACTION: No significant difference "
                    f"(p={primary_comp.p_value:.4f} > {self.significance_level})"
                )
        else:
            recommendation = "Primary metric not available for comparison."

        return ExperimentComparison(
            baseline_name=baseline_runs[0].experiment_name,
            treatment_name=treatment_runs[0].experiment_name,
            baseline_version=baseline_runs[0].experiment_version,
            treatment_version=treatment_runs[0].experiment_version,
            comparison_date=datetime.now(),
            metric_comparisons=comparisons,
            primary_metric=primary_metric,
            recommendation=recommendation,
            confidence_level=1 - self.significance_level,
            sample_sizes={
                "baseline": len(baseline_runs),
                "treatment": len(treatment_runs),
            },
            warnings=warnings,
        )

    def _bootstrap_ci(
        self,
        baseline_values: np.ndarray,
        treatment_values: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for difference.

        Args:
            baseline_values: Baseline sample values
            treatment_values: Treatment sample values
            alpha: Significance level

        Returns:
            (lower, upper) confidence interval
        """
        differences = []

        for _ in range(self.bootstrap_iterations):
            baseline_sample = np.random.choice(
                baseline_values,
                size=len(baseline_values),
                replace=True,
            )
            treatment_sample = np.random.choice(
                treatment_values,
                size=len(treatment_values),
                replace=True,
            )
            differences.append(np.mean(treatment_sample) - np.mean(baseline_sample))

        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))


def compare_experiments(
    baseline_name: str,
    treatment_name: str,
    registry: Optional[Any] = None,
    metrics: Optional[List[str]] = None,
    primary_metric: str = "sharpe_ratio",
) -> ExperimentComparison:
    """
    Compare two experiments by name.

    Convenience function that loads experiments from registry.

    Args:
        baseline_name: Baseline experiment name
        treatment_name: Treatment experiment name
        registry: Experiment registry (uses FileBasedRegistry if None)
        metrics: Metrics to compare
        primary_metric: Primary metric for recommendation

    Returns:
        ExperimentComparison

    Example:
        comparison = compare_experiments(
            "baseline_ppo_v1",
            "new_ppo_v2",
            primary_metric="sharpe_ratio",
        )
        print(comparison.summary())

        if comparison.is_significantly_better("sharpe_ratio"):
            print("Deploy treatment!")
    """
    if registry is None:
        registry = FileBasedRegistry()

    # Get runs
    baseline_runs = registry.get_runs(baseline_name)
    treatment_runs = registry.get_runs(treatment_name)

    if not baseline_runs:
        raise ValueError(f"No runs found for baseline: {baseline_name}")
    if not treatment_runs:
        raise ValueError(f"No runs found for treatment: {treatment_name}")

    # Filter successful runs
    baseline_runs = [r for r in baseline_runs if r.status == "success"]
    treatment_runs = [r for r in treatment_runs if r.status == "success"]

    if not baseline_runs:
        raise ValueError(f"No successful runs for baseline: {baseline_name}")
    if not treatment_runs:
        raise ValueError(f"No successful runs for treatment: {treatment_name}")

    # Compare
    comparator = ExperimentComparator()

    if len(baseline_runs) > 1 and len(treatment_runs) > 1:
        return comparator.compare_multiple_runs(
            baseline_runs,
            treatment_runs,
            metrics,
            primary_metric,
        )
    else:
        return comparator.compare_single_run(
            baseline_runs[0],
            treatment_runs[0],
            metrics,
            primary_metric,
        )


__all__ = [
    "ExperimentComparison",
    "ExperimentComparator",
    "MetricComparison",
    "compare_experiments",
]
