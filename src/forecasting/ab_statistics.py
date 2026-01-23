"""
Forecasting A/B Statistics Module
=================================

Statistical testing framework for comparing forecasting model experiments.
Adapted for direction accuracy and return-based comparisons.

Design Patterns:
    - Strategy Pattern: Different statistical tests as strategies
    - Factory Pattern: Test selection based on metric type
    - SSOT: All thresholds and configs from contracts

Key Differences from RL A/B Testing:
    - Primary metric: Direction Accuracy (not Sharpe)
    - McNemar test for paired binary outcomes
    - Multiple comparison correction (Bonferroni) for horizons
    - No real-time shadow mode (post-hoc comparison)

@version 1.0.0
@contract CTR-FORECAST-AB-001
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS (SSOT)
# =============================================================================

class SignificanceLevel(str, Enum):
    """Statistical significance levels."""
    VERY_STRONG = "very_strong"  # p < 0.001
    STRONG = "strong"            # p < 0.01
    STANDARD = "standard"        # p < 0.05
    WEAK = "weak"                # p < 0.10
    NOT_SIGNIFICANT = "not_significant"  # p >= 0.10


class EffectSize(str, Enum):
    """Effect size interpretation (Cohen's conventions)."""
    NEGLIGIBLE = "negligible"  # |d| < 0.2
    SMALL = "small"            # 0.2 <= |d| < 0.5
    MEDIUM = "medium"          # 0.5 <= |d| < 0.8
    LARGE = "large"            # |d| >= 0.8


class Recommendation(str, Enum):
    """Experiment comparison recommendation."""
    DEPLOY_TREATMENT = "deploy_treatment"
    KEEP_BASELINE = "keep_baseline"
    INCONCLUSIVE = "inconclusive"
    NEEDS_MORE_DATA = "needs_more_data"


# Default configuration (SSOT)
AB_CONFIG = {
    "default_alpha": 0.05,
    "min_samples": 30,
    "bootstrap_iterations": 10000,
    "direction_threshold": 0.0001,  # Min return to count as directional
    "bonferroni_correction": True,
    "effect_size_thresholds": {
        "negligible": 0.2,
        "small": 0.5,
        "medium": 0.8,
    },
}

# Horizons from forecasting contracts (SSOT)
HORIZONS = (1, 5, 10, 15, 20, 25, 30)


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class StatisticalTestResult:
    """Result of a single statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    significance_level: SignificanceLevel
    effect_size: Optional[float] = None
    effect_interpretation: Optional[EffectSize] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size_baseline: int = 0
    sample_size_treatment: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "significance_level": self.significance_level.value,
            "effect_size": self.effect_size,
            "effect_interpretation": self.effect_interpretation.value if self.effect_interpretation else None,
            "confidence_interval": self.confidence_interval,
            "sample_size_baseline": self.sample_size_baseline,
            "sample_size_treatment": self.sample_size_treatment,
            "details": self.details,
        }


@dataclass
class HorizonComparisonResult:
    """Comparison result for a single horizon."""
    horizon: int
    baseline_metric: float
    treatment_metric: float
    metric_difference: float
    metric_difference_pct: float
    statistical_test: StatisticalTestResult
    winner: str  # "baseline", "treatment", "tie"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "baseline_metric": self.baseline_metric,
            "treatment_metric": self.treatment_metric,
            "metric_difference": self.metric_difference,
            "metric_difference_pct": self.metric_difference_pct,
            "statistical_test": self.statistical_test.to_dict(),
            "winner": self.winner,
        }


@dataclass
class ExperimentComparisonResult:
    """Complete A/B comparison result."""
    baseline_experiment: str
    treatment_experiment: str
    comparison_date: str
    primary_metric: str
    horizon_results: Dict[int, HorizonComparisonResult]
    aggregate_result: StatisticalTestResult
    recommendation: Recommendation
    confidence_score: float  # 0-1 confidence in recommendation
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_experiment": self.baseline_experiment,
            "treatment_experiment": self.treatment_experiment,
            "comparison_date": self.comparison_date,
            "primary_metric": self.primary_metric,
            "horizon_results": {h: r.to_dict() for h, r in self.horizon_results.items()},
            "aggregate_result": self.aggregate_result.to_dict(),
            "recommendation": self.recommendation.value,
            "confidence_score": self.confidence_score,
            "summary": self.summary,
            "warnings": self.warnings,
        }


# =============================================================================
# STATISTICAL TEST STRATEGIES (Strategy Pattern)
# =============================================================================

class StatisticalTestStrategy(ABC):
    """Abstract base class for statistical tests."""

    @abstractmethod
    def run(
        self,
        baseline_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run the statistical test."""
        pass

    @staticmethod
    def _interpret_significance(p_value: float) -> SignificanceLevel:
        """Interpret p-value as significance level."""
        if p_value < 0.001:
            return SignificanceLevel.VERY_STRONG
        elif p_value < 0.01:
            return SignificanceLevel.STRONG
        elif p_value < 0.05:
            return SignificanceLevel.STANDARD
        elif p_value < 0.10:
            return SignificanceLevel.WEAK
        return SignificanceLevel.NOT_SIGNIFICANT

    @staticmethod
    def _interpret_effect_size(d: float) -> EffectSize:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return EffectSize.NEGLIGIBLE
        elif d_abs < 0.5:
            return EffectSize.SMALL
        elif d_abs < 0.8:
            return EffectSize.MEDIUM
        return EffectSize.LARGE


class McNemarTest(StatisticalTestStrategy):
    """
    McNemar's test for paired binary outcomes.

    Used for comparing direction accuracy between two models
    on the same set of predictions (paired data).

    H0: Both models have the same accuracy
    H1: Models have different accuracy
    """

    def run(
        self,
        baseline_correct: np.ndarray,
        treatment_correct: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """
        Run McNemar test.

        Args:
            baseline_correct: Boolean array (1=correct, 0=incorrect)
            treatment_correct: Boolean array (1=correct, 0=incorrect)
            alpha: Significance level

        Returns:
            StatisticalTestResult
        """
        if len(baseline_correct) != len(treatment_correct):
            raise ValueError("Arrays must have same length for paired test")

        n = len(baseline_correct)

        # Build contingency table
        # b = baseline correct, treatment incorrect
        # c = baseline incorrect, treatment correct
        b = np.sum((baseline_correct == 1) & (treatment_correct == 0))
        c = np.sum((baseline_correct == 0) & (treatment_correct == 1))

        # McNemar statistic with continuity correction
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

        # Effect size: odds ratio
        odds_ratio = (c + 0.5) / (b + 0.5) if b > 0 else float('inf')
        effect_size = np.log(odds_ratio)  # Log odds ratio

        significant = p_value < alpha

        return StatisticalTestResult(
            test_name="McNemar",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            significance_level=self._interpret_significance(p_value),
            effect_size=effect_size,
            effect_interpretation=self._interpret_effect_size(effect_size),
            sample_size_baseline=n,
            sample_size_treatment=n,
            details={
                "baseline_only_correct": int(b),
                "treatment_only_correct": int(c),
                "both_correct": int(np.sum((baseline_correct == 1) & (treatment_correct == 1))),
                "both_incorrect": int(np.sum((baseline_correct == 0) & (treatment_correct == 0))),
                "odds_ratio": odds_ratio,
            },
        )


class PairedTTest(StatisticalTestStrategy):
    """
    Paired t-test for continuous metrics (e.g., prediction errors).

    Used when comparing paired observations from two conditions.
    """

    def run(
        self,
        baseline_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run paired t-test."""
        if len(baseline_data) != len(treatment_data):
            raise ValueError("Arrays must have same length for paired test")

        n = len(baseline_data)
        differences = treatment_data - baseline_data

        # Paired t-test
        statistic, p_value = stats.ttest_rel(treatment_data, baseline_data)

        # Cohen's d for paired samples
        d_diff = differences.mean()
        s_diff = differences.std(ddof=1)
        cohens_d = d_diff / s_diff if s_diff > 0 else 0

        # Confidence interval for difference
        se = s_diff / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lower = d_diff - t_crit * se
        ci_upper = d_diff + t_crit * se

        significant = p_value < alpha

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            significance_level=self._interpret_significance(p_value),
            effect_size=cohens_d,
            effect_interpretation=self._interpret_effect_size(cohens_d),
            confidence_interval=(ci_lower, ci_upper),
            sample_size_baseline=n,
            sample_size_treatment=n,
            details={
                "mean_difference": d_diff,
                "std_difference": s_diff,
                "baseline_mean": float(baseline_data.mean()),
                "treatment_mean": float(treatment_data.mean()),
            },
        )


class WelchTTest(StatisticalTestStrategy):
    """
    Welch's t-test for independent samples with unequal variances.

    Used when samples are independent (not paired).
    """

    def run(
        self,
        baseline_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run Welch's t-test."""
        n1, n2 = len(baseline_data), len(treatment_data)

        # Welch's t-test
        statistic, p_value = stats.ttest_ind(
            treatment_data, baseline_data, equal_var=False
        )

        # Cohen's d for independent samples
        pooled_std = np.sqrt(
            ((n1 - 1) * baseline_data.std(ddof=1) ** 2 +
             (n2 - 1) * treatment_data.std(ddof=1) ** 2) /
            (n1 + n2 - 2)
        )
        cohens_d = (treatment_data.mean() - baseline_data.mean()) / pooled_std if pooled_std > 0 else 0

        significant = p_value < alpha

        return StatisticalTestResult(
            test_name="Welch t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            significance_level=self._interpret_significance(p_value),
            effect_size=cohens_d,
            effect_interpretation=self._interpret_effect_size(cohens_d),
            sample_size_baseline=n1,
            sample_size_treatment=n2,
            details={
                "baseline_mean": float(baseline_data.mean()),
                "baseline_std": float(baseline_data.std()),
                "treatment_mean": float(treatment_data.mean()),
                "treatment_std": float(treatment_data.std()),
            },
        )


class BootstrapTest(StatisticalTestStrategy):
    """
    Bootstrap test for metric difference.

    Non-parametric approach using bootstrap resampling
    to estimate confidence intervals.
    """

    def __init__(self, n_bootstrap: int = 10000):
        self.n_bootstrap = n_bootstrap

    def run(
        self,
        baseline_data: np.ndarray,
        treatment_data: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Run bootstrap test."""
        n1, n2 = len(baseline_data), len(treatment_data)

        observed_diff = treatment_data.mean() - baseline_data.mean()

        # Bootstrap resampling
        bootstrap_diffs = []
        rng = np.random.default_rng(42)

        for _ in range(self.n_bootstrap):
            b_sample = rng.choice(baseline_data, size=n1, replace=True)
            t_sample = rng.choice(treatment_data, size=n2, replace=True)
            bootstrap_diffs.append(t_sample.mean() - b_sample.mean())

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

        # P-value: proportion of bootstrap samples with opposite sign
        if observed_diff > 0:
            p_value = 2 * np.mean(bootstrap_diffs <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= 0)

        p_value = min(p_value, 1.0)

        # Effect size (standardized)
        pooled_std = np.sqrt((baseline_data.var() + treatment_data.var()) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        significant = p_value < alpha

        return StatisticalTestResult(
            test_name="Bootstrap",
            statistic=observed_diff,
            p_value=p_value,
            significant=significant,
            significance_level=self._interpret_significance(p_value),
            effect_size=effect_size,
            effect_interpretation=self._interpret_effect_size(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            sample_size_baseline=n1,
            sample_size_treatment=n2,
            details={
                "n_bootstrap": self.n_bootstrap,
                "observed_difference": observed_diff,
                "bootstrap_std": float(bootstrap_diffs.std()),
            },
        )


# =============================================================================
# TEST FACTORY (Factory Pattern)
# =============================================================================

class StatisticalTestFactory:
    """Factory for creating appropriate statistical tests."""

    _tests = {
        "mcnemar": McNemarTest,
        "paired_ttest": PairedTTest,
        "welch_ttest": WelchTTest,
        "bootstrap": BootstrapTest,
    }

    @classmethod
    def create(cls, test_name: str, **kwargs) -> StatisticalTestStrategy:
        """Create a statistical test by name."""
        if test_name not in cls._tests:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(cls._tests.keys())}")

        return cls._tests[test_name](**kwargs)

    @classmethod
    def get_available_tests(cls) -> List[str]:
        """Get list of available tests."""
        return list(cls._tests.keys())


# =============================================================================
# MAIN A/B STATISTICS CLASS
# =============================================================================

class ForecastABStatistics:
    """
    A/B Statistics engine for forecasting experiments.

    Provides comprehensive statistical comparison of forecasting models
    with support for multiple horizons and Bonferroni correction.

    Usage:
        ab = ForecastABStatistics(alpha=0.05)
        result = ab.compare_experiments(
            baseline_results=baseline_df,
            treatment_results=treatment_df,
            actual_prices=prices_df,
            primary_metric="direction_accuracy"
        )
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bonferroni_correction: bool = True,
        min_samples: int = 30,
        bootstrap_iterations: int = 10000,
    ):
        """
        Initialize A/B Statistics engine.

        Args:
            alpha: Base significance level
            bonferroni_correction: Apply Bonferroni correction for multiple comparisons
            min_samples: Minimum samples required for valid comparison
            bootstrap_iterations: Number of bootstrap iterations
        """
        self.alpha = alpha
        self.bonferroni_correction = bonferroni_correction
        self.min_samples = min_samples
        self.bootstrap_iterations = bootstrap_iterations

        # Initialize test strategies
        self._mcnemar = McNemarTest()
        self._paired_ttest = PairedTTest()
        self._bootstrap = BootstrapTest(n_bootstrap=bootstrap_iterations)

    def compare_direction_accuracy(
        self,
        baseline_predictions: pd.DataFrame,
        treatment_predictions: pd.DataFrame,
        actual_prices: pd.DataFrame,
        horizon: int,
    ) -> HorizonComparisonResult:
        """
        Compare direction accuracy for a specific horizon.

        Args:
            baseline_predictions: Baseline model predictions
            treatment_predictions: Treatment model predictions
            actual_prices: Actual price data with 'date' and 'close' columns
            horizon: Forecast horizon (days)

        Returns:
            HorizonComparisonResult with statistical test
        """
        # Ensure aligned data
        common_dates = set(baseline_predictions['inference_date']) & \
                      set(treatment_predictions['inference_date'])

        if len(common_dates) < self.min_samples:
            return HorizonComparisonResult(
                horizon=horizon,
                baseline_metric=0.0,
                treatment_metric=0.0,
                metric_difference=0.0,
                metric_difference_pct=0.0,
                statistical_test=StatisticalTestResult(
                    test_name="McNemar",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    significance_level=SignificanceLevel.NOT_SIGNIFICANT,
                    details={"error": "Insufficient samples"},
                ),
                winner="tie",
            )

        # Calculate correctness for each prediction
        baseline_correct = []
        treatment_correct = []

        for date in sorted(common_dates):
            b_pred = baseline_predictions[
                baseline_predictions['inference_date'] == date
            ].iloc[0]
            t_pred = treatment_predictions[
                treatment_predictions['inference_date'] == date
            ].iloc[0]

            # Get actual price movement
            pred_date = pd.to_datetime(date)
            target_date = pred_date + pd.Timedelta(days=horizon)

            if target_date not in actual_prices.index:
                continue

            actual_return = (
                actual_prices.loc[target_date, 'close'] -
                actual_prices.loc[pred_date, 'close']
            ) / actual_prices.loc[pred_date, 'close']

            actual_direction = 1 if actual_return > 0.0001 else (-1 if actual_return < -0.0001 else 0)

            # Check if predictions were correct
            b_direction = 1 if b_pred.get('predicted_return_pct', 0) > 0.0001 else \
                         (-1 if b_pred.get('predicted_return_pct', 0) < -0.0001 else 0)
            t_direction = 1 if t_pred.get('predicted_return_pct', 0) > 0.0001 else \
                         (-1 if t_pred.get('predicted_return_pct', 0) < -0.0001 else 0)

            if actual_direction != 0:  # Only count when there was actual movement
                baseline_correct.append(1 if b_direction == actual_direction else 0)
                treatment_correct.append(1 if t_direction == actual_direction else 0)

        if len(baseline_correct) < self.min_samples:
            return self._insufficient_data_result(horizon)

        baseline_correct = np.array(baseline_correct)
        treatment_correct = np.array(treatment_correct)

        # Calculate accuracies
        baseline_da = baseline_correct.mean()
        treatment_da = treatment_correct.mean()
        da_diff = treatment_da - baseline_da
        da_diff_pct = (da_diff / baseline_da * 100) if baseline_da > 0 else 0

        # Run McNemar test
        test_result = self._mcnemar.run(baseline_correct, treatment_correct, self.alpha)

        # Determine winner
        if test_result.significant:
            winner = "treatment" if da_diff > 0 else "baseline"
        else:
            winner = "tie"

        return HorizonComparisonResult(
            horizon=horizon,
            baseline_metric=baseline_da,
            treatment_metric=treatment_da,
            metric_difference=da_diff,
            metric_difference_pct=da_diff_pct,
            statistical_test=test_result,
            winner=winner,
        )

    def compare_rmse(
        self,
        baseline_errors: pd.Series,
        treatment_errors: pd.Series,
        horizon: int,
    ) -> HorizonComparisonResult:
        """
        Compare RMSE between two models using paired t-test.

        Args:
            baseline_errors: Absolute prediction errors from baseline
            treatment_errors: Absolute prediction errors from treatment
            horizon: Forecast horizon

        Returns:
            HorizonComparisonResult
        """
        if len(baseline_errors) != len(treatment_errors):
            raise ValueError("Error arrays must have same length")

        if len(baseline_errors) < self.min_samples:
            return self._insufficient_data_result(horizon)

        baseline_rmse = np.sqrt((baseline_errors ** 2).mean())
        treatment_rmse = np.sqrt((treatment_errors ** 2).mean())

        # Squared errors for comparison
        baseline_sq = baseline_errors ** 2
        treatment_sq = treatment_errors ** 2

        # Paired t-test on squared errors
        test_result = self._paired_ttest.run(
            baseline_sq.values,
            treatment_sq.values,
            self.alpha
        )

        rmse_diff = treatment_rmse - baseline_rmse
        rmse_diff_pct = (rmse_diff / baseline_rmse * 100) if baseline_rmse > 0 else 0

        # For RMSE, lower is better
        if test_result.significant:
            winner = "treatment" if rmse_diff < 0 else "baseline"
        else:
            winner = "tie"

        return HorizonComparisonResult(
            horizon=horizon,
            baseline_metric=baseline_rmse,
            treatment_metric=treatment_rmse,
            metric_difference=rmse_diff,
            metric_difference_pct=rmse_diff_pct,
            statistical_test=test_result,
            winner=winner,
        )

    def compare_by_horizon(
        self,
        baseline_results: Dict[int, pd.DataFrame],
        treatment_results: Dict[int, pd.DataFrame],
        actual_prices: pd.DataFrame,
        metric: str = "direction_accuracy",
    ) -> Dict[int, HorizonComparisonResult]:
        """
        Compare experiments across all horizons.

        Applies Bonferroni correction if enabled.

        Args:
            baseline_results: {horizon: predictions_df}
            treatment_results: {horizon: predictions_df}
            actual_prices: Actual price data
            metric: "direction_accuracy" or "rmse"

        Returns:
            Dict mapping horizon to comparison result
        """
        horizons = sorted(set(baseline_results.keys()) & set(treatment_results.keys()))
        n_comparisons = len(horizons)

        # Adjust alpha for multiple comparisons
        adjusted_alpha = self.alpha / n_comparisons if self.bonferroni_correction else self.alpha

        results = {}

        for horizon in horizons:
            if metric == "direction_accuracy":
                result = self.compare_direction_accuracy(
                    baseline_results[horizon],
                    treatment_results[horizon],
                    actual_prices,
                    horizon,
                )
            elif metric == "rmse":
                # Need to calculate errors first
                # This would require additional processing
                continue
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Adjust significance for Bonferroni
            if self.bonferroni_correction:
                result.statistical_test.significant = (
                    result.statistical_test.p_value < adjusted_alpha
                )

            results[horizon] = result

        return results

    def compare_experiments(
        self,
        baseline_name: str,
        treatment_name: str,
        baseline_results: Dict[int, pd.DataFrame],
        treatment_results: Dict[int, pd.DataFrame],
        actual_prices: pd.DataFrame,
        primary_metric: str = "direction_accuracy",
    ) -> ExperimentComparisonResult:
        """
        Full experiment comparison with recommendation.

        Args:
            baseline_name: Name of baseline experiment
            treatment_name: Name of treatment experiment
            baseline_results: Baseline predictions by horizon
            treatment_results: Treatment predictions by horizon
            actual_prices: Actual price data
            primary_metric: Primary comparison metric

        Returns:
            ExperimentComparisonResult with full analysis
        """
        from datetime import datetime

        warnings_list = []

        # Compare by horizon
        horizon_results = self.compare_by_horizon(
            baseline_results,
            treatment_results,
            actual_prices,
            primary_metric,
        )

        if not horizon_results:
            return ExperimentComparisonResult(
                baseline_experiment=baseline_name,
                treatment_experiment=treatment_name,
                comparison_date=datetime.now().isoformat(),
                primary_metric=primary_metric,
                horizon_results={},
                aggregate_result=StatisticalTestResult(
                    test_name="aggregate",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    significance_level=SignificanceLevel.NOT_SIGNIFICANT,
                ),
                recommendation=Recommendation.NEEDS_MORE_DATA,
                confidence_score=0.0,
                warnings=["No valid horizon comparisons"],
            )

        # Aggregate results
        treatment_wins = sum(1 for r in horizon_results.values() if r.winner == "treatment")
        baseline_wins = sum(1 for r in horizon_results.values() if r.winner == "baseline")
        ties = sum(1 for r in horizon_results.values() if r.winner == "tie")

        # Compute aggregate metrics
        avg_baseline = np.mean([r.baseline_metric for r in horizon_results.values()])
        avg_treatment = np.mean([r.treatment_metric for r in horizon_results.values()])
        avg_diff = avg_treatment - avg_baseline

        # Aggregate p-value using Fisher's method
        p_values = [r.statistical_test.p_value for r in horizon_results.values()]
        chi2_stat = -2 * np.sum(np.log(np.maximum(p_values, 1e-10)))
        aggregate_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))

        aggregate_significant = aggregate_p < self.alpha

        aggregate_result = StatisticalTestResult(
            test_name="Fisher's combined",
            statistic=chi2_stat,
            p_value=aggregate_p,
            significant=aggregate_significant,
            significance_level=StatisticalTestStrategy._interpret_significance(aggregate_p),
            details={
                "avg_baseline_metric": avg_baseline,
                "avg_treatment_metric": avg_treatment,
                "avg_difference": avg_diff,
                "treatment_wins": treatment_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
            },
        )

        # Generate recommendation
        if not aggregate_significant:
            if ties == len(horizon_results):
                recommendation = Recommendation.INCONCLUSIVE
                confidence = 0.3
            else:
                recommendation = Recommendation.INCONCLUSIVE
                confidence = 0.5
        else:
            if treatment_wins > baseline_wins and avg_diff > 0:
                recommendation = Recommendation.DEPLOY_TREATMENT
                confidence = min(0.95, 0.7 + 0.05 * treatment_wins)
            elif baseline_wins > treatment_wins:
                recommendation = Recommendation.KEEP_BASELINE
                confidence = min(0.95, 0.7 + 0.05 * baseline_wins)
            else:
                recommendation = Recommendation.INCONCLUSIVE
                confidence = 0.4

        # Add warnings
        if len(horizon_results) < len(HORIZONS):
            warnings_list.append(f"Only {len(horizon_results)}/{len(HORIZONS)} horizons compared")

        for h, r in horizon_results.items():
            if r.statistical_test.sample_size_baseline < 50:
                warnings_list.append(f"Low sample size for horizon {h}: {r.statistical_test.sample_size_baseline}")

        summary = {
            "horizons_compared": len(horizon_results),
            "treatment_wins": treatment_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "avg_metric_improvement": avg_diff,
            "avg_metric_improvement_pct": (avg_diff / avg_baseline * 100) if avg_baseline > 0 else 0,
        }

        return ExperimentComparisonResult(
            baseline_experiment=baseline_name,
            treatment_experiment=treatment_name,
            comparison_date=datetime.now().isoformat(),
            primary_metric=primary_metric,
            horizon_results=horizon_results,
            aggregate_result=aggregate_result,
            recommendation=recommendation,
            confidence_score=confidence,
            summary=summary,
            warnings=warnings_list,
        )

    def _insufficient_data_result(self, horizon: int) -> HorizonComparisonResult:
        """Create result for insufficient data."""
        return HorizonComparisonResult(
            horizon=horizon,
            baseline_metric=0.0,
            treatment_metric=0.0,
            metric_difference=0.0,
            metric_difference_pct=0.0,
            statistical_test=StatisticalTestResult(
                test_name="N/A",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                significance_level=SignificanceLevel.NOT_SIGNIFICANT,
                details={"error": f"Insufficient samples (min: {self.min_samples})"},
            ),
            winner="tie",
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_ab_contract_hash() -> str:
    """Compute hash of this module's configuration."""
    config_data = {
        "ab_config": AB_CONFIG,
        "horizons": HORIZONS,
        "version": "1.0.0",
    }
    content = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Contract metadata
AB_CONTRACT_VERSION = "1.0.0"
AB_CONTRACT_HASH = compute_ab_contract_hash()


__all__ = [
    # Enums
    "SignificanceLevel",
    "EffectSize",
    "Recommendation",
    # Data classes
    "StatisticalTestResult",
    "HorizonComparisonResult",
    "ExperimentComparisonResult",
    # Test strategies
    "StatisticalTestStrategy",
    "McNemarTest",
    "PairedTTest",
    "WelchTTest",
    "BootstrapTest",
    "StatisticalTestFactory",
    # Main class
    "ForecastABStatistics",
    # Config
    "AB_CONFIG",
    "AB_CONTRACT_VERSION",
    "AB_CONTRACT_HASH",
]
