"""
A/B Testing Statistical Significance Module
===========================================

Statistical tests for comparing model performance in A/B testing scenarios.
Provides rigorous statistical analysis to determine if performance differences
are significant or due to random chance.

P1: Statistical Significance Testing

Features:
- Chi-square test for win rates
- T-test for Sharpe ratios
- Bootstrap confidence intervals
- Minimum sample size calculator
- Effect size calculations (Cohen's d)
- Bayesian A/B testing support

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SignificanceLevel(Enum):
    """Common significance levels for hypothesis testing."""
    VERY_STRONG = 0.001  # p < 0.001
    STRONG = 0.01        # p < 0.01
    STANDARD = 0.05      # p < 0.05
    WEAK = 0.10          # p < 0.10


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    test_name: str
    metric_name: str

    # Values
    control_value: float
    treatment_value: float
    difference: float
    relative_difference: float  # Percentage change

    # Statistical measures
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float

    # Interpretation
    is_significant: bool
    significance_level: str
    effect_size: Optional[float]
    effect_size_interpretation: Optional[str]

    # Sample info
    control_n: int
    treatment_n: int
    power: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "metric_name": self.metric_name,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "difference": self.difference,
            "relative_difference_pct": self.relative_difference * 100,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "significance_level": self.significance_level,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "control_n": self.control_n,
            "treatment_n": self.treatment_n,
            "power": self.power,
        }


class ABStatistics:
    """
    Statistical analysis for A/B testing of trading models.

    Provides methods to determine if differences in model performance
    metrics are statistically significant.

    Usage:
        ab = ABStatistics(confidence_level=0.95)

        # Compare win rates
        result = ab.compare_win_rates(
            control_wins=120, control_losses=80,
            treatment_wins=140, treatment_losses=60
        )
        print(f"Win rate difference significant: {result.is_significant}")

        # Compare Sharpe ratios
        result = ab.compare_sharpe_ratios(
            control_returns=[0.01, -0.02, 0.015, ...],
            treatment_returns=[0.02, -0.01, 0.025, ...]
        )
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_effect_size: float = 0.05,
    ):
        """
        Initialize the A/B statistics calculator.

        Args:
            confidence_level: Confidence level for tests (default: 0.95)
            min_effect_size: Minimum effect size to detect (for power analysis)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.min_effect_size = min_effect_size

    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value into significance level."""
        if p_value < SignificanceLevel.VERY_STRONG.value:
            return "very_strong (p < 0.001)"
        elif p_value < SignificanceLevel.STRONG.value:
            return "strong (p < 0.01)"
        elif p_value < SignificanceLevel.STANDARD.value:
            return "significant (p < 0.05)"
        elif p_value < SignificanceLevel.WEAK.value:
            return "weak (p < 0.10)"
        else:
            return "not_significant (p >= 0.10)"

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def compare_win_rates(
        self,
        control_wins: int,
        control_losses: int,
        treatment_wins: int,
        treatment_losses: int,
    ) -> ABTestResult:
        """
        Compare win rates using Chi-square test.

        Tests the null hypothesis that both groups have the same win rate.

        Args:
            control_wins: Number of winning trades in control group
            control_losses: Number of losing trades in control group
            treatment_wins: Number of winning trades in treatment group
            treatment_losses: Number of losing trades in treatment group

        Returns:
            ABTestResult with chi-square test results
        """
        # Create contingency table
        observed = np.array([
            [control_wins, control_losses],
            [treatment_wins, treatment_losses]
        ])

        # Calculate win rates
        control_n = control_wins + control_losses
        treatment_n = treatment_wins + treatment_losses
        control_rate = control_wins / control_n if control_n > 0 else 0
        treatment_rate = treatment_wins / treatment_n if treatment_n > 0 else 0

        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        # Calculate confidence interval for difference in proportions
        pooled_rate = (control_wins + treatment_wins) / (control_n + treatment_n)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_n + 1/treatment_n))
        z = stats.norm.ppf(1 - self.alpha / 2)
        diff = treatment_rate - control_rate
        ci = (diff - z * se, diff + z * se)

        # Effect size (Cohen's h for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))

        return ABTestResult(
            test_name="chi_square_test",
            metric_name="win_rate",
            control_value=control_rate,
            treatment_value=treatment_rate,
            difference=diff,
            relative_difference=diff / control_rate if control_rate > 0 else 0,
            statistic=chi2,
            p_value=p_value,
            confidence_interval=ci,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            significance_level=self._interpret_p_value(p_value),
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_cohens_d(effect_size),
            control_n=control_n,
            treatment_n=treatment_n,
            power=None,  # Would require additional calculation
        )

    def compare_sharpe_ratios(
        self,
        control_returns: np.ndarray,
        treatment_returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> ABTestResult:
        """
        Compare Sharpe ratios using Welch's t-test on returns.

        Note: This is an approximation. For more rigorous Sharpe ratio
        comparison, consider bootstrap methods.

        Args:
            control_returns: Array of returns for control model
            treatment_returns: Array of returns for treatment model
            risk_free_rate: Risk-free rate (annualized, daily returns assumed)

        Returns:
            ABTestResult with t-test results
        """
        control_returns = np.asarray(control_returns)
        treatment_returns = np.asarray(treatment_returns)

        # Calculate Sharpe ratios (assuming daily returns, annualized)
        annualization_factor = np.sqrt(252)

        control_mean = np.mean(control_returns) - risk_free_rate / 252
        control_std = np.std(control_returns, ddof=1)
        control_sharpe = (control_mean / control_std * annualization_factor) if control_std > 0 else 0

        treatment_mean = np.mean(treatment_returns) - risk_free_rate / 252
        treatment_std = np.std(treatment_returns, ddof=1)
        treatment_sharpe = (treatment_mean / treatment_std * annualization_factor) if treatment_std > 0 else 0

        # Welch's t-test on returns
        t_stat, p_value = stats.ttest_ind(treatment_returns, control_returns, equal_var=False)

        # Bootstrap confidence interval for Sharpe ratio difference
        ci = self._bootstrap_sharpe_ci(control_returns, treatment_returns)

        # Cohen's d for effect size
        pooled_std = np.sqrt(
            ((len(control_returns) - 1) * control_std**2 +
             (len(treatment_returns) - 1) * treatment_std**2) /
            (len(control_returns) + len(treatment_returns) - 2)
        )
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        return ABTestResult(
            test_name="welch_t_test",
            metric_name="sharpe_ratio",
            control_value=control_sharpe,
            treatment_value=treatment_sharpe,
            difference=treatment_sharpe - control_sharpe,
            relative_difference=(treatment_sharpe - control_sharpe) / abs(control_sharpe) if control_sharpe != 0 else 0,
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            significance_level=self._interpret_p_value(p_value),
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_cohens_d(effect_size),
            control_n=len(control_returns),
            treatment_n=len(treatment_returns),
            power=None,
        )

    def _bootstrap_sharpe_ci(
        self,
        control_returns: np.ndarray,
        treatment_returns: np.ndarray,
        n_bootstrap: int = 10000,
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for Sharpe ratio difference.

        Args:
            control_returns: Control group returns
            treatment_returns: Treatment group returns
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        annualization_factor = np.sqrt(252)

        def calc_sharpe(returns):
            if len(returns) == 0 or np.std(returns) == 0:
                return 0
            return np.mean(returns) / np.std(returns) * annualization_factor

        sharpe_diffs = []
        for _ in range(n_bootstrap):
            control_sample = np.random.choice(control_returns, size=len(control_returns), replace=True)
            treatment_sample = np.random.choice(treatment_returns, size=len(treatment_returns), replace=True)

            control_sharpe = calc_sharpe(control_sample)
            treatment_sharpe = calc_sharpe(treatment_sample)
            sharpe_diffs.append(treatment_sharpe - control_sharpe)

        lower = np.percentile(sharpe_diffs, (1 - self.confidence_level) / 2 * 100)
        upper = np.percentile(sharpe_diffs, (1 + self.confidence_level) / 2 * 100)

        return (lower, upper)

    def calculate_minimum_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.80,
    ) -> int:
        """
        Calculate minimum sample size needed to detect an effect.

        Uses the formula for comparing two proportions.

        Args:
            baseline_rate: Expected baseline conversion/win rate
            minimum_detectable_effect: Minimum effect size to detect (absolute)
            power: Statistical power (default: 0.80)

        Returns:
            Minimum sample size per group
        """
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        p_pooled = (p1 + p2) / 2

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = (
            2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta) ** 2
        ) / (p2 - p1) ** 2

        return int(np.ceil(n))

    def calculate_power(
        self,
        n_control: int,
        n_treatment: int,
        control_rate: float,
        observed_difference: float,
    ) -> float:
        """
        Calculate statistical power of the current test.

        Args:
            n_control: Sample size in control group
            n_treatment: Sample size in treatment group
            control_rate: Baseline rate in control group
            observed_difference: Observed effect size

        Returns:
            Statistical power (0 to 1)
        """
        treatment_rate = control_rate + observed_difference
        pooled_rate = (control_rate + treatment_rate) / 2

        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_control + 1/n_treatment))

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_effect = abs(observed_difference) / se

        power = stats.norm.cdf(z_effect - z_alpha) + stats.norm.cdf(-z_effect - z_alpha)
        return power

    def bayesian_ab_test(
        self,
        control_wins: int,
        control_losses: int,
        treatment_wins: int,
        treatment_losses: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100000,
    ) -> Dict[str, Any]:
        """
        Bayesian A/B test using Beta-Binomial model.

        Calculates the probability that treatment is better than control.

        Args:
            control_wins: Wins in control group
            control_losses: Losses in control group
            treatment_wins: Wins in treatment group
            treatment_losses: Losses in treatment group
            prior_alpha: Prior alpha for Beta distribution
            prior_beta: Prior beta for Beta distribution
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with Bayesian analysis results
        """
        # Posterior parameters
        control_alpha = prior_alpha + control_wins
        control_beta = prior_beta + control_losses
        treatment_alpha = prior_alpha + treatment_wins
        treatment_beta = prior_beta + treatment_losses

        # Sample from posteriors
        control_samples = np.random.beta(control_alpha, control_beta, n_samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)

        # Probability treatment > control
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        # Expected lift
        expected_lift = np.mean((treatment_samples - control_samples) / control_samples)

        # Credible intervals
        diff_samples = treatment_samples - control_samples
        ci_low = np.percentile(diff_samples, 2.5)
        ci_high = np.percentile(diff_samples, 97.5)

        return {
            "probability_treatment_better": prob_treatment_better,
            "probability_control_better": 1 - prob_treatment_better,
            "expected_lift": expected_lift,
            "credible_interval_95": (ci_low, ci_high),
            "control_posterior_mean": control_alpha / (control_alpha + control_beta),
            "treatment_posterior_mean": treatment_alpha / (treatment_alpha + treatment_beta),
            "risk_of_choosing_treatment": 1 - prob_treatment_better,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def compare_models(
    model_a_metrics: Dict[str, Any],
    model_b_metrics: Dict[str, Any],
    confidence_level: float = 0.95,
) -> Dict[str, ABTestResult]:
    """
    Compare two models across multiple metrics.

    Args:
        model_a_metrics: Metrics dict for model A (control)
            Expected keys: win_rate, wins, losses, returns, sharpe_ratio
        model_b_metrics: Metrics dict for model B (treatment)
        confidence_level: Confidence level for tests

    Returns:
        Dictionary of metric name -> ABTestResult
    """
    ab = ABStatistics(confidence_level=confidence_level)
    results = {}

    # Compare win rates if available
    if all(k in model_a_metrics for k in ["wins", "losses"]):
        if all(k in model_b_metrics for k in ["wins", "losses"]):
            results["win_rate"] = ab.compare_win_rates(
                control_wins=model_a_metrics["wins"],
                control_losses=model_a_metrics["losses"],
                treatment_wins=model_b_metrics["wins"],
                treatment_losses=model_b_metrics["losses"],
            )

    # Compare Sharpe ratios if returns available
    if "returns" in model_a_metrics and "returns" in model_b_metrics:
        results["sharpe_ratio"] = ab.compare_sharpe_ratios(
            control_returns=np.array(model_a_metrics["returns"]),
            treatment_returns=np.array(model_b_metrics["returns"]),
        )

    return results


def get_minimum_test_duration(
    trades_per_day: int,
    baseline_win_rate: float,
    minimum_detectable_effect: float = 0.05,
    power: float = 0.80,
    confidence_level: float = 0.95,
) -> int:
    """
    Calculate minimum test duration in days.

    Args:
        trades_per_day: Expected number of trades per day
        baseline_win_rate: Expected baseline win rate
        minimum_detectable_effect: Minimum effect size to detect
        power: Desired statistical power
        confidence_level: Confidence level

    Returns:
        Minimum number of days to run the test
    """
    ab = ABStatistics(confidence_level=confidence_level)
    min_sample = ab.calculate_minimum_sample_size(
        baseline_rate=baseline_win_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        power=power,
    )

    # Need samples for both groups
    total_samples_needed = 2 * min_sample

    days_needed = int(np.ceil(total_samples_needed / trades_per_day))
    return days_needed


__all__ = [
    "SignificanceLevel",
    "ABTestResult",
    "ABStatistics",
    "compare_models",
    "get_minimum_test_duration",
]
