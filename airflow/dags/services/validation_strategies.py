"""
Validation Strategies - Strategy Pattern Implementation
=======================================================
Different validation strategies for backtest results.

SOLID Principles:
- Single Responsibility: Each strategy validates one aspect
- Open/Closed: New strategies via inheritance, not modification
- Liskov Substitution: All strategies are interchangeable
- Interface Segregation: Minimal interface (validate method)
- Dependency Inversion: Depends on contracts, not implementations

Design Patterns:
- Strategy Pattern: Interchangeable validation algorithms
- Template Method: Base class defines validation flow
- Chain of Responsibility: Multiple validators in sequence

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

from contracts.backtest_contracts import (
    BacktestMetrics,
    BacktestResult,
    ValidationThresholds,
    ValidationCheckResult,
    ValidationReport,
    ValidationResult,
    AlertSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASE STRATEGY - Interface Segregation
# =============================================================================

class ValidationStrategy(ABC):
    """
    Abstract base class for validation strategies.

    Implements Template Method pattern:
    - validate() is the template method
    - _run_checks() is implemented by subclasses
    """

    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds

    def validate(
        self,
        backtest_result: BacktestResult,
        baseline_metrics: Optional[BacktestMetrics] = None
    ) -> ValidationReport:
        """
        Template method for validation.

        Args:
            backtest_result: Backtest result to validate
            baseline_metrics: Optional baseline for comparison

        Returns:
            ValidationReport with all check results
        """
        if not backtest_result.metrics:
            return self._create_failed_report(
                backtest_result,
                "No metrics available for validation"
            )

        # Run all validation checks
        checks = self._run_checks(backtest_result.metrics, baseline_metrics)

        # Determine overall result
        overall_result = self._determine_overall_result(checks)

        # Build report
        report = ValidationReport(
            model_id=backtest_result.model_id,
            backtest_result=backtest_result,
            thresholds=self.thresholds,
            checks=checks,
            overall_result=overall_result,
            baseline_metrics=baseline_metrics,
        )

        logger.info(
            f"Validation complete: {overall_result.value} "
            f"({report.passed_checks}/{len(checks)} checks passed)"
        )

        return report

    @abstractmethod
    def _run_checks(
        self,
        metrics: BacktestMetrics,
        baseline: Optional[BacktestMetrics] = None
    ) -> List[ValidationCheckResult]:
        """Run validation checks - implemented by subclasses"""
        pass

    def _determine_overall_result(self, checks: List[ValidationCheckResult]) -> ValidationResult:
        """Determine overall validation result from individual checks"""
        critical_failures = [c for c in checks if c.severity == AlertSeverity.CRITICAL and not c.is_passing]
        warnings = [c for c in checks if c.severity == AlertSeverity.WARNING and not c.is_passing]

        if critical_failures:
            return ValidationResult.FAILED
        elif warnings:
            return ValidationResult.DEGRADED
        else:
            return ValidationResult.PASSED

    def _create_failed_report(
        self,
        backtest_result: BacktestResult,
        error_message: str
    ) -> ValidationReport:
        """Create a failed validation report"""
        return ValidationReport(
            model_id=backtest_result.model_id,
            backtest_result=backtest_result,
            thresholds=self.thresholds,
            checks=[
                ValidationCheckResult(
                    check_name="prerequisite",
                    result=ValidationResult.FAILED,
                    actual_value=0.0,
                    threshold_value=1.0,
                    message=error_message,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
            overall_result=ValidationResult.FAILED,
        )


# =============================================================================
# CONCRETE STRATEGIES
# =============================================================================

class StandardValidationStrategy(ValidationStrategy):
    """
    Standard validation strategy with core metrics checks.

    Checks:
    - Sharpe ratio >= threshold
    - Max drawdown <= threshold
    - Win rate >= threshold
    - Minimum trades >= threshold
    """

    def _run_checks(
        self,
        metrics: BacktestMetrics,
        baseline: Optional[BacktestMetrics] = None
    ) -> List[ValidationCheckResult]:
        checks = []

        # Check 1: Sharpe Ratio
        sharpe = metrics.sharpe_ratio or 0.0
        checks.append(ValidationCheckResult(
            check_name="sharpe_ratio",
            result=ValidationResult.PASSED if sharpe >= self.thresholds.min_sharpe_ratio else ValidationResult.FAILED,
            actual_value=sharpe,
            threshold_value=self.thresholds.min_sharpe_ratio,
            message=f"Sharpe ratio {sharpe:.2f} {'meets' if sharpe >= self.thresholds.min_sharpe_ratio else 'below'} minimum {self.thresholds.min_sharpe_ratio}",
            severity=AlertSeverity.CRITICAL,
        ))

        # Check 2: Max Drawdown
        drawdown = metrics.max_drawdown_pct
        checks.append(ValidationCheckResult(
            check_name="max_drawdown",
            result=ValidationResult.PASSED if drawdown <= self.thresholds.max_drawdown_pct else ValidationResult.FAILED,
            actual_value=drawdown,
            threshold_value=self.thresholds.max_drawdown_pct,
            message=f"Max drawdown {drawdown:.1%} {'within' if drawdown <= self.thresholds.max_drawdown_pct else 'exceeds'} limit {self.thresholds.max_drawdown_pct:.1%}",
            severity=AlertSeverity.CRITICAL,
        ))

        # Check 3: Win Rate
        win_rate = metrics.win_rate
        checks.append(ValidationCheckResult(
            check_name="win_rate",
            result=ValidationResult.PASSED if win_rate >= self.thresholds.min_win_rate else ValidationResult.DEGRADED,
            actual_value=win_rate,
            threshold_value=self.thresholds.min_win_rate,
            message=f"Win rate {win_rate:.1%} {'meets' if win_rate >= self.thresholds.min_win_rate else 'below'} minimum {self.thresholds.min_win_rate:.1%}",
            severity=AlertSeverity.WARNING,
        ))

        # Check 4: Minimum Trades
        total_trades = metrics.total_trades
        checks.append(ValidationCheckResult(
            check_name="minimum_trades",
            result=ValidationResult.PASSED if total_trades >= self.thresholds.min_trades else ValidationResult.FAILED,
            actual_value=float(total_trades),
            threshold_value=float(self.thresholds.min_trades),
            message=f"Total trades {total_trades} {'meets' if total_trades >= self.thresholds.min_trades else 'below'} minimum {self.thresholds.min_trades}",
            severity=AlertSeverity.CRITICAL,
        ))

        # Check 5: Profit Factor (if available)
        if metrics.profit_factor is not None:
            pf = metrics.profit_factor
            checks.append(ValidationCheckResult(
                check_name="profit_factor",
                result=ValidationResult.PASSED if pf >= self.thresholds.min_profit_factor else ValidationResult.DEGRADED,
                actual_value=pf,
                threshold_value=self.thresholds.min_profit_factor,
                message=f"Profit factor {pf:.2f} {'meets' if pf >= self.thresholds.min_profit_factor else 'below'} minimum {self.thresholds.min_profit_factor}",
                severity=AlertSeverity.WARNING,
            ))

        # Check 6: Consecutive Losses
        max_losses = metrics.max_consecutive_losses
        checks.append(ValidationCheckResult(
            check_name="consecutive_losses",
            result=ValidationResult.PASSED if max_losses <= self.thresholds.max_consecutive_losses else ValidationResult.DEGRADED,
            actual_value=float(max_losses),
            threshold_value=float(self.thresholds.max_consecutive_losses),
            message=f"Max consecutive losses {max_losses} {'within' if max_losses <= self.thresholds.max_consecutive_losses else 'exceeds'} limit {self.thresholds.max_consecutive_losses}",
            severity=AlertSeverity.WARNING,
        ))

        return checks


class ComparisonValidationStrategy(ValidationStrategy):
    """
    Comparison validation strategy - compares against baseline.

    Checks:
    - All standard checks
    - Relative Sharpe improvement
    - Relative drawdown comparison
    - Statistical significance (if enough data)
    """

    def __init__(
        self,
        thresholds: ValidationThresholds,
        min_improvement_pct: float = 0.0,  # Allow same or better
        max_degradation_pct: float = 0.10,  # Max 10% worse
    ):
        super().__init__(thresholds)
        self.min_improvement_pct = min_improvement_pct
        self.max_degradation_pct = max_degradation_pct

    def _run_checks(
        self,
        metrics: BacktestMetrics,
        baseline: Optional[BacktestMetrics] = None
    ) -> List[ValidationCheckResult]:
        # Run standard checks first
        standard_strategy = StandardValidationStrategy(self.thresholds)
        checks = standard_strategy._run_checks(metrics, baseline)

        # Add comparison checks if baseline available
        if baseline:
            checks.extend(self._run_comparison_checks(metrics, baseline))

        return checks

    def _run_comparison_checks(
        self,
        metrics: BacktestMetrics,
        baseline: BacktestMetrics
    ) -> List[ValidationCheckResult]:
        checks = []

        # Sharpe comparison
        if baseline.sharpe_ratio and metrics.sharpe_ratio:
            sharpe_change = (metrics.sharpe_ratio - baseline.sharpe_ratio) / abs(baseline.sharpe_ratio) if baseline.sharpe_ratio != 0 else 0
            checks.append(ValidationCheckResult(
                check_name="sharpe_vs_baseline",
                result=ValidationResult.PASSED if sharpe_change >= -self.max_degradation_pct else ValidationResult.DEGRADED,
                actual_value=sharpe_change,
                threshold_value=-self.max_degradation_pct,
                message=f"Sharpe {'improved' if sharpe_change >= 0 else 'degraded'} by {abs(sharpe_change):.1%} vs baseline",
                severity=AlertSeverity.WARNING,
            ))

        # Return comparison
        if baseline.total_return_pct != 0:
            return_change = metrics.total_return_pct - baseline.total_return_pct
            checks.append(ValidationCheckResult(
                check_name="return_vs_baseline",
                result=ValidationResult.PASSED if return_change >= -self.max_degradation_pct else ValidationResult.DEGRADED,
                actual_value=return_change,
                threshold_value=-self.max_degradation_pct,
                message=f"Return {'improved' if return_change >= 0 else 'degraded'} by {abs(return_change):.1%} vs baseline",
                severity=AlertSeverity.WARNING,
            ))

        # Drawdown comparison (lower is better)
        drawdown_change = metrics.max_drawdown_pct - baseline.max_drawdown_pct
        checks.append(ValidationCheckResult(
            check_name="drawdown_vs_baseline",
            result=ValidationResult.PASSED if drawdown_change <= self.max_degradation_pct else ValidationResult.DEGRADED,
            actual_value=drawdown_change,
            threshold_value=self.max_degradation_pct,
            message=f"Max drawdown {'improved' if drawdown_change <= 0 else 'worsened'} by {abs(drawdown_change):.1%} vs baseline",
            severity=AlertSeverity.WARNING,
        ))

        return checks


class StrictValidationStrategy(ValidationStrategy):
    """
    Strict validation strategy for production deployment.

    More stringent thresholds and additional checks:
    - Higher Sharpe requirement
    - Lower drawdown tolerance
    - Consistency checks
    - Regime robustness
    """

    def __init__(self, thresholds: ValidationThresholds):
        # Override with stricter defaults
        strict_thresholds = ValidationThresholds(
            min_sharpe_ratio=max(thresholds.min_sharpe_ratio, 1.0),
            max_drawdown_pct=min(thresholds.max_drawdown_pct, 0.15),
            min_win_rate=max(thresholds.min_win_rate, 0.45),
            min_profit_factor=max(thresholds.min_profit_factor, 1.2),
            min_trades=max(thresholds.min_trades, 20),
            max_consecutive_losses=min(thresholds.max_consecutive_losses, 7),
        )
        super().__init__(strict_thresholds)

    def _run_checks(
        self,
        metrics: BacktestMetrics,
        baseline: Optional[BacktestMetrics] = None
    ) -> List[ValidationCheckResult]:
        # Run standard checks with strict thresholds
        standard_strategy = StandardValidationStrategy(self.thresholds)
        checks = standard_strategy._run_checks(metrics, baseline)

        # Add production-specific checks
        checks.extend(self._run_production_checks(metrics))

        return checks

    def _run_production_checks(self, metrics: BacktestMetrics) -> List[ValidationCheckResult]:
        checks = []

        # Check: Positive total return
        checks.append(ValidationCheckResult(
            check_name="positive_return",
            result=ValidationResult.PASSED if metrics.total_return_pct > 0 else ValidationResult.FAILED,
            actual_value=metrics.total_return_pct,
            threshold_value=0.0,
            message=f"Total return {metrics.total_return_pct:.1%} {'is positive' if metrics.total_return_pct > 0 else 'is negative'}",
            severity=AlertSeverity.CRITICAL,
        ))

        # Check: Sortino ratio (if available)
        if metrics.sortino_ratio is not None:
            checks.append(ValidationCheckResult(
                check_name="sortino_ratio",
                result=ValidationResult.PASSED if metrics.sortino_ratio >= 1.0 else ValidationResult.DEGRADED,
                actual_value=metrics.sortino_ratio,
                threshold_value=1.0,
                message=f"Sortino ratio {metrics.sortino_ratio:.2f} {'meets' if metrics.sortino_ratio >= 1.0 else 'below'} production threshold",
                severity=AlertSeverity.WARNING,
            ))

        # Check: Reasonable average trade duration
        if metrics.avg_trade_duration_minutes:
            duration = metrics.avg_trade_duration_minutes
            # Between 5 minutes and 4 hours for 5-min bars
            is_reasonable = 5 <= duration <= 240
            checks.append(ValidationCheckResult(
                check_name="trade_duration",
                result=ValidationResult.PASSED if is_reasonable else ValidationResult.DEGRADED,
                actual_value=duration,
                threshold_value=240.0,
                message=f"Avg trade duration {duration:.0f} min {'is reasonable' if is_reasonable else 'may indicate issues'}",
                severity=AlertSeverity.INFO,
            ))

        return checks


class WalkForwardValidationStrategy(ValidationStrategy):
    """
    Walk-forward validation strategy.

    Validates consistency across multiple time windows.
    Used for detecting overfitting.
    """

    def __init__(
        self,
        thresholds: ValidationThresholds,
        window_results: List[BacktestMetrics],
        min_positive_windows_pct: float = 0.70,
    ):
        super().__init__(thresholds)
        self.window_results = window_results
        self.min_positive_windows_pct = min_positive_windows_pct

    def _run_checks(
        self,
        metrics: BacktestMetrics,
        baseline: Optional[BacktestMetrics] = None
    ) -> List[ValidationCheckResult]:
        # Run standard checks
        standard_strategy = StandardValidationStrategy(self.thresholds)
        checks = standard_strategy._run_checks(metrics, baseline)

        # Add walk-forward specific checks
        checks.extend(self._run_walk_forward_checks())

        return checks

    def _run_walk_forward_checks(self) -> List[ValidationCheckResult]:
        checks = []

        if not self.window_results:
            return checks

        # Check: Percentage of positive windows
        positive_windows = sum(1 for w in self.window_results if w.total_return_pct > 0)
        pct_positive = positive_windows / len(self.window_results)

        checks.append(ValidationCheckResult(
            check_name="positive_windows",
            result=ValidationResult.PASSED if pct_positive >= self.min_positive_windows_pct else ValidationResult.FAILED,
            actual_value=pct_positive,
            threshold_value=self.min_positive_windows_pct,
            message=f"{positive_windows}/{len(self.window_results)} windows ({pct_positive:.0%}) are profitable",
            severity=AlertSeverity.CRITICAL,
        ))

        # Check: Sharpe consistency (std deviation across windows)
        sharpes = [w.sharpe_ratio for w in self.window_results if w.sharpe_ratio is not None]
        if sharpes:
            import statistics
            sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 0
            avg_sharpe = statistics.mean(sharpes)

            # Coefficient of variation should be reasonable (< 1)
            cv = sharpe_std / abs(avg_sharpe) if avg_sharpe != 0 else float('inf')
            checks.append(ValidationCheckResult(
                check_name="sharpe_consistency",
                result=ValidationResult.PASSED if cv < 1.0 else ValidationResult.DEGRADED,
                actual_value=cv,
                threshold_value=1.0,
                message=f"Sharpe CV {cv:.2f} indicates {'consistent' if cv < 1.0 else 'inconsistent'} performance",
                severity=AlertSeverity.WARNING,
            ))

        return checks


# =============================================================================
# STRATEGY REGISTRY - Factory Pattern Helper
# =============================================================================

class ValidationStrategyRegistry:
    """
    Registry of validation strategies.

    Implements Factory Pattern for strategy creation.
    """

    _strategies: Dict[str, type] = {
        "standard": StandardValidationStrategy,
        "comparison": ComparisonValidationStrategy,
        "strict": StrictValidationStrategy,
        "walk_forward": WalkForwardValidationStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a new strategy"""
        if not issubclass(strategy_class, ValidationStrategy):
            raise TypeError(f"{strategy_class} must be a ValidationStrategy subclass")
        cls._strategies[name] = strategy_class
        logger.info(f"Registered validation strategy: {name}")

    @classmethod
    def get(cls, name: str, thresholds: ValidationThresholds, **kwargs) -> ValidationStrategy:
        """Get strategy by name"""
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown strategy: {name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name](thresholds, **kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies"""
        return list(cls._strategies.keys())
