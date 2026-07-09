#!/usr/bin/env python3
"""
Experiment Comparison CLI Tool
==============================

Compare two or more MLflow experiments/runs with comprehensive metrics,
config diffs, and statistical significance testing.

Part of COMP-01 to COMP-09 remediation from Experimentation Audit.

Features:
- Side-by-side config diff
- Metrics comparison table
- Statistical significance tests (chi-square, t-test, bootstrap)
- Relative improvement/degradation percentages
- Export to markdown/JSON

Usage:
    # Compare two runs by ID
    python scripts/compare_experiments.py --run-a abc123 --run-b def456

    # Compare by experiment names
    python scripts/compare_experiments.py --exp-a baseline --exp-b new_features

    # Export to markdown
    python scripts/compare_experiments.py --run-a abc123 --run-b def456 --output report.md

    # Statistical significance test
    python scripts/compare_experiments.py --run-a abc123 --run-b def456 --stat-test

Author: Trading Team
Date: 2026-01-17
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: numpy and pandas are required. Install with: pip install numpy pandas")
    sys.exit(1)

# Import A/B statistics module
try:
    from src.inference.ab_statistics import (
        ABStatistics,
        ABTestResult,
        compare_models,
        get_minimum_test_duration,
    )
    AB_STATS_AVAILABLE = True
except ImportError:
    AB_STATS_AVAILABLE = False
    print("Warning: ab_statistics module not available. Statistical tests disabled.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentRun:
    """Container for MLflow run data."""
    run_id: str
    run_name: str
    experiment_id: str
    experiment_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]

    @classmethod
    def from_mlflow_run(cls, run: mlflow.entities.Run, client: MlflowClient) -> "ExperimentRun":
        """Create from MLflow Run object."""
        experiment = client.get_experiment(run.info.experiment_id)

        # Get artifacts list
        try:
            artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
        except Exception:
            artifacts = []

        return cls(
            run_id=run.info.run_id,
            run_name=run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            experiment_id=run.info.experiment_id,
            experiment_name=experiment.name,
            status=run.info.status,
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            tags=dict(run.data.tags),
            artifacts=artifacts,
        )


@dataclass
class ConfigDiff:
    """Configuration difference between two experiments."""
    param_name: str
    value_a: Any
    value_b: Any
    is_different: bool
    category: str  # 'hyperparameter', 'environment', 'data', 'other'


@dataclass
class MetricComparison:
    """Comparison of a single metric between experiments."""
    metric_name: str
    value_a: float
    value_b: float
    absolute_diff: float
    relative_diff_pct: float
    is_improvement: bool  # Based on metric type (higher/lower is better)
    significance: Optional[str] = None


@dataclass
class ComparisonReport:
    """Complete comparison report between two experiments."""
    run_a: ExperimentRun
    run_b: ExperimentRun
    config_diffs: List[ConfigDiff]
    metric_comparisons: List[MetricComparison]
    statistical_tests: Dict[str, ABTestResult] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate summary text."""
        lines = []
        lines.append(f"Comparing: {self.run_a.run_name} vs {self.run_b.run_name}")
        lines.append(f"Config differences: {sum(1 for d in self.config_diffs if d.is_different)}")

        improvements = sum(1 for m in self.metric_comparisons if m.is_improvement)
        degradations = len(self.metric_comparisons) - improvements
        lines.append(f"Metric improvements: {improvements}, degradations: {degradations}")

        return "\n".join(lines)


# =============================================================================
# Comparison Logic
# =============================================================================

class ExperimentComparator:
    """Compare MLflow experiments and runs."""

    # Metrics where higher is better
    HIGHER_IS_BETTER = {
        "sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor",
        "total_pnl", "total_trades", "eval_sharpe_ratio", "eval_win_rate",
        "episode_reward_mean", "eval_total_pnl",
    }

    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "max_drawdown", "eval_max_drawdown", "loss", "training_loss",
    }

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the comparator.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def get_run(self, run_id: str) -> ExperimentRun:
        """
        Get experiment run by ID.

        Args:
            run_id: MLflow run ID

        Returns:
            ExperimentRun object
        """
        run = self.client.get_run(run_id)
        return ExperimentRun.from_mlflow_run(run, self.client)

    def get_latest_run(self, experiment_name: str) -> ExperimentRun:
        """
        Get the latest run from an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            ExperimentRun object for the latest run
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_name}")

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment: {experiment_name}")

        return ExperimentRun.from_mlflow_run(runs[0], self.client)

    def compare_configs(
        self,
        run_a: ExperimentRun,
        run_b: ExperimentRun
    ) -> List[ConfigDiff]:
        """
        Compare configurations between two runs.

        Args:
            run_a: First experiment run
            run_b: Second experiment run

        Returns:
            List of ConfigDiff objects
        """
        diffs = []

        # Get all unique parameter names
        all_params = set(run_a.params.keys()) | set(run_b.params.keys())

        for param in sorted(all_params):
            value_a = run_a.params.get(param)
            value_b = run_b.params.get(param)

            # Determine category
            if param.startswith("hp_"):
                category = "hyperparameter"
            elif param.startswith("env_"):
                category = "environment"
            elif param.startswith("data_") or param.startswith("dataset"):
                category = "data"
            else:
                category = "other"

            diffs.append(ConfigDiff(
                param_name=param,
                value_a=value_a,
                value_b=value_b,
                is_different=value_a != value_b,
                category=category,
            ))

        return diffs

    def compare_metrics(
        self,
        run_a: ExperimentRun,
        run_b: ExperimentRun
    ) -> List[MetricComparison]:
        """
        Compare metrics between two runs.

        Args:
            run_a: First experiment run (control/baseline)
            run_b: Second experiment run (treatment/new)

        Returns:
            List of MetricComparison objects
        """
        comparisons = []

        # Get all unique metric names
        all_metrics = set(run_a.metrics.keys()) | set(run_b.metrics.keys())

        for metric in sorted(all_metrics):
            value_a = run_a.metrics.get(metric, 0.0)
            value_b = run_b.metrics.get(metric, 0.0)

            absolute_diff = value_b - value_a
            relative_diff_pct = (absolute_diff / abs(value_a) * 100) if value_a != 0 else 0.0

            # Determine if this is an improvement
            if metric in self.HIGHER_IS_BETTER:
                is_improvement = value_b > value_a
            elif metric in self.LOWER_IS_BETTER:
                is_improvement = value_b < value_a
            else:
                # Default: higher is better
                is_improvement = value_b > value_a

            comparisons.append(MetricComparison(
                metric_name=metric,
                value_a=value_a,
                value_b=value_b,
                absolute_diff=absolute_diff,
                relative_diff_pct=relative_diff_pct,
                is_improvement=is_improvement,
            ))

        return comparisons

    def run_statistical_tests(
        self,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
    ) -> Dict[str, ABTestResult]:
        """
        Run statistical significance tests between two runs.

        Args:
            run_a: Control run
            run_b: Treatment run

        Returns:
            Dictionary of test name -> ABTestResult
        """
        if not AB_STATS_AVAILABLE:
            return {}

        results = {}
        ab = ABStatistics(confidence_level=0.95)

        # Compare win rates if available
        if all(k in run_a.metrics for k in ["eval_win_rate"]):
            if all(k in run_b.metrics for k in ["eval_win_rate"]):
                # Estimate wins/losses from win rate and total trades
                a_trades = int(run_a.metrics.get("eval_total_trades", 100))
                b_trades = int(run_b.metrics.get("eval_total_trades", 100))
                a_win_rate = run_a.metrics.get("eval_win_rate", 0.5)
                b_win_rate = run_b.metrics.get("eval_win_rate", 0.5)

                a_wins = int(a_trades * a_win_rate)
                a_losses = a_trades - a_wins
                b_wins = int(b_trades * b_win_rate)
                b_losses = b_trades - b_wins

                try:
                    results["win_rate"] = ab.compare_win_rates(
                        control_wins=a_wins,
                        control_losses=a_losses,
                        treatment_wins=b_wins,
                        treatment_losses=b_losses,
                    )
                except Exception as e:
                    print(f"Warning: Win rate comparison failed: {e}")

        return results

    def compare(
        self,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
        run_stat_tests: bool = True,
    ) -> ComparisonReport:
        """
        Generate complete comparison report.

        Args:
            run_a: First run (control/baseline)
            run_b: Second run (treatment/new)
            run_stat_tests: Whether to run statistical tests

        Returns:
            ComparisonReport object
        """
        config_diffs = self.compare_configs(run_a, run_b)
        metric_comparisons = self.compare_metrics(run_a, run_b)

        stat_tests = {}
        if run_stat_tests and AB_STATS_AVAILABLE:
            stat_tests = self.run_statistical_tests(run_a, run_b)

        return ComparisonReport(
            run_a=run_a,
            run_b=run_b,
            config_diffs=config_diffs,
            metric_comparisons=metric_comparisons,
            statistical_tests=stat_tests,
        )


# =============================================================================
# Output Formatters
# =============================================================================

def format_config_diff_table(diffs: List[ConfigDiff], show_all: bool = False) -> str:
    """Format config diffs as a markdown table."""
    lines = []
    lines.append("## Configuration Differences\n")
    lines.append("| Parameter | Run A | Run B | Changed |")
    lines.append("|-----------|-------|-------|---------|")

    for diff in diffs:
        if show_all or diff.is_different:
            changed = "Yes" if diff.is_different else "-"
            lines.append(f"| {diff.param_name} | {diff.value_a} | {diff.value_b} | {changed} |")

    return "\n".join(lines)


def format_metric_comparison_table(comparisons: List[MetricComparison]) -> str:
    """Format metric comparisons as a markdown table."""
    lines = []
    lines.append("## Metrics Comparison\n")
    lines.append("| Metric | Run A | Run B | Diff | % Change | Better? |")
    lines.append("|--------|-------|-------|------|----------|---------|")

    for comp in comparisons:
        better = "Yes" if comp.is_improvement else "No"
        sign = "+" if comp.relative_diff_pct >= 0 else ""
        lines.append(
            f"| {comp.metric_name} | {comp.value_a:.4f} | {comp.value_b:.4f} | "
            f"{comp.absolute_diff:+.4f} | {sign}{comp.relative_diff_pct:.2f}% | {better} |"
        )

    return "\n".join(lines)


def format_statistical_tests(tests: Dict[str, ABTestResult]) -> str:
    """Format statistical test results."""
    if not tests:
        return "## Statistical Tests\n\nNo statistical tests performed.\n"

    lines = []
    lines.append("## Statistical Significance Tests\n")

    for name, result in tests.items():
        lines.append(f"### {name.replace('_', ' ').title()}\n")
        lines.append(f"- **Test**: {result.test_name}")
        lines.append(f"- **Control value**: {result.control_value:.4f}")
        lines.append(f"- **Treatment value**: {result.treatment_value:.4f}")
        lines.append(f"- **Difference**: {result.difference:+.4f} ({result.relative_difference*100:+.2f}%)")
        lines.append(f"- **P-value**: {result.p_value:.6f}")
        lines.append(f"- **Significant**: {'Yes' if result.is_significant else 'No'} ({result.significance_level})")
        lines.append(f"- **95% CI**: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        if result.effect_size:
            lines.append(f"- **Effect size**: {result.effect_size:.4f} ({result.effect_size_interpretation})")
        lines.append("")

    return "\n".join(lines)


def format_markdown_report(report: ComparisonReport, show_all_params: bool = False) -> str:
    """Generate full markdown report."""
    lines = []

    # Header
    lines.append("# Experiment Comparison Report\n")
    lines.append(f"**Generated**: {report.generated_at.isoformat()}\n")

    # Run info
    lines.append("## Runs Compared\n")
    lines.append(f"| | Run A (Control) | Run B (Treatment) |")
    lines.append("|--|-----------------|-------------------|")
    lines.append(f"| **Run ID** | `{report.run_a.run_id[:8]}...` | `{report.run_b.run_id[:8]}...` |")
    lines.append(f"| **Run Name** | {report.run_a.run_name} | {report.run_b.run_name} |")
    lines.append(f"| **Experiment** | {report.run_a.experiment_name} | {report.run_b.experiment_name} |")
    lines.append(f"| **Status** | {report.run_a.status} | {report.run_b.status} |")
    lines.append(f"| **Started** | {report.run_a.start_time} | {report.run_b.start_time} |")
    lines.append("")

    # Config diffs
    lines.append(format_config_diff_table(report.config_diffs, show_all_params))
    lines.append("")

    # Metrics comparison
    lines.append(format_metric_comparison_table(report.metric_comparisons))
    lines.append("")

    # Statistical tests
    lines.append(format_statistical_tests(report.statistical_tests))

    # Summary
    lines.append("## Summary\n")
    improvements = sum(1 for m in report.metric_comparisons if m.is_improvement)
    total = len(report.metric_comparisons)
    lines.append(f"- **Config changes**: {sum(1 for d in report.config_diffs if d.is_different)}")
    lines.append(f"- **Metric improvements**: {improvements}/{total}")

    # Key metrics summary
    for comp in report.metric_comparisons:
        if comp.metric_name in ["eval_sharpe_ratio", "sharpe_ratio"]:
            sign = "+" if comp.relative_diff_pct >= 0 else ""
            status = "IMPROVED" if comp.is_improvement else "DEGRADED"
            lines.append(f"- **Sharpe Ratio**: {comp.value_a:.2f} -> {comp.value_b:.2f} ({sign}{comp.relative_diff_pct:.1f}%) - {status}")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by compare_experiments.py*")

    return "\n".join(lines)


def format_json_report(report: ComparisonReport) -> str:
    """Generate JSON report."""
    data = {
        "generated_at": report.generated_at.isoformat(),
        "run_a": {
            "run_id": report.run_a.run_id,
            "run_name": report.run_a.run_name,
            "experiment_name": report.run_a.experiment_name,
            "params": report.run_a.params,
            "metrics": report.run_a.metrics,
        },
        "run_b": {
            "run_id": report.run_b.run_id,
            "run_name": report.run_b.run_name,
            "experiment_name": report.run_b.experiment_name,
            "params": report.run_b.params,
            "metrics": report.run_b.metrics,
        },
        "config_diffs": [
            {
                "param": d.param_name,
                "value_a": d.value_a,
                "value_b": d.value_b,
                "is_different": d.is_different,
                "category": d.category,
            }
            for d in report.config_diffs
        ],
        "metric_comparisons": [
            {
                "metric": m.metric_name,
                "value_a": m.value_a,
                "value_b": m.value_b,
                "absolute_diff": m.absolute_diff,
                "relative_diff_pct": m.relative_diff_pct,
                "is_improvement": m.is_improvement,
            }
            for m in report.metric_comparisons
        ],
        "statistical_tests": {
            name: result.to_dict()
            for name, result in report.statistical_tests.items()
        },
    }
    return json.dumps(data, indent=2)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare MLflow experiments and runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs by ID
  python compare_experiments.py --run-a abc123 --run-b def456

  # Compare latest runs from two experiments
  python compare_experiments.py --exp-a baseline --exp-b new_features

  # Export to markdown file
  python compare_experiments.py --run-a abc123 --run-b def456 -o report.md

  # JSON output
  python compare_experiments.py --run-a abc123 --run-b def456 --format json
        """
    )

    # Run selection
    run_group = parser.add_argument_group("Run Selection")
    run_group.add_argument(
        "--run-a", "--run_a",
        type=str,
        help="MLflow run ID for control/baseline (Run A)"
    )
    run_group.add_argument(
        "--run-b", "--run_b",
        type=str,
        help="MLflow run ID for treatment/new (Run B)"
    )
    run_group.add_argument(
        "--exp-a", "--exp_a",
        type=str,
        help="Experiment name for control (uses latest run)"
    )
    run_group.add_argument(
        "--exp-b", "--exp_b",
        type=str,
        help="Experiment name for treatment (uses latest run)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    output_group.add_argument(
        "--format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    output_group.add_argument(
        "--show-all-params",
        action="store_true",
        help="Show all parameters, not just differences"
    )

    # Statistical options
    stat_group = parser.add_argument_group("Statistical Options")
    stat_group.add_argument(
        "--stat-test", "--stat_test",
        action="store_true",
        default=True,
        help="Run statistical significance tests (default: True)"
    )
    stat_group.add_argument(
        "--no-stat-test",
        action="store_true",
        help="Skip statistical significance tests"
    )
    stat_group.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for tests (default: 0.95)"
    )

    # MLflow options
    mlflow_group = parser.add_argument_group("MLflow Options")
    mlflow_group.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if not (args.run_a or args.exp_a):
        print("Error: Must specify either --run-a or --exp-a")
        return 1

    if not (args.run_b or args.exp_b):
        print("Error: Must specify either --run-b or --exp-b")
        return 1

    try:
        comparator = ExperimentComparator(tracking_uri=args.tracking_uri)

        # Get runs
        if args.run_a:
            run_a = comparator.get_run(args.run_a)
        else:
            run_a = comparator.get_latest_run(args.exp_a)

        if args.run_b:
            run_b = comparator.get_run(args.run_b)
        else:
            run_b = comparator.get_latest_run(args.exp_b)

        print(f"Comparing: {run_a.run_name} (control) vs {run_b.run_name} (treatment)")

        # Run comparison
        run_stat_tests = args.stat_test and not args.no_stat_test
        report = comparator.compare(run_a, run_b, run_stat_tests=run_stat_tests)

        # Format output
        if args.format == "markdown":
            output = format_markdown_report(report, show_all_params=args.show_all_params)
        elif args.format == "json":
            output = format_json_report(report)
        else:
            output = report.summary()

        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Report saved to: {output_path}")
        else:
            print("\n" + output)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
