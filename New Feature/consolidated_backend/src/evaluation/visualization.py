# backend/src/evaluation/visualization.py
"""
Visualization and Reporting for ML Training Results.

Generates:
1. DA (Direction Accuracy) tables by model/horizon
2. Variance ratio heatmaps
3. Overfitting gap analysis
4. Equity curves
5. DA by window plots
6. Training reports

Usage:
    from src.evaluation.visualization import TrainingReporter

    reporter = TrainingReporter(output_dir='reports/')
    reporter.generate_da_table(results)
    reporter.plot_equity_curves(backtest_results)
    reporter.create_full_report(training_results, backtest_results)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    model_name: str
    horizon: int
    da_train: float
    da_val: float
    da_gap: float
    variance_ratio: float
    mse_val: float
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: Path
    include_plots: bool = True
    save_csv: bool = True
    save_json: bool = True
    figure_format: str = 'png'
    dpi: int = 150


class TrainingReporter:
    """
    Generates training reports and visualizations.

    Attributes:
        config: ReportConfig with output settings
        results_cache: Cache of processed results
    """

    def __init__(
        self,
        output_dir: str = 'reports',
        include_plots: bool = True
    ):
        """
        Initialize reporter.

        Args:
            output_dir: Directory for output files
            include_plots: Whether to generate matplotlib plots
        """
        self.config = ReportConfig(
            output_dir=Path(output_dir),
            include_plots=include_plots
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache = {}

        # Check if matplotlib is available
        self._has_matplotlib = False
        if include_plots:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                self._has_matplotlib = True
            except ImportError:
                logger.warning("Matplotlib not available. Plots disabled.")

    def generate_da_table(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate Direction Accuracy table by model and horizon.

        Args:
            results: Nested dict {model: {horizon: {da_train, da_val, ...}}}
            save: Whether to save to file

        Returns:
            DataFrame with DA values, formatted for display

        Example output:
            Model           H=1    H=5   H=10   H=15   H=20   H=25   H=30
            ridge          52.1%  51.8%  50.2%  49.5%  48.3%  47.1%  46.2%
            xgboost_pure   53.2%  52.4%  51.1%  50.3%  49.2%  48.1%  47.0%
        """
        # Build table
        rows = []
        horizons = sorted(set(
            h for model_results in results.values()
            for h in model_results.keys()
        ))

        for model_name, horizon_results in results.items():
            row = {'model': model_name}
            for horizon in horizons:
                if horizon in horizon_results:
                    da = horizon_results[horizon].get('da_val', 0)
                    row[f'H={horizon}'] = da
                else:
                    row[f'H={horizon}'] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('model')

        # Sort by average DA
        df['avg_da'] = df.mean(axis=1)
        df = df.sort_values('avg_da', ascending=False)
        df = df.drop('avg_da', axis=1)

        if save:
            # Save as CSV
            csv_path = self.config.output_dir / 'da_table.csv'
            df.to_csv(csv_path)

            # Save formatted version
            df_formatted = df.apply(
                lambda x: x.apply(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
            )
            txt_path = self.config.output_dir / 'da_table.txt'
            with open(txt_path, 'w') as f:
                f.write("Direction Accuracy by Model and Horizon\n")
                f.write("=" * 60 + "\n\n")
                f.write(df_formatted.to_string())
                f.write("\n")

            logger.info(f"DA table saved to {csv_path}")

        return df

    def generate_variance_table(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate Variance Ratio table by model and horizon.

        Variance ratio < 0.1 indicates potential model collapse.

        Args:
            results: Nested dict with variance_ratio per model/horizon
            save: Whether to save to file

        Returns:
            DataFrame with variance ratios
        """
        rows = []
        horizons = sorted(set(
            h for model_results in results.values()
            for h in model_results.keys()
        ))

        for model_name, horizon_results in results.items():
            row = {'model': model_name}
            for horizon in horizons:
                if horizon in horizon_results:
                    vr = horizon_results[horizon].get('variance_ratio', 0)
                    row[f'H={horizon}'] = vr
                else:
                    row[f'H={horizon}'] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('model')

        if save:
            csv_path = self.config.output_dir / 'variance_ratio_table.csv'
            df.to_csv(csv_path)

            # Flag low variance ratios
            df_flagged = df.apply(
                lambda x: x.apply(
                    lambda v: f"{v:.3f} !!LOW!!" if pd.notna(v) and v < 0.1
                    else (f"{v:.3f}" if pd.notna(v) else "-")
                )
            )
            txt_path = self.config.output_dir / 'variance_ratio_table.txt'
            with open(txt_path, 'w') as f:
                f.write("Variance Ratio by Model and Horizon\n")
                f.write("(Values < 0.1 indicate potential model collapse)\n")
                f.write("=" * 60 + "\n\n")
                f.write(df_flagged.to_string())
                f.write("\n")

            logger.info(f"Variance ratio table saved to {csv_path}")

        return df

    def generate_gap_table(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate Overfitting Gap table (DA_train - DA_val).

        Gap > 15% indicates potential overfitting.

        Args:
            results: Nested dict with da_train and da_val per model/horizon
            save: Whether to save to file

        Returns:
            DataFrame with overfitting gaps
        """
        rows = []
        horizons = sorted(set(
            h for model_results in results.values()
            for h in model_results.keys()
        ))

        for model_name, horizon_results in results.items():
            row = {'model': model_name}
            for horizon in horizons:
                if horizon in horizon_results:
                    da_train = horizon_results[horizon].get('da_train', 0)
                    da_val = horizon_results[horizon].get('da_val', 0)
                    gap = da_train - da_val
                    row[f'H={horizon}'] = gap
                else:
                    row[f'H={horizon}'] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('model')

        if save:
            csv_path = self.config.output_dir / 'overfitting_gap_table.csv'
            df.to_csv(csv_path)

            # Flag high gaps
            df_flagged = df.apply(
                lambda x: x.apply(
                    lambda v: f"{v:.1%} !!OVERFIT!!" if pd.notna(v) and v > 0.15
                    else (f"{v:.1%}" if pd.notna(v) else "-")
                )
            )
            txt_path = self.config.output_dir / 'overfitting_gap_table.txt'
            with open(txt_path, 'w') as f:
                f.write("Overfitting Gap (DA_train - DA_val) by Model and Horizon\n")
                f.write("(Gap > 15% indicates potential overfitting)\n")
                f.write("=" * 60 + "\n\n")
                f.write(df_flagged.to_string())
                f.write("\n")

            logger.info(f"Overfitting gap table saved to {csv_path}")

        return df

    def plot_equity_curves(
        self,
        backtest_results: Dict[str, Any],
        horizon: int = 15,
        save: bool = True
    ) -> Optional[str]:
        """
        Plot equity curves from walk-forward backtest results.

        Args:
            backtest_results: Dict {model_name: BacktestResult}
            horizon: Horizon to plot (if results are by horizon)
            save: Whether to save plot

        Returns:
            Path to saved figure, or None if plotting disabled
        """
        if not self._has_matplotlib:
            logger.warning("Matplotlib not available. Skipping equity curve plot.")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for model_name, result in backtest_results.items():
            if hasattr(result, 'equity_curve'):
                equity = result.equity_curve
            elif isinstance(result, dict) and 'equity_curve' in result:
                equity = result['equity_curve']
            else:
                continue

            ax.plot(equity, label=model_name, linewidth=1.5)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Equity (Starting = 1.0)')
        ax.set_title(f'Walk-Forward Equity Curves - H={horizon}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        if save:
            path = self.config.output_dir / f'equity_curves_h{horizon}.{self.config.figure_format}'
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Equity curves saved to {path}")
            return str(path)

        plt.close(fig)
        return None

    def plot_da_by_window(
        self,
        backtest_results: Dict[str, Any],
        horizon: int = 15,
        save: bool = True
    ) -> Optional[str]:
        """
        Plot DA by window for each model.

        Args:
            backtest_results: Dict {model_name: BacktestResult}
            horizon: Horizon to plot
            save: Whether to save plot

        Returns:
            Path to saved figure, or None if plotting disabled
        """
        if not self._has_matplotlib:
            logger.warning("Matplotlib not available. Skipping DA by window plot.")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        x_positions = None
        width = 0.8 / len(backtest_results)

        for i, (model_name, result) in enumerate(backtest_results.items()):
            if hasattr(result, 'da_by_window'):
                da_values = result.da_by_window
            elif isinstance(result, dict) and 'da_by_window' in result:
                da_values = result['da_by_window']
            else:
                continue

            if x_positions is None:
                x_positions = np.arange(len(da_values))

            offset = (i - len(backtest_results) / 2 + 0.5) * width
            ax.bar(
                x_positions + offset,
                da_values,
                width=width,
                label=model_name,
                alpha=0.8
            )

        if x_positions is not None:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'W{i+1}' for i in range(len(x_positions))])

        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.set_xlabel('Window')
        ax.set_ylabel('Direction Accuracy')
        ax.set_title(f'DA by Walk-Forward Window - H={horizon}')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0.4, 0.7)
        ax.grid(True, alpha=0.3, axis='y')

        if save:
            path = self.config.output_dir / f'da_by_window_h{horizon}.{self.config.figure_format}'
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"DA by window plot saved to {path}")
            return str(path)

        plt.close(fig)
        return None

    def plot_da_heatmap(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        metric: str = 'da_val',
        save: bool = True
    ) -> Optional[str]:
        """
        Plot DA heatmap (models x horizons).

        Args:
            results: Nested dict with metrics
            metric: Which metric to plot ('da_val', 'variance_ratio', etc.)
            save: Whether to save plot

        Returns:
            Path to saved figure, or None if plotting disabled
        """
        if not self._has_matplotlib:
            return None

        import matplotlib.pyplot as plt

        # Build matrix
        models = list(results.keys())
        horizons = sorted(set(
            h for model_results in results.values()
            for h in model_results.keys()
        ))

        matrix = np.zeros((len(models), len(horizons)))
        for i, model in enumerate(models):
            for j, horizon in enumerate(horizons):
                if horizon in results[model]:
                    matrix[i, j] = results[model][horizon].get(metric, np.nan)
                else:
                    matrix[i, j] = np.nan

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.60)

        # Labels
        ax.set_xticks(np.arange(len(horizons)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([f'H={h}' for h in horizons])
        ax.set_yticklabels(models)

        # Annotations
        for i in range(len(models)):
            for j in range(len(horizons)):
                value = matrix[i, j]
                if not np.isnan(value):
                    text = ax.text(
                        j, i, f'{value:.1%}',
                        ha='center', va='center',
                        color='black' if 0.48 < value < 0.58 else 'white',
                        fontsize=9
                    )

        ax.set_title(f'{metric.upper()} Heatmap by Model and Horizon')
        fig.colorbar(im, ax=ax, label=metric)

        if save:
            path = self.config.output_dir / f'{metric}_heatmap.{self.config.figure_format}'
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Heatmap saved to {path}")
            return str(path)

        plt.close(fig)
        return None

    def create_full_report(
        self,
        training_results: Dict[str, Dict[int, Dict[str, float]]],
        backtest_results: Optional[Dict[str, Dict[int, Any]]] = None,
        report_name: str = None
    ) -> Dict[str, Any]:
        """
        Create full training report with all tables and plots.

        Args:
            training_results: Results from training phase
            backtest_results: Results from walk-forward backtest
            report_name: Name for the report (default: timestamp)

        Returns:
            Dict with paths to all generated files
        """
        report_name = report_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = self.config.output_dir / report_name
        report_dir.mkdir(parents=True, exist_ok=True)

        # Update output dir temporarily
        original_output = self.config.output_dir
        self.config.output_dir = report_dir

        generated_files = {'report_dir': str(report_dir)}

        try:
            # Generate tables
            logger.info("Generating DA table...")
            da_df = self.generate_da_table(training_results)
            generated_files['da_table'] = str(report_dir / 'da_table.csv')

            logger.info("Generating variance ratio table...")
            vr_df = self.generate_variance_table(training_results)
            generated_files['variance_table'] = str(report_dir / 'variance_ratio_table.csv')

            logger.info("Generating overfitting gap table...")
            gap_df = self.generate_gap_table(training_results)
            generated_files['gap_table'] = str(report_dir / 'overfitting_gap_table.csv')

            # Generate plots if matplotlib available
            if self._has_matplotlib:
                logger.info("Generating DA heatmap...")
                heatmap_path = self.plot_da_heatmap(training_results)
                if heatmap_path:
                    generated_files['da_heatmap'] = heatmap_path

            # Backtest plots
            if backtest_results:
                horizons = set()
                for model_results in backtest_results.values():
                    if isinstance(model_results, dict):
                        horizons.update(model_results.keys())

                for horizon in sorted(horizons):
                    horizon_results = {
                        model: results.get(horizon)
                        for model, results in backtest_results.items()
                        if isinstance(results, dict) and horizon in results
                    }

                    if horizon_results:
                        eq_path = self.plot_equity_curves(horizon_results, horizon)
                        if eq_path:
                            generated_files[f'equity_h{horizon}'] = eq_path

                        da_path = self.plot_da_by_window(horizon_results, horizon)
                        if da_path:
                            generated_files[f'da_window_h{horizon}'] = da_path

            # Summary JSON
            summary = {
                'report_name': report_name,
                'generated_at': datetime.now().isoformat(),
                'n_models': len(training_results),
                'n_horizons': len(set(
                    h for m in training_results.values() for h in m.keys()
                )),
                'files': generated_files
            }

            # Best models summary
            best_models = {}
            horizons = sorted(set(
                h for m in training_results.values() for h in m.keys()
            ))
            for horizon in horizons:
                best_da = 0
                best_model = None
                for model, results in training_results.items():
                    if horizon in results:
                        da = results[horizon].get('da_val', 0)
                        if da > best_da:
                            best_da = da
                            best_model = model
                if best_model:
                    best_models[f'H={horizon}'] = {
                        'model': best_model,
                        'da_val': best_da
                    }
            summary['best_models'] = best_models

            # Save summary
            summary_path = report_dir / 'report_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            generated_files['summary'] = str(summary_path)

            logger.info(f"Full report generated at {report_dir}")

        finally:
            self.config.output_dir = original_output

        return generated_files

    def print_summary(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        show_flags: bool = True
    ) -> str:
        """
        Print formatted summary to console.

        Args:
            results: Training results
            show_flags: Whether to show warning flags

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("TRAINING SUMMARY REPORT")
        lines.append("=" * 70 + "\n")

        # Get all horizons
        horizons = sorted(set(
            h for model_results in results.values()
            for h in model_results.keys()
        ))

        # Header
        header = f"{'Model':<20}"
        for h in horizons:
            header += f"{'H=' + str(h):>8}"
        header += f"{'Avg':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        model_avgs = []
        for model_name, horizon_results in sorted(results.items()):
            row = f"{model_name:<20}"
            das = []
            for horizon in horizons:
                if horizon in horizon_results:
                    da = horizon_results[horizon].get('da_val', 0)
                    vr = horizon_results[horizon].get('variance_ratio', 1)
                    gap = horizon_results[horizon].get('da_train', 0) - da

                    # Flags
                    flag = ""
                    if show_flags:
                        if vr < 0.1:
                            flag = "*"  # Low variance
                        elif gap > 0.15:
                            flag = "!"  # Overfitting

                    row += f"{da*100:>7.1f}{flag}"
                    das.append(da)
                else:
                    row += f"{'--':>8}"

            avg_da = np.mean(das) if das else 0
            model_avgs.append((model_name, avg_da))
            row += f"{avg_da*100:>8.1f}"
            lines.append(row)

        lines.append("-" * len(header))

        # Legend
        if show_flags:
            lines.append("\nFlags: * = Low variance (<0.1), ! = Overfitting (gap >15%)")

        # Best model per horizon
        lines.append("\nBest Models per Horizon:")
        for horizon in horizons:
            best_da = 0
            best_model = "-"
            for model_name, horizon_results in results.items():
                if horizon in horizon_results:
                    da = horizon_results[horizon].get('da_val', 0)
                    if da > best_da:
                        best_da = da
                        best_model = model_name
            lines.append(f"  H={horizon:>2}: {best_model:<20} ({best_da:.1%})")

        # Overall best
        lines.append("\nOverall Best (by average DA):")
        sorted_models = sorted(model_avgs, key=lambda x: x[1], reverse=True)[:3]
        for i, (model, avg) in enumerate(sorted_models):
            lines.append(f"  {i+1}. {model:<20} ({avg:.1%})")

        lines.append("\n" + "=" * 70)

        summary = "\n".join(lines)
        print(summary)
        return summary


def create_training_report(
    results: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str = 'reports',
    backtest_results: Dict = None
) -> Dict[str, Any]:
    """
    Convenience function to create a full training report.

    Args:
        results: Training results dict
        output_dir: Output directory
        backtest_results: Optional backtest results

    Returns:
        Dict with paths to generated files
    """
    reporter = TrainingReporter(output_dir=output_dir)
    reporter.print_summary(results)
    return reporter.create_full_report(results, backtest_results)
