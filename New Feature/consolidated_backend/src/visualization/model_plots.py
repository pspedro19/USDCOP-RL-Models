# pipeline_limpio_regresion/visualization/model_plots.py
"""
Model comparison visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ModelComparisonPlotter:
    """
    Creates model comparison visualizations for reports.

    Plots:
    - Model ranking bar charts
    - Heatmaps of metrics by horizon
    - Feature importance
    - Training time comparison
    """

    def __init__(
        self,
        figsize: tuple = (12, 6),
        dpi: int = 150
    ):
        self.figsize = figsize
        self.dpi = dpi

    def plot_model_ranking(
        self,
        results_df: pd.DataFrame,
        metric: str = 'direction_accuracy',
        save_path: Path = None
    ):
        """
        Plot model ranking by metric.

        Args:
            results_df: DataFrame with model results
            metric: Metric to rank by
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        # Average metric by model
        avg_by_model = results_df.groupby('model')[metric].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg_by_model)))

        bars = ax.barh(range(len(avg_by_model)), avg_by_model.values * 100, color=colors)

        ax.set_yticks(range(len(avg_by_model)))
        ax.set_yticklabels([m.upper() for m in avg_by_model.index])
        ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)')
        ax.set_title(f'Model Ranking by {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, avg_by_model.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val*100:.1f}%', va='center', fontsize=10)

        # Reference lines
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.legend()

        ax.set_xlim(0, max(avg_by_model.values) * 100 + 10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved ranking plot to {save_path}")

        plt.close()
        return fig

    def plot_metrics_heatmap(
        self,
        results_df: pd.DataFrame,
        metric: str = 'direction_accuracy',
        save_path: Path = None
    ):
        """
        Plot heatmap of metric by model and horizon.

        Args:
            results_df: DataFrame with results
            metric: Metric to display
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        # Pivot table
        pivot = results_df.pivot_table(
            values=metric,
            index='model',
            columns='horizon',
            aggfunc='mean'
        )

        # For DA, multiply by 100
        if metric == 'direction_accuracy':
            pivot = pivot * 100

        fig, ax = plt.subplots(figsize=self.figsize)

        cmap = 'RdYlGn' if metric == 'direction_accuracy' else 'RdYlGn_r'
        vmin = 40 if metric == 'direction_accuracy' else None
        vmax = 70 if metric == 'direction_accuracy' else None

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                   vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label': metric})

        ax.set_title(f'{metric.replace("_", " ").title()} by Model and Horizon',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Model')

        # Highlight best per horizon
        for col in pivot.columns:
            best_idx = pivot[col].idxmax()
            row_idx = list(pivot.index).index(best_idx)
            col_idx = list(pivot.columns).index(col)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False,
                                       edgecolor='blue', linewidth=2))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")

        plt.close()
        return fig

    def plot_training_time(
        self,
        results_df: pd.DataFrame,
        save_path: Path = None
    ):
        """
        Plot training time comparison.

        Args:
            results_df: DataFrame with results
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        # Total time by model
        time_by_model = results_df.groupby('model')['train_time'].sum()

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(time_by_model.index, time_by_model.values, color='steelblue', alpha=0.7)

        ax.set_xlabel('Model')
        ax.set_ylabel('Total Training Time (seconds)')
        ax.set_title('Training Time by Model', fontsize=14, fontweight='bold')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved training time plot to {save_path}")

        plt.close()
        return fig

    def plot_all_metrics_summary(
        self,
        results_df: pd.DataFrame,
        save_path: Path = None
    ):
        """
        Create comprehensive metrics summary plot.

        Args:
            results_df: DataFrame with results
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # DA by model
        ax1 = axes[0, 0]
        avg_da = results_df.groupby('model')['direction_accuracy'].mean() * 100
        avg_da.sort_values().plot(kind='barh', ax=ax1, color='steelblue')
        ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Direction Accuracy (%)')
        ax1.set_title('Average DA by Model')

        # RMSE by model
        ax2 = axes[0, 1]
        avg_rmse = results_df.groupby('model')['rmse'].mean()
        avg_rmse.sort_values(ascending=False).plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('RMSE')
        ax2.set_title('Average RMSE by Model')

        # R2 by model
        ax3 = axes[1, 0]
        avg_r2 = results_df.groupby('model')['r2'].mean()
        avg_r2.sort_values().plot(kind='barh', ax=ax3, color='mediumseagreen')
        ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_xlabel('R²')
        ax3.set_title('Average R² by Model')

        # Correlation by model
        ax4 = axes[1, 1]
        if 'correlation' in results_df.columns:
            avg_corr = results_df.groupby('model')['correlation'].mean()
            avg_corr.sort_values().plot(kind='barh', ax=ax4, color='purple', alpha=0.7)
            ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Correlation')
            ax4.set_title('Average Correlation by Model')

        plt.suptitle('Model Comparison: All Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved metrics summary to {save_path}")

        plt.close()
        return fig

    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 20,
        save_path: Path = None
    ):
        """
        Plot feature importance.

        Args:
            importance: Dictionary of feature -> importance
            top_n: Number of top features to show
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        # Sort and get top N
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        features, values = zip(*sorted_imp)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='gray', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        plt.close()
        return fig
