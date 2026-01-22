# pipeline_limpio_regresion/visualization/backtest_plots.py
"""
Professional backtest visualization for model evaluation.

Includes both price and return visualizations for comprehensive analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BacktestPlotter:
    """
    Creates professional backtest visualizations for model evaluation.

    Shows both prices and returns for comprehensive analysis.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 150,
        style: str = 'default'
    ):
        self.figsize = figsize
        self.dpi = dpi

        if HAS_MATPLOTLIB:
            plt.style.use(style)
            # Set professional defaults
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 9

    def plot_complete_backtest(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: pd.Series,
        dates: pd.DatetimeIndex,
        model_name: str = "Model",
        horizon: int = None,
        save_path: Path = None
    ):
        """
        Create comprehensive backtest visualization with prices AND returns.

        This is the main visualization showing:
        - Top row: Prices (actual vs predicted price levels)
        - Middle row: Returns (actual vs predicted returns)
        - Bottom row: Performance metrics and analysis

        Args:
            y_true: True returns
            y_pred: Predicted returns
            prices: Historical price series (aligned with y_true/y_pred)
            dates: Date index for the test period
            model_name: Model name for title
            horizon: Prediction horizon
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        # Create figure with custom grid
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1.2, 1, 1, 0.8],
                     hspace=0.35, wspace=0.25)

        title_suffix = f" H={horizon}" if horizon else ""

        # Calculate metrics
        da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0

        # Calculate predicted prices
        if len(prices) > len(y_true):
            prices_aligned = prices.iloc[-len(y_true):].copy()
        else:
            prices_aligned = prices.copy()

        # Reconstruct predicted prices from returns
        base_price = prices_aligned.iloc[0]
        actual_prices = prices_aligned.values
        pred_prices = np.zeros_like(actual_prices)
        pred_prices[0] = base_price

        for i in range(1, len(pred_prices)):
            pred_prices[i] = pred_prices[i-1] * np.exp(y_pred[i-1] if i-1 < len(y_pred) else 0)

        # ============================================================
        # ROW 1: PRICE ANALYSIS (Full width)
        # ============================================================
        ax_price = fig.add_subplot(gs[0, :])

        ax_price.plot(dates, actual_prices, 'b-', linewidth=1.5,
                     label='Precio Real', alpha=0.9)
        ax_price.plot(dates, pred_prices, 'r--', linewidth=1.5,
                     label='Precio Predicho', alpha=0.8)

        # Highlight areas where prediction was correct direction
        correct_mask = np.sign(y_true) == np.sign(y_pred)
        for i in range(len(dates)-1):
            if i < len(correct_mask) and correct_mask[i]:
                ax_price.axvspan(dates[i], dates[i+1], alpha=0.1, color='green')

        ax_price.set_title(f'{model_name.upper()}{title_suffix} - Backtest: Precios Reales vs Predichos',
                          fontsize=14, fontweight='bold')
        ax_price.set_xlabel('Fecha')
        ax_price.set_ylabel('Precio USD/COP')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Add metrics annotation
        metrics_text = f'DA: {da:.1f}% | RMSE: {rmse:.6f} | Corr: {corr:.3f}'
        ax_price.text(0.98, 0.98, metrics_text, transform=ax_price.transAxes,
                     fontsize=11, fontweight='bold', ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

        # ============================================================
        # ROW 2: RETURNS TIME SERIES
        # ============================================================
        ax_ret_ts = fig.add_subplot(gs[1, :2])

        ax_ret_ts.plot(dates[:len(y_true)], y_true * 100, 'b-', linewidth=1,
                      label='Retorno Real', alpha=0.8)
        ax_ret_ts.plot(dates[:len(y_pred)], y_pred * 100, 'r-', linewidth=1,
                      label='Retorno Predicho', alpha=0.8)
        ax_ret_ts.axhline(y=0, color='black', linewidth=0.5)

        ax_ret_ts.set_title('Retornos: Real vs Predicho', fontsize=12, fontweight='bold')
        ax_ret_ts.set_xlabel('Fecha')
        ax_ret_ts.set_ylabel('Retorno (%)')
        ax_ret_ts.legend(loc='upper right')
        ax_ret_ts.grid(True, alpha=0.3)
        ax_ret_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_ret_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ============================================================
        # ROW 2 RIGHT: SCATTER PLOT
        # ============================================================
        ax_scatter = fig.add_subplot(gs[1, 2])

        # Color by correct/incorrect direction
        colors = ['green' if np.sign(t) == np.sign(p) else 'red'
                 for t, p in zip(y_true, y_pred)]

        ax_scatter.scatter(y_true * 100, y_pred * 100, c=colors, alpha=0.5, s=30)

        min_val = min(y_true.min(), y_pred.min()) * 100
        max_val = max(y_true.max(), y_pred.max()) * 100
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
                       label='Prediccion Perfecta')
        ax_scatter.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax_scatter.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)

        ax_scatter.set_title('Scatter: Real vs Predicho', fontsize=12, fontweight='bold')
        ax_scatter.set_xlabel('Retorno Real (%)')
        ax_scatter.set_ylabel('Retorno Predicho (%)')

        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=10, label='Direccion Correcta'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=10, label='Direccion Incorrecta'),
            Line2D([0], [0], color='black', linestyle='--', label='Perfecta')
        ]
        ax_scatter.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax_scatter.grid(True, alpha=0.3)

        # ============================================================
        # ROW 3 LEFT: CUMULATIVE RETURNS
        # ============================================================
        ax_cum = fig.add_subplot(gs[2, 0])

        cum_true = np.cumsum(y_true) * 100
        cum_pred = np.cumsum(y_pred) * 100

        # Strategy: follow predictions
        strategy_returns = y_true * np.sign(y_pred)
        cum_strategy = np.cumsum(strategy_returns) * 100

        ax_cum.plot(dates[:len(cum_true)], cum_true, 'b-', linewidth=1.5, label='Real')
        ax_cum.plot(dates[:len(cum_pred)], cum_pred, 'r-', linewidth=1.5, label='Predicho')
        ax_cum.plot(dates[:len(cum_strategy)], cum_strategy, 'g-', linewidth=2, label='Estrategia')
        ax_cum.axhline(y=0, color='black', linewidth=0.5)
        ax_cum.fill_between(dates[:len(cum_strategy)], 0, cum_strategy, alpha=0.1, color='green')

        ax_cum.set_title('Retornos Acumulados', fontsize=12, fontweight='bold')
        ax_cum.set_xlabel('Fecha')
        ax_cum.set_ylabel('Retorno Acumulado (%)')
        ax_cum.legend(loc='best', fontsize=8)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ============================================================
        # ROW 3 CENTER: ERROR DISTRIBUTION
        # ============================================================
        ax_error = fig.add_subplot(gs[2, 1])

        errors = (y_pred - y_true) * 100
        ax_error.hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
        ax_error.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cero')
        ax_error.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
                        label=f'Media: {np.mean(errors):.3f}%')
        ax_error.axvline(x=np.median(errors), color='orange', linestyle='--', linewidth=2,
                        label=f'Mediana: {np.median(errors):.3f}%')

        ax_error.set_title('Distribucion de Errores', fontsize=12, fontweight='bold')
        ax_error.set_xlabel('Error de Prediccion (%)')
        ax_error.set_ylabel('Frecuencia')
        ax_error.legend(loc='upper right', fontsize=8)

        # ============================================================
        # ROW 3 RIGHT: ROLLING DIRECTION ACCURACY
        # ============================================================
        ax_da = fig.add_subplot(gs[2, 2])

        correct = (np.sign(y_true) == np.sign(y_pred)).astype(int)
        window = min(20, len(correct) // 5)
        if window > 1:
            rolling_da = pd.Series(correct).rolling(window, min_periods=1).mean() * 100
            ax_da.plot(dates[:len(rolling_da)], rolling_da, 'g-', linewidth=1.5)

        ax_da.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax_da.axhline(y=da, color='blue', linestyle='--', alpha=0.7, label=f'Media: {da:.1f}%')
        ax_da.fill_between(dates[:len(rolling_da)], 50, rolling_da,
                          where=rolling_da >= 50, alpha=0.2, color='green')
        ax_da.fill_between(dates[:len(rolling_da)], 50, rolling_da,
                          where=rolling_da < 50, alpha=0.2, color='red')

        ax_da.set_title(f'Direction Accuracy Rolling (w={window})', fontsize=12, fontweight='bold')
        ax_da.set_xlabel('Fecha')
        ax_da.set_ylabel('DA (%)')
        ax_da.legend(loc='lower right', fontsize=8)
        ax_da.set_ylim(20, 80)
        ax_da.grid(True, alpha=0.3)
        ax_da.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax_da.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ============================================================
        # ROW 4: METRICS SUMMARY TABLE
        # ============================================================
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')

        # Calculate additional metrics
        sharpe_strategy = (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                         if np.std(strategy_returns) > 0 else 0)
        max_dd = self._calculate_max_drawdown(cum_strategy)
        win_rate = da
        profit_factor = (np.sum(strategy_returns[strategy_returns > 0]) /
                        abs(np.sum(strategy_returns[strategy_returns < 0]))
                        if np.sum(strategy_returns[strategy_returns < 0]) != 0 else np.inf)

        # Create table data
        col_labels = ['Metrica', 'Valor', 'Metrica', 'Valor', 'Metrica', 'Valor']
        table_data = [
            ['Direction Accuracy', f'{da:.2f}%',
             'RMSE', f'{rmse:.6f}',
             'Correlacion', f'{corr:.4f}'],
            ['MAE', f'{mae:.6f}',
             'Sharpe Ratio', f'{sharpe_strategy:.3f}',
             'Max Drawdown', f'{max_dd:.2f}%'],
            ['Retorno Acumulado', f'{cum_strategy[-1]:.2f}%',
             'Win Rate', f'{win_rate:.1f}%',
             'Profit Factor', f'{profit_factor:.2f}' if profit_factor < 100 else '>100'],
        ]

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.12, 0.15, 0.12, 0.15, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style table
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#3498DB')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Color code values
        for i in range(1, len(table_data) + 1):
            for j in [1, 3, 5]:
                cell = table[(i, j)]
                value = table_data[i-1][j]
                if '%' in value:
                    try:
                        num = float(value.replace('%', '').replace('>', ''))
                        if 'DA' in table_data[i-1][j-1] or 'Win' in table_data[i-1][j-1]:
                            if num > 55:
                                cell.set_facecolor('#90EE90')
                            elif num < 45:
                                cell.set_facecolor('#FFB6C1')
                    except:
                        pass

        plt.suptitle(f'BACKTEST COMPLETO - {model_name.upper()}{title_suffix}',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved backtest plot to {save_path}")

        plt.close()
        return fig

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns)
        return np.max(drawdown) if len(drawdown) > 0 else 0

    def plot_predicted_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex = None,
        model_name: str = "Model",
        horizon: int = None,
        save_path: Path = None
    ):
        """
        Plot predicted vs actual returns (simplified version).
        Kept for backward compatibility.
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        title_suffix = f" (H={horizon})" if horizon else ""

        # Time series plot
        ax1 = axes[0, 0]
        if dates is not None:
            ax1.plot(dates, y_true * 100, 'b-', alpha=0.7, linewidth=1, label='Real')
            ax1.plot(dates, y_pred * 100, 'r-', alpha=0.7, linewidth=1, label='Predicho')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax1.plot(y_true * 100, 'b-', alpha=0.7, linewidth=1, label='Real')
            ax1.plot(y_pred * 100, 'r-', alpha=0.7, linewidth=1, label='Predicho')

        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_title(f'{model_name}{title_suffix}: Retornos Predicho vs Real')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Retorno (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2 = axes[0, 1]
        colors = ['green' if np.sign(t) == np.sign(p) else 'red'
                 for t, p in zip(y_true, y_pred)]
        ax2.scatter(y_true * 100, y_pred * 100, c=colors, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min()) * 100
        max_val = max(y_true.max(), y_pred.max()) * 100
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
        ax2.set_title('Scatter: Real vs Predicho')
        ax2.set_xlabel('Retorno Real (%)')
        ax2.set_ylabel('Retorno Predicho (%)')
        ax2.grid(True, alpha=0.3)

        # Error distribution
        ax3 = axes[1, 0]
        errors = (y_pred - y_true) * 100
        ax3.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
                   label=f'Media: {np.mean(errors):.3f}%')
        ax3.set_title('Distribucion de Errores')
        ax3.set_xlabel('Error (%)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()

        # Direction accuracy rolling
        ax4 = axes[1, 1]
        correct = (np.sign(y_true) == np.sign(y_pred)).astype(int)
        da = np.mean(correct) * 100
        window = min(20, len(correct) // 5)
        if window > 0:
            rolling_da = pd.Series(correct).rolling(window, min_periods=1).mean() * 100
            if dates is not None:
                ax4.plot(dates[:len(rolling_da)], rolling_da, 'g-', linewidth=1.5)
            else:
                ax4.plot(rolling_da, 'g-', linewidth=1.5)
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax4.axhline(y=da, color='blue', linestyle='--', alpha=0.7, label=f'Media: {da:.1f}%')
        ax4.set_title(f'Direction Accuracy Rolling (w={window})')
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('DA (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved backtest plot to {save_path}")

        plt.close()
        return fig

    def plot_cumulative_returns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex = None,
        model_name: str = "Model",
        save_path: Path = None
    ):
        """Plot cumulative returns comparison."""
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        cum_true = np.cumsum(y_true) * 100
        cum_pred = np.cumsum(y_pred) * 100
        strategy_returns = y_true * np.sign(y_pred)
        cum_strategy = np.cumsum(strategy_returns) * 100

        x = dates if dates is not None else range(len(y_true))

        ax.plot(x, cum_true, 'b-', linewidth=1.5, label='Real', alpha=0.8)
        ax.plot(x, cum_pred, 'r-', linewidth=1.5, label='Predicho', alpha=0.8)
        ax.plot(x, cum_strategy, 'g-', linewidth=2.5, label='Estrategia', alpha=0.9)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.fill_between(x, 0, cum_strategy, alpha=0.1, color='green')

        ax.set_title(f'{model_name}: Retornos Acumulados', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Retorno Acumulado (%)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved cumulative returns plot to {save_path}")

        plt.close()
        return fig

    def plot_multi_model_comparison(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        save_path: Path = None
    ):
        """Compare multiple models side by side."""
        if not HAS_MATPLOTLIB:
            return

        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for i, (model_name, data) in enumerate(results.items()):
            y_true = data['y_true']
            y_pred = data['y_pred']

            # Scatter with colors
            colors = ['green' if np.sign(t) == np.sign(p) else 'red'
                     for t, p in zip(y_true, y_pred)]
            axes[0, i].scatter(y_true * 100, y_pred * 100, c=colors, alpha=0.5, s=20)
            min_val = min(y_true.min(), y_pred.min()) * 100
            max_val = max(y_true.max(), y_pred.max()) * 100
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'k--')
            axes[0, i].axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            axes[0, i].axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
            axes[0, i].set_title(f'{model_name.upper()}', fontweight='bold')
            axes[0, i].set_xlabel('Real (%)')
            axes[0, i].set_ylabel('Predicho (%)')

            da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            axes[0, i].text(0.05, 0.95, f'DA: {da:.1f}%\nRMSE: {rmse:.5f}',
                          transform=axes[0, i].transAxes,
                          verticalalignment='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Error histogram
            errors = (y_pred - y_true) * 100
            axes[1, i].hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
            axes[1, i].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1, i].set_xlabel('Error (%)')
            axes[1, i].set_ylabel('Frecuencia')

        plt.suptitle('Comparacion de Modelos: Backtest', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved multi-model comparison to {save_path}")

        plt.close()
        return fig

    def plot_horizon_comparison(
        self,
        results_by_horizon: Dict[int, Dict[str, np.ndarray]],
        model_name: str = "Model",
        save_path: Path = None
    ):
        """Compare model performance across horizons."""
        if not HAS_MATPLOTLIB:
            return

        horizons = sorted(results_by_horizon.keys())
        das = []
        rmses = []
        corrs = []

        for h in horizons:
            y_true = results_by_horizon[h]['y_true']
            y_pred = results_by_horizon[h]['y_pred']

            da = np.mean(np.sign(y_true) == np.sign(y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0

            das.append(da * 100)
            rmses.append(rmse)
            corrs.append(corr)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # DA by horizon
        bars1 = axes[0].bar(horizons, das, color='steelblue', alpha=0.8)
        axes[0].axhline(y=50, color='red', linestyle='--', label='Random', linewidth=2)
        axes[0].set_xlabel('Horizonte (dias)')
        axes[0].set_ylabel('Direction Accuracy (%)')
        axes[0].set_title('DA por Horizonte', fontweight='bold')
        axes[0].legend()
        # Color bars based on value
        for bar, da in zip(bars1, das):
            bar.set_color('green' if da > 55 else 'orange' if da > 50 else 'red')

        # RMSE by horizon
        axes[1].bar(horizons, rmses, color='coral', alpha=0.8)
        axes[1].set_xlabel('Horizonte (dias)')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE por Horizonte', fontweight='bold')

        # Correlation by horizon
        bars3 = axes[2].bar(horizons, corrs, color='mediumseagreen', alpha=0.8)
        axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axes[2].set_xlabel('Horizonte (dias)')
        axes[2].set_ylabel('Correlacion')
        axes[2].set_title('Correlacion por Horizonte', fontweight='bold')
        for bar, corr in zip(bars3, corrs):
            bar.set_color('green' if corr > 0.1 else 'orange' if corr > 0 else 'red')

        plt.suptitle(f'{model_name.upper()}: Performance por Horizonte',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved horizon comparison to {save_path}")

        plt.close()
        return fig
