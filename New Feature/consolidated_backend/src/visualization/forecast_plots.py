# pipeline_limpio_regresion/visualization/forecast_plots.py
"""
Forecast visualization for academic papers and investors.

Includes fan charts with confidence intervals for forward forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - visualization disabled")


class ForecastPlotter:
    """
    Creates publication-quality forecast visualizations.

    Plots:
    - Forward forecast with confidence intervals
    - Fan chart
    - Forecast table
    - Multi-horizon comparison
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 150
    ):
        self.figsize = figsize
        self.dpi = dpi

        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                plt.style.use('default')

    def plot_forward_forecast(
        self,
        prices: pd.Series,
        forecasts: Dict[int, float],
        confidence_intervals: Dict[int, Tuple[float, float]] = None,
        current_price: float = None,
        save_path: Path = None,
        title: str = "USD/COP Forward Forecast"
    ):
        """
        Plot forward forecast with confidence intervals.

        Args:
            prices: Historical prices
            forecasts: Dictionary of horizon -> forecasted return
            confidence_intervals: Dictionary of horizon -> (lower, upper)
            current_price: Current price for reference
            save_path: Path to save figure
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))

        # Get current price
        current_price = current_price or prices.iloc[-1]
        last_date = prices.index[-1]

        # Plot historical prices
        ax1.plot(prices.index[-60:], prices.iloc[-60:], 'b-', linewidth=1.5, label='Historical')
        ax1.axhline(y=current_price, color='gray', linestyle='--', alpha=0.5)

        # Plot forecasts as price levels
        horizons = sorted(forecasts.keys())
        forecast_dates = [last_date + pd.Timedelta(days=h) for h in horizons]
        forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

        ax1.plot(forecast_dates, forecast_prices, 'ro-', linewidth=2, markersize=8, label='Forecast')

        # Confidence intervals
        if confidence_intervals:
            lower = [current_price * np.exp(confidence_intervals[h][0]) for h in horizons]
            upper = [current_price * np.exp(confidence_intervals[h][1]) for h in horizons]
            ax1.fill_between(forecast_dates, lower, upper, alpha=0.2, color='red')

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (COP)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot returns
        ax2.bar(horizons, [forecasts[h] * 100 for h in horizons], color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Horizon (days)')
        ax2.set_ylabel('Expected Return (%)')
        ax2.set_title('Forecasted Returns by Horizon')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved forecast plot to {save_path}")

        plt.close()
        return fig

    def plot_fan_chart(
        self,
        prices: pd.Series,
        forecasts_by_model: Dict[str, Dict[int, float]],
        save_path: Path = None,
        title: str = "Forecast Fan Chart"
    ):
        """
        Plot fan chart showing forecasts from multiple models.

        Args:
            prices: Historical prices
            forecasts_by_model: model_name -> horizon -> forecast
            save_path: Path to save figure
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        current_price = prices.iloc[-1]
        last_date = prices.index[-1]

        # Historical
        ax.plot(prices.index[-60:], prices.iloc[-60:], 'b-', linewidth=1.5, label='Historical')

        # Colors for models
        colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts_by_model)))

        # Plot each model
        for (model_name, forecasts), color in zip(forecasts_by_model.items(), colors):
            horizons = sorted(forecasts.keys())
            forecast_dates = [last_date + pd.Timedelta(days=h) for h in horizons]
            forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

            ax.plot(forecast_dates, forecast_prices, 'o-', color=color,
                   linewidth=1.5, markersize=6, label=model_name.upper(), alpha=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (COP)')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved fan chart to {save_path}")

        plt.close()
        return fig

    def plot_forecast_table(
        self,
        forecasts: Dict[str, Dict[int, float]],
        metrics: Dict[str, Dict[int, Dict]] = None,
        save_path: Path = None,
        title: str = "Forecast Summary Table"
    ):
        """
        Create a visual table of forecasts.

        Args:
            forecasts: model_name -> horizon -> forecast
            metrics: model_name -> horizon -> metrics dict
            save_path: Path to save
            title: Plot title
        """
        if not HAS_MATPLOTLIB:
            return

        # Collect data
        models = list(forecasts.keys())
        horizons = sorted(set(h for f in forecasts.values() for h in f.keys()))

        # Create table data
        cell_text = []
        row_labels = []

        for model in models:
            row = []
            for h in horizons:
                if h in forecasts[model]:
                    ret = forecasts[model][h] * 100
                    direction = "↑" if ret > 0 else "↓" if ret < 0 else "→"
                    row.append(f"{direction} {abs(ret):.2f}%")
                else:
                    row.append("-")
            cell_text.append(row)
            row_labels.append(model.upper())

        # Add metrics row if available
        if metrics:
            for model in models:
                row = []
                for h in horizons:
                    if model in metrics and h in metrics[model]:
                        da = metrics[model][h].get('direction_accuracy', 0) * 100
                        row.append(f"DA: {da:.1f}%")
                    else:
                        row.append("-")
                cell_text.append(row)
                row_labels.append(f"{model.upper()} (DA)")

        # Create figure
        fig, ax = plt.subplots(figsize=(len(horizons) * 1.5 + 2, len(row_labels) * 0.6 + 1))
        ax.axis('tight')
        ax.axis('off')

        col_labels = [f"H={h}" for h in horizons]

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color code
        for i, row in enumerate(cell_text):
            for j, cell in enumerate(row):
                if "↑" in cell:
                    table[(i + 1, j)].set_facecolor('#90EE90')  # Light green
                elif "↓" in cell:
                    table[(i + 1, j)].set_facecolor('#FFB6C1')  # Light red

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved forecast table to {save_path}")

        plt.close()
        return fig

    def plot_signal_dashboard(
        self,
        prices: pd.Series,
        signal: str,
        confidence: float,
        horizon: int,
        forecast_return: float,
        metrics: Dict[str, float] = None,
        save_path: Path = None
    ):
        """
        Create a signal dashboard for investors.

        Args:
            prices: Historical prices
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Signal confidence (0-1)
            horizon: Forecast horizon
            forecast_return: Expected return
            metrics: Model metrics
            save_path: Path to save
        """
        if not HAS_MATPLOTLIB:
            return

        fig = plt.figure(figsize=(14, 8))

        # Grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Signal box
        ax1 = fig.add_subplot(gs[0, 0])
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1,
                     facecolor=signal_colors.get(signal, 'gray'),
                     alpha=0.3))
        ax1.text(0.5, 0.6, signal, ha='center', va='center',
                fontsize=36, fontweight='bold',
                color=signal_colors.get(signal, 'gray'))
        ax1.text(0.5, 0.3, f"Confidence: {confidence*100:.0f}%",
                ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Current Signal', fontsize=12, fontweight='bold')

        # Price chart
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(prices.index[-60:], prices.iloc[-60:], 'b-', linewidth=1.5)
        ax2.axhline(y=prices.iloc[-1], color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Recent Price History', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)

        # Forecast details
        ax3 = fig.add_subplot(gs[1, 0])
        details = [
            f"Horizon: {horizon} days",
            f"Expected Return: {forecast_return*100:.2f}%",
            f"Current Price: {prices.iloc[-1]:,.2f}",
            f"Target Price: {prices.iloc[-1] * np.exp(forecast_return):,.2f}"
        ]
        ax3.text(0.1, 0.8, "\n".join(details), transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax3.axis('off')
        ax3.set_title('Forecast Details', fontsize=12, fontweight='bold')

        # Metrics
        ax4 = fig.add_subplot(gs[1, 1])
        if metrics:
            metrics_text = [
                f"Direction Accuracy: {metrics.get('direction_accuracy', 0)*100:.1f}%",
                f"RMSE: {metrics.get('rmse', 0):.6f}",
                f"R²: {metrics.get('r2', 0):.4f}",
                f"Correlation: {metrics.get('correlation', 0):.4f}"
            ]
            ax4.text(0.1, 0.8, "\n".join(metrics_text), transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        ax4.set_title('Model Performance', fontsize=12, fontweight='bold')

        # Disclaimer
        ax5 = fig.add_subplot(gs[1, 2])
        disclaimer = "DISCLAIMER:\nThis is a model forecast.\nPast performance does not\nguarantee future results.\nUse at your own risk."
        ax5.text(0.5, 0.5, disclaimer, ha='center', va='center',
                fontsize=9, color='gray', style='italic')
        ax5.axis('off')

        plt.suptitle('USD/COP Forecast Dashboard', fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved signal dashboard to {save_path}")

        plt.close()
        return fig

    def plot_model_fan_chart(
        self,
        prices: pd.Series,
        forecasts: Dict[int, float],
        model_name: str,
        historical_volatility: float = None,
        n_history_days: int = 100,
        save_path: Path = None
    ):
        """
        Plot fan chart for a single model with expanding confidence intervals.

        Args:
            prices: Historical price series
            forecasts: Dictionary of horizon -> predicted return
            model_name: Name of the model
            historical_volatility: Historical volatility for CI calculation
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Calculate historical volatility if not provided
        if historical_volatility is None:
            returns = np.log(prices / prices.shift(1)).dropna()
            historical_volatility = returns.std()

        # Plot historical prices with dates on x-axis
        hist_prices = prices.iloc[-n_history_days:]
        ax.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2, label='Historico')

        # Current price marker
        ax.plot(current_date, current_price, 'ko', markersize=12, zorder=5)
        ax.annotate(f'{current_price:,.0f}', xy=(current_date, current_price),
                   xytext=(3, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold')

        # Vertical line at today
        ax.axvline(x=current_date, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Horizontal line at current price
        ax.axhline(y=current_price, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Forecasts with dates
        horizons = sorted(forecasts.keys())
        forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
        forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

        # Plot central prediction
        all_dates = [current_date] + forecast_dates
        all_prices = [current_price] + forecast_prices
        ax.plot(all_dates, all_prices, 'r-o', linewidth=2, markersize=8,
               label='Prediccion Central', zorder=4)

        # Fan chart - expanding confidence intervals
        # Use multiple confidence levels (50%, 70%, 90%, 95%)
        confidence_levels = [0.50, 0.70, 0.90, 0.95]
        z_scores = [0.674, 1.036, 1.645, 1.96]  # Normal distribution z-scores

        # Colors from light to dark for the fan
        colors_upper = ['#90EE90', '#98FB98', '#7CFC00', '#32CD32']  # Greens (upside)
        colors_lower = ['#FFB6C1', '#FFA07A', '#FA8072', '#FF6347']  # Reds (downside)
        alphas = [0.4, 0.35, 0.3, 0.25]

        for i, (conf, z) in enumerate(zip(confidence_levels, z_scores)):
            upper_prices = []
            lower_prices = []

            for h in horizons:
                # Volatility scales with sqrt of time
                vol_h = historical_volatility * np.sqrt(h)
                pred = forecasts[h]

                upper_ret = pred + z * vol_h
                lower_ret = pred - z * vol_h

                upper_prices.append(current_price * np.exp(upper_ret))
                lower_prices.append(current_price * np.exp(lower_ret))

            # Create fan from current price
            fan_dates = [current_date] + forecast_dates
            upper_fan = [current_price] + upper_prices
            lower_fan = [current_price] + lower_prices

            # Fill between upper and central prediction
            ax.fill_between(fan_dates, all_prices, upper_fan,
                          color=colors_upper[i], alpha=alphas[i])
            # Fill between central prediction and lower
            ax.fill_between(fan_dates, lower_fan, all_prices,
                          color=colors_lower[i], alpha=alphas[i])

        # Formatting
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio USD/COP', fontsize=12)
        ax.set_title(f'Forecast Fan Chart - {model_name.upper()}', fontsize=14, fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='Historico'),
            Line2D([0], [0], color='red', marker='o', linewidth=2, label='Prediccion Central'),
            Patch(facecolor='green', alpha=0.3, label='Intervalo Alcista'),
            Patch(facecolor='red', alpha=0.3, label='Intervalo Bajista')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        ax.grid(True, alpha=0.3)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved fan chart to {save_path}")

        plt.close()
        return fig

    def plot_forward_forecast_with_ci(
        self,
        prices: pd.Series,
        forecasts: Dict[int, float],
        model_name: str,
        historical_volatility: float = None,
        confidence_level: float = 0.90,
        n_history_days: int = 70,
        save_path: Path = None
    ):
        """
        Plot forward forecast with single confidence interval band.

        Args:
            prices: Historical price series
            forecasts: Dictionary of horizon -> predicted return
            model_name: Name of the model
            historical_volatility: Historical volatility for CI calculation
            confidence_level: Confidence level for interval (default 90%)
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Calculate historical volatility if not provided
        if historical_volatility is None:
            returns = np.log(prices / prices.shift(1)).dropna()
            historical_volatility = returns.std()

        # Z-score for confidence level
        z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.645)

        # Plot historical prices with dates
        hist_prices = prices.iloc[-n_history_days:]
        ax.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2, label='Historico')

        # Current price marker
        ax.plot(current_date, current_price, 'ko', markersize=12, zorder=5)
        ax.annotate(f'{current_price:,.0f}', xy=(current_date, current_price),
                   xytext=(-40, 5), textcoords='offset points',
                   fontsize=11, fontweight='bold')

        # Vertical line at today
        ax.axvline(x=current_date, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Horizontal line at current price
        ax.axhline(y=current_price, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Forecasts with dates
        horizons = sorted(forecasts.keys())
        forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
        forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

        # Calculate confidence intervals
        upper_prices = []
        lower_prices = []

        for h in horizons:
            vol_h = historical_volatility * np.sqrt(h)
            pred = forecasts[h]

            upper_ret = pred + z_score * vol_h
            lower_ret = pred - z_score * vol_h

            upper_prices.append(current_price * np.exp(upper_ret))
            lower_prices.append(current_price * np.exp(lower_ret))

        # Create smooth confidence band
        fan_dates = [current_date] + forecast_dates
        upper_fan = [current_price] + upper_prices
        lower_fan = [current_price] + lower_prices

        # Fill confidence interval
        ax.fill_between(fan_dates, lower_fan, upper_fan,
                       color='lightblue', alpha=0.5,
                       label=f'{int(confidence_level*100)}% Intervalo')

        # Plot central prediction
        all_dates = [current_date] + forecast_dates
        all_prices = [current_price] + forecast_prices
        ax.plot(all_dates, all_prices, 'r-o', linewidth=2, markersize=8,
               label='Prediccion', zorder=4)

        # Formatting
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio USD/COP', fontsize=12)

        # Title with date and price
        title = f'Forward Forecast USD/COP - {model_name.upper()}\n'
        title += f'Fecha: {current_date.strftime("%Y-%m-%d")} | Precio: {current_price:,.2f}'
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved forward forecast plot to {save_path}")

        plt.close()
        return fig

    def plot_all_models_comparison(
        self,
        prices: pd.Series,
        forecasts_by_model: Dict[str, Dict[int, float]],
        n_history_days: int = 70,
        save_path: Path = None
    ):
        """
        Plot all model forecasts on a single chart for comparison.

        Args:
            prices: Historical price series
            forecasts_by_model: model_name -> horizon -> predicted return
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Plot historical with dates
        hist_prices = prices.iloc[-n_history_days:]
        ax.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2.5, label='Historico')

        # Current price marker
        ax.plot(current_date, current_price, 'ko', markersize=12, zorder=5)
        ax.axvline(x=current_date, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Color palette for models
        model_colors = {
            'ridge': '#E74C3C',      # Red
            'xgboost': '#3498DB',    # Blue
            'lightgbm': '#2ECC71',   # Green
            'catboost': '#9B59B6',   # Purple
            'arima': '#F39C12',      # Orange
            'garch': '#1ABC9C',      # Teal
            'ensemble': '#E91E63'    # Pink
        }

        # Plot each model
        for model_name, forecasts in forecasts_by_model.items():
            if not forecasts:
                continue

            horizons = sorted(forecasts.keys())
            forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
            forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

            all_dates = [current_date] + forecast_dates
            all_prices = [current_price] + forecast_prices

            color = model_colors.get(model_name.lower(), 'gray')
            marker = 's' if model_name.lower() == 'ensemble' else 'o'
            linewidth = 2.5 if model_name.lower() == 'ensemble' else 1.5

            ax.plot(all_dates, all_prices, marker=marker, linewidth=linewidth,
                   color=color, markersize=7, label=model_name.upper(), alpha=0.8)

        # Formatting
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio USD/COP', fontsize=12)
        ax.set_title(f'Comparacion de Modelos - Forward Forecast\n'
                    f'Fecha: {current_date.strftime("%Y-%m-%d")} | Precio Actual: {current_price:,.2f}',
                    fontsize=13, fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.legend(loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved model comparison plot to {save_path}")

        plt.close()
        return fig

    def plot_consensus_all_models(
        self,
        prices: pd.Series,
        forecasts_by_model: Dict[str, Dict[int, float]],
        n_history_days: int = 90,
        save_path: Path = None
    ):
        """
        Plot consensus chart with ALL models + ensemble + average in ONE graph.

        This is the key visualization for comparing all model forecasts and their
        consensus (average) prediction.

        Args:
            prices: Historical price series
            forecasts_by_model: model_name -> horizon -> predicted return
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Plot historical prices
        hist_prices = prices.iloc[-n_history_days:]
        ax.plot(hist_prices.index, hist_prices.values, 'k-', linewidth=2.5,
                label='Historical USD/COP', zorder=10)

        # Mark last historical point
        ax.scatter([current_date], [current_price], color='black', s=100, zorder=11)

        # Vertical line at forecast start
        ax.axvline(x=current_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)

        # Color palette for models
        model_colors = {
            'ridge': '#1f77b4',
            'bayesian_ridge': '#ff7f0e',
            'ard': '#2ca02c',
            'xgboost_pure': '#d62728',
            'lightgbm_pure': '#9467bd',
            'catboost_pure': '#8c564b',
            'hybrid_xgboost': '#e377c2',
            'hybrid_lightgbm': '#7f7f7f',
            'hybrid_catboost': '#bcbd22',
            'ensemble': '#17becf'
        }

        # Collect all forecasts for consensus calculation
        all_forecasts = {}  # horizon -> list of prices

        # Plot each model
        for model_name, forecasts in forecasts_by_model.items():
            if not forecasts:
                continue

            horizons = sorted(forecasts.keys())
            forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
            forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]

            # Store for consensus
            for i, h in enumerate(horizons):
                if h not in all_forecasts:
                    all_forecasts[h] = []
                all_forecasts[h].append(forecast_prices[i])

            # Plot dates with current price as starting point
            plot_dates = [current_date] + forecast_dates
            plot_prices = [current_price] + forecast_prices

            color = model_colors.get(model_name.lower().replace(' ', '_'), '#333333')
            linewidth = 2.5 if 'ensemble' in model_name.lower() else 1.5
            linestyle = '--' if 'pure' in model_name.lower() else '-'
            alpha = 1.0 if 'ensemble' in model_name.lower() else 0.7

            ax.plot(plot_dates, plot_prices, color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, label=model_name.upper(),
                    marker='o', markersize=4)

        # Calculate and plot consensus (average of all models)
        if all_forecasts:
            consensus_horizons = sorted(all_forecasts.keys())
            consensus_dates = [current_date + pd.Timedelta(days=h) for h in consensus_horizons]
            consensus_prices = [np.mean(all_forecasts[h]) for h in consensus_horizons]

            # Min and max for range
            min_prices = [min(all_forecasts[h]) for h in consensus_horizons]
            max_prices = [max(all_forecasts[h]) for h in consensus_horizons]

            # Plot consensus with thick red line
            plot_dates = [current_date] + consensus_dates
            plot_consensus = [current_price] + consensus_prices
            plot_min = [current_price] + min_prices
            plot_max = [current_price] + max_prices

            ax.plot(plot_dates, plot_consensus, color='red', linewidth=4,
                    linestyle='-', label='CONSENSUS (Average)', marker='s',
                    markersize=8, zorder=15)

            # Fill model range
            ax.fill_between(plot_dates, plot_min, plot_max,
                            alpha=0.2, color='red', label='Model Range')

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('USD/COP Exchange Rate', fontsize=12, fontweight='bold')
        ax.set_title('USD/COP Forecast Consensus - All Models\n'
                     f'(Historical {n_history_days} days + 30-day Forecast)',
                     fontsize=14, fontweight='bold')

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add annotation
        if all_forecasts and consensus_prices:
            final_consensus = consensus_prices[-1]
            pct_change = ((final_consensus / current_price) - 1) * 100
            textstr = (f'Last Price: ${current_price:,.2f}\n'
                       f'Consensus H30: ${final_consensus:,.2f}\n'
                       f'Change: {pct_change:+.2f}%')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved consensus plot to {save_path}")

        plt.close()
        return fig

    def plot_single_model_all_horizons(
        self,
        prices: pd.Series,
        forecasts: Dict[int, float],
        model_name: str,
        historical_volatility: float = None,
        n_history_days: int = 70,
        save_path: Path = None
    ):
        """
        Plot a single model's forecasts with ALL horizons, each horizon in a distinct color.

        This is the key visualization showing how a model predicts across different
        time horizons, with each horizon clearly distinguishable by color.

        Args:
            prices: Historical price series
            forecasts: Dictionary of horizon -> predicted return
            model_name: Name of the model
            historical_volatility: Historical volatility for CI calculation
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(16, 8))

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Calculate historical volatility if not provided
        if historical_volatility is None:
            returns = np.log(prices / prices.shift(1)).dropna()
            historical_volatility = returns.std()

        # Plot historical prices with dates
        hist_prices = prices.iloc[-n_history_days:]
        ax.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2.5, label='Historico')

        # Vertical line at today
        ax.axvline(x=current_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Horizontal line at current price
        ax.axhline(y=current_price, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Current price marker
        ax.plot(current_date, current_price, 'ko', markersize=14, zorder=10)
        ax.annotate(f'{current_price:,.0f}', xy=(current_date, current_price),
                   xytext=(-45, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Color palette for horizons - warm to cool (short to long term)
        horizon_colors = {
            1: '#E74C3C',    # Red - immediate
            5: '#F39C12',    # Orange - short term
            10: '#F1C40F',   # Yellow/Gold - transition
            15: '#2ECC71',   # Green - medium term
            22: '#1ABC9C',   # Teal - medium-long
            30: '#9B59B6',   # Purple - long term
        }

        # Fallback color palette for any horizons
        default_colors = plt.cm.viridis(np.linspace(0, 0.9, len(forecasts)))

        # Sort horizons for plotting
        horizons = sorted(forecasts.keys())

        # Plot each horizon with its color
        legend_handles = []
        for i, h in enumerate(horizons):
            ret = forecasts[h]
            forecast_price = current_price * np.exp(ret)
            forecast_date = current_date + pd.Timedelta(days=h)

            # Get color
            color = horizon_colors.get(h, default_colors[i])

            # Calculate confidence interval
            vol_h = historical_volatility * np.sqrt(h)
            z_90 = 1.645
            upper_price = current_price * np.exp(ret + z_90 * vol_h)
            lower_price = current_price * np.exp(ret - z_90 * vol_h)

            # Error bar for confidence interval
            ax.errorbar(forecast_date, forecast_price,
                       yerr=[[forecast_price - lower_price], [upper_price - forecast_price]],
                       fmt='o', color=color, markersize=12, capsize=5, capthick=2,
                       elinewidth=2, markeredgecolor='white', markeredgewidth=2,
                       zorder=5)

            # Annotation with price and return
            change_pct = ret * 100
            sign = '+' if change_pct >= 0 else ''
            ax.annotate(f'{forecast_price:,.0f}\n({sign}{change_pct:.1f}%)',
                       xy=(forecast_date, forecast_price),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=9, ha='left', va='center',
                       color=color, fontweight='bold')

            # Legend handle with date
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                      markersize=10, label=f'H={h} ({forecast_date.strftime("%Y-%m-%d")})')
            )

        # Connect forecasts with a dashed line
        forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
        forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]
        all_dates = [current_date] + forecast_dates
        all_prices = [current_price] + forecast_prices
        ax.plot(all_dates, all_prices, '--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)

        # Formatting
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio USD/COP', fontsize=12)

        title = f'{model_name.upper()} - Forward Forecast por Horizonte\n'
        title += f'Fecha: {current_date.strftime("%Y-%m-%d")} | Precio Actual: {current_price:,.2f}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Custom legend
        hist_handle = Line2D([0], [0], color='blue', linewidth=2.5, label='Historico')
        current_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                               markersize=10, label='Precio Actual')

        all_handles = [hist_handle, current_handle] + legend_handles
        ax.legend(handles=all_handles, loc='upper left', framealpha=0.95,
                 fontsize=9, ncol=2, title='Horizontes de Prediccion')

        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved multi-horizon plot to {save_path}")

        plt.close()
        return fig

    def plot_complete_forward_forecast(
        self,
        prices: pd.Series,
        forecasts: Dict[int, float],
        model_name: str,
        historical_volatility: float = None,
        n_history_days: int = 60,
        save_path: Path = None
    ):
        """
        Create comprehensive forward forecast visualization with PRICES and RETURNS.

        This is the premium visualization showing:
        - Top panel: Price forecast with confidence intervals (fan chart)
        - Middle-left: Returns by horizon (bar chart)
        - Middle-right: Price targets table
        - Bottom: Complete forecast summary

        Args:
            prices: Historical price series
            forecasts: Dictionary of horizon -> predicted return
            model_name: Name of the model
            historical_volatility: Historical volatility for CI calculation
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.4, 1, 0.6],
                     hspace=0.3, wspace=0.25)

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Calculate historical volatility if not provided
        if historical_volatility is None:
            returns = np.log(prices / prices.shift(1)).dropna()
            historical_volatility = returns.std()

        horizons = sorted(forecasts.keys())
        forecast_dates = [current_date + pd.Timedelta(days=h) for h in horizons]
        forecast_prices = [current_price * np.exp(forecasts[h]) for h in horizons]
        forecast_returns = [forecasts[h] * 100 for h in horizons]

        # Z-score for 90% CI
        z_90 = 1.645

        # ============================================================
        # TOP PANEL: PRICE FORECAST WITH FAN CHART (Full width)
        # ============================================================
        ax_price = fig.add_subplot(gs[0, :])

        # Historical prices
        hist_prices = prices.iloc[-n_history_days:]
        ax_price.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2.5,
                     label='Historico')

        # Current price marker
        ax_price.plot(current_date, current_price, 'ko', markersize=14, zorder=10)
        ax_price.annotate(f'{current_price:,.0f}', xy=(current_date, current_price),
                         xytext=(-50, 10), textcoords='offset points',
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        # Vertical line at today
        ax_price.axvline(x=current_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Fan chart with confidence intervals
        confidence_levels = [0.50, 0.70, 0.90]
        z_scores = [0.674, 1.036, 1.645]
        colors_upper = ['#98FB98', '#7CFC00', '#32CD32']
        colors_lower = ['#FFB6C1', '#FA8072', '#FF6347']
        alphas = [0.4, 0.35, 0.3]

        for i, (conf, z) in enumerate(zip(confidence_levels, z_scores)):
            upper_prices = []
            lower_prices = []
            for h in horizons:
                vol_h = historical_volatility * np.sqrt(h)
                pred = forecasts[h]
                upper_prices.append(current_price * np.exp(pred + z * vol_h))
                lower_prices.append(current_price * np.exp(pred - z * vol_h))

            fan_dates = [current_date] + forecast_dates
            upper_fan = [current_price] + upper_prices
            lower_fan = [current_price] + lower_prices
            central = [current_price] + forecast_prices

            ax_price.fill_between(fan_dates, central, upper_fan,
                                 color=colors_upper[i], alpha=alphas[i])
            ax_price.fill_between(fan_dates, lower_fan, central,
                                 color=colors_lower[i], alpha=alphas[i])

        # Central forecast line
        all_dates = [current_date] + forecast_dates
        all_prices = [current_price] + forecast_prices
        ax_price.plot(all_dates, all_prices, 'r-o', linewidth=2.5, markersize=10,
                     label='Prediccion Central', zorder=5)

        # Annotate each forecast point
        for i, (date, price, ret) in enumerate(zip(forecast_dates, forecast_prices, forecast_returns)):
            h = horizons[i]
            sign = '+' if ret >= 0 else ''
            ax_price.annotate(f'H={h}\n{price:,.0f}\n({sign}{ret:.1f}%)',
                            xy=(date, price),
                            xytext=(0, 20 if ret >= 0 else -35), textcoords='offset points',
                            fontsize=9, ha='center', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        ax_price.set_title(f'{model_name.upper()} - Forward Forecast Completo\n'
                          f'Fecha Base: {current_date.strftime("%Y-%m-%d")} | Precio Actual: {current_price:,.2f}',
                          fontsize=14, fontweight='bold')
        ax_price.set_xlabel('Fecha')
        ax_price.set_ylabel('Precio USD/COP')
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2.5, label='Historico'),
            Line2D([0], [0], color='red', marker='o', linewidth=2.5, label='Prediccion'),
            Patch(facecolor='green', alpha=0.3, label='CI Alcista'),
            Patch(facecolor='red', alpha=0.3, label='CI Bajista')
        ]
        ax_price.legend(handles=legend_elements, loc='upper left', framealpha=0.95)
        ax_price.grid(True, alpha=0.3)

        # ============================================================
        # MIDDLE LEFT: RETURNS BAR CHART
        # ============================================================
        ax_returns = fig.add_subplot(gs[1, 0])

        colors_bars = ['green' if r >= 0 else 'red' for r in forecast_returns]
        bars = ax_returns.bar([f'H={h}\n{d.strftime("%m/%d")}' for h, d in zip(horizons, forecast_dates)],
                             forecast_returns, color=colors_bars, alpha=0.8, edgecolor='black')

        ax_returns.axhline(y=0, color='black', linewidth=1)

        # Add value labels on bars
        for bar, ret in zip(bars, forecast_returns):
            height = bar.get_height()
            ax_returns.annotate(f'{ret:+.2f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3 if height >= 0 else -12),
                              textcoords="offset points",
                              ha='center', va='bottom' if height >= 0 else 'top',
                              fontsize=10, fontweight='bold')

        ax_returns.set_title('Retornos Esperados por Horizonte', fontsize=12, fontweight='bold')
        ax_returns.set_xlabel('Horizonte / Fecha')
        ax_returns.set_ylabel('Retorno Esperado (%)')
        ax_returns.grid(True, alpha=0.3, axis='y')

        # ============================================================
        # MIDDLE RIGHT: PRICE TARGETS WITH CI
        # ============================================================
        ax_table = fig.add_subplot(gs[1, 1])
        ax_table.axis('off')

        # Create detailed table
        table_data = []
        for h, date, price, ret in zip(horizons, forecast_dates, forecast_prices, forecast_returns):
            vol_h = historical_volatility * np.sqrt(h)
            lower_90 = current_price * np.exp(forecasts[h] - z_90 * vol_h)
            upper_90 = current_price * np.exp(forecasts[h] + z_90 * vol_h)
            change = price - current_price

            sign_ret = '+' if ret >= 0 else ''
            sign_chg = '+' if change >= 0 else ''

            table_data.append([
                f'H={h}',
                date.strftime('%Y-%m-%d'),
                f'{price:,.0f}',
                f'{sign_ret}{ret:.2f}%',
                f'{sign_chg}{change:,.0f}',
                f'{lower_90:,.0f}',
                f'{upper_90:,.0f}'
            ])

        col_labels = ['Horizonte', 'Fecha', 'Precio', 'Retorno', 'Cambio', 'CI 90% Inf', 'CI 90% Sup']

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style header
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#3498DB')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Color code returns
        for i, ret in enumerate(forecast_returns):
            table[(i + 1, 3)].set_facecolor('#90EE90' if ret > 0.5 else '#FFB6C1' if ret < -0.5 else '#F5F5F5')
            table[(i + 1, 4)].set_facecolor('#90EE90' if ret > 0.5 else '#FFB6C1' if ret < -0.5 else '#F5F5F5')

        ax_table.set_title('Tabla de Predicciones Detallada', fontsize=12, fontweight='bold', pad=20)

        # ============================================================
        # BOTTOM: SUMMARY METRICS
        # ============================================================
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')

        # Calculate summary statistics
        avg_return = np.mean(forecast_returns)
        max_return = max(forecast_returns)
        min_return = min(forecast_returns)
        max_h = horizons[forecast_returns.index(max_return)]
        min_h = horizons[forecast_returns.index(min_return)]

        # Direction signal
        if avg_return > 1:
            signal = "ALCISTA FUERTE"
            signal_color = "green"
        elif avg_return > 0.2:
            signal = "ALCISTA MODERADO"
            signal_color = "lightgreen"
        elif avg_return > -0.2:
            signal = "NEUTRAL"
            signal_color = "gray"
        elif avg_return > -1:
            signal = "BAJISTA MODERADO"
            signal_color = "orange"
        else:
            signal = "BAJISTA FUERTE"
            signal_color = "red"

        summary_text = (
            f"RESUMEN DE PREDICCION | Modelo: {model_name.upper()}\n\n"
            f"Senal: {signal} | Retorno Promedio: {avg_return:+.2f}%\n"
            f"Mejor Horizonte: H={max_h} ({max_return:+.2f}%) | "
            f"Peor Horizonte: H={min_h} ({min_return:+.2f}%)\n"
            f"Volatilidad Diaria: {historical_volatility*100:.2f}% | "
            f"Volatilidad 30d: {historical_volatility*np.sqrt(30)*100:.2f}%"
        )

        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=12, ha='center', va='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved complete forward forecast to {save_path}")

        plt.close()
        return fig

    def plot_horizons_summary_table(
        self,
        forecasts_by_model: Dict[str, Dict[int, float]],
        current_price: float,
        save_path: Path = None
    ):
        """
        Create a visual summary table showing all model forecasts by horizon.

        Args:
            forecasts_by_model: model_name -> horizon -> predicted return
            current_price: Current price for price calculations
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        # Collect all horizons
        all_horizons = sorted(set(h for f in forecasts_by_model.values() for h in f.keys()))
        models = list(forecasts_by_model.keys())

        fig, ax = plt.subplots(figsize=(len(all_horizons) * 2 + 3, len(models) * 0.8 + 2))

        # Create table data
        cell_text = []
        cell_colors = []

        for model in models:
            row_text = []
            row_colors = []
            for h in all_horizons:
                if h in forecasts_by_model[model]:
                    ret = forecasts_by_model[model][h]
                    price = current_price * np.exp(ret)
                    pct = ret * 100

                    # Format cell
                    if pct > 0.5:
                        row_text.append(f'{price:,.0f}\n+{pct:.1f}%')
                        row_colors.append('#90EE90')  # Light green
                    elif pct < -0.5:
                        row_text.append(f'{price:,.0f}\n{pct:.1f}%')
                        row_colors.append('#FFB6C1')  # Light red
                    else:
                        row_text.append(f'{price:,.0f}\n{pct:+.1f}%')
                        row_colors.append('#F5F5F5')  # Light gray
                else:
                    row_text.append('-')
                    row_colors.append('white')

            cell_text.append(row_text)
            cell_colors.append(row_colors)

        ax.axis('off')

        table = ax.table(
            cellText=cell_text,
            rowLabels=[m.upper() for m in models],
            colLabels=[f'H={h}' for h in all_horizons],
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 2.0)

        # Style header
        for j in range(len(all_horizons)):
            table[(0, j)].set_facecolor('#3498DB')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'Resumen de Forecasts por Modelo y Horizonte\nPrecio Actual: {current_price:,.2f}',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved horizons summary table to {save_path}")

        plt.close()
        return fig


    def plot_best_model_per_horizon(
        self,
        prices: pd.Series,
        forecasts_by_model: Dict[str, Dict[int, float]],
        metrics_by_model: Dict[str, Dict[int, Dict]] = None,
        historical_volatility: float = None,
        n_history_days: int = 60,
        save_path: Path = None
    ):
        """
        Plot forward forecast using the BEST model for each horizon.

        This creates a "Best of Best" visualization where each horizon
        uses the prediction from the model with highest DA for that horizon.

        Args:
            prices: Historical price series
            forecasts_by_model: model_name -> horizon -> predicted return
            metrics_by_model: model_name -> horizon -> metrics dict (with 'direction_accuracy')
            historical_volatility: Historical volatility for CI calculation
            n_history_days: Number of historical days to show
            save_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], hspace=0.3, wspace=0.25)

        current_price = prices.iloc[-1]
        current_date = prices.index[-1]

        # Calculate historical volatility if not provided
        if historical_volatility is None:
            returns = np.log(prices / prices.shift(1)).dropna()
            historical_volatility = returns.std()

        # Collect all horizons
        all_horizons = sorted(set(h for f in forecasts_by_model.values() for h in f.keys()))

        # Determine best model per horizon
        best_per_horizon = {}
        for h in all_horizons:
            best_model = None
            best_da = -1

            for model_name, forecasts in forecasts_by_model.items():
                if h not in forecasts:
                    continue

                # Get DA from metrics if available, otherwise use 50%
                da = 50.0
                if metrics_by_model and model_name in metrics_by_model:
                    if h in metrics_by_model[model_name]:
                        da = metrics_by_model[model_name][h].get('direction_accuracy', 0.5) * 100

                if da > best_da:
                    best_da = da
                    best_model = model_name

            if best_model:
                best_per_horizon[h] = {
                    'model': best_model,
                    'forecast': forecasts_by_model[best_model][h],
                    'da': best_da
                }

        # Model colors
        model_colors = {
            'ridge': '#E74C3C',      # Red
            'xgboost': '#3498DB',    # Blue
            'lightgbm': '#2ECC71',   # Green
            'catboost': '#9B59B6',   # Purple
            'ensemble': '#E91E63'    # Pink
        }

        # ============================================================
        # TOP PANEL: PRICE CHART WITH BEST MODELS
        # ============================================================
        ax_price = fig.add_subplot(gs[0, :])

        # Historical prices
        hist_prices = prices.iloc[-n_history_days:]
        ax_price.plot(hist_prices.index, hist_prices.values, 'b-', linewidth=2.5,
                     label='Historico')

        # Current price marker
        ax_price.plot(current_date, current_price, 'ko', markersize=14, zorder=10)
        ax_price.annotate(f'{current_price:,.0f}', xy=(current_date, current_price),
                         xytext=(-50, 10), textcoords='offset points',
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        # Vertical line at today
        ax_price.axvline(x=current_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Plot best model forecast for each horizon
        legend_handles = [Line2D([0], [0], color='blue', linewidth=2.5, label='Historico')]

        forecast_dates = []
        forecast_prices = []
        forecast_colors = []

        for h in all_horizons:
            if h not in best_per_horizon:
                continue

            info = best_per_horizon[h]
            model = info['model']
            ret = info['forecast']
            da = info['da']

            forecast_date = current_date + pd.Timedelta(days=h)
            forecast_price = current_price * np.exp(ret)
            color = model_colors.get(model.lower(), 'gray')

            forecast_dates.append(forecast_date)
            forecast_prices.append(forecast_price)
            forecast_colors.append(color)

            # Calculate CI
            vol_h = historical_volatility * np.sqrt(h)
            z_90 = 1.645
            upper = current_price * np.exp(ret + z_90 * vol_h)
            lower = current_price * np.exp(ret - z_90 * vol_h)

            # Error bar
            ax_price.errorbar(forecast_date, forecast_price,
                            yerr=[[forecast_price - lower], [upper - forecast_price]],
                            fmt='o', color=color, markersize=14, capsize=6, capthick=2,
                            elinewidth=2, markeredgecolor='white', markeredgewidth=2,
                            zorder=5)

            # Annotation
            ret_pct = ret * 100
            sign = '+' if ret_pct >= 0 else ''
            ax_price.annotate(
                f'H={h}\n{model.upper()}\n{forecast_price:,.0f}\n({sign}{ret_pct:.1f}%)\nDA:{da:.0f}%',
                xy=(forecast_date, forecast_price),
                xytext=(0, 30 if ret_pct >= 0 else -50), textcoords='offset points',
                fontsize=8, ha='center', fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                         edgecolor=color, linewidth=2)
            )

        # Connect with dashed line
        all_dates = [current_date] + forecast_dates
        all_prices = [current_price] + forecast_prices
        ax_price.plot(all_dates, all_prices, '--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)

        ax_price.set_title(f'BEST MODEL PER HORIZON - Forward Forecast\n'
                          f'Fecha: {current_date.strftime("%Y-%m-%d")} | Precio Actual: {current_price:,.2f}',
                          fontsize=14, fontweight='bold')
        ax_price.set_xlabel('Fecha')
        ax_price.set_ylabel('Precio USD/COP')
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Create legend for models used
        used_models = set(best_per_horizon[h]['model'] for h in best_per_horizon)
        for model in used_models:
            color = model_colors.get(model.lower(), 'gray')
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                      markersize=12, markeredgecolor='white', markeredgewidth=2,
                      label=model.upper())
            )
        ax_price.legend(handles=legend_handles, loc='upper left', framealpha=0.95)
        ax_price.grid(True, alpha=0.3)

        # ============================================================
        # BOTTOM LEFT: RETURNS BAR CHART
        # ============================================================
        ax_returns = fig.add_subplot(gs[1, 0])

        horizons = list(best_per_horizon.keys())
        returns = [best_per_horizon[h]['forecast'] * 100 for h in horizons]
        colors = [model_colors.get(best_per_horizon[h]['model'].lower(), 'gray') for h in horizons]
        labels = [f"H={h}\n{best_per_horizon[h]['model'][:4].upper()}" for h in horizons]

        bars = ax_returns.bar(labels, returns, color=colors, alpha=0.8, edgecolor='black')
        ax_returns.axhline(y=0, color='black', linewidth=1)

        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax_returns.annotate(f'{ret:+.2f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3 if height >= 0 else -12),
                              textcoords="offset points",
                              ha='center', va='bottom' if height >= 0 else 'top',
                              fontsize=10, fontweight='bold')

        ax_returns.set_title('Retornos por Horizonte (Mejor Modelo)', fontsize=12, fontweight='bold')
        ax_returns.set_xlabel('Horizonte / Modelo')
        ax_returns.set_ylabel('Retorno Esperado (%)')
        ax_returns.grid(True, alpha=0.3, axis='y')

        # ============================================================
        # BOTTOM RIGHT: SUMMARY TABLE
        # ============================================================
        ax_table = fig.add_subplot(gs[1, 1])
        ax_table.axis('off')

        table_data = []
        for h in horizons:
            info = best_per_horizon[h]
            model = info['model']
            ret = info['forecast']
            da = info['da']
            forecast_date = current_date + pd.Timedelta(days=h)
            forecast_price = current_price * np.exp(ret)

            ret_pct = ret * 100
            sign = '+' if ret_pct >= 0 else ''

            table_data.append([
                f'H={h}',
                forecast_date.strftime('%Y-%m-%d'),
                model.upper(),
                f'{da:.1f}%',
                f'{forecast_price:,.0f}',
                f'{sign}{ret_pct:.2f}%'
            ])

        col_labels = ['Horizonte', 'Fecha', 'Mejor Modelo', 'DA', 'Precio', 'Retorno']

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style header
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#2C3E50')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Color code by model
        for i, h in enumerate(horizons):
            model = best_per_horizon[h]['model']
            color = model_colors.get(model.lower(), 'gray')
            table[(i + 1, 2)].set_facecolor(color)
            table[(i + 1, 2)].set_text_props(color='white', fontweight='bold')

            # Color returns
            ret = best_per_horizon[h]['forecast'] * 100
            table[(i + 1, 5)].set_facecolor('#90EE90' if ret > 0.5 else '#FFB6C1' if ret < -0.5 else '#F5F5F5')

        ax_table.set_title('Resumen: Mejor Modelo por Horizonte', fontsize=12, fontweight='bold', pad=20)

        plt.suptitle('FORWARD FORECAST - BEST MODEL PER HORIZON',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved best model per horizon plot to {save_path}")

        plt.close()
        return fig


def generate_all_forecast_plots(
    prices: pd.Series,
    forecasts_by_model: Dict[str, Dict[int, float]],
    ensemble_forecasts: Dict[int, float] = None,
    all_ensembles: Dict[str, Dict[int, float]] = None,
    output_dir: Path = None,
    historical_volatility: float = None,
    metrics_by_model: Dict[str, Dict[int, Dict]] = None
):
    """
    Generate all forward forecast visualizations.

    Args:
        prices: Historical price series
        forecasts_by_model: model_name -> horizon -> predicted return
        ensemble_forecasts: Single ensemble forecasts (horizon -> return) - backward compatible
        all_ensembles: Dict of ensemble_name -> (horizon -> return) for multiple ensembles
                       Keys: 'best_of_breed', 'top_3', 'top_6_mean'
        output_dir: Directory to save plots
        historical_volatility: Volatility for CI calculation

    Returns:
        List of generated file paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available")
        return []

    output_dir = Path(output_dir or 'forecasts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate volatility if not provided
    if historical_volatility is None:
        returns = np.log(prices / prices.shift(1)).dropna()
        historical_volatility = returns.std()

    plotter = ForecastPlotter(figsize=(14, 7), dpi=150)
    generated_files = []

    # Generate COMPLETE forward forecast for each model (PRICES + RETURNS)
    for model_name, forecasts in forecasts_by_model.items():
        if not forecasts:
            continue

        # COMPLETE forward forecast (precios + retornos en una sola imagen)
        complete_path = output_dir / f'complete_forecast_{model_name.lower()}.png'
        plotter.plot_complete_forward_forecast(
            prices=prices,
            forecasts=forecasts,
            model_name=model_name,
            historical_volatility=historical_volatility,
            save_path=complete_path
        )
        generated_files.append(complete_path)

        # Fan chart with multiple confidence levels
        fan_path = output_dir / f'fan_chart_{model_name.lower()}.png'
        plotter.plot_model_fan_chart(
            prices=prices,
            forecasts=forecasts,
            model_name=model_name,
            historical_volatility=historical_volatility,
            save_path=fan_path
        )
        generated_files.append(fan_path)

        # Forward forecast with 90% CI
        forward_path = output_dir / f'forward_forecast_{model_name.lower()}.png'
        plotter.plot_forward_forecast_with_ci(
            prices=prices,
            forecasts=forecasts,
            model_name=model_name,
            historical_volatility=historical_volatility,
            confidence_level=0.90,
            save_path=forward_path
        )
        generated_files.append(forward_path)

    # Generate ensemble plots if available (single ensemble - backward compatible)
    if ensemble_forecasts:
        # COMPLETE ensemble forecast
        complete_ensemble_path = output_dir / 'complete_forecast_ensemble.png'
        plotter.plot_complete_forward_forecast(
            prices=prices,
            forecasts=ensemble_forecasts,
            model_name='Ensemble',
            historical_volatility=historical_volatility,
            save_path=complete_ensemble_path
        )
        generated_files.append(complete_ensemble_path)

        # Ensemble fan chart
        fan_path = output_dir / 'fan_chart_ensemble.png'
        plotter.plot_model_fan_chart(
            prices=prices,
            forecasts=ensemble_forecasts,
            model_name='Ensemble',
            historical_volatility=historical_volatility,
            save_path=fan_path
        )
        generated_files.append(fan_path)

        # Ensemble forward forecast
        forward_path = output_dir / 'forward_forecast_ensemble.png'
        plotter.plot_forward_forecast_with_ci(
            prices=prices,
            forecasts=ensemble_forecasts,
            model_name='Ensemble',
            historical_volatility=historical_volatility,
            confidence_level=0.90,
            save_path=forward_path
        )
        generated_files.append(forward_path)

    # Generate plots for each ensemble type (best_of_breed, top_3, top_6_mean)
    ensemble_display_names = {
        'best_of_breed': 'Best of Breed',
        'top_3': 'Top 3 Average',
        'top_6_mean': 'Top 6 Average'
    }

    if all_ensembles:
        for ensemble_name, forecasts in all_ensembles.items():
            if not forecasts:
                continue

            display_name = ensemble_display_names.get(ensemble_name, ensemble_name)

            # Forward forecast for this ensemble type
            forward_path = output_dir / f'forward_forecast_ensemble_{ensemble_name}.png'
            plotter.plot_forward_forecast_with_ci(
                prices=prices,
                forecasts=forecasts,
                model_name=display_name,
                historical_volatility=historical_volatility,
                confidence_level=0.90,
                save_path=forward_path,
                n_history_days=90
            )
            generated_files.append(forward_path)
            logger.info(f"Generated {forward_path}")

    # All models comparison
    all_forecasts = dict(forecasts_by_model)
    if ensemble_forecasts:
        all_forecasts['ensemble'] = ensemble_forecasts

    comparison_path = output_dir / 'models_comparison.png'
    plotter.plot_all_models_comparison(
        prices=prices,
        forecasts_by_model=all_forecasts,
        save_path=comparison_path
    )
    generated_files.append(comparison_path)

    # CONSENSUS chart - All models + ensemble + average in ONE graph
    consensus_path = output_dir / 'forward_forecast_consensus_all_models.png'
    plotter.plot_consensus_all_models(
        prices=prices,
        forecasts_by_model=all_forecasts,
        n_history_days=90,
        save_path=consensus_path
    )
    generated_files.append(consensus_path)

    # Multi-horizon plots for each model (shows all horizons with distinct colors)
    for model_name, forecasts in forecasts_by_model.items():
        if not forecasts:
            continue

        multi_horizon_path = output_dir / f'horizons_{model_name.lower()}.png'
        plotter.plot_single_model_all_horizons(
            prices=prices,
            forecasts=forecasts,
            model_name=model_name,
            historical_volatility=historical_volatility,
            save_path=multi_horizon_path
        )
        generated_files.append(multi_horizon_path)

    # Ensemble multi-horizon plot
    if ensemble_forecasts:
        ensemble_horizons_path = output_dir / 'horizons_ensemble.png'
        plotter.plot_single_model_all_horizons(
            prices=prices,
            forecasts=ensemble_forecasts,
            model_name='Ensemble',
            historical_volatility=historical_volatility,
            save_path=ensemble_horizons_path
        )
        generated_files.append(ensemble_horizons_path)

    # Summary table of all forecasts
    summary_table_path = output_dir / 'forecasts_summary_table.png'
    plotter.plot_horizons_summary_table(
        forecasts_by_model=all_forecasts,
        current_price=prices.iloc[-1],
        save_path=summary_table_path
    )
    generated_files.append(summary_table_path)

    # BEST MODEL PER HORIZON - Premium visualization
    best_per_horizon_path = output_dir / 'best_model_per_horizon.png'
    plotter.plot_best_model_per_horizon(
        prices=prices,
        forecasts_by_model=forecasts_by_model,
        metrics_by_model=metrics_by_model,
        historical_volatility=historical_volatility,
        save_path=best_per_horizon_path
    )
    generated_files.append(best_per_horizon_path)

    logger.info(f"Generated {len(generated_files)} forecast plots in {output_dir}")
    return generated_files
