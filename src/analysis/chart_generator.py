"""
Chart Generator (SDD-07 §5)
==============================
Generates matplotlib PNG charts for macro variables.
Dark background (#0f172a), 90-day lookback, SMA + Bollinger + RSI overlay.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_variable_chart(
    series: pd.Series,
    variable_key: str,
    variable_name: str,
    end_date: date,
    output_dir: str,
    week_label: str = "",
    lookback_days: int = 90,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> str | None:
    """Generate a PNG chart for a single macro variable.

    Chart includes:
    - Main plot: Price line + SMA-20 (dashed) + Bollinger bands (shaded)
    - Bottom subplot: RSI (14-period) with overbought/oversold zones

    Returns:
        Path to generated PNG, or None if generation fails.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed — pip install matplotlib")
        return None

    if series.empty:
        return None

    start_date = end_date - timedelta(days=lookback_days)
    mask = (series.index >= pd.Timestamp(start_date)) & (series.index <= pd.Timestamp(end_date))
    data = series[mask].dropna()

    if len(data) < 20:
        logger.warning(f"Insufficient data for {variable_key} chart ({len(data)} points)")
        return None

    # Compute indicators
    sma20 = series.rolling(window=20, min_periods=1).mean()[mask].dropna()
    std20 = series.rolling(window=20, min_periods=1).std()[mask].dropna()
    bb_upper = (sma20 + 2 * std20)
    bb_lower = (sma20 - 2 * std20)

    # RSI
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi = (100.0 - (100.0 / (1.0 + rs)))[mask].dropna()

    # Create figure
    bg_color = "#0f172a"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi,
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=bg_color,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(bg_color)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, alpha=0.15, color="#334155")

    # Main plot: Price + SMA + Bollinger
    ax1.plot(data.index, data.values, color="white", linewidth=1.5, label=variable_name)
    ax1.plot(sma20.index, sma20.values, color="#22d3ee", linewidth=1, linestyle="--", alpha=0.8, label="SMA 20")
    ax1.fill_between(
        bb_upper.index, bb_upper.values, bb_lower.values,
        alpha=0.1, color="#64748b", label="Bollinger (2σ)",
    )
    ax1.set_title(
        f"{variable_name}  |  {week_label}" if week_label else variable_name,
        color="white", fontsize=12, fontweight="bold", pad=10,
    )
    ax1.legend(loc="upper left", fontsize=7, facecolor=bg_color, edgecolor="#334155", labelcolor="white")
    ax1.yaxis.label.set_color("white")

    # RSI subplot
    ax2.plot(rsi.index, rsi.values, color="#a78bfa", linewidth=1)
    ax2.axhline(y=70, color="#ef4444", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.axhline(y=30, color="#22c55e", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.fill_between(rsi.index, 70, rsi.values, where=rsi.values >= 70, alpha=0.2, color="#ef4444")
    ax2.fill_between(rsi.index, 30, rsi.values, where=rsi.values <= 30, alpha=0.2, color="#22c55e")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI 14", color="white", fontsize=8)

    # Format x-axis
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"macro_{variable_key}_{week_label}.png" if week_label else f"macro_{variable_key}.png"
    filepath = output_path / filename

    fig.savefig(filepath, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated chart: {filepath}")
    return str(filepath)


def generate_all_charts(
    macro_df: pd.DataFrame,
    variables: dict,
    end_date: date,
    output_dir: str,
    week_label: str = "",
) -> dict[str, str]:
    """Generate charts for multiple variables.

    Args:
        macro_df: DataFrame with date index and variable columns
        variables: Dict of {variable_key: column_name}
        end_date: End date for chart
        output_dir: Directory to save PNGs
        week_label: Label for filename (e.g., "2026-W09")

    Returns:
        Dict of {variable_key: filepath}
    """
    charts = {}
    for var_key, col_name in variables.items():
        if col_name not in macro_df.columns:
            continue
        series = macro_df[col_name].dropna()
        if series.empty:
            continue

        from src.contracts.analysis_schema import DISPLAY_NAMES
        name = DISPLAY_NAMES.get(var_key, var_key)

        path = generate_variable_chart(
            series=series,
            variable_key=var_key,
            variable_name=name,
            end_date=end_date,
            output_dir=output_dir,
            week_label=week_label,
        )
        if path:
            charts[var_key] = path

    return charts
