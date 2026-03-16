"""
Macro Analyzer (SDD-07 §3)
============================
Computes technical indicators (SMA, Bollinger, RSI, MACD, ROC) for macro variables.
Reads from existing macro_indicators_daily table (populated by L0 DAGs).

Reuses Wilder's RSI pattern from src/features/calculator_registry.py.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.contracts.analysis_schema import MacroSnapshot, KEY_MACRO_VARIABLES, DISPLAY_NAMES

logger = logging.getLogger(__name__)


class MacroAnalyzer:
    """Computes technical indicators for macro variables.

    Reads macro data from:
    - PostgreSQL macro_indicators_daily (production)
    - Parquet file (fallback/local dev)
    """

    def __init__(
        self,
        sma_periods: list[int] = None,
        bollinger_period: int = 20,
        bollinger_std: int = 2,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        roc_periods: list[int] = None,
        lookback_days: int = 90,
    ):
        self.sma_periods = sma_periods or [5, 10, 20, 50]
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.roc_periods = roc_periods or [5, 20]
        self.lookback_days = lookback_days

    def compute_snapshot(
        self,
        series: pd.Series,
        variable_key: str,
        as_of_date: date,
    ) -> Optional[MacroSnapshot]:
        """Compute all indicators for a single variable as of a given date.

        Args:
            series: Time series indexed by date (at least lookback_days of data)
            variable_key: Variable identifier (e.g., 'dxy', 'vix')
            as_of_date: Date to compute indicators for

        Returns:
            MacroSnapshot with all indicators, or None if insufficient data.
        """
        if series.empty or len(series) < self.macd_slow + self.macd_signal:
            logger.warning(f"Insufficient data for {variable_key}: {len(series)} rows")
            return None

        # Get current value
        try:
            current_value = float(series.loc[:as_of_date].iloc[-1])
        except (IndexError, KeyError):
            return None

        # SMAs
        smas = {}
        for period in self.sma_periods:
            sma = series.rolling(window=period, min_periods=1).mean()
            smas[period] = _safe_float(sma.loc[:as_of_date].iloc[-1])

        # Bollinger Bands
        bb = self.compute_bollinger_bands(series, self.bollinger_period, self.bollinger_std)
        bb_upper = _safe_float(bb["upper"].loc[:as_of_date].iloc[-1])
        bb_lower = _safe_float(bb["lower"].loc[:as_of_date].iloc[-1])
        bb_width = None
        if bb_upper is not None and bb_lower is not None and smas.get(20):
            bb_width = (bb_upper - bb_lower) / smas[20] if smas[20] != 0 else None

        # RSI (Wilder's)
        rsi = self.compute_rsi(series, self.rsi_period)
        rsi_val = _safe_float(rsi.loc[:as_of_date].iloc[-1])

        # MACD
        macd = self.compute_macd(series, self.macd_fast, self.macd_slow, self.macd_signal)
        macd_line = _safe_float(macd["macd_line"].loc[:as_of_date].iloc[-1])
        macd_signal_val = _safe_float(macd["signal"].loc[:as_of_date].iloc[-1])
        macd_hist = _safe_float(macd["histogram"].loc[:as_of_date].iloc[-1])

        # ROC
        roc = {}
        for period in self.roc_periods:
            roc_series = self.compute_roc(series, period)
            roc[period] = _safe_float(roc_series.loc[:as_of_date].iloc[-1])

        # Z-score (20-period)
        z_score = self._compute_z_score(series, 20, as_of_date)

        # Trend
        trend = self._compute_trend(current_value, smas, series, as_of_date)

        # Signal
        signal = self._compute_signal(rsi_val, bb_upper, bb_lower, current_value, bb_width)

        return MacroSnapshot(
            snapshot_date=as_of_date,
            variable_key=variable_key,
            variable_name=DISPLAY_NAMES.get(variable_key, variable_key),
            value=current_value,
            sma_5=smas.get(5),
            sma_10=smas.get(10),
            sma_20=smas.get(20),
            sma_50=smas.get(50),
            bollinger_upper_20=bb_upper,
            bollinger_lower_20=bb_lower,
            bollinger_width_20=bb_width,
            rsi_14=rsi_val,
            macd_line=macd_line,
            macd_signal=macd_signal_val,
            macd_histogram=macd_hist,
            roc_5=roc.get(5),
            roc_20=roc.get(20),
            z_score_20=z_score,
            trend=trend,
            signal=signal,
        )

    def compute_all_snapshots(
        self,
        macro_df: pd.DataFrame,
        as_of_date: date,
        variables: Optional[list[str]] = None,
    ) -> dict[str, MacroSnapshot]:
        """Compute snapshots for all key macro variables.

        Args:
            macro_df: DataFrame with date index and variable columns
            as_of_date: Date to compute indicators for
            variables: List of variable keys (defaults to KEY_MACRO_VARIABLES)

        Returns:
            Dict of {variable_key: MacroSnapshot}
        """
        variables = variables or list(KEY_MACRO_VARIABLES)
        snapshots = {}

        for var_key in variables:
            # Map variable key to column name
            col = self._find_column(macro_df, var_key)
            if col is None:
                logger.debug(f"Column not found for {var_key}")
                continue

            series = macro_df[col].dropna()
            if series.empty:
                continue

            snapshot = self.compute_snapshot(series, var_key, as_of_date)
            if snapshot:
                snapshots[var_key] = snapshot

        logger.info(f"Computed {len(snapshots)} macro snapshots for {as_of_date}")
        return snapshots

    def get_chart_data(
        self,
        series: pd.Series,
        variable_key: str,
        end_date: date,
        lookback_days: Optional[int] = None,
    ) -> list[dict]:
        """Get time series data formatted for Recharts frontend.

        Returns list of dicts: [{date, value, sma20, bb_upper, bb_lower, rsi}, ...]
        """
        lookback = lookback_days or self.lookback_days
        start_date = end_date - timedelta(days=lookback)

        # Filter to date range
        mask = (series.index >= pd.Timestamp(start_date)) & (series.index <= pd.Timestamp(end_date))
        filtered = series[mask]

        if filtered.empty:
            return []

        # Compute indicators on full series, then slice
        sma20 = series.rolling(window=20, min_periods=1).mean()
        bb = self.compute_bollinger_bands(series, 20, 2)
        rsi = self.compute_rsi(series, 14)

        points = []
        for dt in filtered.index:
            points.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "value": _safe_float(filtered.get(dt)),
                "sma20": _safe_float(sma20.get(dt)),
                "bb_upper": _safe_float(bb["upper"].get(dt)),
                "bb_lower": _safe_float(bb["lower"].get(dt)),
                "rsi": _safe_float(rsi.get(dt)),
            })

        return points

    # ------------------------------------------------------------------
    # Technical indicator computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: int = 2,
    ) -> dict:
        """Compute Bollinger Bands."""
        sma = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        return {
            "middle": sma,
            "upper": sma + num_std * std,
            "lower": sma - num_std * std,
        }

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using Wilder's EMA (alpha=1/period).

        Identical to src/features/calculator_registry.py:calculate_rsi_wilders.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.clip(lower=1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.clip(0, 100)

    @staticmethod
    def compute_macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        """Compute MACD line, signal line, and histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd_line": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def compute_roc(series: pd.Series, period: int = 5) -> pd.Series:
        """Compute Rate of Change (%)."""
        shifted = series.shift(period)
        return ((series - shifted) / shifted.clip(lower=1e-10)) * 100

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------

    def _compute_z_score(
        self,
        series: pd.Series,
        period: int,
        as_of_date: date,
    ) -> Optional[float]:
        """Compute z-score vs rolling mean/std."""
        rolling_mean = series.rolling(window=period, min_periods=1).mean()
        rolling_std = series.rolling(window=period, min_periods=1).std()
        try:
            val = float(series.loc[:as_of_date].iloc[-1])
            mean = float(rolling_mean.loc[:as_of_date].iloc[-1])
            std = float(rolling_std.loc[:as_of_date].iloc[-1])
            if std > 0:
                return round((val - mean) / std, 4)
        except (IndexError, KeyError):
            pass
        return None

    @staticmethod
    def _compute_trend(
        value: float,
        smas: dict,
        series: pd.Series,
        as_of_date: date,
    ) -> str:
        """Detect trend signal."""
        sma20 = smas.get(20)
        sma50 = smas.get(50)

        if sma20 is not None and sma50 is not None:
            # Check for golden/death cross (SMA20 crossing SMA50)
            sma20_series = series.rolling(window=20, min_periods=1).mean()
            sma50_series = series.rolling(window=50, min_periods=1).mean()
            try:
                prev_diff = float(sma20_series.loc[:as_of_date].iloc[-2]) - float(sma50_series.loc[:as_of_date].iloc[-2])
                curr_diff = sma20 - sma50
                if prev_diff < 0 and curr_diff >= 0:
                    return "golden_cross"
                elif prev_diff > 0 and curr_diff <= 0:
                    return "death_cross"
            except (IndexError, KeyError):
                pass

        if sma20 is not None:
            return "above_sma20" if value > sma20 else "below_sma20"

        return "neutral"

    @staticmethod
    def _compute_signal(
        rsi: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        value: float,
        bb_width: Optional[float],
    ) -> str:
        """Compute composite signal."""
        if rsi is not None:
            if rsi >= 70:
                return "overbought"
            elif rsi <= 30:
                return "oversold"

        if bb_upper is not None and bb_lower is not None:
            if value >= bb_upper:
                return "bb_upper_touch"
            elif value <= bb_lower:
                return "bb_lower_touch"
            if bb_width is not None and bb_width < 0.02:
                return "bb_squeeze"

        return "neutral"

    @staticmethod
    def _find_column(df: pd.DataFrame, variable_key: str) -> Optional[str]:
        """Find the column name in macro DataFrame for a variable key."""
        # Direct match
        if variable_key in df.columns:
            return variable_key

        # Common mappings (short names + SSOT long names from MACRO_DAILY_CLEAN.parquet)
        mappings = {
            "dxy": ["dxy_close", "dxy", "FXRT_INDEX_DXY_USA_D_DXY"],
            "vix": ["vix_close", "vix", "VOLT_VIX_USA_D_VIX"],
            "wti": ["wti_close", "oil_close", "wti", "COMM_OIL_WTI_GLB_D_WTI"],
            "embi_col": ["embi_col", "embi_close", "embi", "CRSK_SPREAD_EMBI_COL_D_EMBI"],
            "ust10y": ["ust10y", "us_10y_yield", "FINC_BOND_YIELD10Y_USA_D_UST10Y"],
            "ust2y": ["ust2y", "us_2y_yield", "FINC_BOND_YIELD2Y_USA_D_DGS2"],
            "ibr": ["ibr", "ibr_overnight", "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"],
            "tpm": ["tpm", "tasa_politica", "POLR_POLICY_RATE_COL_M_TPM"],
            "fedfunds": ["fedfunds", "fed_funds_rate", "POLR_PRIME_RATE_USA_D_PRIME"],
            "gold": ["gold_close", "gold", "COMM_METAL_GOLD_GLB_D_GOLD"],
            "brent": ["brent_close", "brent", "COMM_OIL_BRENT_GLB_D_BRENT"],
            "cpi_us": ["cpi_us", "us_cpi"],
            "cpi_col": ["cpi_col", "col_cpi"],
        }
        for alt in mappings.get(variable_key, []):
            if alt in df.columns:
                return alt
        return None


def _safe_float(val) -> Optional[float]:
    """Convert to float safely, returning None for NaN/inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    except (ValueError, TypeError):
        return None
