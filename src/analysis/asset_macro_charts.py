"""Shared macro-driver snapshot/chart builder for the multi-asset analysis.

Reads the GLOBAL macro CLEAN parquet (DXY / VIX / WTI / Gold / UST10Y / Brent —
all global market drivers, none COP-specific) and produces the SAME
``macro_snapshots`` / ``macro_charts`` blocks the USD/COP weekly analysis emits.
That way Gold and BTC render the identical /analysis modules (MacroSnapshotBar,
UnifiedMacroChart, MacroChartGrid) as USD/COP — real data only, no fabrication,
no DB dependency (reads the git-tracked parquet).

Contract shapes (see lib/contracts/weekly-analysis.contract.ts):
  macro_charts[key]    = { png_url: null, data: [MacroChartPoint...] }
  macro_snapshots[key] = MacroVariableSnapshot (value required, non-null)
  MacroChartPoint      = { date, value, sma20, bb_upper, bb_lower, rsi }
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MACRO_CLEAN = PROJECT_ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"

# macro key -> (CLEAN column name, frontend display name). The keys match the
# ones MacroSnapshotBar/MacroChartGrid already know (DISPLAY_NAMES/KEY_VARIABLES).
_MACRO_COLS: dict[str, tuple[str, str]] = {
    "dxy":    ("FXRT_INDEX_DXY_USA_D_DXY",        "DXY (Dollar Index)"),
    "vix":    ("VOLT_VIX_USA_D_VIX",              "VIX (Volatilidad)"),
    "wti":    ("COMM_OIL_WTI_GLB_D_WTI",          "WTI Petroleo"),
    "brent":  ("COMM_OIL_BRENT_GLB_D_BRENT",      "Brent"),
    "gold":   ("COMM_METAL_GOLD_GLB_D_GOLD",      "Oro"),
    "ust10y": ("FINC_BOND_YIELD10Y_USA_D_UST10Y", "US Treasury 10Y"),
}

# Per-asset macro drivers (subset above; ordered by relevance). All are global.
# Gold leads with its own spot (gold) + real-rate/dollar drivers; BTC leads with
# the dollar + risk sentiment (VIX) + gold as a store-of-value comparator.
ASSET_DRIVERS: dict[str, list[str]] = {
    "xauusd":  ["gold", "dxy", "ust10y", "vix", "wti", "brent"],
    "btcusdt": ["dxy", "vix", "gold", "ust10y", "wti", "brent"],
}

_CHART_WINDOW = 120   # daily points in each mini-chart series (~6 months)

_macro_cache: dict[str, pd.DataFrame] = {}


def _load_macro() -> pd.DataFrame:
    if "df" not in _macro_cache:
        if not _MACRO_CLEAN.exists():
            raise FileNotFoundError(f"macro CLEAN parquet not found: {_MACRO_CLEAN}")
        df = pd.read_parquet(_MACRO_CLEAN)
        df.index = pd.to_datetime(df.index)
        _macro_cache["df"] = df.sort_index()
    return _macro_cache["df"]


def _round(v: Any, n: int = 6) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if (np.isnan(f) or np.isinf(f)) else round(f, n)


def _series_indicators(s: pd.Series) -> pd.DataFrame:
    """Rolling SMA20 / Bollinger(20,2) / Wilder-RSI(14) over the whole series."""
    out = pd.DataFrame({"value": s})
    out["sma20"] = s.rolling(20, min_periods=5).mean()
    std20 = s.rolling(20, min_periods=5).std(ddof=0)
    out["bb_upper"] = out["sma20"] + 2 * std20
    out["bb_lower"] = out["sma20"] - 2 * std20
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).where(avg_loss != 0, 100.0)
    return out


def _snapshot(key: str, name: str, hist: pd.Series, ind: pd.DataFrame) -> dict[str, Any] | None:
    """Full MacroVariableSnapshot for the last observation up to the week end."""
    if hist.empty:
        return None
    last_val = hist.iloc[-1]
    if pd.isna(last_val):
        return None
    tail = hist.tail(60)

    def sma(n: int) -> float | None:
        return _round(hist.tail(n).mean()) if len(hist) >= max(2, n // 4) else None

    mean20, std20 = hist.tail(20).mean(), hist.tail(20).std(ddof=0)
    ema12 = hist.ewm(span=12, adjust=False).mean()
    ema26 = hist.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    rsi = ind["rsi"].iloc[-1]
    sma20 = ind["sma20"].iloc[-1]
    z20 = (last_val - mean20) / std20 if std20 and not pd.isna(std20) and std20 != 0 else None

    def roc(n: int) -> float | None:
        if len(hist) <= n or pd.isna(hist.iloc[-n - 1]) or hist.iloc[-n - 1] == 0:
            return None
        return _round((last_val / hist.iloc[-n - 1] - 1) * 100, 6)

    trend = None
    if not pd.isna(sma20):
        trend = "above_sma20" if last_val >= sma20 else "below_sma20"
    signal = None
    if not pd.isna(rsi):
        signal = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"

    return {
        "snapshot_date": str(hist.index[-1].date()),
        "variable_key": key,
        "variable_name": name,
        "value": _round(last_val, 4),
        "sma_5": sma(5),
        "sma_10": sma(10),
        "sma_20": _round(sma20, 6),
        "sma_50": sma(50),
        "bollinger_upper_20": _round(ind["bb_upper"].iloc[-1]),
        "bollinger_lower_20": _round(ind["bb_lower"].iloc[-1]),
        "bollinger_width_20": _round((ind["bb_upper"].iloc[-1] - ind["bb_lower"].iloc[-1]) / mean20)
        if mean20 else None,
        "rsi_14": _round(rsi),
        "macd_line": _round(macd.iloc[-1]),
        "macd_signal": _round(macd_sig.iloc[-1]),
        "macd_histogram": _round(macd.iloc[-1] - macd_sig.iloc[-1]),
        "roc_5": roc(5),
        "roc_20": roc(20),
        "z_score_20": _round(z20, 4),
        "z_score_60": _round((last_val - tail.mean()) / tail.std(ddof=0), 4)
        if tail.std(ddof=0) else None,
        "z_score_252": None,
        "trend": trend,
        "signal": signal,
        "chart_data": None,  # the series lives in macro_charts[key].data (shared)
        "png_url": None,
    }


def build_macro_blocks(week_end: date, asset_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (macro_snapshots, macro_charts) for one asset up to ``week_end``.

    Uses only observations at or before ``week_end`` (no look-ahead). Drivers
    missing from the parquet are skipped silently. Returns ({}, {}) if the macro
    parquet is unavailable so callers degrade to the previous empty behaviour.
    """
    drivers = ASSET_DRIVERS.get(asset_id, [])
    if not drivers:
        return {}, {}
    try:
        macro = _load_macro()
    except FileNotFoundError as e:
        logger.warning("asset_macro_charts: %s — emitting empty macro blocks", e)
        return {}, {}

    cutoff = pd.Timestamp(week_end)
    snapshots: dict[str, Any] = {}
    charts: dict[str, Any] = {}
    for key in drivers:
        col, name = _MACRO_COLS[key]
        if col not in macro.columns:
            continue
        s = macro[col].loc[:cutoff].dropna()
        if len(s) < 20:
            continue
        ind = _series_indicators(s)
        snap = _snapshot(key, name, s, ind)
        if snap is None:
            continue
        window = ind.tail(_CHART_WINDOW)
        data = [
            {
                "date": str(idx.date()),
                "value": _round(row["value"], 6),
                "sma20": _round(row["sma20"], 6),
                "bb_upper": _round(row["bb_upper"], 6),
                "bb_lower": _round(row["bb_lower"], 6),
                "rsi": _round(row["rsi"], 6),
            }
            for idx, row in window.iterrows()
        ]
        snapshots[key] = snap
        charts[key] = {"png_url": None, "data": data}
    return snapshots, charts
