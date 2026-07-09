"""
Smart Simple v2.0 feature enhancement (SHARED SSOT).
====================================================

Single source of truth for the v2.0 feature enhancement used by BOTH the
backtest/production export (`scripts/train_and_export_smart_simple.py`) AND the
live weekly Airflow pipeline (H5-L3 training + H5-L5 signal).

Historically this logic lived only in the export script, so the live weekly
pipeline trained/inferred on the 21 base features while the approved +25.63%
backtest used 23 (base + vol_regime_ratio + trend_slope_60d). That divergence
(audit A3-02) meant production models were not the approved models. Keeping the
enhancement here — imported everywhere — guarantees train == backtest == infer.

With the current MACRO_DAILY_CLEAN.parquet (which lacks ibr/fedfunds/UST columns
under these names), the carry_diff/term_spread merge is a no-op, so the enhanced
set is deterministically: base (21) + vol_regime_ratio + trend_slope_60d = 23.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = two levels up from this file (src/forecasting/enhance_v2.py).
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def enhance_features_v2(df, base_feature_cols, project_root=None):
    """
    Smart Simple v2.0 feature enhancement.

    Adds regime + carry + term spread features.
    Transforms macro levels to 5d returns (stationary).
    Keeps raw prices (Ridge needs them; XGBoost handles non-stationarity).

    Args:
        df: DataFrame with at least `close`, `volatility_5d`, `volatility_20d`,
            `date` columns.
        base_feature_cols: iterable of the base feature column names.
        project_root: repo root used to locate MACRO_DAILY_CLEAN.parquet. Defaults
            to the repo root inferred from this file's location. The live DAGs
            pass their own root (e.g. /opt/airflow).

    Returns: (df_enhanced, v2_feature_cols)
    """
    df = df.copy()
    if project_root is None:
        project_root = _DEFAULT_PROJECT_ROOT

    # --- New features ---
    # Vol regime ratio (short vol / long vol)
    vol5 = df["volatility_5d"].replace(0, np.nan)
    vol20 = df["volatility_20d"].replace(0, np.nan)
    df["vol_regime_ratio"] = (vol5 / vol20).clip(-5, 5).fillna(1.0)

    # Trend slope 60d (normalized)
    def _trend_slope(series, window=60):
        result = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            chunk = series.iloc[i - window:i].values
            if len(chunk) == window and np.std(chunk) > 0:
                x = np.arange(window)
                slope = np.polyfit(x, chunk, 1)[0]
                result.iloc[i] = slope / np.mean(chunk)
        return result

    df["trend_slope_60d"] = _trend_slope(df["close"])

    # Carry differential (IBR - FedFunds) if available in macro
    # These come from merge_asof in dataset_loader already lagged T-1
    macro_path = Path(project_root) / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    if macro_path.exists():
        try:
            macro = pd.read_parquet(macro_path)
            macro["date"] = pd.to_datetime(macro["fecha"]).dt.tz_localize(None)
            for col_pair in [("ibr_overnight", "fedfunds_rate")]:
                c1, c2 = col_pair
                if c1 in macro.columns and c2 in macro.columns:
                    macro["carry_diff"] = (macro[c1] - macro[c2]).shift(1)
                    df = pd.merge_asof(
                        df.sort_values("date"),
                        macro[["date", "carry_diff"]].dropna().sort_values("date"),
                        on="date", direction="backward",
                    )
            if "ust10y_close" in macro.columns and "ust2y_close" in macro.columns:
                macro["term_spread"] = (macro["ust10y_close"] - macro["ust2y_close"]).shift(1)
                df = pd.merge_asof(
                    df.sort_values("date"),
                    macro[["date", "term_spread"]].dropna().sort_values("date"),
                    on="date", direction="backward",
                )
        except Exception as e:
            print(f"    [v2] Macro enhancement failed: {e}")

    # Fill NaN for new features
    for col in ["vol_regime_ratio", "trend_slope_60d", "carry_diff", "term_spread"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    # Build v2 feature list: base + new
    v2_features = list(base_feature_cols)
    for new_col in ["vol_regime_ratio", "trend_slope_60d", "carry_diff", "term_spread"]:
        if new_col in df.columns:
            v2_features.append(new_col)

    return df, v2_features
