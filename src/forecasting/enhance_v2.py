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

The macro merge USED to be a silent no-op: it read `macro["fecha"]` (that is the
DatetimeIndex, not a column) and looked for lowercase names that MACRO_DAILY_CLEAN
does not use, so `carry_diff`/`term_spread` were filled with 0.0 and the enhanced
set was effectively base (21) + vol_regime_ratio + trend_slope_60d = 23.

Fixed 2026-07-20: the merge now resolves the UPPERCASE SSOT columns and produces
25 features. `carry_diff` was renamed `rate_diff_ibr_ust2y` because it is a COP
overnight vs US 2Y rate differential, not a pure FX carry (FedFunds is monthly and
absent from the daily CLEAN) — naming it `carry_diff` overstated what it measures.

NOTE: this changes the feature set, so it is a NEW TRIAL under quant-constitution.md.
Any edge claim must be re-established on the forward period, not on 2025.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    # MACRO_DAILY_CLEAN stores `fecha` as the DatetimeIndex (not a column) and uses the
    # UPPERCASE SSOT variable names from config/macro_variables_ssot.yaml. This block used
    # to read `macro["fecha"]` and look for lowercase names (`ibr_overnight`,
    # `fedfunds_rate`, `ust10y_close`, `ust2y_close`) — none of which exist. The KeyError
    # was caught and printed, then the fill-NaN block below set both features to 0.0.
    #
    # Net effect: `carry_diff` and `term_spread` were identically zero in every training
    # window since v2 shipped. Two dead features that the contract claimed were live.
    #
    # FedFunds is monthly (macro_indicators_monthly), so it is not in the daily CLEAN;
    # the US 2Y yield is the daily short-rate proxy for the carry leg.
    IBR = "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"
    UST10Y = "FINC_BOND_YIELD10Y_USA_D_UST10Y"
    UST2Y = "FINC_BOND_YIELD2Y_USA_D_DGS2"

    if macro_path.exists():
        try:
            macro = pd.read_parquet(macro_path).reset_index()
            date_col = "fecha" if "fecha" in macro.columns else macro.columns[0]
            macro["date"] = pd.to_datetime(macro[date_col]).dt.tz_localize(None)

            missing = [c for c in (IBR, UST10Y, UST2Y) if c not in macro.columns]
            if missing:
                raise KeyError(
                    f"MACRO_DAILY_CLEAN is missing expected SSOT columns: {missing}. "
                    f"Available: {sorted(macro.columns)[:8]}..."
                )

            # .shift(1) keeps the T-1 availability rule (anti-leakage, data-governance).
            macro["rate_diff_ibr_ust2y"] = (macro[IBR] - macro[UST2Y]).shift(1)
            macro["term_spread"] = (macro[UST10Y] - macro[UST2Y]).shift(1)

            for feat in ("rate_diff_ibr_ust2y", "term_spread"):
                df = pd.merge_asof(
                    df.sort_values("date"),
                    macro[["date", feat]].dropna().sort_values("date"),
                    on="date", direction="backward",
                )
        except Exception as e:
            # Loud, not decorative: a silent failure here produced two constant-zero
            # features that nobody noticed. Callers must be able to detect it.
            logger.error(
                "[v2] Macro enhancement FAILED (%s). carry_diff/term_spread will be "
                "constant 0.0 and the model is training with two dead features.", e
            )
            print(f"    [v2] ERROR Macro enhancement failed: {e}")

    # Fill NaN for new features
    for col in ["vol_regime_ratio", "trend_slope_60d", "rate_diff_ibr_ust2y", "term_spread"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    # Build v2 feature list: base + new
    v2_features = list(base_feature_cols)
    for new_col in ["vol_regime_ratio", "trend_slope_60d", "rate_diff_ibr_ust2y", "term_spread"]:
        if new_col in df.columns:
            v2_features.append(new_col)

    return df, v2_features
