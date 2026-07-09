"""
Root Cause Analysis: Why 2025 Good, 2026 Bad?
==============================================

Investigates every possible cause:
1. Macro regime change (DXY, WTI, VIX, EMBI)
2. Volatility regime change
3. Trend structure change
4. Model prediction quality decomposition
5. Feature distribution shift
6. Trailing stop behavior change
7. Direction bias analysis (LONG vs SHORT)
8. Calendar effects

@date 2026-02-16
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import MODEL_IDS, get_horizon_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")


# =============================================================================
# DATA
# =============================================================================

def load_data():
    """Load daily data with features + macro raw values."""
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)

    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_raw_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_raw",
        "COMM_OIL_WTI_GLB_D_WTI": "wti_raw",
        "VOLT_VIX_USA_D_VIX": "vix_raw",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_raw",
    }
    macro_lag_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }

    df_macro_sub = df_macro[["date"] + list(macro_raw_cols.keys())].copy()

    # Raw (same day) for analysis
    for orig, new in macro_raw_cols.items():
        df_macro_sub[new] = df_macro_sub[orig]

    # Lagged for features
    for orig, new in macro_lag_cols.items():
        df_macro_sub[new] = df_macro_sub[orig].shift(1)

    df_macro_sub = df_macro_sub.sort_values("date")

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub[["date"] + list(macro_raw_cols.values()) + list(macro_lag_cols.values())].sort_values("date"),
        on="date", direction="backward",
    )

    # Build features
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss_s.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
    for c in macro_lag_cols.values():
        df[c] = df[c].ffill()
    for c in macro_raw_cols.values():
        df[c] = df[c].ffill()

    df["target_1d"] = np.log(df["close"].shift(-1) / df["close"])

    # Realized volatility (annualized)
    df["realized_vol_21d"] = df["return_1d"].rolling(21).std() * np.sqrt(252)

    # Additional derived
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["dxy_return_5d"] = df["dxy_raw"].pct_change(5)
    df["wti_return_5d"] = df["wti_raw"].pct_change(5)
    df["vix_change_5d"] = df["vix_raw"].diff(5)
    df["embi_change_5d"] = df["embi_raw"].diff(5)

    # COP-DXY correlation (rolling 21d)
    df["cop_dxy_corr_21d"] = df["return_1d"].rolling(21).corr(df["dxy_raw"].pct_change(1))

    return df


def get_predictions(df):
    """Train on 2020-2024, predict for each OOS day with monthly re-training."""
    feature_cols = list(FEATURE_COLUMNS)
    horizon_config = get_horizon_config(1)
    models_to_use = list(MODEL_IDS)

    df_oos = df[(df["date"] >= pd.Timestamp("2025-01-01")) & (df["target_1d"].notna())].copy()
    months = sorted(df_oos["date"].dt.to_period("M").unique())

    results = []
    per_model_preds = {m: [] for m in models_to_use}

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        if month_idx == 0:
            df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
        else:
            df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()

        df_train = df_train[df_train["target_1d"].notna()]
        df_month = df_oos[(df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)]

        if len(df_month) == 0 or len(df_train) < 200:
            continue

        X_train = df_train[feature_cols].values.astype(np.float64)
        y_train = df_train["target_1d"].values.astype(np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        trained = {}
        for model_id in models_to_use:
            try:
                params = None
                if model_id in {"catboost_pure"}:
                    params = {"iterations": 50, "depth": 3, "learning_rate": 0.05,
                              "verbose": False, "allow_writing_files": False}
                elif model_id in {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}:
                    if "catboost" in model_id:
                        params = {"iterations": 50, "depth": 3, "learning_rate": 0.05,
                                  "verbose": False, "allow_writing_files": False}
                    else:
                        params = horizon_config
                elif model_id not in {"ridge", "bayesian_ridge", "ard"}:
                    params = horizon_config

                model = ModelFactory.create(model_id, params=params, horizon=1)
                if model.requires_scaling:
                    model.fit(X_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                trained[model_id] = model
            except Exception:
                pass

        for _, row in df_month.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
            X_day_scaled = scaler.transform(X_day)

            preds = {}
            for model_id, model in trained.items():
                try:
                    if model.requires_scaling:
                        p = model.predict(X_day_scaled)[0]
                    else:
                        p = model.predict(X_day)[0]
                    preds[model_id] = p
                except Exception:
                    pass

            if len(preds) < 3:
                continue

            # Top-3 ensemble
            sorted_m = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
            top3 = sorted_m[:3]
            ensemble = np.mean([preds[m] for m in top3])
            direction = 1 if ensemble > 0 else -1

            # Per-model direction
            model_directions = {m: (1 if p > 0 else -1) for m, p in preds.items()}
            actual_dir = 1 if row["target_1d"] > 0 else -1

            # Model agreement
            dirs = list(model_directions.values())
            agreement = max(sum(1 for d in dirs if d == 1), sum(1 for d in dirs if d == -1)) / len(dirs)

            results.append({
                "date": row["date"],
                "close": row["close"],
                "actual_return": row["target_1d"],
                "actual_dir": actual_dir,
                "ensemble_pred": ensemble,
                "pred_dir": direction,
                "correct": direction == actual_dir,
                "top3": top3,
                "n_models_long": sum(1 for d in dirs if d == 1),
                "n_models_short": sum(1 for d in dirs if d == -1),
                "model_agreement": agreement,
                "pred_magnitude": abs(ensemble),
                "actual_magnitude": abs(row["target_1d"]),
                # Per-model correct
                **{f"{m}_correct": (model_directions.get(m, 0) == actual_dir) for m in models_to_use if m in preds},
                **{f"{m}_pred": preds.get(m, np.nan) for m in models_to_use},
            })

    return pd.DataFrame(results)


# =============================================================================
# DIAGNOSTIC ANALYSES
# =============================================================================

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze_price_regime(df):
    """Compare price dynamics 2025 vs 2026."""
    print_section("1. PRICE & VOLATILITY REGIME")

    for label, start, end in [
        ("2024 (training tail)", "2024-01-01", "2024-12-31"),
        ("2025 H1 (Jan-Jun)", "2025-01-01", "2025-06-30"),
        ("2025 H2 (Jul-Dec)", "2025-07-01", "2025-12-31"),
        ("2025 FULL", "2025-01-01", "2025-12-31"),
        ("2026 YTD", "2026-01-01", "2026-12-31"),
    ]:
        mask = (df["date"] >= start) & (df["date"] <= end) & df["return_1d"].notna()
        sub = df[mask]
        if len(sub) < 5:
            continue

        rets = sub["return_1d"].values
        close_start = sub["close"].iloc[0]
        close_end = sub["close"].iloc[-1]
        period_ret = (close_end - close_start) / close_start * 100

        print(f"\n  {label} ({len(sub)} days, {close_start:.0f} -> {close_end:.0f}, {period_ret:+.1f}%)")
        print(f"    Daily vol (ann.): {np.std(rets)*np.sqrt(252)*100:.1f}%")
        print(f"    Mean daily ret:   {np.mean(rets)*100:.4f}%")
        print(f"    Skew:             {float(pd.Series(rets).skew()):.3f}")
        print(f"    Kurtosis:         {float(pd.Series(rets).kurtosis()):.3f}")
        print(f"    Max daily move:   {np.max(np.abs(rets))*100:.2f}%")
        print(f"    Days up:          {np.sum(rets>0)}/{len(rets)} ({np.sum(rets>0)/len(rets)*100:.1f}%)")
        print(f"    Avg range (H-L):  {sub['range_pct'].mean():.3f}%")

        if "realized_vol_21d" in sub.columns:
            rv = sub["realized_vol_21d"].dropna()
            if len(rv) > 0:
                print(f"    Realized vol 21d: {rv.mean()*100:.1f}% (min={rv.min()*100:.1f}%, max={rv.max()*100:.1f}%)")


def analyze_macro_regime(df):
    """Compare macro environment 2025 vs 2026."""
    print_section("2. MACRO REGIME SHIFT")

    macro_vars = [
        ("DXY", "dxy_raw"),
        ("WTI Oil", "wti_raw"),
        ("VIX", "vix_raw"),
        ("EMBI Col", "embi_raw"),
    ]

    periods = [
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025 H1", "2025-01-01", "2025-06-30"),
        ("2025 H2", "2025-07-01", "2025-12-31"),
        ("2026 YTD", "2026-01-01", "2026-12-31"),
    ]

    for name, col in macro_vars:
        print(f"\n  {name}:")
        print(f"    {'Period':<12} {'Mean':>10} {'Start':>10} {'End':>10} {'Change%':>10} {'Std':>10}")
        print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for plabel, start, end in periods:
            mask = (df["date"] >= start) & (df["date"] <= end) & df[col].notna()
            sub = df[mask][col]
            if len(sub) < 2:
                continue
            s_val = sub.iloc[0]
            e_val = sub.iloc[-1]
            chg = (e_val - s_val) / s_val * 100
            print(f"    {plabel:<12} {sub.mean():>10.2f} {s_val:>10.2f} {e_val:>10.2f} {chg:>+9.1f}% {sub.std():>10.2f}")

    # COP-DXY correlation
    print(f"\n  COP-DXY Rolling Correlation (21d):")
    for plabel, start, end in periods:
        mask = (df["date"] >= start) & (df["date"] <= end) & df["cop_dxy_corr_21d"].notna()
        sub = df[mask]["cop_dxy_corr_21d"]
        if len(sub) > 0:
            print(f"    {plabel:<12} mean={sub.mean():+.3f}  min={sub.min():+.3f}  max={sub.max():+.3f}")


def analyze_feature_distribution_shift(df):
    """Check if feature distributions shifted between train, 2025, and 2026."""
    print_section("3. FEATURE DISTRIBUTION SHIFT (KS-test)")

    feature_cols = list(FEATURE_COLUMNS)
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    df_2025 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")]
    df_2026 = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-12-31")]

    print(f"\n  {'Feature':<20} {'KS train-2025':>15} {'p':>8} {'KS train-2026':>15} {'p':>8} {'Shift?':>8}")
    print(f"  {'-'*20} {'-'*15} {'-'*8} {'-'*15} {'-'*8} {'-'*8}")

    significant_shifts = []

    for feat in feature_cols:
        train_vals = df_train[feat].dropna().values
        vals_2025 = df_2025[feat].dropna().values
        vals_2026 = df_2026[feat].dropna().values

        if len(train_vals) < 10 or len(vals_2025) < 5:
            continue

        ks_25, p_25 = stats.ks_2samp(train_vals, vals_2025)

        if len(vals_2026) >= 5:
            ks_26, p_26 = stats.ks_2samp(train_vals, vals_2026)
        else:
            ks_26, p_26 = np.nan, np.nan

        shift = ""
        if p_26 < 0.05:
            shift = "***"
            significant_shifts.append((feat, ks_26, p_26))
        elif p_26 < 0.10:
            shift = "*"

        print(f"  {feat:<20} {ks_25:>15.4f} {p_25:>8.4f} {ks_26:>15.4f} {p_26:>8.4f} {shift:>8}")

    if significant_shifts:
        print(f"\n  SIGNIFICANT SHIFTS (p<0.05 between training and 2026):")
        for feat, ks, p in sorted(significant_shifts, key=lambda x: x[2]):
            print(f"    {feat}: KS={ks:.4f}, p={p:.4f}")
    else:
        print(f"\n  No statistically significant feature shifts detected.")


def analyze_prediction_quality(preds):
    """Decompose prediction quality 2025 vs 2026."""
    print_section("4. PREDICTION QUALITY DECOMPOSITION")

    preds["year"] = preds["date"].dt.year
    preds["month"] = preds["date"].dt.to_period("M")

    for year in [2025, 2026]:
        sub = preds[preds["year"] == year]
        if len(sub) == 0:
            continue

        n = len(sub)
        correct = sub["correct"].sum()
        da = correct / n * 100

        # Direction breakdown
        long_mask = sub["pred_dir"] == 1
        short_mask = sub["pred_dir"] == -1
        n_long = long_mask.sum()
        n_short = short_mask.sum()

        long_correct = sub[long_mask]["correct"].sum() if n_long > 0 else 0
        short_correct = sub[short_mask]["correct"].sum() if n_short > 0 else 0
        long_da = long_correct / n_long * 100 if n_long > 0 else 0
        short_da = short_correct / n_short * 100 if n_short > 0 else 0

        # Actual market direction
        actual_up = (sub["actual_dir"] == 1).sum()
        actual_down = (sub["actual_dir"] == -1).sum()

        # Prediction confidence vs accuracy
        high_conf = sub[sub["pred_magnitude"] > sub["pred_magnitude"].median()]
        low_conf = sub[sub["pred_magnitude"] <= sub["pred_magnitude"].median()]
        high_da = high_conf["correct"].mean() * 100 if len(high_conf) > 0 else 0
        low_da = low_conf["correct"].mean() * 100 if len(low_conf) > 0 else 0

        # Model agreement vs accuracy
        high_agree = sub[sub["model_agreement"] > 0.7]
        low_agree = sub[sub["model_agreement"] <= 0.7]
        agree_da = high_agree["correct"].mean() * 100 if len(high_agree) > 0 else 0
        disagree_da = low_agree["correct"].mean() * 100 if len(low_agree) > 0 else 0

        # Prediction magnitude
        avg_pred_mag = sub["pred_magnitude"].mean()
        avg_actual_mag = sub["actual_magnitude"].mean()

        print(f"\n  {year} ({n} days):")
        print(f"    Overall DA:            {da:.1f}% ({correct}/{n})")
        print(f"    Market: {actual_up} up / {actual_down} down ({actual_up/n*100:.1f}% up)")
        print(f"    Predicted: {n_long} LONG / {n_short} SHORT ({n_long/n*100:.1f}% long)")
        print(f"    LONG accuracy:         {long_da:.1f}% ({long_correct}/{n_long})")
        print(f"    SHORT accuracy:        {short_da:.1f}% ({short_correct}/{n_short})")
        print(f"    High-confidence DA:    {high_da:.1f}% ({len(high_conf)} days)")
        print(f"    Low-confidence DA:     {low_da:.1f}% ({len(low_conf)} days)")
        print(f"    High-agreement DA:     {agree_da:.1f}% ({len(high_agree)} days)")
        print(f"    Low-agreement DA:      {disagree_da:.1f}% ({len(low_agree)} days)")
        print(f"    Avg |prediction|:      {avg_pred_mag:.6f}")
        print(f"    Avg |actual return|:   {avg_actual_mag:.6f}")
        print(f"    Pred/Actual ratio:     {avg_pred_mag/avg_actual_mag:.3f}" if avg_actual_mag > 0 else "")

    # Monthly breakdown
    print(f"\n  Monthly DA breakdown:")
    print(f"  {'Month':<10} {'Days':>5} {'DA%':>7} {'Long':>6} {'Short':>6} {'LongDA':>8} {'ShortDA':>8} {'Agreement':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")

    for month in sorted(preds["month"].unique()):
        sub = preds[preds["month"] == month]
        n = len(sub)
        da = sub["correct"].mean() * 100
        n_long = (sub["pred_dir"] == 1).sum()
        n_short = (sub["pred_dir"] == -1).sum()
        l_da = sub[sub["pred_dir"]==1]["correct"].mean()*100 if n_long > 0 else 0
        s_da = sub[sub["pred_dir"]==-1]["correct"].mean()*100 if n_short > 0 else 0
        agree = sub["model_agreement"].mean()
        print(f"  {str(month):<10} {n:>5} {da:>6.1f}% {n_long:>6} {n_short:>6} {l_da:>7.1f}% {s_da:>7.1f}% {agree:>9.1%}")


def analyze_per_model_breakdown(preds):
    """Check which models broke in 2026."""
    print_section("5. PER-MODEL ACCURACY (2025 vs 2026)")

    model_ids = list(MODEL_IDS)
    preds_copy = preds.copy()
    preds_copy["year"] = preds_copy["date"].dt.year

    print(f"\n  {'Model':<20} {'2025 DA%':>10} {'2026 DA%':>10} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    for m in model_ids:
        col = f"{m}_correct"
        if col not in preds_copy.columns:
            continue

        da_25 = preds_copy[preds_copy["year"]==2025][col].mean() * 100
        sub_26 = preds_copy[preds_copy["year"]==2026][col]
        da_26 = sub_26.mean() * 100 if len(sub_26) > 0 else float("nan")
        delta = da_26 - da_25

        marker = " <-- BROKEN" if delta < -10 else ""
        print(f"  {m:<20} {da_25:>9.1f}% {da_26:>9.1f}% {delta:>+7.1f}{marker}")


def analyze_autocorrelation(df):
    """Check if return predictability structure changed."""
    print_section("6. RETURN PREDICTABILITY (Autocorrelation)")

    for label, start, end in [
        ("Training (2020-2024)", "2020-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026 YTD", "2026-01-01", "2026-12-31"),
    ]:
        mask = (df["date"] >= start) & (df["date"] <= end) & df["return_1d"].notna()
        rets = df[mask]["return_1d"].values
        if len(rets) < 20:
            continue

        # Autocorrelation lags 1-5
        ac = [pd.Series(rets).autocorr(lag=i) for i in range(1, 6)]

        # Mean reversion test: correlation of return with lagged return
        # Positive = momentum, negative = mean reversion
        print(f"\n  {label} ({len(rets)} days):")
        print(f"    AC(1)={ac[0]:+.4f}  AC(2)={ac[1]:+.4f}  AC(3)={ac[2]:+.4f}  AC(4)={ac[3]:+.4f}  AC(5)={ac[4]:+.4f}")

        # Serial correlation LM test (Ljung-Box)
        from scipy.stats import chi2
        n = len(rets)
        q_stat = n * (n + 2) * sum((ac_i**2) / (n - lag) for lag, ac_i in enumerate(ac, 1))
        p_lb = 1 - chi2.cdf(q_stat, df=5)
        print(f"    Ljung-Box Q(5)={q_stat:.3f}, p={p_lb:.4f} {'(serial corr!)' if p_lb < 0.05 else '(random walk)'}")

        # Streak analysis
        signs = np.sign(rets)
        streaks = []
        current = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1]:
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)
        print(f"    Avg streak: {np.mean(streaks):.2f} days | Max streak: {max(streaks)} days")


def analyze_error_patterns(preds, df):
    """Look at WHEN the model fails — is it during specific conditions?"""
    print_section("7. WHEN DOES THE MODEL FAIL? (Error Pattern Analysis)")

    preds_merged = preds.merge(
        df[["date", "realized_vol_21d", "rsi_14d", "ma_ratio_20d", "dxy_raw", "vix_raw", "range_pct"]],
        on="date", how="left"
    )

    # Split by year
    for year in [2025, 2026]:
        sub = preds_merged[preds_merged["date"].dt.year == year]
        if len(sub) < 5:
            continue

        correct = sub[sub["correct"] == True]
        wrong = sub[sub["correct"] == False]

        print(f"\n  {year}: {len(correct)} correct, {len(wrong)} wrong")

        if len(wrong) < 3:
            continue

        conditions = [
            ("Realized Vol", "realized_vol_21d"),
            ("RSI", "rsi_14d"),
            ("MA Ratio 20d", "ma_ratio_20d"),
            ("VIX", "vix_raw"),
            ("Daily Range%", "range_pct"),
        ]

        print(f"    {'Condition':<20} {'Correct mean':>14} {'Wrong mean':>14} {'Diff':>10}")
        print(f"    {'-'*20} {'-'*14} {'-'*14} {'-'*10}")

        for name, col in conditions:
            c_val = correct[col].dropna().mean()
            w_val = wrong[col].dropna().mean()
            if np.isnan(c_val) or np.isnan(w_val):
                continue
            print(f"    {name:<20} {c_val:>14.4f} {w_val:>14.4f} {w_val-c_val:>+10.4f}")

        # Big move analysis
        big_moves = sub[sub["actual_magnitude"] > sub["actual_magnitude"].quantile(0.75)]
        small_moves = sub[sub["actual_magnitude"] <= sub["actual_magnitude"].quantile(0.25)]
        big_da = big_moves["correct"].mean() * 100 if len(big_moves) > 0 else 0
        small_da = small_moves["correct"].mean() * 100 if len(small_moves) > 0 else 0
        print(f"\n    Big moves (top 25%) DA:    {big_da:.1f}% ({len(big_moves)} days)")
        print(f"    Small moves (bot 25%) DA:  {small_da:.1f}% ({len(small_moves)} days)")


def analyze_trailing_stop_effect(preds, df):
    """Analyze if trailing stop hurts or helps in each period."""
    print_section("8. TRAILING STOP DECOMPOSITION")

    preds_merged = preds.merge(
        df[["date", "realized_vol_21d", "range_pct"]],
        on="date", how="left"
    )

    for year in [2025, 2026]:
        sub = preds_merged[preds_merged["date"].dt.year == year]
        if len(sub) < 5:
            continue

        correct_mask = sub["correct"]
        wrong_mask = ~sub["correct"]

        # When correct: does trailing stop capture enough of the move?
        # When wrong: does trailing stop limit the loss?
        correct_sub = sub[correct_mask]
        wrong_sub = sub[wrong_mask]

        avg_correct_move = correct_sub["actual_magnitude"].mean() * 100 if len(correct_sub) > 0 else 0
        avg_wrong_move = wrong_sub["actual_magnitude"].mean() * 100 if len(wrong_sub) > 0 else 0

        # Average intraday range
        avg_range = sub["range_pct"].mean()

        print(f"\n  {year}:")
        print(f"    Avg |move| when correct:  {avg_correct_move:.4f}%")
        print(f"    Avg |move| when wrong:    {avg_wrong_move:.4f}%")
        print(f"    Avg intraday range (H-L): {avg_range:.3f}%")
        print(f"    Trail activation:         0.200%")
        print(f"    Trail distance:           0.300%")
        print(f"    Hard stop:                1.500%")

        # Is the move smaller than trailing stop activation?
        small_move_pct = (sub["actual_magnitude"] * 100 < 0.20).mean() * 100
        print(f"    Moves < activation (0.2%): {small_move_pct:.1f}% of days")


def summarize_root_causes(preds, df):
    """Final summary of root causes."""
    print_section("DIAGNOSIS SUMMARY")

    preds_25 = preds[preds["date"].dt.year == 2025]
    preds_26 = preds[preds["date"].dt.year == 2026]

    df_25 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")]
    df_26 = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-12-31")]

    # Key metrics
    da_25 = preds_25["correct"].mean() * 100
    da_26 = preds_26["correct"].mean() * 100 if len(preds_26) > 0 else 0

    vol_25 = df_25["return_1d"].std() * np.sqrt(252) * 100
    vol_26 = df_26["return_1d"].std() * np.sqrt(252) * 100 if len(df_26) > 5 else 0

    dxy_25_start = df_25["dxy_raw"].iloc[0] if len(df_25) > 0 else 0
    dxy_25_end = df_25["dxy_raw"].iloc[-1] if len(df_25) > 0 else 0
    dxy_26_start = df_26["dxy_raw"].iloc[0] if len(df_26) > 0 else 0
    dxy_26_end = df_26["dxy_raw"].iloc[-1] if len(df_26) > 0 else 0

    vix_25 = df_25["vix_raw"].mean()
    vix_26 = df_26["vix_raw"].mean() if len(df_26) > 0 else 0

    range_25 = df_25["range_pct"].mean()
    range_26 = df_26["range_pct"].mean() if len(df_26) > 0 else 0

    print(f"""
  Key findings:

  1. DIRECTION ACCURACY
     2025: {da_25:.1f}%  |  2026: {da_26:.1f}%  |  Drop: {da_26-da_25:.1f}pp

  2. VOLATILITY
     2025: {vol_25:.1f}% ann  |  2026: {vol_26:.1f}% ann
     2025 avg range: {range_25:.3f}%  |  2026 avg range: {range_26:.3f}%

  3. MACRO
     DXY 2025: {dxy_25_start:.1f} -> {dxy_25_end:.1f} ({(dxy_25_end-dxy_25_start)/dxy_25_start*100:+.1f}%)
     DXY 2026: {dxy_26_start:.1f} -> {dxy_26_end:.1f} ({(dxy_26_end-dxy_26_start)/dxy_26_start*100:+.1f}%)
     VIX 2025 avg: {vix_25:.1f}  |  VIX 2026 avg: {vix_26:.1f}

  4. ROOT CAUSE CANDIDATES (ranked by likelihood):
""")

    # Automated root cause detection
    causes = []

    # Check volatility regime
    if abs(vol_26 - vol_25) / vol_25 > 0.3:
        causes.append(("HIGH", f"Volatility regime change: {vol_25:.1f}% -> {vol_26:.1f}% ({(vol_26-vol_25)/vol_25*100:+.0f}%)"))

    # Check DXY regime
    dxy_25_change = (dxy_25_end - dxy_25_start) / dxy_25_start * 100
    dxy_26_change = (dxy_26_end - dxy_26_start) / dxy_26_start * 100
    if np.sign(dxy_25_change) != np.sign(dxy_26_change) and abs(dxy_26_change) > 2:
        causes.append(("HIGH", f"DXY trend reversal: 2025 {dxy_25_change:+.1f}% -> 2026 {dxy_26_change:+.1f}%"))

    # Check VIX regime
    if abs(vix_26 - vix_25) / vix_25 > 0.3:
        causes.append(("HIGH", f"VIX regime: {vix_25:.1f} -> {vix_26:.1f} ({(vix_26-vix_25)/vix_25*100:+.0f}%)"))

    # Check LONG vs SHORT accuracy drop
    long_25 = preds_25[preds_25["pred_dir"]==1]["correct"].mean()*100 if (preds_25["pred_dir"]==1).sum() > 0 else 50
    short_25 = preds_25[preds_25["pred_dir"]==-1]["correct"].mean()*100 if (preds_25["pred_dir"]==-1).sum() > 0 else 50
    long_26 = preds_26[preds_26["pred_dir"]==1]["correct"].mean()*100 if len(preds_26) > 0 and (preds_26["pred_dir"]==1).sum() > 0 else 50
    short_26 = preds_26[preds_26["pred_dir"]==-1]["correct"].mean()*100 if len(preds_26) > 0 and (preds_26["pred_dir"]==-1).sum() > 0 else 50

    if abs(long_26 - long_25) > 15:
        causes.append(("MEDIUM", f"LONG accuracy collapse: {long_25:.1f}% -> {long_26:.1f}%"))
    if abs(short_26 - short_25) > 15:
        causes.append(("MEDIUM", f"SHORT accuracy collapse: {short_25:.1f}% -> {short_26:.1f}%"))

    # Check model agreement
    agree_25 = preds_25["model_agreement"].mean()
    agree_26 = preds_26["model_agreement"].mean() if len(preds_26) > 0 else 0
    if agree_26 < agree_25 - 0.05:
        causes.append(("MEDIUM", f"Model agreement dropped: {agree_25:.1%} -> {agree_26:.1%}"))

    # Check range vs trailing stop
    if range_26 < 0.20:
        causes.append(("MEDIUM", f"Avg range ({range_26:.3f}%) < trailing activation (0.200%) - stop can't capture moves"))

    # Check sample size
    if len(preds_26) < 30:
        causes.append(("LOW", f"Small 2026 sample ({len(preds_26)} days) - may be noise not regime change"))

    # Print causes
    for severity, cause in causes:
        print(f"     [{severity}] {cause}")

    if not causes:
        print(f"     No clear root cause detected. May be normal variance with small sample.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()

    df = load_data()
    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    preds = get_predictions(df)
    logger.info(f"Predictions: {len(preds)} days")

    analyze_price_regime(df)
    analyze_macro_regime(df)
    analyze_feature_distribution_shift(df)
    analyze_prediction_quality(preds)
    analyze_per_model_breakdown(preds)
    analyze_autocorrelation(df)
    analyze_error_patterns(preds, df)
    analyze_trailing_stop_effect(preds, df)
    summarize_root_causes(preds, df)

    # Save
    output = PROJECT_ROOT / "results" / "regime_diagnosis_2025_vs_2026.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "da_2025": round(preds[preds["date"].dt.year==2025]["correct"].mean()*100, 1),
        "da_2026": round(preds[preds["date"].dt.year==2026]["correct"].mean()*100, 1) if len(preds[preds["date"].dt.year==2026]) > 0 else None,
        "n_days_2025": len(preds[preds["date"].dt.year==2025]),
        "n_days_2026": len(preds[preds["date"].dt.year==2026]),
    }
    with open(output, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
