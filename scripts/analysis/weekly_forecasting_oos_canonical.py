"""Canonical weekly OOS directional audit for USD/COP.

One forecast origin per ISO week. Feature selection is frozen using pre-2025
data. At every origin only labels matured by ``origin - horizon`` are used.
Macro observations become available no earlier than T+1.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis._causal_backtest import matured_label_indices_before
from src.forecasting.dataset_loader import ForecastingDatasetLoader
from src.forecasting.ssot_config import ForecastingSSOTConfig

HORIZONS = (1, 5, 10, 15, 20, 25, 30)
RAW_MACRO = [
    "FXRT_INDEX_DXY_USA_D_DXY", "FXRT_SPOT_USDMXN_MEX_D_USDMXN",
    "FXRT_SPOT_USDCLP_CHL_D_USDCLP", "VOLT_VIX_USA_D_VIX",
    "CRSK_SPREAD_EMBI_COL_D_EMBI", "COMM_OIL_WTI_GLB_D_WTI",
    "COMM_OIL_BRENT_GLB_D_BRENT", "COMM_METAL_GOLD_GLB_D_GOLD",
    "COMM_AGRI_COFFEE_GLB_D_COFFEE", "FINC_BOND_YIELD10Y_USA_D_UST10Y",
    "FINC_BOND_YIELD2Y_USA_D_DGS2", "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR",
    "POLR_POLICY_RATE_COL_M_TPM", "FINC_BOND_YIELD10Y_COL_D_COL10Y",
    "FINC_BOND_YIELD5Y_COL_D_COL5Y", "EQTY_INDEX_COLCAP_COL_D_COLCAP",
]


def build_frame(
    *, include_forward_pit: bool = True, promotion_only: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cfg = ForecastingSSOTConfig.load()
    df, _ = ForecastingDatasetLoader(cfg, project_root=ROOT).load_dataset()
    df = df.sort_values("date").reset_index(drop=True)
    macro = pd.read_parquet(
        ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
    ).reset_index()
    macro = macro.rename(columns={macro.columns[0]: "date"})
    # Conservative availability contract: observation dated T is usable T+1.
    macro["date"] = pd.to_datetime(macro["date"]) + pd.Timedelta(days=1)
    macro = macro[["date"] + RAW_MACRO].sort_values("date").drop_duplicates("date")
    df = pd.merge_asof(df, macro, on="date", direction="backward")

    price_features = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
    ]
    macro_features: list[str] = []
    rate_cols = {c for c in RAW_MACRO if "RATE" in c or "YIELD" in c}
    for col in RAW_MACRO:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        for lag in (1, 5, 20):
            name = f"{col}_{'diff' if col in rate_cols else 'ret'}_{lag}"
            if col in rate_cols:
                df[name] = df[col].diff(lag)
            else:
                df[name] = df[col].pct_change(lag)
            macro_features.append(name)
    # Economically interpretable carry/slope features.
    df["col_curve_10y_5y"] = (
        df["FINC_BOND_YIELD10Y_COL_D_COL10Y"] - df["FINC_BOND_YIELD5Y_COL_D_COL5Y"]
    )
    df["carry_ibr_us2y"] = (
        df["FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"] - df["FINC_BOND_YIELD2Y_USA_D_DGS2"]
    )
    df["carry_col10y_us10y"] = (
        df["FINC_BOND_YIELD10Y_COL_D_COL10Y"]
        - df["FINC_BOND_YIELD10Y_USA_D_UST10Y"]
    )
    df["carry_policy_us2y"] = (
        df["POLR_POLICY_RATE_COL_M_TPM"] - df["FINC_BOND_YIELD2Y_USA_D_DGS2"]
    )
    macro_features += [
        "col_curve_10y_5y", "carry_ibr_us2y", "carry_col10y_us10y",
        "carry_policy_us2y",
    ]

    # Relative/cross-asset features are more stationary than standalone levels.
    for lag in (1, 5, 20):
        mxn = f"FXRT_SPOT_USDMXN_MEX_D_USDMXN_ret_{lag}"
        clp = f"FXRT_SPOT_USDCLP_CHL_D_USDCLP_ret_{lag}"
        dxy = f"FXRT_INDEX_DXY_USA_D_DXY_ret_{lag}"
        vix = f"VOLT_VIX_USA_D_VIX_ret_{lag}"
        embi = f"CRSK_SPREAD_EMBI_COL_D_EMBI_ret_{lag}"
        brent = f"COMM_OIL_BRENT_GLB_D_BRENT_ret_{lag}"
        coffee = f"COMM_AGRI_COFFEE_GLB_D_COFFEE_ret_{lag}"
        colcap = f"EQTY_INDEX_COLCAP_COL_D_COLCAP_ret_{lag}"
        cop = f"return_{lag}d"
        df[f"latam_fx_ret_{lag}"] = 0.5 * (df[mxn] + df[clp])
        df[f"cop_latam_residual_{lag}"] = df[cop] - df[f"latam_fx_ret_{lag}"]
        df[f"cop_dxy_residual_{lag}"] = df[cop] - df[dxy]
        df[f"terms_of_trade_impulse_{lag}"] = df[brent] + df[coffee]
        raw_risk = df[dxy] + df[vix] + df[embi] - df[brent] - df[colcap]
        mean = raw_risk.rolling(252, min_periods=60).mean()
        std = raw_risk.rolling(252, min_periods=60).std().replace(0, np.nan)
        df[f"risk_off_impulse_z_{lag}"] = (raw_risk - mean) / std
        macro_features += [
            f"latam_fx_ret_{lag}", f"cop_latam_residual_{lag}",
            f"cop_dxy_residual_{lag}", f"terms_of_trade_impulse_{lag}",
            f"risk_off_impulse_z_{lag}",
        ]
    pit_features: list[str] = []
    if include_forward_pit:
        from src.data.usdcop_forward_macro import attach_forward_macro_features

        df, pit_features = attach_forward_macro_features(
            df,
            ROOT / "data/pipeline/04_cleaning/output/USDCOP_FORWARD_MACRO_PIT.parquet",
            promotion_only=promotion_only,
        )
    df.attrs["forward_pit_features"] = pit_features
    df.attrs["classic_macro_features"] = list(macro_features)
    return df, price_features, price_features + macro_features + pit_features


def select_features(df: pd.DataFrame, candidates: list[str], horizon: int, k: int = 12) -> list[str]:
    close = df["close"]
    future_return = np.log(close.shift(-horizon) / close)
    target = (future_return > 0).where(future_return.notna()).astype(float)
    mask = pd.Series(False, index=df.index)
    selection_idx = matured_label_indices_before(
        df["date"], horizon=horizon, cutoff="2025-01-01"
    )
    mask.iloc[selection_idx] = True
    mask &= target.notna()
    usable = [c for c in candidates if df.loc[mask, c].notna().sum() >= 250]
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(df.loc[mask, usable])
    y = target.loc[mask].astype(int).to_numpy()
    scores = mutual_info_classif(x, y, random_state=17)
    return [usable[i] for i in np.argsort(scores)[::-1][: min(k, len(usable))]]


def evaluate(df: pd.DataFrame, candidates: list[str], feature_set: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    origins = df.assign(iso_week=dates.dt.strftime("%G-W%V")).groupby("iso_week").tail(1)
    origins = origins[(origins["date"] >= "2025-01-01") & (origins["date"] <= "2026-12-31")]
    rows: list[dict] = []
    for horizon in HORIZONS:
        features = select_features(df, candidates, horizon)
        future_return = np.log(df["close"].shift(-horizon) / df["close"])
        y = (future_return > 0).where(future_return.notna()).astype(float)
        for origin_idx, origin in origins.iterrows():
            target_idx = origin_idx + horizon
            if target_idx >= len(df) or pd.isna(y.iloc[origin_idx]):
                continue
            # Purge every training label whose target reaches the origin.
            train_idx = np.arange(0, max(0, origin_idx - horizon + 1))
            train_idx = train_idx[y.iloc[train_idx].notna().to_numpy()]
            if len(train_idx) < 300:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=0.1, max_iter=2000),
            )
            model.fit(df.loc[train_idx, features], y.iloc[train_idx].astype(int))
            pred = int(model.predict(df.loc[[origin_idx], features])[0])
            actual = int(y.iloc[origin_idx])
            majority = int(y.iloc[train_idx].mean() >= 0.5)
            rows.append({
                "feature_set": feature_set, "week": origin["iso_week"],
                "origin_date": origin["date"], "target_date": df.loc[target_idx, "date"],
                "horizon": horizon, "prediction": pred, "actual": actual,
                "correct": int(pred == actual), "majority_correct": int(majority == actual),
                "selected_features": "|".join(features),
            })
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.copy()
    x["period"] = np.where(x["week"].str.startswith("2025"), "2025_OOS", "2026_FORWARD")
    rows = []
    for keys, g in x.groupby(["feature_set", "period", "horizon"]):
        rows.append({
            "feature_set": keys[0], "period": keys[1], "horizon": keys[2],
            "n_weeks": len(g), "da": g["correct"].mean(),
            "majority_da": g["majority_correct"].mean(),
            "balanced_da": balanced_accuracy_score(g["actual"], g["prediction"]),
            "actual_up_rate": g["actual"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-only", action="store_true")
    parser.add_argument("--output-prefix", default="weekly_forecasting_oos_canonical")
    args = parser.parse_args()
    df, base, enriched = build_frame(promotion_only=args.promotion_only)
    pit_features = df.attrs.get("forward_pit_features", [])
    classic = [feature for feature in enriched if feature not in set(pit_features)]
    evaluations = [
        evaluate(df, base, "stationary_price"),
        evaluate(df, classic, "stationary_price_plus_macro"),
    ]
    if pit_features:
        suffix = "promotion_pit" if args.promotion_only else "all_pit"
        evaluations.append(evaluate(df, enriched, f"stationary_price_plus_macro_{suffix}"))
    predictions = pd.concat(evaluations, ignore_index=True)
    summary = summarize(predictions)
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    predictions.to_csv(report / f"{args.output_prefix}_predictions.csv", index=False)
    summary.to_csv(report / f"{args.output_prefix}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
