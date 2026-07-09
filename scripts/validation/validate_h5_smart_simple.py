"""
Smoke test for Smart Simple v1.0 pipeline.
==========================================

Validates the complete pipeline offline (no Airflow, no DB):
1. Load daily OHLCV and compute features
2. Train Ridge + BayesianRidge on 2020-2024
3. Generate 4 weeks of signals from 2025-01 data
4. Score confidence on each signal
5. Compute vol-targeting + adaptive stops
6. Simulate TP/HS/Friday close execution
7. Aggregate PnL and verify pipeline correctness

Usage:
    python scripts/validate_h5_smart_simple.py

@version 1.0.0
@contract FC-H5-SMOKE-001
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.confidence_scorer import (
    ConfidenceConfig,
    ConfidenceScore,
    score_confidence,
)
from src.forecasting.adaptive_stops import (
    AdaptiveStopsConfig,
    compute_adaptive_stops,
    check_hard_stop,
    check_take_profit,
    get_exit_price,
)
from src.forecasting.vol_targeting import (
    VolTargetConfig,
    compute_realized_vol,
    compute_vol_target_signal,
    apply_asymmetric_sizing,
)


def load_config():
    """Load Smart Simple v1 config."""
    config_path = PROJECT_ROOT / "config" / "execution" / "smart_simple_v1.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute features from daily OHLCV (simplified)."""
    df = df_daily.copy()
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi_14d"] = 100 - 100 / (1 + rs)

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(float)

    # Placeholder macro features (use zeros for smoke test)
    df["dxy_close_lag1"] = 0.0
    df["oil_close_lag1"] = 0.0
    df["vix_close_lag1"] = 0.0
    df["embi_close_lag1"] = 0.0

    return df.dropna()


def run_smoke_test():
    """Execute the full Smart Simple pipeline smoke test."""
    print("=" * 70)
    print("Smart Simple v1.0 — Offline Smoke Test")
    print("=" * 70)

    # Load config
    config = load_config()
    print(f"\n[1] Loaded config: {config['executor']['name']} v{config['executor']['version']}")

    # Load daily OHLCV
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df = pd.read_parquet(ohlcv_path)
    df = df.reset_index()
    if "time" in df.columns:
        df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    print(f"[2] Loaded daily OHLCV: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")

    # Build features
    df_feat = build_features(df)
    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1",
    ]

    # Target: log return H=5
    df_feat["target_h5"] = np.log(df_feat["close"].shift(-5) / df_feat["close"])
    df_feat = df_feat.dropna(subset=["target_h5"])
    print(f"[3] Features built: {len(df_feat)} rows with H=5 target")

    # Split: train 2020-2024, test first 4 Mondays of 2025
    train = df_feat["2020":"2024"]
    test_days = df_feat["2025-01":"2025-02"]

    X_train = train[feature_cols].values
    y_train = train["target_h5"].values

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Ridge + BayesianRidge
    from sklearn.linear_model import Ridge, BayesianRidge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    br = BayesianRidge(max_iter=300)
    br.fit(X_train_scaled, y_train)
    print(f"[4] Trained Ridge + BayesianRidge on {len(train)} training samples")

    # Find Mondays in test period
    mondays = test_days[test_days.index.dayofweek == 0].index[:4]
    print(f"[5] Simulating {len(mondays)} weeks: {[str(m.date()) for m in mondays]}")

    # Confidence config
    cc = config.get("confidence", {})
    conf_config = ConfidenceConfig(
        agreement_tight=cc.get("agreement_tight", 0.001),
        agreement_loose=cc.get("agreement_loose", 0.005),
        magnitude_high=cc.get("magnitude_high", 0.010),
        magnitude_medium=cc.get("magnitude_medium", 0.005),
        short_high=cc.get("short", {}).get("HIGH", 2.0),
        short_medium=cc.get("short", {}).get("MEDIUM", 1.5),
        short_low=cc.get("short", {}).get("LOW", 1.0),
        long_high=cc.get("long", {}).get("HIGH", 1.0),
        long_medium=cc.get("long", {}).get("MEDIUM", 0.5),
        long_low=cc.get("long", {}).get("LOW", 0.0),
    )

    # Vol-targeting config
    vt_cfg = config.get("vol_targeting", {})
    vt_config = VolTargetConfig(
        target_vol=vt_cfg.get("target_vol", 0.15),
        max_leverage=vt_cfg.get("max_leverage", 2.0),
        min_leverage=vt_cfg.get("min_leverage", 0.5),
        vol_lookback=vt_cfg.get("vol_lookback", 21),
    )

    # Adaptive stops config
    as_cfg = config.get("adaptive_stops", {})
    stops_config = AdaptiveStopsConfig(
        vol_multiplier=as_cfg.get("vol_multiplier", 1.5),
        hard_stop_min_pct=as_cfg.get("hard_stop_min_pct", 0.01),
        hard_stop_max_pct=as_cfg.get("hard_stop_max_pct", 0.03),
        tp_ratio=as_cfg.get("tp_ratio", 0.5),
    )

    results = []
    total_pnl = 0.0

    for monday in mondays:
        # Get prediction features (Friday before Monday)
        friday_idx = df_feat.index.get_indexer([monday], method="ffill")[0]
        if friday_idx < 0:
            continue
        friday = df_feat.index[friday_idx]
        if friday >= monday:
            friday_idx -= 1
            friday = df_feat.index[friday_idx]

        X_pred = scaler.transform(df_feat[feature_cols].iloc[friday_idx:friday_idx+1].values)

        # Predict
        pred_ridge = float(ridge.predict(X_pred)[0])
        pred_br = float(br.predict(X_pred)[0])
        ensemble_return = (pred_ridge + pred_br) / 2.0
        direction = 1 if ensemble_return > 0 else -1

        # Confidence scoring
        confidence = score_confidence(pred_ridge, pred_br, direction, conf_config)

        # Realized vol
        returns = df["close"].loc[:friday].pct_change().dropna().values
        realized_vol = compute_realized_vol(returns, lookback=21, annualization=252.0)

        # Vol-targeting
        vt_signal = compute_vol_target_signal(
            forecast_direction=direction,
            forecast_return=ensemble_return,
            realized_vol_21d=realized_vol,
            config=vt_config,
        )
        base_lev = apply_asymmetric_sizing(vt_signal.clipped_leverage, direction, 0.5, 1.0)
        adjusted_lev = max(0.5, min(base_lev * confidence.sizing_multiplier, 2.0))

        # Adaptive stops
        stops = compute_adaptive_stops(realized_vol, stops_config)

        # Simulate execution
        dir_str = "LONG" if direction == 1 else "SHORT"
        entry_price = float(df_feat["close"].loc[friday])

        if confidence.skip_trade:
            week_pnl = 0.0
            exit_reason = "skip"
        else:
            # Get Friday close (5 bdays after Monday)
            target_idx = min(friday_idx + 5, len(df_feat) - 1)
            friday_close = float(df_feat["close"].iloc[target_idx])

            # Simple TP/HS check using Friday close
            if direction == 1:
                tp_hit = friday_close >= entry_price * (1 + stops.take_profit_pct)
                hs_hit = friday_close <= entry_price * (1 - stops.hard_stop_pct)
            else:
                tp_hit = friday_close <= entry_price * (1 - stops.take_profit_pct)
                hs_hit = friday_close >= entry_price * (1 + stops.hard_stop_pct)

            if hs_hit:
                exit_price = get_exit_price(direction, entry_price, "hard_stop",
                                            stops.hard_stop_pct, stops.take_profit_pct, friday_close)
                exit_reason = "hard_stop"
            elif tp_hit:
                exit_price = get_exit_price(direction, entry_price, "take_profit",
                                            stops.hard_stop_pct, stops.take_profit_pct, friday_close)
                exit_reason = "take_profit"
            else:
                exit_price = friday_close
                exit_reason = "week_end"

            raw_pnl = direction * (exit_price - entry_price) / entry_price
            week_pnl = raw_pnl * adjusted_lev

        total_pnl += week_pnl

        results.append({
            "monday": str(monday.date()),
            "direction": dir_str,
            "confidence": confidence.tier.value,
            "sizing_mult": confidence.sizing_multiplier,
            "skip": confidence.skip_trade,
            "leverage": round(adjusted_lev, 3),
            "HS%": round(stops.hard_stop_pct * 100, 2),
            "TP%": round(stops.take_profit_pct * 100, 2),
            "exit": exit_reason,
            "pnl%": round(week_pnl * 100, 4),
        })

    # Print results
    print(f"\n{'='*70}")
    print("WEEK RESULTS")
    print(f"{'='*70}")
    print(f"{'Monday':<12} {'Dir':<6} {'Conf':<8} {'Mult':<6} {'Lev':<6} "
          f"{'HS%':<7} {'TP%':<7} {'Exit':<12} {'PnL%':>8}")
    print("-" * 80)

    for r in results:
        skip_str = "[SKIP]" if r["skip"] else ""
        print(
            f"{r['monday']:<12} {r['direction']:<6} {r['confidence']:<8} "
            f"{r['sizing_mult']:<6.2f} {r['leverage']:<6.3f} "
            f"{r['HS%']:<7.2f} {r['TP%']:<7.2f} {r['exit']:<12} "
            f"{r['pnl%']:>+8.4f} {skip_str}"
        )

    print("-" * 80)
    print(f"Total PnL: {total_pnl * 100:+.4f}%")
    print(f"Trades executed: {sum(1 for r in results if not r['skip'])}/{len(results)}")
    print(f"Skipped (LOW confidence LONG): {sum(1 for r in results if r['skip'])}")

    # Validation checks
    errors = []

    # Check confidence scoring works
    for r in results:
        if r["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
            errors.append(f"Invalid confidence tier: {r['confidence']}")

    # Check adaptive stops are within bounds
    for r in results:
        if r["HS%"] < 1.0 or r["HS%"] > 3.0:
            errors.append(f"HS% out of bounds: {r['HS%']}%")
        if r["TP%"] < 0.5 or r["TP%"] > 1.5:
            errors.append(f"TP% out of bounds: {r['TP%']}%")

    # Check skip logic
    for r in results:
        if r["direction"] == "LONG" and r["confidence"] == "LOW" and not r["skip"]:
            errors.append(f"LOW confidence LONG should be skipped: {r['monday']}")
        if r["direction"] == "SHORT" and r["skip"]:
            errors.append(f"SHORT should never be skipped: {r['monday']}")

    # Check leverage bounds
    for r in results:
        if not r["skip"] and (r["leverage"] < 0.5 or r["leverage"] > 2.0):
            errors.append(f"Leverage out of bounds: {r['leverage']}")

    print(f"\n{'='*70}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: All checks OK")
        print(f"  - Confidence tiers: valid")
        print(f"  - Adaptive stops: within [1%, 3%] bounds")
        print(f"  - Skip logic: LOW confidence LONGs skipped, SHORTs never skipped")
        print(f"  - Leverage: within [0.5, 2.0] bounds")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_smoke_test()
