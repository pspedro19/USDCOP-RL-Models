"""
Smoke Test: H=5 Linear-Only Paper Trading Pipeline
===================================================

Runs 4 weeks of simulated paper trading offline (no Airflow, no DB):
    Week N: train -> signal -> vol-target -> asymmetric sizing -> execute -> monitor

Uses 2024 data as a mini-validation. Validates that:
    1. Training produces non-collapsed Ridge + BayesianRidge models
    2. Signal generation produces valid direction + return
    3. Vol-targeting + asymmetric sizing applies correctly
    4. MultiDayExecutor enters, monitors, and closes positions
    5. PnL aggregation is correct

Usage:
    python scripts/validate_h5_paper_trading.py

Exit code 0 = all checks pass. Non-zero = failure.

Contract: FC-H5-SMOKE-001
Date: 2026-02-16
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data():
    """Load daily OHLCV + macro, build features, compute H=5 target."""
    ohlcv_path = PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_daily_ohlcv.parquet'
    macro_path = PROJECT_ROOT / 'data' / 'pipeline' / '04_cleaning' / 'output' / 'MACRO_DAILY_CLEAN.parquet'

    if not ohlcv_path.exists():
        print(f"ERROR: OHLCV not found: {ohlcv_path}")
        sys.exit(1)
    if not macro_path.exists():
        print(f"ERROR: Macro not found: {macro_path}")
        sys.exit(1)

    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)

    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_sub = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_sub.rename(columns=macro_cols, inplace=True)
    df_macro_sub = df_macro_sub.sort_values("date")
    for col in macro_cols.values():
        df_macro_sub[col] = df_macro_sub[col].shift(1)

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date",
        direction="backward",
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
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    for col in ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]:
        df[col] = df[col].ffill()

    # H=5 target
    df["target_return_5d"] = np.log(df["close"].shift(-5) / df["close"])

    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
        "vix_close_lag1", "embi_close_lag1",
    ]

    mask = df[feature_cols].notna().all(axis=1) & df["target_return_5d"].notna()
    df_clean = df[mask].reset_index(drop=True)

    return df_clean, feature_cols


def run_smoke_test():
    """Run 4 simulated weeks of H=5 paper trading."""
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.vol_targeting import (
        VolTargetConfig,
        compute_vol_target_signal,
        compute_realized_vol,
        apply_asymmetric_sizing,
    )
    from src.execution.multiday_executor import MultiDayConfig, MultiDayExecutor

    print("=" * 60)
    print("H5 PAPER TRADING SMOKE TEST")
    print("=" * 60)

    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # Use 4 weeks in late 2024 for testing
    # Train on 2020-2024-Q3, test weeks in 2024-Q4
    test_start = pd.Timestamp("2024-10-01")
    test_weeks = df[df["date"] >= test_start].copy()

    # Get Monday dates (simulate weekly signal generation)
    test_weeks["dow"] = test_weeks["date"].dt.dayofweek
    mondays = test_weeks[test_weeks["dow"] == 0]["date"].unique()[:4]

    if len(mondays) < 4:
        print(f"WARNING: Only {len(mondays)} Mondays found in test period")
        if len(mondays) == 0:
            print("ERROR: No Mondays in test period")
            sys.exit(1)

    print(f"Testing {len(mondays)} weeks: {[str(m)[:10] for m in mondays]}")

    # MultiDayExecutor config
    exec_config = MultiDayConfig(
        activation_pct=0.002,
        trail_pct=0.001,
        hard_stop_pct=0.035,
        cooldown_minutes=20,
    )
    executor = MultiDayExecutor(exec_config)

    vol_config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5)
    results = []
    checks_passed = 0
    checks_total = 0

    for i, monday in enumerate(mondays):
        monday_ts = pd.Timestamp(monday)
        friday_ts = monday_ts + pd.offsets.BDay(4)
        print(f"\n--- Week {i+1}: {str(monday_ts.date())} to {str(friday_ts.date())} ---")

        # 1. TRAIN: expanding window up to Friday before signal
        train_end = monday_ts - timedelta(days=1)
        df_train = df[df["date"] <= train_end].copy()
        X_train = df_train[feature_cols].values.astype(np.float64)
        y_train = df_train["target_return_5d"].values.astype(np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y_train)

        br = BayesianRidge(max_iter=300)
        br.fit(X_scaled, y_train)

        # Check: models produce non-zero predictions
        checks_total += 1
        ridge_preds = ridge.predict(X_scaled)
        br_preds = br.predict(X_scaled)
        ridge_std = np.std(ridge_preds)
        br_std = np.std(br_preds)
        if ridge_std > 0.001 and br_std > 0.001:
            checks_passed += 1
            print(f"  [PASS] Models not collapsed (ridge_std={ridge_std:.5f}, br_std={br_std:.5f})")
        else:
            print(f"  [FAIL] Model collapsed! ridge_std={ridge_std:.5f}, br_std={br_std:.5f}")

        # 2. SIGNAL: predict on latest features
        latest_row = df_train[feature_cols].iloc[-1:].values.astype(np.float64)
        latest_scaled = scaler.transform(latest_row)

        pred_ridge = float(ridge.predict(latest_scaled)[0])
        pred_br = float(br.predict(latest_scaled)[0])
        ensemble_return = (pred_ridge + pred_br) / 2.0
        direction = 1 if ensemble_return > 0 else -1

        checks_total += 1
        if abs(ensemble_return) > 0:
            checks_passed += 1
            dir_str = "LONG" if direction == 1 else "SHORT"
            print(f"  [PASS] Signal: {dir_str}, return={ensemble_return:+.6f}")
        else:
            print(f"  [FAIL] Zero ensemble return")

        # 3. VOL-TARGET + ASYMMETRIC SIZING
        returns = df_train["return_1d"].dropna().values
        realized_vol = compute_realized_vol(returns, lookback=21)

        vt_signal = compute_vol_target_signal(
            forecast_direction=direction,
            forecast_return=ensemble_return,
            realized_vol_21d=realized_vol,
            config=vol_config,
        )

        asymmetric_lev = apply_asymmetric_sizing(
            leverage=vt_signal.clipped_leverage,
            direction=direction,
            long_mult=0.5,
            short_mult=1.0,
        )

        checks_total += 1
        if direction == 1:
            expected_mult = 0.5
        else:
            expected_mult = 1.0
        if abs(asymmetric_lev - vt_signal.clipped_leverage * expected_mult) < 1e-10:
            checks_passed += 1
            print(f"  [PASS] Asymmetric: clipped={vt_signal.clipped_leverage:.3f} x{expected_mult} = {asymmetric_lev:.3f}")
        else:
            print(f"  [FAIL] Asymmetric sizing mismatch")

        # 4. EXECUTE: simulate 5-day hold with trailing stop
        # Use actual close price on Monday as entry
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            # Find nearest trading day
            mask = df["date"] >= monday_ts
            if mask.any():
                monday_row = df[mask].iloc[:1]
            else:
                print(f"  [SKIP] No data for {monday_ts.date()}")
                continue

        entry_price = float(monday_row["close"].iloc[0])
        entry_ts = datetime(monday_ts.year, monday_ts.month, monday_ts.day, 14, 0, tzinfo=timezone.utc)

        state = executor.enter(str(monday_ts.date()), direction, asymmetric_lev, entry_price, entry_ts)

        checks_total += 1
        if state.status.value == "positioned" and len(state.subtrades) == 1:
            checks_passed += 1
            print(f"  [PASS] Entered: price={entry_price:.2f}, leverage={asymmetric_lev:.3f}")
        else:
            print(f"  [FAIL] Entry failed: status={state.status}")

        # Simulate daily close bars (Tue-Fri) as 30-min monitoring
        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        for day in week_dates:
            day_row = df[df["date"] == day]
            if day_row.empty:
                continue

            bar_high = float(day_row["high"].iloc[0])
            bar_low = float(day_row["low"].iloc[0])
            bar_close = float(day_row["close"].iloc[0])
            bar_ts = datetime(day.year, day.month, day.day, 15, 0, tzinfo=timezone.utc)

            if executor.should_monitor(state):
                state, event = executor.update(state, bar_high, bar_low, bar_close, bar_ts)
                if event in ("trailing_exit", "hard_stop"):
                    print(f"    {day.date()}: {event} at {state.subtrades[-1].exit_price:.2f}")
                    # No re-entry in smoke test for simplicity

        # Close week
        if executor.should_monitor(state):
            friday_row = df[df["date"] == friday_ts]
            if not friday_row.empty:
                last_close = float(friday_row["close"].iloc[0])
            else:
                last_close = entry_price  # Fallback
            close_ts = datetime(friday_ts.year, friday_ts.month, friday_ts.day, 17, 50, tzinfo=timezone.utc)
            state = executor.close_week(state, last_close, close_ts)

        # 5. PnL CHECK
        checks_total += 1
        if state.week_pnl_pct is not None:
            checks_passed += 1
            print(f"  [PASS] Week PnL: {state.week_pnl_pct:+.4f}% ({len(state.subtrades)} subtrades)")
        else:
            print(f"  [FAIL] No PnL computed")

        results.append({
            "week": str(monday_ts.date()),
            "direction": direction,
            "leverage": asymmetric_lev,
            "pnl_pct": state.week_pnl_pct,
            "n_subtrades": len(state.subtrades),
        })

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Checks passed: {checks_passed}/{checks_total}")
    print()

    for r in results:
        dir_str = "LONG" if r["direction"] == 1 else "SHORT"
        pnl_str = f"{r['pnl_pct']:+.4f}%" if r["pnl_pct"] is not None else "N/A"
        print(f"  {r['week']}  {dir_str:5s}  lev={r['leverage']:.3f}  pnl={pnl_str}  subs={r['n_subtrades']}")

    total_pnl = sum(r["pnl_pct"] for r in results if r["pnl_pct"] is not None)
    print(f"\nTotal PnL (4 weeks): {total_pnl:+.4f}%")

    if checks_passed == checks_total:
        print("\n*** ALL CHECKS PASSED ***")
        return 0
    else:
        print(f"\n*** {checks_total - checks_passed} CHECKS FAILED ***")
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_test()
    sys.exit(exit_code)
