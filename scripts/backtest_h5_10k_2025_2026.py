"""
H=5 LINEAR-ONLY $10K Backtest for 2025 and 2026.

Simulates weekly paper trading:
  - Train Ridge + BayesianRidge on expanding window (2020 onward)
  - Signal: ensemble mean of 2 models -> direction
  - Vol-targeting + asymmetric sizing (SHORT 1.0x, LONG 0.5x)
  - Trailing stop: activation=0.40%, trail=0.30%, hard_stop=4.00%
  - 5-day hold, re-entry after 30-min cooldown (simplified: no intraday re-entry)

Usage:
    python scripts/backtest_h5_10k_2025_2026.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data():
    """Load daily OHLCV + macro, build 21 features, compute H=5 target."""
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"

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
        df_macro_sub[col] = df_macro_sub[col].shift(1)  # T-1 anti-leakage

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date",
        direction="backward",
    )

    # Build 21 features
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

    return df, feature_cols


def simulate_trailing_stop(direction, entry_price, daily_bars, activation_pct=0.004, trail_pct=0.003, hard_stop_pct=0.04):
    """
    Simulate trailing stop on daily bars for a week.
    Returns (exit_price, exit_reason, exit_day_idx).
    """
    peak_price = entry_price
    trailing_activated = False

    for i, bar in enumerate(daily_bars):
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        if direction == 1:  # LONG
            # Hard stop check
            if (entry_price - bar_low) / entry_price >= hard_stop_pct:
                exit_price = entry_price * (1 - hard_stop_pct)
                return exit_price, "hard_stop", i

            # Update peak
            if bar_high > peak_price:
                peak_price = bar_high

            # Activation check
            peak_pnl = (peak_price - entry_price) / entry_price
            if peak_pnl >= activation_pct:
                trailing_activated = True

            # Trail trigger
            if trailing_activated:
                drawback = (peak_price - bar_low) / peak_price
                if drawback >= trail_pct:
                    exit_price = peak_price * (1 - trail_pct)
                    return exit_price, "trailing_stop", i

        else:  # SHORT
            # Hard stop check
            if (bar_high - entry_price) / entry_price >= hard_stop_pct:
                exit_price = entry_price * (1 + hard_stop_pct)
                return exit_price, "hard_stop", i

            # Update peak (lowest price for short)
            if bar_low < peak_price:
                peak_price = bar_low

            # Activation check
            peak_pnl = (entry_price - peak_price) / entry_price
            if peak_pnl >= activation_pct:
                trailing_activated = True

            # Trail trigger
            if trailing_activated:
                drawback = (bar_high - peak_price) / peak_price
                if drawback >= trail_pct:
                    exit_price = peak_price * (1 + trail_pct)
                    return exit_price, "trailing_stop", i

    # Week end - close at last bar's close
    return daily_bars[-1]["close"], "week_end", len(daily_bars) - 1


def run_backtest():
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # Vol-target params
    tv = 0.15
    max_lev = 2.0
    min_lev = 0.5
    vol_lookback = 21

    for year in [2025, 2026]:
        print(f"\n{'=' * 60}")
        print(f"  H=5 LINEAR-ONLY BACKTEST — {year}")
        print(f"{'=' * 60}")

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        # Get all Mondays in test period
        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        test_data["dow"] = test_data["date"].dt.dayofweek
        mondays = test_data[test_data["dow"] == 0]["date"].unique()

        if len(mondays) == 0:
            print(f"  No Mondays found in {year}")
            continue

        equity = 10000.0
        trades = []

        for monday in mondays:
            monday_ts = pd.Timestamp(monday)
            friday_ts = monday_ts + pd.offsets.BDay(4)

            # Train on everything before this Monday
            train_end = monday_ts - timedelta(days=1)
            df_train = df[(df["date"] <= train_end) & df["target_return_5d"].notna()].copy()
            mask = df_train[feature_cols].notna().all(axis=1) & df_train["target_return_5d"].notna()
            df_train = df_train[mask]

            if len(df_train) < 100:
                continue

            X_train = df_train[feature_cols].values.astype(np.float64)
            y_train = df_train["target_return_5d"].values.astype(np.float64)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_scaled, y_train)

            br = BayesianRidge(max_iter=300)
            br.fit(X_scaled, y_train)

            # Predict on latest available features
            latest_row = df_train[feature_cols].iloc[-1:].values.astype(np.float64)
            latest_scaled = scaler.transform(latest_row)

            pred_ridge = float(ridge.predict(latest_scaled)[0])
            pred_br = float(br.predict(latest_scaled)[0])
            ensemble_return = (pred_ridge + pred_br) / 2.0
            direction = 1 if ensemble_return > 0 else -1

            # Vol-targeting
            returns = df_train["return_1d"].dropna().values
            realized_vol = np.std(returns[-vol_lookback:]) * np.sqrt(252) if len(returns) >= vol_lookback else 0.15
            if realized_vol < 0.001:
                realized_vol = 0.15
            raw_lev = tv / realized_vol
            clipped_lev = np.clip(raw_lev, min_lev, max_lev)

            # Asymmetric sizing
            if direction == 1:
                final_lev = clipped_lev * 0.5  # LONG = half
            else:
                final_lev = clipped_lev * 1.0  # SHORT = full

            # Get entry price (Monday close)
            monday_row = df[df["date"] == monday_ts]
            if monday_row.empty:
                mask_after = df["date"] >= monday_ts
                if mask_after.any():
                    monday_row = df[mask_after].iloc[:1]
                else:
                    continue

            entry_price = float(monday_row["close"].iloc[0])

            # Get daily bars Tue-Fri for trailing stop simulation
            week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
            daily_bars = []
            for day in week_dates:
                day_row = df[df["date"] == day]
                if not day_row.empty:
                    daily_bars.append({
                        "high": float(day_row["high"].iloc[0]),
                        "low": float(day_row["low"].iloc[0]),
                        "close": float(day_row["close"].iloc[0]),
                    })

            if len(daily_bars) == 0:
                # No bars to trade - skip
                continue

            # Simulate trailing stop
            exit_price, exit_reason, exit_day = simulate_trailing_stop(
                direction, entry_price, daily_bars,
                activation_pct=0.004, trail_pct=0.003, hard_stop_pct=0.04,
            )

            # PnL
            if direction == 1:
                pnl_unleveraged = (exit_price - entry_price) / entry_price
            else:
                pnl_unleveraged = (entry_price - exit_price) / entry_price

            pnl_leveraged = pnl_unleveraged * final_lev

            # Apply to equity
            equity_before = equity
            equity *= (1 + pnl_leveraged)

            actual_close_5d = daily_bars[-1]["close"] if daily_bars else entry_price
            if direction == 1:
                actual_return = (actual_close_5d - entry_price) / entry_price
            else:
                actual_return = (entry_price - actual_close_5d) / entry_price

            trades.append({
                "week": str(monday_ts.date()),
                "direction": "LONG" if direction == 1 else "SHORT",
                "leverage": final_lev,
                "entry": entry_price,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_leveraged * 100,
                "equity": equity,
                "correct": actual_return > 0,
            })

        if not trades:
            print(f"  No trades executed in {year}")
            continue

        # Results
        df_trades = pd.DataFrame(trades)
        total_return = (equity / 10000.0 - 1) * 100
        n_trades = len(df_trades)
        n_long = len(df_trades[df_trades["direction"] == "LONG"])
        n_short = len(df_trades[df_trades["direction"] == "SHORT"])
        win_rate = len(df_trades[df_trades["pnl_pct"] > 0]) / n_trades * 100
        da = len(df_trades[df_trades["correct"]]) / n_trades * 100

        # Sharpe
        weekly_returns = df_trades["pnl_pct"].values / 100
        if np.std(weekly_returns) > 0:
            sharpe = np.mean(weekly_returns) / np.std(weekly_returns) * np.sqrt(52)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_curve = [10000.0]
        for r in weekly_returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - peak) / peak
        max_dd = np.min(dd) * 100

        # Win rate by direction
        long_trades = df_trades[df_trades["direction"] == "LONG"]
        short_trades = df_trades[df_trades["direction"] == "SHORT"]
        long_wr = len(long_trades[long_trades["pnl_pct"] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_wr = len(short_trades[short_trades["pnl_pct"] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0

        # Hard stops / trailing stops
        n_hard = len(df_trades[df_trades["exit_reason"] == "hard_stop"])
        n_trail = len(df_trades[df_trades["exit_reason"] == "trailing_stop"])
        n_weekend = len(df_trades[df_trades["exit_reason"] == "week_end"])

        # Bootstrap p-value
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(weekly_returns, size=len(weekly_returns), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        p_value = np.mean(boot_means <= 0)

        print(f"\n  $10,000 -> ${equity:,.2f}  ({total_return:+.2f}%)")
        print(f"  Trades: {n_trades} ({n_long}L / {n_short}S)")
        print(f"  Win Rate: {win_rate:.1f}% (LONG {long_wr:.1f}%, SHORT {short_wr:.1f}%)")
        print(f"  DA: {da:.1f}%")
        print(f"  Sharpe: {sharpe:.3f}")
        print(f"  Max DD: {max_dd:.2f}%")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Exits: {n_weekend} week_end, {n_trail} trailing, {n_hard} hard_stop")
        print(f"\n  Weekly PnL breakdown:")
        print(f"    Mean: {np.mean(weekly_returns)*100:+.3f}%")
        print(f"    Std:  {np.std(weekly_returns)*100:.3f}%")
        print(f"    Best: {np.max(weekly_returns)*100:+.3f}%")
        print(f"    Worst:{np.min(weekly_returns)*100:+.3f}%")

        # Print first/last 5 trades
        print(f"\n  First 5 trades:")
        for _, t in df_trades.head(5).iterrows():
            print(f"    {t['week']}  {t['direction']:5s}  lev={t['leverage']:.3f}  "
                  f"entry={t['entry']:.0f}  exit={t['exit']:.0f}  "
                  f"pnl={t['pnl_pct']:+.3f}%  [{t['exit_reason']}]")

        if len(df_trades) > 5:
            print(f"\n  Last 5 trades:")
            for _, t in df_trades.tail(5).iterrows():
                print(f"    {t['week']}  {t['direction']:5s}  lev={t['leverage']:.3f}  "
                      f"entry={t['entry']:.0f}  exit={t['exit']:.0f}  "
                      f"pnl={t['pnl_pct']:+.3f}%  [{t['exit_reason']}]")


if __name__ == "__main__":
    run_backtest()
