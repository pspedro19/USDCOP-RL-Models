"""
H=5 Return Analysis: Why 19% with trailing vs 32% without?
And: what combination of strategies reaches 30% APR?

Tests:
  A) H=5 simple 5-day hold (no trailing stop) — the raw signal alpha
  B) H=5 with trailing on daily bars (what we just measured: ~19%)
  C) Combined: Track A (H=1 SHORT-only) + Track B (H=5 bidirectional)
  D) H=5 with higher leverage caps
"""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data():
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
        df_macro_sub[col] = df_macro_sub[col].shift(1)

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date",
        direction="backward",
    )

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

    df["target_return_5d"] = np.log(df["close"].shift(-5) / df["close"])

    # Also H=1 target for Track A simulation
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

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


def simulate_trailing(direction, entry_price, daily_bars, act_pct, trail_pct, hard_pct):
    peak = entry_price
    activated = False
    for i, bar in enumerate(daily_bars):
        h, l, c = bar["high"], bar["low"], bar["close"]
        if direction == 1:
            if (entry_price - l) / entry_price >= hard_pct:
                return entry_price * (1 - hard_pct), "hard_stop", i
            if h > peak:
                peak = h
            if (peak - entry_price) / entry_price >= act_pct:
                activated = True
            if activated and (peak - l) / peak >= trail_pct:
                return peak * (1 - trail_pct), "trailing", i
        else:
            if (h - entry_price) / entry_price >= hard_pct:
                return entry_price * (1 + hard_pct), "hard_stop", i
            if l < peak:
                peak = l
            if (entry_price - peak) / entry_price >= act_pct:
                activated = True
            if activated and (h - peak) / peak >= trail_pct:
                return peak * (1 + trail_pct), "trailing", i
    return daily_bars[-1]["close"], "week_end", len(daily_bars) - 1


def compute_metrics(weekly_returns, initial=10000.0):
    eq = initial
    eq_curve = [eq]
    for r in weekly_returns:
        eq *= (1 + r)
        eq_curve.append(eq)
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = np.min(dd) * 100
    ret = (eq / initial - 1) * 100
    sharpe = np.mean(weekly_returns) / np.std(weekly_returns) * np.sqrt(52) if np.std(weekly_returns) > 0 else 0.0
    wr = np.mean(np.array(weekly_returns) > 0) * 100

    # Bootstrap p-value
    boot = [np.mean(np.random.choice(weekly_returns, size=len(weekly_returns), replace=True)) for _ in range(10000)]
    p_value = np.mean(np.array(boot) <= 0)

    return {"equity": eq, "return_pct": ret, "sharpe": sharpe, "max_dd": max_dd, "wr": wr, "p_value": p_value}


def run_h5(df, feature_cols, year, mode="simple_hold", act_pct=0.002, trail_pct=0.001,
           hard_pct=0.04, long_mult=1.0, short_mult=1.0, max_lev=2.0, min_lev=0.5, tv=0.15):
    """Run H=5 strategy for a given year.
    mode: 'simple_hold' (Friday close), 'trailing' (with trailing stop), 'hard_only' (hard stop but no trailing)
    """
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    test_data["dow"] = test_data["date"].dt.dayofweek
    mondays = test_data[test_data["dow"] == 0]["date"].unique()

    weekly_returns = []
    directions = []

    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        friday_ts = monday_ts + pd.offsets.BDay(4)

        train_end = monday_ts - timedelta(days=1)
        df_train = df[(df["date"] <= train_end) & df["target_return_5d"].notna()].copy()
        mask = df_train[feature_cols].notna().all(axis=1) & df_train["target_return_5d"].notna()
        df_train = df_train[mask]
        if len(df_train) < 100:
            continue

        X = df_train[feature_cols].values.astype(np.float64)
        y = df_train["target_return_5d"].values.astype(np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=1.0).fit(Xs, y)
        br = BayesianRidge(max_iter=300).fit(Xs, y)

        latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
        ens = (float(ridge.predict(latest)[0]) + float(br.predict(latest)[0])) / 2.0
        direction = 1 if ens > 0 else -1

        # Vol-target
        rets = df_train["return_1d"].dropna().values
        rv = np.std(rets[-21:]) * np.sqrt(252) if len(rets) >= 21 else 0.15
        if rv < 0.001:
            rv = 0.15
        lev = np.clip(tv / rv, min_lev, max_lev)

        # Asymmetric
        mult = long_mult if direction == 1 else short_mult
        if mult == 0.0 and direction == 1:
            continue  # SHORT-only
        lev *= mult

        # Entry
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])

        # Week bars
        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        bars = []
        for day in week_dates:
            r = df[df["date"] == day]
            if not r.empty:
                bars.append({"high": float(r["high"].iloc[0]), "low": float(r["low"].iloc[0]), "close": float(r["close"].iloc[0])})
        if not bars:
            continue

        if mode == "simple_hold":
            exit_p = bars[-1]["close"]
        elif mode == "hard_only":
            # Only hard stop, no trailing
            exit_p = bars[-1]["close"]
            for bar in bars:
                if direction == 1 and (entry - bar["low"]) / entry >= hard_pct:
                    exit_p = entry * (1 - hard_pct)
                    break
                elif direction == -1 and (bar["high"] - entry) / entry >= hard_pct:
                    exit_p = entry * (1 + hard_pct)
                    break
        elif mode == "trailing":
            exit_p, _, _ = simulate_trailing(direction, entry, bars, act_pct, trail_pct, hard_pct)

        if direction == 1:
            pnl = (exit_p - entry) / entry * lev
        else:
            pnl = (entry - exit_p) / entry * lev

        weekly_returns.append(pnl)
        directions.append(direction)

    return weekly_returns, directions


def run_h1_daily(df, feature_cols, year, mode="short_only", act_pct=0.002, trail_pct=0.003,
                 hard_pct=0.015, max_lev=2.0, min_lev=0.5, tv=0.15):
    """Run H=1 daily strategy (Track A)."""
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    trading_days = test_data["date"].unique()

    daily_returns = []

    for day in trading_days:
        day_ts = pd.Timestamp(day)
        train_end = day_ts - timedelta(days=1)
        df_train = df[(df["date"] <= train_end) & df["target_return_1d"].notna()].copy()
        mask = df_train[feature_cols].notna().all(axis=1) & df_train["target_return_1d"].notna()
        df_train = df_train[mask]
        if len(df_train) < 100:
            continue

        X = df_train[feature_cols].values.astype(np.float64)
        y = df_train["target_return_1d"].values.astype(np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=1.0).fit(Xs, y)
        br = BayesianRidge(max_iter=300).fit(Xs, y)

        latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
        ens = (float(ridge.predict(latest)[0]) + float(br.predict(latest)[0])) / 2.0
        direction = 1 if ens > 0 else -1

        if mode == "short_only" and direction == 1:
            daily_returns.append(0.0)  # Flat when LONG signal
            continue

        # Vol-target
        rets = df_train["return_1d"].dropna().values
        rv = np.std(rets[-21:]) * np.sqrt(252) if len(rets) >= 21 else 0.15
        if rv < 0.001:
            rv = 0.15
        lev = np.clip(tv / rv, min_lev, max_lev)

        # Entry & exit
        day_row = df[df["date"] == day_ts]
        if day_row.empty:
            continue
        entry = float(day_row["close"].iloc[0])

        # Next day
        next_rows = df[df["date"] > day_ts]
        if next_rows.empty:
            continue
        next_day = next_rows.iloc[0]
        exit_p = float(next_day["close"])

        # Simple hard stop check using next day's high/low
        next_high = float(next_day["high"])
        next_low = float(next_day["low"])

        if direction == 1:
            if (entry - next_low) / entry >= hard_pct:
                exit_p = entry * (1 - hard_pct)
            pnl = (exit_p - entry) / entry * lev
        else:
            if (next_high - entry) / entry >= hard_pct:
                exit_p = entry * (1 + hard_pct)
            pnl = (entry - exit_p) / entry * lev

        daily_returns.append(pnl)

    return daily_returns


def main():
    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows\n")

    for year in [2025, 2026]:
        print(f"\n{'#' * 70}")
        print(f"#  YEAR: {year}")
        print(f"{'#' * 70}")

        # ===== STRATEGY A: H=5 Simple Hold (no trailing) =====
        wr_a, dirs_a = run_h5(df, feature_cols, year, mode="simple_hold",
                              long_mult=1.0, short_mult=1.0, max_lev=2.0)
        m_a = compute_metrics(wr_a) if wr_a else None

        # ===== STRATEGY B: H=5 Simple Hold + Higher Leverage =====
        wr_b, dirs_b = run_h5(df, feature_cols, year, mode="simple_hold",
                              long_mult=1.0, short_mult=1.0, max_lev=3.0)
        m_b = compute_metrics(wr_b) if wr_b else None

        # ===== STRATEGY C: H=5 Simple Hold + Asymmetric (L=0.5, S=1.0) + MaxLev=3.0 =====
        wr_c, dirs_c = run_h5(df, feature_cols, year, mode="simple_hold",
                              long_mult=0.5, short_mult=1.0, max_lev=3.0)
        m_c = compute_metrics(wr_c) if wr_c else None

        # ===== STRATEGY D: H=5 Hard Stop only (no trailing) =====
        wr_d, dirs_d = run_h5(df, feature_cols, year, mode="hard_only", hard_pct=0.02,
                              long_mult=1.0, short_mult=1.0, max_lev=2.0)
        m_d = compute_metrics(wr_d) if wr_d else None

        # ===== STRATEGY E: H=5 Trailing (best from grid: act=0.20%, trail=0.10%) =====
        wr_e, dirs_e = run_h5(df, feature_cols, year, mode="trailing",
                              act_pct=0.002, trail_pct=0.001, hard_pct=0.04,
                              long_mult=1.0, short_mult=1.0, max_lev=3.0)
        m_e = compute_metrics(wr_e) if wr_e else None

        # ===== STRATEGY F: H=5 SHORT-ONLY, simple hold =====
        wr_f, dirs_f = run_h5(df, feature_cols, year, mode="simple_hold",
                              long_mult=0.0, short_mult=1.0, max_lev=3.0)
        m_f = compute_metrics(wr_f) if wr_f else None

        # ===== STRATEGY G: H=5 Simple Hold + Aggressive (L=0.7, S=1.0, MaxLev=3.0) =====
        wr_g, dirs_g = run_h5(df, feature_cols, year, mode="simple_hold",
                              long_mult=0.7, short_mult=1.0, max_lev=3.0)
        m_g = compute_metrics(wr_g) if wr_g else None

        strategies = [
            ("A) H5 hold, sym, lev=2.0", wr_a, dirs_a, m_a),
            ("B) H5 hold, sym, lev=3.0", wr_b, dirs_b, m_b),
            ("C) H5 hold, asym(L=0.5), lev=3.0", wr_c, dirs_c, m_c),
            ("D) H5 hard_stop=2%, sym, lev=2.0", wr_d, dirs_d, m_d),
            ("E) H5 trail(0.2/0.1%), lev=3.0", wr_e, dirs_e, m_e),
            ("F) H5 SHORT-ONLY, lev=3.0", wr_f, dirs_f, m_f),
            ("G) H5 hold, asym(L=0.7), lev=3.0", wr_g, dirs_g, m_g),
        ]

        print(f"\n{'Strategy':<40} {'$10K->':>10} {'Ret%':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR%':>6} {'Trd':>4} {'p-val':>7}")
        print("-" * 98)
        for name, wr, dirs, m in strategies:
            if m:
                n_l = sum(1 for d in dirs if d == 1)
                n_s = sum(1 for d in dirs if d == -1)
                ls_str = f"{n_l}L/{n_s}S"
                print(f"{name:<40} ${m['equity']:>9,.0f} {m['return_pct']:>+7.2f}% "
                      f"{m['sharpe']:>7.3f} {m['max_dd']:>6.2f}% {m['wr']:>5.1f} {len(wr):>4} {m['p_value']:>7.4f}")
            else:
                print(f"{name:<40} {'N/A':>10}")

        # ===== COMBINED: Track A (H=1 SHORT-only) + Track B (H=5) =====
        # Track A: daily H=1, SHORT-only, existing production params
        print(f"\n  --- Track A + B Combinations ({year}) ---")
        print(f"  (50% capital Track A H=1 SHORT-only + 50% Track B H=5)")

        # This takes much longer so only run a couple combos
        # For speed, approximate Track A with weekly aggregation
        # Train once per week (same Monday) for H=1 to save time
        daily_rets_a = run_h1_daily(df, feature_cols, year, mode="short_only",
                                    hard_pct=0.015, max_lev=2.0)
        if daily_rets_a:
            m_h1 = compute_metrics(daily_rets_a, initial=5000.0)  # Half capital
            print(f"\n  Track A alone (H=1 SHORT-only, $5K): ${m_h1['equity']:,.0f} ({m_h1['return_pct']:+.2f}%) "
                  f"Sharpe={m_h1['sharpe']:.3f} trades={len(daily_rets_a)}")

            # Best H=5 for combining: Strategy B (simple hold, sym, lev=3.0)
            for label, wr_x, m_x in [
                ("B) sym lev=3.0", wr_b, m_b),
                ("G) asym(L=0.7) lev=3.0", wr_g, m_g),
            ]:
                if not m_x:
                    continue
                # Interleave returns: Track A runs daily, Track B runs weekly
                # Simple: final equity = (Track A equity on $5K) + (Track B equity on $5K)
                m_x_half = compute_metrics(wr_x, initial=5000.0)
                combined_eq = m_h1["equity"] + m_x_half["equity"]
                combined_ret = (combined_eq / 10000.0 - 1) * 100
                print(f"  A + {label} ($5K+$5K): ${combined_eq:,.0f} ({combined_ret:+.2f}%)")

        # ===== AGGRESSIVE: Just maxing out leverage =====
        print(f"\n  --- High Leverage Scenarios ({year}) ---")
        for ml in [3.0, 4.0, 5.0]:
            wr_agg, dirs_agg = run_h5(df, feature_cols, year, mode="simple_hold",
                                       long_mult=0.7, short_mult=1.0, max_lev=ml, min_lev=0.5)
            if wr_agg:
                m_agg = compute_metrics(wr_agg)
                print(f"  H5 hold asym(L=0.7) max_lev={ml:.0f}: ${m_agg['equity']:>9,.0f} "
                      f"({m_agg['return_pct']:>+7.2f}%) Sharpe={m_agg['sharpe']:.3f} "
                      f"MaxDD={m_agg['max_dd']:.2f}%")


if __name__ == "__main__":
    main()
