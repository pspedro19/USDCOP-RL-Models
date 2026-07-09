"""
H=5 Sensitivity Analysis: Find config for ~30% APR.

Grid search over:
  - activation_pct: [0.10%, 0.15%, 0.20%, 0.30%, 0.40%]
  - trail_pct:      [0.10%, 0.15%, 0.20%, 0.30%]
  - hard_stop_pct:  [1.0%, 2.0%, 3.0%, 4.0%]
  - long_mult:      [0.0 (SHORT-only), 0.3, 0.5, 0.7, 1.0]
  - max_leverage:   [2.0, 2.5, 3.0]

Reports top configs by 2025 return, with 2026 OOS check.
"""

import sys
from pathlib import Path
from datetime import timedelta
from itertools import product

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


def run_year(df, feature_cols, year, act_pct, trail_pct, hard_pct, long_mult, short_mult, max_lev, min_lev=0.5, tv=0.15):
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    test_data["dow"] = test_data["date"].dt.dayofweek
    mondays = test_data[test_data["dow"] == 0]["date"].unique()

    equity = 10000.0
    weekly_returns = []
    n_long = 0
    n_short = 0
    n_hard = 0

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
        if mult == 0.0:
            # SHORT-only: skip LONG signals
            if direction == 1:
                continue
        lev *= mult

        if direction == 1:
            n_long += 1
        else:
            n_short += 1

        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])

        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        bars = []
        for day in week_dates:
            r = df[df["date"] == day]
            if not r.empty:
                bars.append({"high": float(r["high"].iloc[0]), "low": float(r["low"].iloc[0]), "close": float(r["close"].iloc[0])})
        if not bars:
            continue

        exit_p, reason, _ = simulate_trailing(direction, entry, bars, act_pct, trail_pct, hard_pct)
        if reason == "hard_stop":
            n_hard += 1

        if direction == 1:
            pnl = (exit_p - entry) / entry * lev
        else:
            pnl = (entry - exit_p) / entry * lev

        equity *= (1 + pnl)
        weekly_returns.append(pnl)

    if not weekly_returns:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0, "n_long": 0, "n_short": 0, "n_hard": 0, "wr": 0.0}

    ret = (equity / 10000.0 - 1) * 100
    wr_arr = np.array(weekly_returns)
    wr_pct = np.mean(wr_arr > 0) * 100
    sharpe = np.mean(wr_arr) / np.std(wr_arr) * np.sqrt(52) if np.std(wr_arr) > 0 else 0.0

    eq_curve = [10000.0]
    for r in weekly_returns:
        eq_curve.append(eq_curve[-1] * (1 + r))
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = np.min(dd) * 100

    return {
        "return_pct": ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": len(weekly_returns),
        "n_long": n_long,
        "n_short": n_short,
        "n_hard": n_hard,
        "wr": wr_pct,
    }


def main():
    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # Grid
    activations = [0.001, 0.0015, 0.002, 0.003, 0.004]
    trails = [0.001, 0.0015, 0.002, 0.003]
    hard_stops = [0.010, 0.015, 0.020, 0.030, 0.040]
    long_mults = [0.0, 0.3, 0.5, 0.7, 1.0]
    max_levs = [2.0, 2.5, 3.0]

    combos = list(product(activations, trails, hard_stops, long_mults, max_levs))
    print(f"Total combinations: {len(combos)}")
    print("Running 2025 grid search...\n")

    results = []
    for i, (act, trail, hard, lm, ml) in enumerate(combos):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(combos)} ({i/len(combos)*100:.0f}%)")

        r25 = run_year(df, feature_cols, 2025, act, trail, hard, lm, 1.0, ml)
        results.append({
            "act": act, "trail": trail, "hard": hard,
            "long_mult": lm, "max_lev": ml,
            "ret_2025": r25["return_pct"],
            "sharpe_2025": r25["sharpe"],
            "max_dd_2025": r25["max_dd"],
            "trades_2025": r25["trades"],
            "n_long_2025": r25["n_long"],
            "n_short_2025": r25["n_short"],
            "n_hard_2025": r25["n_hard"],
            "wr_2025": r25["wr"],
        })

    df_results = pd.DataFrame(results)

    # Filter: Sharpe > 1.0, max_dd > -15%, trades >= 15
    viable = df_results[
        (df_results["sharpe_2025"] > 1.0) &
        (df_results["max_dd_2025"] > -15.0) &
        (df_results["trades_2025"] >= 15)
    ].sort_values("ret_2025", ascending=False)

    print(f"\n{'=' * 90}")
    print(f"TOP 20 CONFIGS (2025 return, Sharpe > 1.0, DD > -15%, trades >= 15)")
    print(f"{'=' * 90}")
    print(f"{'#':>3} {'Act%':>5} {'Trl%':>5} {'Hard%':>6} {'L_mult':>6} {'MaxLev':>6} | "
          f"{'Ret%':>7} {'Shrpe':>6} {'MaxDD':>7} {'Trd':>4} {'L/S':>7} {'Hard':>4} {'WR%':>5}")
    print("-" * 90)

    top20 = viable.head(20)
    for idx, (_, row) in enumerate(top20.iterrows()):
        ls = f"{int(row['n_long_2025'])}L/{int(row['n_short_2025'])}S"
        print(f"{idx+1:>3} {row['act']*100:>5.2f} {row['trail']*100:>5.2f} {row['hard']*100:>6.2f} "
              f"{row['long_mult']:>6.1f} {row['max_lev']:>6.1f} | "
              f"{row['ret_2025']:>+7.2f} {row['sharpe_2025']:>6.3f} {row['max_dd_2025']:>7.2f} "
              f"{int(row['trades_2025']):>4} {ls:>7} {int(row['n_hard_2025']):>4} {row['wr_2025']:>5.1f}")

    # Now run TOP 5 on 2026 as OOS validation
    print(f"\n{'=' * 90}")
    print(f"TOP 5 — 2026 OOS VALIDATION")
    print(f"{'=' * 90}")

    for idx, (_, row) in enumerate(top20.head(5).iterrows()):
        r26 = run_year(df, feature_cols, 2026, row["act"], row["trail"], row["hard"],
                       row["long_mult"], 1.0, row["max_lev"])
        ls_25 = f"{int(row['n_long_2025'])}L/{int(row['n_short_2025'])}S"
        ls_26 = f"{r26['n_long']}L/{r26['n_short']}S"
        print(f"\n  Config #{idx+1}: act={row['act']*100:.2f}% trail={row['trail']*100:.2f}% "
              f"hard={row['hard']*100:.2f}% long_mult={row['long_mult']:.1f} max_lev={row['max_lev']:.1f}")
        print(f"    2025: ${10000*(1+row['ret_2025']/100):>10,.2f} ({row['ret_2025']:>+.2f}%) "
              f"Sharpe={row['sharpe_2025']:.3f} DD={row['max_dd_2025']:.2f}% "
              f"trades={int(row['trades_2025'])} {ls_25} hard={int(row['n_hard_2025'])} WR={row['wr_2025']:.1f}%")
        print(f"    2026: ${10000*(1+r26['return_pct']/100):>10,.2f} ({r26['return_pct']:>+.2f}%) "
              f"Sharpe={r26['sharpe']:.3f} DD={r26['max_dd']:.2f}% "
              f"trades={r26['trades']} {ls_26} hard={r26['n_hard']} WR={r26['wr']:.1f}%")
        combined = (1 + row['ret_2025']/100) * (1 + r26['return_pct']/100)
        print(f"    Combined: $10K -> ${10000*combined:,.2f} ({(combined-1)*100:+.2f}%)")

    # Also check: what % of combos achieve >= 25% in 2025?
    print(f"\n{'=' * 60}")
    print(f"DISTRIBUTION OF 2025 RETURNS")
    print(f"{'=' * 60}")
    for threshold in [10, 15, 20, 25, 30, 35, 40, 50]:
        n = len(df_results[df_results["ret_2025"] >= threshold])
        pct = n / len(df_results) * 100
        print(f"  >= {threshold}%: {n}/{len(df_results)} combos ({pct:.1f}%)")

    # SHORT-only vs bidirectional comparison
    print(f"\n{'=' * 60}")
    print(f"SHORT-ONLY vs BIDIRECTIONAL (best of each)")
    print(f"{'=' * 60}")
    short_only = viable[viable["long_mult"] == 0.0]
    bidirectional = viable[viable["long_mult"] > 0.0]
    if not short_only.empty:
        best_so = short_only.iloc[0]
        print(f"  SHORT-only best:  {best_so['ret_2025']:+.2f}% Sharpe={best_so['sharpe_2025']:.3f} "
              f"act={best_so['act']*100:.2f}% trail={best_so['trail']*100:.2f}% hard={best_so['hard']*100:.2f}% "
              f"max_lev={best_so['max_lev']:.1f}")
    if not bidirectional.empty:
        best_bi = bidirectional.iloc[0]
        print(f"  Bidirectional best: {best_bi['ret_2025']:+.2f}% Sharpe={best_bi['sharpe_2025']:.3f} "
              f"act={best_bi['act']*100:.2f}% trail={best_bi['trail']*100:.2f}% hard={best_bi['hard']*100:.2f}% "
              f"long_mult={best_bi['long_mult']:.1f} max_lev={best_bi['max_lev']:.1f}")


if __name__ == "__main__":
    main()
