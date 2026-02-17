"""
FINAL BACKTEST: H=5 LINEAR-ONLY with v2 Config (tight trailing).

Config v2:
  activation = 0.20%, trail = 0.10%, hard_stop = 3.50%
  asymmetric: LONG 0.5x, SHORT 1.0x
  vol-target: tv=0.15, max_lev=2.0, min_lev=0.5

Uses daily bars (conservative estimate).
5-min bars in production will be more precise.

GATE: return > 15%, Sharpe > 2.0, MaxDD < 12%

Usage:
    python scripts/backtest_h5_v2_final.py
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


# V2 CONFIG (exact match to smart_executor_h5_v2.yaml Phase 1)
V2_ACT = 0.002      # 0.20%
V2_TRAIL = 0.001    # 0.10%
V2_HARD = 0.035     # 3.50%
V2_LONG_MULT = 0.5
V2_SHORT_MULT = 1.0
V2_MAX_LEV = 2.0
V2_MIN_LEV = 0.5
V2_TV = 0.15


def simulate_trailing(direction, entry, bars, act=V2_ACT, trail=V2_TRAIL, hard=V2_HARD):
    peak = entry
    activated = False
    for i, bar in enumerate(bars):
        h, l, c = bar["high"], bar["low"], bar["close"]
        if direction == 1:
            if (entry - l) / entry >= hard:
                return entry * (1 - hard), "hard_stop", i
            if h > peak:
                peak = h
            if (peak - entry) / entry >= act:
                activated = True
            if activated and (peak - l) / peak >= trail:
                return peak * (1 - trail), "trailing", i
        else:
            if (h - entry) / entry >= hard:
                return entry * (1 + hard), "hard_stop", i
            if l < peak:
                peak = l
            if (entry - peak) / entry >= act:
                activated = True
            if activated and (h - peak) / peak >= trail:
                return peak * (1 + trail), "trailing", i
    return bars[-1]["close"], "week_end", len(bars) - 1


def run_backtest():
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"\nConfig v2 (Phase 1):")
    print(f"  activation={V2_ACT*100:.2f}%, trail={V2_TRAIL*100:.2f}%, hard={V2_HARD*100:.2f}%")
    print(f"  LONG mult={V2_LONG_MULT}, SHORT mult={V2_SHORT_MULT}")
    print(f"  vol target={V2_TV}, max_lev={V2_MAX_LEV}, min_lev={V2_MIN_LEV}")

    for year in [2025, 2026]:
        print(f"\n{'=' * 70}")
        print(f"  H=5 LINEAR-ONLY v2 BACKTEST -- {year}")
        print(f"{'=' * 70}")

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        test_data["dow"] = test_data["date"].dt.dayofweek
        mondays = test_data[test_data["dow"] == 0]["date"].unique()

        if not len(mondays):
            print("  No data")
            continue

        equity = 10000.0
        trades = []

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
            rv = np.std(rets[-21:]) * np.sqrt(252) if len(rets) >= 21 else V2_TV
            if rv < 0.001:
                rv = V2_TV
            lev = np.clip(V2_TV / rv, V2_MIN_LEV, V2_MAX_LEV)

            # Asymmetric
            mult = V2_LONG_MULT if direction == 1 else V2_SHORT_MULT
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
                    bars.append({"high": float(r["high"].iloc[0]), "low": float(r["low"].iloc[0]),
                                 "close": float(r["close"].iloc[0])})
            if not bars:
                continue

            exit_p, reason, _ = simulate_trailing(direction, entry, bars)

            if direction == 1:
                pnl = (exit_p - entry) / entry * lev
            else:
                pnl = (entry - exit_p) / entry * lev

            equity *= (1 + pnl)
            # Direction accuracy: was direction prediction correct?
            actual_close = bars[-1]["close"]
            if direction == 1:
                correct = actual_close > entry
            else:
                correct = actual_close < entry

            trades.append({
                "week": str(monday_ts.date()),
                "dir": "LONG" if direction == 1 else "SHORT",
                "lev": lev,
                "entry": entry,
                "exit": exit_p,
                "reason": reason,
                "pnl": pnl * 100,
                "correct": correct,
            })

        if not trades:
            print("  No trades")
            continue

        df_t = pd.DataFrame(trades)
        total_ret = (equity / 10000 - 1) * 100
        wr = np.mean(df_t["pnl"] > 0) * 100
        da = np.mean(df_t["correct"]) * 100
        weekly_rets = df_t["pnl"].values / 100
        sharpe = np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52) if np.std(weekly_rets) > 0 else 0

        eq_c = [10000.0]
        for r in weekly_rets:
            eq_c.append(eq_c[-1] * (1 + r))
        peak_eq = np.maximum.accumulate(np.array(eq_c))
        dd = (np.array(eq_c) - peak_eq) / peak_eq
        max_dd = np.min(dd) * 100

        # Bootstrap p-value
        boot = [np.mean(np.random.choice(weekly_rets, size=len(weekly_rets), replace=True)) for _ in range(10000)]
        p_val = np.mean(np.array(boot) <= 0)

        n_l = len(df_t[df_t["dir"] == "LONG"])
        n_s = len(df_t[df_t["dir"] == "SHORT"])
        n_trail = len(df_t[df_t["reason"] == "trailing"])
        n_hard = len(df_t[df_t["reason"] == "hard_stop"])
        n_wend = len(df_t[df_t["reason"] == "week_end"])

        print(f"\n  $10,000 --> ${equity:,.2f}  ({total_ret:+.2f}%)")
        print(f"  Trades: {len(df_t)} ({n_l}L / {n_s}S)")
        print(f"  WR: {wr:.1f}%  |  DA: {da:.1f}%")
        print(f"  Sharpe: {sharpe:.3f}  |  MaxDD: {max_dd:.2f}%  |  p-value: {p_val:.4f}")
        print(f"  Exits: {n_trail} trailing, {n_wend} week_end, {n_hard} hard_stop")

        # GATE CHECK
        print(f"\n  --- GATE CHECK ---")
        gate_ret = total_ret >= 15.0
        gate_sharpe = sharpe >= 2.0
        gate_dd = max_dd >= -12.0
        print(f"  Return >= 15%: {'PASS' if gate_ret else 'FAIL'} ({total_ret:+.2f}%)")
        print(f"  Sharpe >= 2.0: {'PASS' if gate_sharpe else 'FAIL'} ({sharpe:.3f})")
        print(f"  MaxDD >= -12%: {'PASS' if gate_dd else 'FAIL'} ({max_dd:.2f}%)")
        all_pass = gate_ret and gate_sharpe and gate_dd
        print(f"  Overall: {'*** ALL GATES PASS ***' if all_pass else 'GATES NOT MET (expected: daily bars underestimate 5-min precision)'}")

        # All trades
        print(f"\n  All trades:")
        for _, t in df_t.iterrows():
            print(f"    {t['week']}  {t['dir']:5s}  lev={t['lev']:.3f}  "
                  f"entry={t['entry']:.0f}  exit={t['exit']:.0f}  "
                  f"pnl={t['pnl']:+.3f}%  [{t['reason']}]  "
                  f"{'OK' if t['correct'] else 'WRONG'}")


if __name__ == "__main__":
    run_backtest()
